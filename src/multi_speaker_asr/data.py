import os
import librosa
import re
import io
from num2words import num2words
from datasets import load_dataset, Audio, Dataset
from torch.utils.data import IterableDataset
from faster_whisper.vad import VadOptions, get_speech_timestamps
import soundfile as sf
from whisperx.schema import SingleSegment
from .utils.utils import profile, LOGGING_CONFIG
from .utils.vad import collect_audio_chunks, get_timestamps
import logging
import logging.config
import datetime
import numpy as np

logging.config.dictConfig(LOGGING_CONFIG)


CWD = os.getcwd()

class AudioData(IterableDataset):
    """
    Data wrapper class to load either local or Huggingface datasets. Perform preprocessing, resampling and formatting as preparation for model training and inference.
    """

    DATA = {
        'coral': {
            'name': 'CoRal-project/coral-v3',
            'type': 'conversation',
            'split': 'test',
            'path': 'root/.cache/huggingface/datasets/CoRal-project___coral-v3/conversation/0.0.0/01f7c93c21fc9dec87fe9f7149c79569cc433f08'
        }
    }
    logger = logging.getLogger(name='AudioData')

    def __init__(
            self, 
            path, 
            audio_path, 
            hpc=True, 
            target_sr=16000, 
            max_segment_duration=30, 
            vad_filter=True, 
            clip_timestamps=False,
            is_asr=True
            ):
        super().__init__()
        self.is_asr=is_asr
        self.audio_path = audio_path
        self.clip_timestamps = clip_timestamps
        self.vad_filter = vad_filter
        self.hpc = hpc
        self.target_sr = target_sr
        self.max_segment_duration = max_segment_duration
        try:
            self.path = self.DATA[path] # path to an online dataset e.g. from Huggingface
        except:
            self.path = path    # path to a local folder
            
        self.load() # Initialize the dataset...
        data_memory = self.load.memory_stats[0]
        self.logger.info('Dataset Memory Stats...: Before load: %f, After load: %f, Delta: %f', data_memory['before'], data_memory['after'], data_memory['delta'])
    

    @profile
    def load(self):
        """
        Loads a dataset either from local or online resource.
        If the path does not exist in the DATA dict, then it is assumed local, and will be loaded manually.
        """
        data_path = self.path

        if type(data_path) == dict:
            # If the path is from an online resource, then load it using datasets built-in function:
            self.ds = load_dataset(
                path=data_path['name'],
                name=data_path['type'],
                split=data_path['split'],
                streaming=True
            )
            # Ensure it does not decode audio using torchDecoder
            self.ds = self.ds.cast_column('audio', Audio(decode=False))
            self.ds = self.ds.rename_column('id_conversation', 'id')
        else:
            # Assumes it is a local data folder:
            self.ds = Dataset.from_csv(path_or_paths=data_path, split='test').to_iterable_dataset()
            self.root_path = self.audio_path if not self.hpc else '/zhome/28/9/151118/thesis/'        
        

    def __iter__(self):
        for item in self.ds:
            yield from self.preprocess(sample=item)
      

    def preprocess(self, sample):
        try:
            audio = sample['audio'] if 'audio' in sample.keys() else None

            if audio == None:
                raise Exception('Missong audio in dataset sample!')
            
            audio = sample['audio']['bytes'] if type(sample['audio']) == dict else sample['audio']
            audio = self.read_audio(audio=audio)

            offset, clip_timestamps, audio = get_timestamps(
                data_sample=sample,
                audio=audio,
                duration=audio.shape[0],
                clip_timestamps=self.clip_timestamps,
                vad_filter=self.vad_filter
            )
            
            yield from collect_audio_chunks(
                data_sample=sample,
                is_asr=self.is_asr,
                audio=audio,
                clip_timestamps=clip_timestamps,
                offset=offset
            )
        except Exception as e:
            self.logger.error('Failed in preprocess() with error... ', e)

    def read_audio(self, audio, target_sr=16000):
        try: 
            if isinstance(audio, bytes):
                audio_ = io.BytesIO(audio)
            else:
                audio_ = self.root_path + audio
            wav, sr = sf.read(audio_, dtype='float32')
            
            # Check for multiple channels and convert to mono
            if wav.ndim > 1:
                wav = wav.mean(axis=1)

            if sr != target_sr:
                wav = librosa.resample(
                    wav,
                    orig_sr=sr,
                    target_sr=target_sr,
                )
                wav = wav.astype("float32")
            return wav
        except Exception as e:
            print('Failed with exception: ', e)

    
    def collator_fn(self, batch):
        """Should ensure that it returns batch object of the same format, i.e. same parameter names and types"""
        return batch

def cast(object: dict):
    return SingleSegment(
        start=object['start'],
        end=object['end'],
        text=object['text'],
        avg_logprob=object['avg_logprob']
    )


def stream_audio(audio, sr: int = 16000, block_length_sec=30):
        
        frame_size = (2048 * sr) // 16000
        hop_length = (1024 * sr) // 16000

        block_length = int(block_length_sec * sr) // hop_length

        stream = librosa.stream(
                    path=audio,
                    block_length=block_length,
                    frame_length=frame_size,
                    hop_length=hop_length
                )
        # Returns a Generator object that when called in a loop will yield one item at a time
        return stream


def clean_transcription(sentence: str):
    """
    Function to preprocess the ground truth and predicted transcripts before computing the performance using WER, CER etc...
    Should standardize the text to lowercase, no punctuations or special characters.
    It should also map all occurrences of numbers to textual representations using library function.
    """
    sentence = str.lower(sentence)
    sentence = re.sub(r'-(?!\d)', '', sentence)             # Remove - that are not followed by a number
    sentence = re.sub(r'(?<!\d)\.|\.?(?!\d)', '', sentence) # Remove . that are not enclosed by two numbers
    sentence = re.sub(r'[^\w\s.-]', '', sentence)           # Remove all punctuation except for the - and .
    sentence = re.sub(' +', ' ', sentence)                  # Replacing all duplicate spaces with single space.
    
    sentence_copy = str(sentence)

    for s in sentence.split():
        try: 
            num = float(s)
            word_rep = str(num2words(number=num))
            sentence_copy = sentence_copy.replace(s, word_rep)
        except ValueError as e:
            continue

    return sentence_copy    

