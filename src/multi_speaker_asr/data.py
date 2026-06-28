import os
import librosa
import re
import io
from pathlib import Path
from num2words import num2words
from datasets import load_dataset, Audio, Dataset
from torch.utils.data import IterableDataset, DataLoader
from faster_whisper.vad import collect_chunks, VadOptions, get_speech_timestamps
import tracemalloc
import numpy as np
import soundfile as sf
from whisperx.schema import SingleSegment
from faster_whisper.transcribe import Segment
from .utils.utils import profile, LOGGING_CONFIG, process_memory
import logging
import logging.config

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

    def __init__(self, path, segmented=True, hpc=True, target_sr=16000, max_segment_duration=30):
        super().__init__()
        self.segmented = segmented
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
            self.root_path = '/root/master_thesis/' if not self.hpc else '/zhome/28/9/151118/thesis/'
            self.len_estimate = len(os.listdir(path=self.root_path + 'thesis_multi_speaker_asr/data/coral-v3-long-form-conversations'))
        
        

    def __iter__(self):
        for item in self.ds:
            if ('start' not in item.keys()) | ('end' not in item.keys()):
                timestamps = None
            else:
                timestamps = {
                    'start': item['start'],
                    'end': item['end']
                }
            
            if 'path' in item.keys():
                audio = item['path']
            else:
                audio = item['audio']['bytes']

            yield {
                'id': item['id'],
                'timestamp': timestamps,
                'text': item['text'] if 'text' in item.keys() else None,
                'audio': audio
            }


    def read_audio(self, audio, target_sr=16000):
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

    
    def collator_fn(self, batch):
        """Should ensure that it returns batch object of the same format, i.e. same parameter names and types"""

        for sample in batch:
            wav = self.read_audio(sample['audio'])
            sample['audio'] = wav

            start, end = None, None
            if sample['timestamp']  == None:
                start, end = 0, wav.shape[0]
                sample['timestamp'] = {
                    'start': start,
                    'end': end
                }
        
        return batch



def chunk_batch(batch: list[dict], max_duration=(30 * 16000), sr=16000):
    """
    Takes a list of data samples (dict objects) and if the samples are less than 30 seconds, it will combine them
    until it reaches a chunk of size 30 seconds, and create a new chunk to fill until all are chunked.
    If the samples are greater than 30 seconds it will split them using Silero VAD and combine them to chunks 
    of size 30 seconds.

    It will run VAD on all audio segments longer than 5 seconds, but not less in order to avoid the risk of removing
    the entire audio segment.
    """
    clip_timestamps = []
    audio_chunks, chunks_metadata = [], []
    for sample in batch:
        audio = sample['audio']
        if audio.shape[0] > max_duration:
            vad_parameters = VadOptions(
                        max_speech_duration_s=30,
                        min_silence_duration_ms=160,
                    )
            clip_timestamps = get_speech_timestamps(audio, vad_parameters)
            audio_chunks, chunks_metadata = collect_chunks(
                audio=audio, 
                chunks=clip_timestamps,
                max_duration=30
                )
            
        else:
            clip_timestamps = clip_timestamps + [{'start': 0, 'end': audio.shape[0]}]
            audio_chunk, chunk_metadata = collect_chunks(audio=audio, chunks=clip_timestamps)
            audio_chunks = audio_chunks + audio_chunk
            chunks_metadata = chunks_metadata + chunk_metadata

    return audio_chunks, chunks_metadata, clip_timestamps


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

