import os
import pandas as pd
import librosa
import re
import io
from num2words import num2words
from memory_profiler import profile
from datasets import load_dataset, Audio, Dataset
from torch.utils.data import IterableDataset
from faster_whisper.vad import collect_chunks, VadOptions, get_speech_timestamps
import numpy as np



CWD = os.getcwd()

class AudioData(IterableDataset):
    """
    Data wrapper class to load either local or Huggingface datasets. Perform preprocessing, resampling and formatting as preparation for model training and inference.
    """

    DATA = {
        'coral': {
            'name': 'CoRal-project/coral-v3',
            'type': 'conversation',
            'split': 'test'
        }
    }

    def __init__(self, path, target_sr=16000, max_segment_duration=30):
        super().__init__()
        self.target_sr = target_sr
        self.max_segment_duration = max_segment_duration
        try:
            self.path = self.DATA[path] # path to an online dataset e.g. from Huggingface
        except:
            self.path = path    # path to a local folder

    def load(self):
        """
        Loads a dataset either from local or online resource.
        If the path does not exist in the DATA dict, then it is assumed local, and will be loaded manually.
        """
        data_path = self.path

        if data_path in self.DATA.keys():
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
        
    def __iter__(self):
        print(self.ds.features)
        for item in self.ds:
            # Return the bytes instead of the whole loaded audio array:
            yield {
                'id': item['id'],
                'path': item['path'] if 'audio' not in item.keys() else item['audio']['path'],
                'start': item['start'],
                'end': item['end']
            }


    #@profile
    def preprocess(self) -> None:
        """Preprocess the raw data and save it to the output folder."""
        self.df = self.df.dropna(subset=['path']) # Removing any rows missing wav files or transcriptions
        

def stream_audio(audio, sr: int = 16000):
        frame_size = (2048 * sr) // 22050
        hop_length = (512 * sr) // 22050

        stream = librosa.stream(
                path=audio,
                block_length=128,
                frame_length=frame_size,
                hop_length=hop_length
            )
        
        return stream # Returns a Generator object that when called in a loop will yield one item at a time


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



"""        
        item = self.df.iloc[index]
        print(f'Get Item: {item}')

        audio_path = os.path.join(self.audio_path, item['path'])
        wav, sr = librosa.load(path=audio_path, sr=self.target_sr)

        # Check the shape of the array
        if wav.ndim == 1:
            print("This is a mono file.")
        elif wav.ndim == 2:
            print(f"This is a stereo file with {wav.shape[0]} channels.")

        duration = librosa.get_duration(y=wav, sr=sr)
        return {
            'id': item['meeting_id'],
            'audio': wav,
            'sr': sr,
            'duration': duration,
            'path': item['path']
        }   
""" 