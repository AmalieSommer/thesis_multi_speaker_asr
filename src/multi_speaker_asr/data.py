from pathlib import Path
from torch.utils.data import Dataset
import os
import pandas as pd
import librosa
from pathlib import Path
from pandas import DataFrame
import uuid
import re
from num2words import num2words


CWD = os.getcwd()
DATA_PATH = {
    'coraal': {
        'metadata': f'{CWD}/data/CORAAL/metadata.csv',
        'audio': f'{CWD}/data/CORAAL/wav'
    },
    'cv': {
        'metadata': f'{CWD}/data/cv/metadata.csv',
        'audio': f'{CWD}/data/cv/wav'
    },
    'amicorpus': {
        'metadata': f'{CWD}/data/amicorpus/metadata.csv',
        'audio': f'{CWD}/data/amicorpus/test_split'
    }
}


class Data(Dataset):
    """
    Data wrapper class to load either local or Huggingface datasets. Perform preprocessing, resampling and formatting as preparation for model training and inference.
    """
    def __init__(self, path, target_sr=16000):
        super().__init__()
        self.target_sr = target_sr

        datapath = DATA_PATH[path]
        self.path = datapath
        self.df = None


    def load(self):
        """
        Loads a dataset from a local path.
        Assumes the dataset can be loaded to a dataset from a metadata.jsonl file.
        """
        self.df = pd.read_csv(self.path['metadata'])
        self.preprocess()
        print(self.df.shape)

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.df)


    def __getitem__(self, index: int):
        """Return a given sample from the dataset. (Useful for streaming dataset from HF)"""
        row = self.df.iloc[index]
        return row     
    
        
    def delete_dataset(self):
        self.df = None


    def preprocess(self) -> None:
        """Preprocess the raw data and save it to the output folder."""
        #TODO: Should iterate through the metadata file and ensure type consistency, e.g. column names and value types to avoid errors...
        df = self.df.dropna(subset=['text', 'path']) # Removing any rows missing wav files or transcriptions
        columns = df.columns
        if 'id' not in columns:
            # Add a unique identifier (UUID)
            df['id'] = [str(uuid.uuid4()) for _ in range(len(df))]
        
        # Decode audio path to an Audio object containing audio array, sample rate and audio path
        audio_arr = []
        for i, row in df.iterrows():
            audio_path = row['path']
            path = Path(os.path.join(self.path['audio'], audio_path))
            try:
                if path.exists():
                    waveform, sr = librosa.load(path=path, sr=self.target_sr)
                    duration = librosa.get_duration(y=waveform, sr=sr)
                    item = {
                        'array': waveform,
                        'samplerate': sr,
                        'path': path,
                        'duration': duration
                    }
                    audio_arr.append(item)
                else:
                    raise Exception(f'Audio path, {path}, does not exist.')
            except Exception as e:
                print(f'Failed with error: {e}')
        df['audio'] = audio_arr
        
        # Set the modified dataframe as class param:
        self.df = df


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
