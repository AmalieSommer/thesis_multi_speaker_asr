from torch.utils.data import Dataset
import os
import pandas as pd
import librosa
import re
from num2words import num2words
from memory_profiler import profile


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


class AudioData(Dataset):
    """
    Data wrapper class to load either local or Huggingface datasets. Perform preprocessing, resampling and formatting as preparation for model training and inference.
    """
    def __init__(self, path, target_sr=16000):
        super().__init__()
        self.target_sr = target_sr
        datapath = DATA_PATH[path]
        self.audio_path = datapath['audio']
        self.df = pd.read_csv(datapath['metadata'])
        self.preprocess()

    def __len__(self) -> int:
        """Return the length of the audio dataset."""
        return len(self.df['path'].unique())


    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        
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
    
        
    def delete_dataset(self):
        self.df = None


    #@profile
    def preprocess(self) -> None:
        """Preprocess the raw data and save it to the output folder."""
        self.df = self.df.dropna(subset=['path']) # Removing any rows missing wav files or transcriptions
        


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
