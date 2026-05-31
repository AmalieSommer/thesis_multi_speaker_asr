from pathlib import Path
from torch.utils.data import Dataset
import os
import pandas as pd
import librosa
from pathlib import Path

class Data(Dataset):
    """
    Data wrapper class to load either local or Huggingface datasets. Perform preprocessing, resampling and formatting as preparation for model training and inference.
    """
    def __init__(self, path, target_sr=16000):
        super().__init__()
        self.target_sr = target_sr
        self.path = path
        self.df = None


    def load(self):
        """
        Loads a dataset from a local path.
        Assumes the dataset can be loaded to a dataset from a metadata.jsonl file.
        """
        self.df = pd.read_csv(self.path)
        print(self.df.shape)

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.df)


    def __getitem__(self, index: int):
        """Return a given sample from the dataset. (Useful for streaming dataset from HF)"""
        row = self.df.iloc[index]
        audio_path = row['path']
        path = Path(os.path.join(os.getcwd(), 'data/en/clips', audio_path))
        try:
            if path.exists():
                waveform, sr = librosa.load(path=path, sr=self.target_sr)
            else:
                raise Exception('Audio path does not exist.')
        except Exception as e:
            print(f'Failed with error: {e}')
        return {
            'uuid': row['uuid'],
            'audio_path': audio_path,
            'audio': waveform,
            'samplerate': sr,
            'transcription': row['sentence'],
            'client_id': row['client_id'],
            'sentence_id': row['sentence_id'],
            'age': row['age'],
            'gender': row['gender'],
            'accent': row['accents'],
            'duration': row['duration[ms]']
        }

        
    def delete_dataset(self):
        self.df = None

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""
        #TODO: Should iterate through the metadata file and ensure type consistency, e.g. column names and value types to avoid errors...



def preprocess(data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    

