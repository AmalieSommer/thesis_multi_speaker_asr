from pathlib import Path
import json
import os
from torch.utils.data import Dataset
from datasets import load_dataset, Audio
import soundfile as sf
import librosa
from omegaconf import DictConfig



class Data(Dataset):
    """
    Data wrapper class to load either local or Huggingface datasets. Perform preprocessing, resampling and formatting as preparation for model training and inference.
    """
    def __init__(self, target_sr=16000):
        super().__init__()
        self.target_sr = target_sr


    def load_from_hf(self, config: DictConfig):
        """
        Loads dataset from Huggingface, without decoding the audio files.
        This will avoid any need for FFMPEG installation.
        """
        data = load_dataset(path=config.data.path, name=config.data.name, split=config.data.split, streaming=True)
        data = data.cast_column('audio', Audio(decode=False, sampling_rate=self.target_sr))
        data = data.iter(batch_size=4)
        self.dataset = data


    def load_from_local(self, path):
        """
        Loads a dataset from a local path.
        Assumes the dataset can be loaded to a dataset from a metadata.jsonl file.
        """
        with open(path, 'r') as file:
            self.dataset = [json.loads(line) for line in file]


    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.dataset)


    def __getitem__(self, index: int):
        """Return a given sample from the dataset. (Useful for streaming dataset from HF)"""
        return self.dataset[index]
        


    def delete_dataset(self):
        self.dataset = None

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""

def preprocess(data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    

