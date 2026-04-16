from pathlib import Path
import typer
import json
import os
from torch.utils.data import Dataset
from datasets import load_dataset
import torchaudio

class Data(Dataset):
    """
    Data wrapper class to load either local or Huggingface datasets. Perform preprocessing, resampling and formatting as preparation for model training and inference.
    """
    def __init__(self, data_path, metadata, target_sr=16000):
        self.data_path = data_path
        self.metadata = metadata
        self.target_sr = target_sr

        # Load data into memory
        self.load()


    def load(self):
        """
        Loading data from either local path or Huggingface.
        """
        self.datasamples = []
        with open(self.metadata, "r") as file:
            for line in file:
                self.datasamples.append(json.loads(line))


    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.datasamples)

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        sample = self.datasamples[index]

        audio_path = os.path.join(self.data_path, sample["audio_filepath"])
        wav, sr = torchaudio.load(audio_path)

        if sr != self.target_sr:
            wav = torchaudio.functional.resample(waveform=wav, orig_freq=sr, new_freq=self.target_sr)

        return {
            "audio": wav,
            "text": sample["text"],
            "sampling_rate": self.target_sr,
            "audio_path": audio_path
        }


    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""

def preprocess(data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    

