from pathlib import Path
import json
import os
from torch.utils.data import Dataset
from datasets import load_dataset
import torchaudio

class Data(Dataset):
    """
    Data wrapper class to load either local or Huggingface datasets. Perform preprocessing, resampling and formatting as preparation for model training and inference.
    """
    def __init__(self, local_data=True, data_path=None, metadata=None, target_sr=16000):
        super().__init__()
        self.data_path = data_path
        self.metadata = metadata
        self.target_sr = target_sr

        # Load data into memory
        if local_data:
            self.load()


    def load_hf(self, name, configuration, split):
        self.datasamples = load_dataset(
            path=name,
            name=configuration,
            split=split,
            decode=False
        )


    def load(self):
        """
        Loading data from either local path or Huggingface.
        """
        self.datasamples = []
        metadata_path = os.path.join(self.data_path, self.metadata)
        with open(metadata_path, "r") as file:
            for line in file:
                self.datasamples.append(json.loads(line))


    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.datasamples)


    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        sample = self.datasamples[index]

        if self.data_path == None:
            audio = sample["audio"]
            wav = audio["array"]
            sr = audio["sampling_rate"]
        else:
            audio_path = os.path.join(self.data_path, sample["audio_filepath"])
            wav, sr = torchaudio.load(audio_path)

        if sr != self.target_sr:
            wav = torchaudio.functional.resample(waveform=wav, orig_freq=sr, new_freq=self.target_sr)
        
        sample_id = [value for key, value in sample.items() if "id" in key][0]


        return {
            "audio": wav,
            "text": sample["text"],
            "sampling_rate": self.target_sr,
            "audio_path": audio_path,
            "id": sample_id
        }


    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""

def preprocess(data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    

