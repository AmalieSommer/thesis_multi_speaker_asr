from pathlib import Path

import typer
from torch.utils.data import Dataset
from datasets import load_dataset

class Data(Dataset):
    """
    Data wrapper class to load either local or Huggingface datasets. Perform preprocessing, resampling and formatting as preparation for model training and inference.
    """
    DATASET_DICT = {
        "emotale": "data/emotale_wav"
    }

    def __init__(self, dataset_id: str = "emotale") -> None:

        if isinstance(dataset_id, str):
            self.dataset_id = dataset_id
        else:
            raise ValueError(f"Incorrect data path type, {type(data_path)}, is passed.")

        if dataset_id not in self.DATASET_DICT:
            raise ValueError(f"Unknown data path is passed.")
                
        self.data_path = self.DATASET_DICT[dataset_id]
        

    def load_data(self):
        """
        Loading data from either local path or Huggingface.
        """
        dataset = load_dataset(self.data_path)
        return dataset


    def __len__(self) -> int:
        """Return the length of the dataset."""

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""

def preprocess(data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    

