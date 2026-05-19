import torch
import whisperx
from omegaconf import DictConfig
import pickle
import os
import torch.nn as nn

MODEL_PATH = "src\\saved_models\\alignment"


class Wav2Vec2(nn.Module):
    """
    Wrapper module class to load the phoneme model to use for timestamp alignment.

    Allows for saving and loading the model from local.
    If not saved local, it will load from Huggingface using WhisperX.
    """
    def __init__(self, config: DictConfig = None):
        super().__init__()

        alignment = config.alignment
        if alignment:
            self.model_name = alignment.name
            self.device = alignment.device
        


    def save(self):
        filename = f'{self.model_name}.pkl'
        path = os.path.join(os.getcwd(), MODEL_PATH)
        with open(path, 'wb') as file:
            pickle.dump(filename, file)


    def load(self):
        cwdir = os.listdir(os.path.join(os.getcwd(), MODEL_PATH))
        if self.model_name in cwdir:
            self.model = pickle.load(self.model_name)
        else:
            self.model = whisperx.load_align_model(self.model_name)