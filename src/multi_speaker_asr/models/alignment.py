import torch
import whisperx
from omegaconf import DictConfig
import pickle
import os
import torch.nn as nn

class Wav2Vec2(nn.Module):
    """
    Wrapper module class to load the phoneme model to use for timestamp alignment.

    Allows for saving and loading the model from local.
    If not saved local, it will load from Huggingface using WhisperX.
    """
    def __init__(self):
        super().__init__()

    def load(self, config: DictConfig):
        self.model_name = config.alignment.name
        self.device = config.alignment.device
        align_model = whisperx.load_align_model(
            language_code='da',
            model_name=self.model_name, 
            device=self.device
            )
        self.model = align_model[0]
        self.metadata = align_model[1]