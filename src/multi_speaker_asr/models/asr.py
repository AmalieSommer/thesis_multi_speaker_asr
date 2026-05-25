import torch
from omegaconf import DictConfig
import pickle
import os
import torch.nn as nn
#from faster_whisper import WhisperModel
from whisperx.asr import WhisperModel
from whisperx.asr import load_model


class Whisper:
    """
    A wrapper class for the ASR models.

    If not already generated, it loads using Faster-Whisper.
    Otherwise, it will load model from local path.
    """

    def __init__(self, device='cpu'):
        self.model = None
        self.device = device
        self.threads = 3

    def load(self, config: DictConfig):
        """To be called when wanting to instantiate the model"""
        self.model = load_model(
            whisper_arch=config.asr.name,
            language='da',
            device=self.device,
            compute_type=config.asr.compute_type,
            threads=self.threads
        )
      
    def unload(self):
        self.model = None