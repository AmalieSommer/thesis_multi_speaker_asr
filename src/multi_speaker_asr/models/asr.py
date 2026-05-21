import torch
from omegaconf import DictConfig
import pickle
import os
import torch.nn as nn
from faster_whisper import WhisperModel, BatchedInferencePipeline
from whisperx import load_model


MODEL_PATH = "src\\saved_models\\asr"

class Whisper(nn.Module):
    """
    A wrapper class for the ASR models.

    If not already generated, it loads using Faster-Whisper.
    Otherwise, it will load model from local path.
    """

    def __init__(self):
        self.model = None
        self.name = None
        self.compute_type = None
        self.device = 'cpu'

    def save(self):
        """Will save the compressed models locally to folder; \\saved_models\\asr"""
        if self.model:
            filename = f'{self.name}.pkl'
            with open(filename, 'wb') as file:
                pickle.dump(filename, file)

    def load(self, config: DictConfig):
        """To be called when wanting to instantiate the model"""
        self.name = config.asr.name
        self.compute_type = config.asr.compute_type
        self.device = config.asr.device
        
        cwdir = os.listdir(os.path.join(os.getcwd(), MODEL_PATH))
        if self.name in cwdir:
            self.model = pickle.load(open(self.name, 'rb')).to(self.device)
        else:
            model = WhisperModel(
                model_size_or_path=self.name, 
                device=self.device, 
                compute_type=self.compute_type,
                cpu_threads=6)
            self.model = BatchedInferencePipeline(model=model)
      