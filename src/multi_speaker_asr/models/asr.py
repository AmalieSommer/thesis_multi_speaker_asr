from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration, Pipeline
import torch
import whisperx
from omegaconf import DictConfig
import pickle
import os
import torch.nn as nn


MODEL_PATH = "src\\saved_models\\asr"

class WhisperPipeline(nn.Module):
    """
    A wrapper class for the ASR models.

    Loaded using WhisperX (Faster-Whisper). If not already generated, it loads using WhisperX.
    Otherwise, it will load model from local path.
    """

    def __init__(self, model: str = "test", device: str = "cpu", config: DictConfig = None):
        if isinstance(device, torch.device):
            self.device = device
        elif isinstance(device, str):
            self.device = torch.device(device)

        # Check if a local model is specified in config file:
        self.name = config.asr.name
        path = os.path.join(os.getcwd(), MODEL_PATH)
        if self.name in os.listdir(path):
            self.model = pickle.load(open(path, 'rb')) # load from local
        else:
            self.model = whisperx.load_model(self.name, device, compute_type=config.asr.compute_type)


    def save(self):
        """Will save the compressed models locally to folder; \\saved_models\\asr"""
        if self.model:
            filename = f'{self.name}.pkl'
            with open(filename, 'wb') as file:
                pickle.dump(filename, file)

    
    def load(self):
        cwdir = os.listdir(os.path.join(os.getcwd(), MODEL_PATH))
        if self.name in cwdir:
            self.model = pickle.load(self.name)
        else:
            self.model = whisperx.load_align_model(self.name)



    def load_pipeline(self):
        """
        Loads the pipeline, specified by the passed model selection, from Huggingface transformers.
        """
        processor = WhisperProcessor.from_pretrained(self.model_id)
        model = WhisperForConditionalGeneration.from_pretrained(self.model_id).to(self.device)

        result = pipeline(
            "automatic-speech-recognition",
            model=model,
            chunk_length_s=30,
            device=self.device,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor
        )
        return result

