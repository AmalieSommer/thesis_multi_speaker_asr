import logging
from .engines import BaseEngine, CT2, OnnxEngine, PytorchEngine, WhisperCPP
import os
from pathlib import Path
from huggingface_hub.utils import validate_repo_id
from huggingface_hub.errors import HFValidationError
from transformers import Wav2Vec2Processor, Wav2Vec2ProcessorWithLM
from multi_speaker_asr.utils.logging_config import get_logger

log = get_logger(__name__)

class ASR:
    def __init__(self, config: dict):
        self.engine = None
        self.pipeline_config = config

        if not isinstance(config['model_type'], str) or len(config['model_type'].strip()) == 0:
            raise ValueError('Parameter: model_type must be a non-empty string.')
        if not isinstance(config['model_path'], str) or len(config['model_path'].strip()) == 0:
            raise ValueError('Parameter: model_path must be a non-empty string.')
        if not isinstance(config['model_name'], str) or len(config['model_name'].strip()) == 0:
            raise ValueError('Parameter: model_name must be a non-empty string.')

        path = Path(config['model_path'])
        if not path.exists():
            try:
                validate_repo_id(repo_id=config['model_path'])
            except HFValidationError as e:
                raise ValueError('Parameter: model_path must be either a valid path to a local directory, or a valid ID to a Huggingface repo.')
        
    
    def load(self, backend: str) -> None:
        """
        Loads the specified engine type.

        Args:
            compute_type (str): Precision type for computational arithmetic
            cpu_threads (int): Number of CPU threads allowed to be used by the OpenMP library when running the model
        """
        if backend == 'ct2':
            self.engine = CT2(config=self.pipeline_config)
        elif backend == 'onnx':
            self.engine = OnnxEngine(config=self.pipeline_config)
        elif backend == 'torch':
            self.engine = PytorchEngine(config=self.pipeline_config)
        elif backend == 'cpp':
            self.engine = WhisperCPP(config=self.pipeline_config)
        else:
            self.engine = BaseEngine(**self.pipeline_config)


    def _get_engine(self):
        return self.engine
        

    def transcribe(self, audio_batch, return_timestamps: bool) -> list:
        """
        A catch-all function for calling the engine's transcribe function

        Args:
            audio_batch (ndarray, list, tuple): The audio arrays to process
            language (str): Transcription language to use if the model is multilingual
        
        Returns:
            list: The transcribed speech segments
        """
        if not isinstance(audio_batch, (list, tuple)):
            audio_batch = [audio_batch]
        if self.engine.model_type not in ('seq2seq', 'ctc'):
            raise ValueError('Unknown model_type...')

        return self.engine.transcribe(audio_batch, return_timestamps=return_timestamps)



    def save_model(self, path_dir: str) -> None:
        """
        Function to call for saving a transformer model to a local directory
        
        Args:
            path_dir (str): String representation of a path to a local directory
        """
        path_dir = Path(path_dir)
        if 'model' in os.listdir(path_dir):
            self.engine.model.config.save_pretrained(path_dir)
        else:
            self.engine.model.save_pretrained(path_dir)

    
    def save_processor(self, path_dir: str) -> None:
        """
        Function to call for saving a transformer model to a local directory

        Note, make sure to save the model and processor to the same folder!
        Args: 
            path_dir (str): String representation of a path to a local directory
        """
        # Make sure to check if the processor is the Wav2Vec2ProcessorWithLM, and if so, it should downgrade, since it does not 
        # currently support an added LM, but only decodes with CTC beam-search!
        if type(self.engine.processor) is Wav2Vec2ProcessorWithLM:
            downgraded_processor = Wav2Vec2Processor(
                feature_extractor=self.engine.processor.feature_extractor,
                tokenizer=self.engine.processor.tokenizer
            )
            downgraded_processor.save_pretrained(path_dir)
        else:
            self.engine.processor.save_pretrained(path_dir)

