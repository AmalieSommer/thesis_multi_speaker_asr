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
    def __init__(self, asr_cfg, engine_cfg):
        if not isinstance(asr_cfg['type'], str) or len(asr_cfg['type'].strip()) == 0:
            raise ValueError('Parameter: model_type must be a non-empty string.')
        if not isinstance(asr_cfg['name'], str) or len(asr_cfg['name'].strip()) == 0:
            raise ValueError('Parameter: model_name must be a non-empty string.')
        self.name = asr_cfg['name']
        self.type = asr_cfg['type']
        self.sr = asr_cfg['target_sr']
        self.engine = self.load_engine(
            config=engine_cfg
        )

    
    def load_engine(self, config) -> None:
        """
        Loads the specified engine type.

        Args:
            compute_type (str): Precision type for computational arithmetic
            cpu_threads (int): Number of CPU threads allowed to be used by the OpenMP library when running the model
        """
        backend = config['name']
        if backend == 'ct2':
            self.engine = CT2(self.name, self.type, config)
        elif backend == 'onnx':
            self.engine = OnnxEngine(self.name, self.type, config)
        elif backend == 'torch':
            self.engine = PytorchEngine(self.name, self.type, config)
        elif backend == 'cpp':
            self.engine = WhisperCPP(self.name, self.type, config)
        else:
            self.engine = BaseEngine(self.name, self.type, config)


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

        output = self.engine.transcribe(audio_batch)
        return self.engine.format_output(
            output=output,
            audio=audio_batch,
            return_timestamps=return_timestamps,
            samplerate=self.sr
        )
        #return self.engine.transcribe(audio_batch, return_timestamps=return_timestamps)



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

