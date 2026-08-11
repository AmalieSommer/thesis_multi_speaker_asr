import logging
from .engines import BaseEngine, CT2, OnnxEngine, PytorchEngine, WhisperCPP
import os
from pathlib import Path
from huggingface_hub.utils import validate_repo_id
from huggingface_hub.errors import HFValidationError
from transformers import Wav2Vec2Processor, Wav2Vec2ProcessorWithLM


log = logging.getLogger(__name__)


class ASR:
    def __init__(self, model_type: str, model_name: str, model_path: str, backend: str, device: str = 'cpu', batch_size=4):
        self.batch_size = batch_size
        self.device = device
        self.engine = None

        if not isinstance(model_type, str) or not model_type.strip():
            raise ValueError('Parameter: model_type must be a non-empty string.')
        if not isinstance(model_path, str) or not model_path.strip():
            raise ValueError('Parameter: model_path must be a non-empty string.')
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError('Parameter: model_name must be a non-empty string.')
        if not isinstance(backend, str) or not backend.strip():
            raise ValueError('Parameter: backend must be a non-empty string.')

        path = Path(model_path)
        if path.exists():
            self.model_path = model_path
        else:
            try:
                validate_repo_id(repo_id=model_path)
                self.model_path = model_path
            except HFValidationError as e:
                raise ValueError('Parameter: model_path must be either a valid path to a local directory, or a valid ID to a Huggingface repo.')
            
        self.model_type = model_type
        self.backend = backend
        self.model_name = model_name
        
    
    def load(self, compute_type: str = 'int8', cpu_threads: int = 6) -> None:
        """
        Loads the specified engine type.

        Args:
            compute_type (str): Precision type for computational arithmetic
            cpu_threads (int): Number of CPU threads allowed to be used by the OpenMP library when running the model
        """

        if self.backend == 'ct2':
            self.engine = CT2(
                model_path=self.model_path,
                model_name=self.model_name,
                model_type=self.model_type,
                compute_type=compute_type, 
                cpu_threads=cpu_threads
                )
        elif self.backend == 'onnx':
            self.engine = OnnxEngine(
                model_path=self.model_path,
                model_type=self.model_type,
                model_name=self.model_name,
                device=self.device,
                cpu_threads=cpu_threads,
                compute_type=compute_type,
                )
        elif self.backend == 'torch':
            self.engine = PytorchEngine(
                model_path=self.model_path,
                model_name=self.model_name, 
                model_type=self.model_type,
                cpu_threads=cpu_threads,
                compute_type=compute_type
                )
        elif self.backend == 'cpp':
            self.engine = WhisperCPP(
                model_path=self.model_path,
                model_type=self.model_type,
                device=self.device,
                cpu_threads=cpu_threads,
                compute_type=compute_type
            )
        else:
            self.engine = BaseEngine(
                model_path=self.model_path,
                model_name=self.model_name,
                model_type=self.model_type,
                device=self.device
                )


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
        if self.model_type not in ('seq2seq', 'ctc'):
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

