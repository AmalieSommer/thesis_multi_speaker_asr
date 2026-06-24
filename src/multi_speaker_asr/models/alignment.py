from transformers import pipeline
from whisperx.alignment import load_align_model, align
from ..utils.utils import profile, LOGGING_CONFIG
import logging
import logging.config

logging.config.dictConfig(LOGGING_CONFIG)

class Wav2Vec2:
    """
    Wrapper module class to load the phoneme model to use for timestamp alignment.

    Allows for saving and loading the model from local.
    If not saved local, it will load from Huggingface using WhisperX.
    """
    logger = logging.getLogger(name='Wav2Vec2')

    def __init__(self, config, device='cpu'):
        self.metadata = None
        self.device = device

        self.load(config=config)
        model_memory = self.load.memory_stats[0]
        self.logger.info('Alignment Model Memory Stats...: Before load: %f, After load: %f, Delta: %f', model_memory['before'], model_memory['after'], model_memory['delta'])

    @profile
    def load(self, config):
        self.pipeline, self.metadata = load_align_model(
            language_code='da',
            device=self.device,
            model_name=config['alignment_model']
        )

    def run_alignment(self, transcript, audio):
        return align(
            transcript=transcript,
            model=self.pipeline,
            align_model_metadata=self.metadata,
            audio=audio,
            device=self.device,
            print_progress=True
        )


    def run_pipeline(self, input, chunk_length: int = 10, stride: int = 2):
        """When running inference, the model needs context in order to produce good results. Using chunking with strides on both sides to improve performance"""
        output = self.pipeline(
            inputs=input,
            chunk_length_s=chunk_length,
            stride_length_s=stride
        )
        print(output)
        return output

    def unload(self):
        self.model = None
        self.metadata = None