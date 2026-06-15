from faster_whisper import WhisperModel
from faster_whisper import BatchedInferencePipeline
from transformers import pipeline


class Whisper:
    """
    A wrapper class for the ASR models.

    If not already generated, it loads using Faster-Whisper.
    Otherwise, it will load model from local path.
    """

    MODEL = {
        'roest-whisper': 'CoRal-project/roest-v3-whisper-1.5b',
        'roest-wav2vec2': 'CoRal-project/roest-v3-wav2vec2-315m'
    }

    def __init__(self, device='cpu'):
        self.model = None
        self.device = device


    def load_model(self, name):
        print(f'Loading model {name}...')
        model_name = self.MODEL[name]
        self.pipeline = pipeline("automatic-speech-recognition", model=model_name)


    def load(self, model_size, compute_type, cpu_threads):
        """To be called when wanting to instantiate the model"""
        print(f'Loading model {model_size}...')
        whisper_model = WhisperModel(
            model_size_or_path=model_size,
            device=self.device,
            compute_type=compute_type,
            cpu_threads=int(cpu_threads),
            num_workers=1
        )
        self.model = BatchedInferencePipeline(
            model=whisper_model
        )
      
    def unload(self):
        self.model = None