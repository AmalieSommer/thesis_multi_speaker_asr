from faster_whisper import WhisperModel
from faster_whisper import BatchedInferencePipeline
from transformers import pipeline
from ..utils.utils import profile

class ASR:
    """
    Wrapper class for ASR models.

    It will contain the Whisper and Wav2Vec2 models, which will inherit basic functions such as unload()
    """
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.pipeline = None


    def unload(self):
        self.model = None



class Whisper(ASR):
    def __init__(self, compute_type, cpu_threads, device='cpu', model='CoRal-project/roest-v3-whisper-1.5b'):
        super().__init__(model, device)
        self.load(
            compute_type=compute_type,
            cpu_threads=cpu_threads
        )
        self.model_memory = self.load.memory_stats[0]


    @profile
    def load(self, compute_type, cpu_threads):

        print(f'Loading model...')

        whisper_model = WhisperModel(
            model_size_or_path=self.model,
            device=self.device,
            compute_type=compute_type,
            cpu_threads=int(cpu_threads),
            num_workers=1
        )
        self.pipeline = BatchedInferencePipeline(
            model=whisper_model
        )



class Wav2Vec2(ASR):
    def __init__(self, model='CoRal-project/roest-v3-wav2vec2-315m', device='cpu'):
        super().__init__(model, device)

    def load(self):
        self.pipeline = pipeline(
            task='automatic-speech-recognition',
            model=self.model,
            device=self.device
        )

