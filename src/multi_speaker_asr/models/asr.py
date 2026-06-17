from faster_whisper import WhisperModel
from faster_whisper import BatchedInferencePipeline
from transformers import pipeline
from memory_profiler import profile



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
        self.load()

    def load(self):
        self.pipeline = pipeline(
            task='automatic-speech-recognition',
            model=self.model,
            device=self.device
        )



    def run_pipeline(self, input, chunk_length: int = 10, stride: int = 2):
        """When running inference, the model needs context in order to produce good results. Using chunking with strides on both sides to improve performance"""
        output = self.pipeline(
            inputs=input,
            chunk_length_s=chunk_length,
            stride_length_s=stride
        )
        return output





"""
class Whisper:

    def __init__(self, device='cpu'):
        self.model = None
        self.device = device

    @profile
    def load(self, model_size, compute_type, cpu_threads):
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

class Wav2Vec2:
    def __init__(self, device='cpu', model='CoRal-project/roest-v3-wav2vec2-315m'):
        self.model = model
        self.device = device


    def load(self, config):
        self.pipeline = pipeline(
            task='automatic-speech-recognition',
            model=self.model,
            device=self.device
        )

    def run_pipeline(self, input, chunk_length: int = 10, stride: int = 2):
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
"""