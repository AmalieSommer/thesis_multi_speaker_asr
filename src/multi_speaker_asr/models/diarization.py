import pandas as pd
from pyannote.audio import Pipeline


MODEL = {
    'py3': 'pyannote/speaker-diarization-3.0',
    'py3.1': 'pyannote/speaker-diarization-3.1',
    'custom': '' # TODO: implement custom option based on code in evaluate.py
}


class Diarize:
    def __init__(self, device='cpu', pipeline='py3'):
        self.device = device
        self.model = None
        self.model_path = MODEL[pipeline]

    
    def load(self, token):
        self.model = Pipeline.from_pretrained(
            self.model_path, 
            use_auth_token=token
            )
        self.model.embedding_batch_size = 1

    # TODO: Implement speaker assignment given the asr and diarization output...
    def assign_wordlevel_speakers(self):
        return None

    def unload(self):
        self.model = None
        
