from omegaconf import DictConfig
from whisperx.diarize import DiarizationPipeline
from whisperx.diarize import assign_word_speakers
import pandas as pd

class Diarize:
    def __init__(self, device='cpu'):
        self.device = device
        self.model = None

    
    def load(self, config: DictConfig):
        self.model = DiarizationPipeline(
            model_name=config.name,
            device=self.device,
            use_auth_token=config.hf_token
        )

    def assign_wordlevel_speakers(self, diarize_segments, transcript):
        return assign_word_speakers(
            diarize_df=diarize_segments,
            transcript_result=transcript,
            speaker_embeddings=True
        )
    

    def unload(self):
        self.model = None
        
