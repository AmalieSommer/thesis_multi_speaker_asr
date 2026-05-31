#import whisperx
#from omegaconf import DictConfig

class Wav2Vec2:
    """
    Wrapper module class to load the phoneme model to use for timestamp alignment.

    Allows for saving and loading the model from local.
    If not saved local, it will load from Huggingface using WhisperX.
    """
    def __init__(self, device='cpu'):
        self.model = None
        self.metadata = None
        self.device = device


    def load(self, config):
        self.model, self.metadata = whisperx.load_align_model(
            language_code='da',
            model_name=config.alignment.name, 
            device=self.device
            )


    def unload(self):
        self.model = None
        self.metadata = None