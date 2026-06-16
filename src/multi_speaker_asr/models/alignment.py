from transformers import pipeline

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
        self.pipeline = pipeline(
            task='automatic-speech-recognition',
            model=config['model'],
            device=self.device
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