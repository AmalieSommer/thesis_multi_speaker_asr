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

    def __init__(self, model_name, device='cpu'):
        self.metadata = None
        self.device = device

        self.load(model=model_name)
        model_memory = self.load.memory_stats[0]
        self.logger.info('Alignment Model Memory Stats...: Before load: %f, After load: %f, Delta: %f', model_memory['before'], model_memory['after'], model_memory['delta'])

    @profile
    def load(self, model):
        self.pipeline, self.metadata = load_align_model(
            language_code='da',
            device=self.device,
            model_name=model
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
        return output
    

    def find_first_idx(self, words, target_start, offset):
        left = 0
        right = len(words) - 1
        res = -1

        while left <= right:
            mid = (left + right) // 2

            word_start = words[mid]
            if word_start >= target_start:
                res = mid
                right = mid - 1
            else:
                left = mid + 1

        return res
    
    def find_last_idx(self, words, target_start, offset):
        left = 0
        right = len(words) - 1
        res = -1

        while left <= right:
            mid = (left + right) // 2

            word_start = words[mid] 
            if word_start <= target_start:
                res = mid
                left = mid + 1
            else:
                right = mid - 1

        return res
    
    def get_chunk_generator(self, words, chunk_offset=0, chunk_size=10, segment_duration=None):

        if segment_duration is None:
            if words:
                segment_duration = words[-1]['end'] - words[0]['start']
            else:
                segment_duration = chunk_size
        current_time = 0
        base_time = words[0]['start'] if words else 0

        starting_index = self.find_first_idx([word['start'] for word in words], base_time + current_time, chunk_offset)
        while current_time < segment_duration:
            next_chunk_time = current_time + chunk_size
            ending_index = self.find_last_idx([word['end'] for word in words], base_time + next_chunk_time, chunk_offset)

            yield starting_index, ending_index

            starting_index = ending_index
            current_time = next_chunk_time


    def unload(self):
        self.pipeline = None
        self.metadata = None