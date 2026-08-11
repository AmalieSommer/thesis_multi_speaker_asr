import logging
import numpy as np
from whisperx import load_align_model, align, assign_word_speakers
from whisperx.types import AlignedTranscriptionResult, SingleSegment
import pandas as pd
import os


HF_TOKEN = os.getenv('HF_TOKEN')
log = logging.getLogger(__name__)

class Alignment:
    def __init__(self, align_config: dict):
        """
        Initialize the alignment object with a dict object (kwargs) containing at least the following:
            - language_code
            - device
            - model_name
            - model_dir
        Passing in a language code assumes that WhisperX already contains a model instance for that language.
        If you want to use a specific model from a remote source, pass in the values for model_name and model_dir
        """
        if 'language_code' not in align_config.keys():
            raise ValueError('Language_code was not found in the config parameter. Please pass in a language code.')
        if 'device' not in align_config.keys():
            log.info('Device configuration is missing. Setting it to a default value of CPU.')
            align_config['device'] = 'cpu'
        if not isinstance(align_config['language_code'], str):
            raise TypeError('Language code must be a string. Instead it was type: %s', type(align_config['language_code']))
        elif len(align_config['language_code'].strip()) == 0:
            raise ValueError('Language code must be a non-empty string.')
        if 'model_name' in align_config.keys():
            if not isinstance(align_config['model_name'], str):
                raise TypeError('Model name must be a string type. Instead it was type: %s', type(align_config['model_name']))
            if len(align_config['model_name'].strip()) == 0:
                raise ValueError('Model name must be a non-empty string.')
        else:
            raise ValueError('Missing a parameter: model_name.')
        self.device = align_config['device']
        self.model, self.metadata = load_align_model(**align_config)


    def align(self, prediction: list, audio: np.ndarray) -> list[dict]:
        if prediction is None:
            raise ValueError('Predictions to align are None')
        if audio is None:
            raise ValueError('Audio is None')
        if not isinstance(prediction, list):
            raise TypeError('Prediction must be a list, but found: %s', type(prediction))
        if not isinstance(audio, np.ndarray):
            raise TypeError('Audio must be a numpy array, but found: %s', type(audio))
      
        try:
            aligned_result = align(
                transcript=prediction,
                model=self.model,
                align_model_metadata=self.metadata,
                audio=audio,
                device=self.device,
                return_char_alignments=False
            )
            log.debug('Result is: %s', aligned_result)
        except Exception as e:
            log.error('Failed with error: %s', e)

        return aligned_result



    def format_model_input(self, asr_output: list[dict]) -> list[SingleSegment]:
        """
        This function can be called to reframe the output of the ASR module to the correctly accepted format of the align() function, which accept input of the form Iterable[SingleSegment]
        
        Args:
            asr_output (list[dict]): The output from a given ASR model is a list of dictionary objects containing either only text or text and start/end timestamps.
        
        Returns:
            list[SingleSegment]: A list of the segments mapped to the custom object type SingleSegment    
        """
        if any(('start' not in item.keys() or 'end' not in item.keys()) for item in asr_output):
            raise ValueError('All items in asr_output must have start and end keys.')

        segments = [SingleSegment(
            start=item['start'],
            end=item['end'],
            text=item['text']
        ) for item in asr_output]
        log.debug('Formatted segments: %s', segments)
        return segments


    def format_model_output(self, model_result: AlignedTranscriptionResult) -> list[dict]:
        if model_result is None:
            raise ValueError('Parameter model_result is None')
        if not isinstance(model_result, AlignedTranscriptionResult):
            raise TypeError('Parameter model_result must have type AlignedTranscriptionResult, but the type %s was found.', type(model_result))
        if 'word_segments' not in model_result.keys():
            raise ValueError('AlignedTranscriptionResult is missing a key: word_segments')

        return [{
            'word': word['word'],
            'start': word['start'],
            'end': word['end'],
            'score': word['score']
        } for word in model_result['word_segments']]
        


    def align_words_speakers(self, sd_output: list[dict], asr_output: AlignedTranscriptionResult) -> dict:
        """
        Function to be called on the combined output from speaker diarization and speech recognition modules.
        It calls the assign_word_speakers() from WhisperX to provide a complete SA-ASR transcript

        Args:
            sd_output (): The resulting output from the speaker diarization module
            asr_output (): The resulting output from the automatic speech recognition module

        Returns:
            dict: A dictionary of words and speakers aligned for each segment
        """
        if sd_output is None:
            raise ValueError('Parameter sd_output is None')
        if asr_output is None:
            raise ValueError('Parameter asr_output is None')
        if type(asr_output) != dict:
            raise TypeError('Parameter asr_output must have the type AlignedTranscriptionResult, but instead the type %s was found.', type(asr_output))
        if not isinstance(sd_output, list):
            raise TypeError('Parameter sd_output must be a list, but instead the type %s was found.', type(sd_output))
        if any(not isinstance(item, dict) for item in sd_output):
            raise TypeError('Parameter sd_output must be a list containing dictionary objects.')
        if any(
                "start" not in item or
                "end" not in item or
                "speaker" not in item
                for item in sd_output
            ):
            raise ValueError('The list of dictionary objects must contain \'start\', \'end\', and \'speaker\'.')

        diarization_df = pd.DataFrame(data=sd_output, columns=['start', 'end', 'speaker'])
        try:
            return  assign_word_speakers(
                        diarize_df=diarization_df,
                        transcript_result=asr_output
                    )
        except Exception as e:
            log.error('Failed with error: %s', e)
        pass



















    """
    def __init__(self, model_name, device='cpu'):
        self.metadata = None
        self.device = device

        self.load(model=model_name)
        model_memory = self.load.memory_stats[0]
        self.logger.info('Alignment Model Memory Stats...: Before load: %f, After load: %f, Delta: %f', model_memory['before'], model_memory['after'], model_memory['delta'])


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
        output = self.pipeline(
            inputs=input,
            chunk_length_s=chunk_length,
            stride_length_s=stride
        )
        return output
    

    def find_first_idx(self, words, target_start):
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
    
    def find_last_idx(self, words, target_start):
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
    
    def get_chunk_generator(self, words, offset, chunk_size=10):

        current_time = offset

        starting_index = self.find_first_idx([word['start'] for word in words], current_time)
        while current_time < (offset + chunk_size):
            next_chunk_time = current_time + chunk_size
            ending_index = self.find_last_idx([word['end'] for word in words], next_chunk_time)

            yield starting_index, ending_index
            current_time = next_chunk_time


    def unload(self):
        self.pipeline = None
        self.metadata = None
        """