import pandas as pd
from pyannote.audio import Pipeline
from whisperx.diarize import IntervalTree
from ..utils.utils import profile, LOGGING_CONFIG
import logging
import logging.config
import tracemalloc

logging.config.dictConfig(LOGGING_CONFIG)


MODEL = {
    'py3': 'pyannote/speaker-diarization-3.0',
    'py3.1': 'pyannote/speaker-diarization-3.1',
    'custom': '' # TODO: implement custom option based on code in evaluate.py
}


class Diarize:

    logger = logging.getLogger(name='Diarization')

    def __init__(self, token, device='cpu', pipeline='py3'):
        self.device = device
        self.model_path = MODEL[pipeline]
        
        self.load(token=token)
        model_memory = self.load.memory_stats[0]
        self.logger.info('Diarization Model Memory Stats...: Before load: %f, After load: %f, Delta: %f', model_memory['before'], model_memory['after'], model_memory['delta'])

    @profile
    def load(self, token):
        self.model = Pipeline.from_pretrained(
            self.model_path, 
            token=token
            )
        self.model.embedding_batch_size = 1
        

    def unload(self):
        self.model = None
        

############################################################################
########### Modified version of the original from WhisperX #################
########### To support different input format for speakers #################
############################################################################

def assign_word_speakers(id: str, segments_list: list[dict], speaker_times: list[dict]):
    intervals = [(item['start'], item['end'], item['speaker']) for item in speaker_times]
    interval_tree = IntervalTree(intervals=intervals)

    # Iterate the list of transcription segments:
    for segment in segments_list:
        segment['id'] = id
        start_segment = segment['start']
        end_segment = segment['end']

        overlapping_intervals = interval_tree.query(start=start_segment, end=end_segment)

        if overlapping_intervals:

            speaker_intersections: dict[str, float] = {}
            for speaker, intersection in overlapping_intervals:
                speaker_intersections[speaker] = speaker_intersections.get(speaker, 0.0) + intersection
            segment['speaker'] = max(speaker_intersections.items(), key=lambda x: x[1])[0]
        else:
            root = (start_segment + end_segment) / 2
            nearest_speaker = interval_tree.find_nearest(time=root)
            if nearest_speaker:
                segment['speaker'] = nearest_speaker
            
        if 'words' in segment.keys():
            # Iterate list of words in the segment:
            for word in segment['words']:
                if ('start' not in word) | ('end' not in word):
                    continue

                start_word = word['start']
                end_word = word.get('end', start_word)

                word_overlaps = interval_tree.query(start=start_word, end=end_word)

                if word_overlaps:
                    # Multiple speakers speak at this time interval:
                    intersections = {}
                    for speaker, intersection in word_overlaps:
                        intersections[speaker] = intersections.get(speaker, 0.0) + intersection
                    word['speaker'] = max(intersections.items(), key=lambda x: x[1])[0]
                else:
                    root = (start_word + end_word) / 2
                    nearest = interval_tree.find_nearest(time=root)
                    if nearest:
                        word['speaker'] = nearest

    return segments_list