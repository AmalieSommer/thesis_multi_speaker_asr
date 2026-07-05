from typing import List, Optional
from bisect import bisect
from faster_whisper.vad import SpeechTimestampsMap

class VAD(SpeechTimestampsMap):

    def __init__(self, sampling_rate = 16000, time_precision = 2):
        
        self.sampling_rate = sampling_rate
        self.time_precision = time_precision
        self.chunk_end_sample = []
        self.total_silence_before = []
        self.global_previous_end = 0
        self.global_silent_samples = 0


    def build_mapping(self, chunks):
        previous_end = 0
        silent_samples = 0

        chunk_end_sample = []
        total_silence_before = []

        for chunk in chunks:
            silent_samples += chunk['start'] - previous_end
            previous_end = chunk['end']

            chunk_end_sample.append(chunk['end'] - silent_samples)
            total_silence_before.append(silent_samples / self.sampling_rate)
        
        self.chunk_end_sample = chunk_end_sample
        self.total_silence_before = total_silence_before

        # Previous end is needed to know where to start the next section relative to, and the 
        # silent samples variable is needed to know the accumulation of previous silent segments...
        self.global_previous_end = previous_end # Is needed for generating the updates to existing mappings
        self.global_silent_samples = silent_samples # Is needed for generating updates to existing mappings


    def update_mapping(self, chunks):
        """
        This is used for making an update on the mapping from augmented to original audio timeline, by simply extending
        the existing mapping with new segment chunks timestamps from the same audio recording.
        """
        for chunk in chunks:
            self.global_silent_samples += chunk['start'] - self.global_previous_end
            self.global_previous_end = chunk['end']

            self.chunk_end_sample.append(chunk['end'] - self.global_silent_samples)
            self.total_silence_before.append(self.global_silent_samples / self.sampling_rate)


    def get_original_time(self, time, chunk_index = None, is_end = False):
        return super().get_original_time(time, chunk_index, is_end)


    def get_chunk_index(self, time, is_end = False):
        return super().get_chunk_index(time, is_end)