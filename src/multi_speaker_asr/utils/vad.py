from typing import List, Optional
from bisect import bisect
from faster_whisper.vad import SpeechTimestampsMap, VadOptions, get_speech_timestamps
import numpy as np
from ..utils.utils import profile, LOGGING_CONFIG
import logging
import logging.config

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(name='AudioData')


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
    

def get_timestamps(
        data_sample,
        audio: np.ndarray,
        duration: int,
        clip_timestamps: bool,
        vad_filter: bool,
        sampling_rate: int = 16000,
        max_duration: int = 30
    ):
    
    offset = 0 if 'start' not in data_sample.keys() else data_sample['start']
    timestamps = []

    if not clip_timestamps:
        # The sample has no specified start and end times for the audio segment:
        if vad_filter:
            # There are no timestamps provided with the audio and the vad_filter flag is enabled.
            vad_parameters = VadOptions(
                        max_speech_duration_s=max_duration,
                        min_silence_duration_ms=160,
                    )
            timestamps = get_speech_timestamps(audio=audio, vad_options=vad_parameters)
        elif duration < max_duration * sampling_rate:
            timestamps = [{'start': 0, 'end': audio.shape[0]}]
        else:
            timestamps = [{'start': 0, 'end': audio.shape[0]}]
            logger.info('Timestamps are not provided, and VAD is not enabled. The audio is longer than 30 second limit, so either provide clip_timestamps or enable VAD filtering.')
    else:
        # There are provided clipped timestamps for the audio
        # It will clip the audio and check if the audio is within the max_duration.
        # If clipped audio is within range, then return the clip_timestamp
        # Otherwise, check if vad_filter is enabled and if so, then apply VAD, and
        # if not, then log an errormessage to enable VAD or provide timestamps.
        start = int(data_sample['start'] * sampling_rate)
        end = int(data_sample['end'] * sampling_rate)
        audio = audio[start : end]

        if audio.shape[0] < max_duration * sampling_rate:
            timestamps = [{'start': start, 'end': end}]
            
        else:
            if vad_filter:
                vad_parameters = VadOptions(
                        max_speech_duration_s=max_duration,
                        min_silence_duration_ms=160,
                    )
                timestamps = get_speech_timestamps(audio=audio, vad_options=vad_parameters)
            else:
                timestamps = [{'start': 0, 'end': audio.shape[0]}]
                logger.info('The clip_timestamps clip the audio to a duration longer than the 30 second limit. It will only process the first 30 seconds. Otherwise, enable VAD filtering.')
    return offset, timestamps, audio

def collect_audio_chunks(
        id,
        audio: np.ndarray,
        offset: float,
        clip_timestamps: List[dict],
        max_duration: int = 30,
        sampling_rate: int = 16000
        ):
    
    current_segments = []
    current_duration = 0
    total_duration = 0
    current_audio = np.array([], dtype=np.float32)
    curr_id = 0

    for i, clip in enumerate(clip_timestamps):
        
        if (
             current_duration + clip['end'] - clip['start'] > max_duration * sampling_rate
        ):
            curr_id += 1
            sample = {
                'audio_id': id,
                'audio': current_audio,
                'chunk_metadata': {
                    'chunk_id': curr_id,
                    'offset': total_duration / sampling_rate,
                    'duration': current_duration / sampling_rate,
                    'segments': current_segments
                }
            }
        
            total_duration += current_duration

            current_segments = [{
                'segment_id': i,
                'start': clip['start'], 
                'end': clip['end'] 
            }]
            current_audio = audio[clip['start'] : clip['end']]
            current_duration = clip['end'] - clip['start']

            yield sample
                            
        else:
            current_segments.append({
                'segment_id': i,
                'start': clip['start'], # could add the offset if the audio is clipped
                'end': clip['end'] # could add the offset if the audio is clipped
            })
            current_audio = np.concatenate(
                (current_audio, audio[clip['start'] : clip['end']])
            )
            current_duration += clip['end'] - clip['start']

    curr_id += 1
    yield {
            'audio_id': id,
            'audio': current_audio,
            'offset': offset,
            'chunk_metadata': {
                'chunk_id': curr_id,
                'offset': total_duration / sampling_rate,
                'duration': current_duration / sampling_rate,
                'segments': current_segments
            }
        }
    




def collect_word_chunks(
        clip_timestamps: List[dict],
        max_duration: int = 30,
        sampling_rate: int = 16000
        ):
    
    new_chunk_start = clip_timestamps[0]['start'] # To take the start of every new iteration...
    start_chunk = clip_timestamps[0]['start']
    end_chunk = 0
    text_chunk = ""

    for i, clip in enumerate(clip_timestamps):

        if (
             clip['end'] - new_chunk_start > max_duration
        ) & (len(str.strip(text_chunk)) > 0):
            yield {
                'start': start_chunk,
                'end': end_chunk,
                'text': text_chunk
            }
            text_chunk = clip['word']
            new_chunk_start = end_chunk
            start_chunk = clip['start']
            end_chunk = clip['end']

        else:
            end_chunk = clip['end']
            text_chunk += clip['word']

    yield {
            'start': start_chunk,
            'end': end_chunk,
            'text': text_chunk
        }