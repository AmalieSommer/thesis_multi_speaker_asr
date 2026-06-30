
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from whisperx.diarize import IntervalTree
from ..utils.utils import profile, LOGGING_CONFIG
import logging
import logging.config
from ..data import AudioData
import torch
import os


logging.config.dictConfig(LOGGING_CONFIG)

HF_TOKEN = os.getenv('HF_TOKEN')


class Diarize:

    logger = logging.getLogger(name='Diarization')

    def __init__(self, device='cpu'):
        self.device = device        
        self.load()

        model_memory = self.load.memory_stats[0]
        self.logger.info('Diarization Model Memory Stats...: Before load: %f, After load: %f, Delta: %f', model_memory['before'], model_memory['after'], model_memory['delta'])

    @profile
    def load(self):
        self.model = Pipeline.from_pretrained(
            'pyannote/speaker-diarization-3.0', 
            token=HF_TOKEN
            )
        self.model.embedding_batch_size = 1
        

    def unload(self):
        self.model = None


    def diarize(self, sample: AudioData, target_sr=16000):
        audio = sample['audio']
        audio_time = audio.shape[0]
        wav = torch.tensor(audio).unsqueeze(0)   # To get the correct format of (channel, time) Tensor.

        speaker_segments = []
        print('Diarization... Model type: ', type(self.model))
        with ProgressHook() as hook:
            output = self.model({
                'waveform': wav,
                'sample_rate': target_sr
            },
            hook,
            min_speakers=1,
            max_speakers=2
            )

        for segment, _, speaker in output.speaker_diarization.itertracks(yield_label=True):
            speaker_segments.append({
                'speaker': speaker,
                'start': segment.start,
                'end': segment.end,
                'duration': segment.duration
            })
        
        return speaker_segments, audio_time

############################################################################
########### Modified version of the original from WhisperX #################
########### To support different input format for speakers #################
############################################################################

def assign_word_speakers(segments: list[dict], speaker_times: list[dict]):
    intervals = [(item['start'], item['end'], item['speaker']) for item in speaker_times]
    interval_tree = IntervalTree(intervals=intervals)

    # Iterate the list of transcription segments:
    for segment in segments:
        for word in segment['words']:

            start_word = word['start']
            end_word = word['end']

            overlapping_intervals = interval_tree.query(start=start_word, end=end_word)

            if overlapping_intervals:
                speaker_intersections: dict[str, float] = {}
                for speaker, intersection in overlapping_intervals:
                    speaker_intersections[speaker] = speaker_intersections.get(speaker, 0.0) + intersection

                word['speaker'] = max(speaker_intersections.items(), key=lambda x: x[1])[0]
            else:
                root = (start_word + end_word) / 2
                nearest_speaker = interval_tree.find_nearest(time=root)
                if nearest_speaker:
                    word['speaker'] = nearest_speaker
            """
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
            """
    return segments