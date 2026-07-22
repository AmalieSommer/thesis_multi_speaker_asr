
from pyannote.audio import Pipeline, Model, Inference
from pyannote.audio.pipelines.utils.hook import ProgressHook
from ..utils.utils import profile, LOGGING_CONFIG
import logging
import logging.config
from ..data import AudioData
import torch
import os
import numpy as np
from scipy.spatial.distance import cdist



logging.config.dictConfig(LOGGING_CONFIG)

HF_TOKEN = os.getenv('HF_TOKEN')


class Diarize:

    logger = logging.getLogger(name='Diarization')

    def __init__(self, device='cpu'):
        self.device = device
        self.speaker_embedding_map = {}     

        #model_memory = self.load.memory_stats[0]
        #self.logger.info('Diarization Model Memory Stats...: Before load: %f, After load: %f, Delta: %f', model_memory['before'], model_memory['after'], model_memory['delta'])

    @profile
    def load(self, type='pyannote/speaker-diarization-3.0'):
        self.model = Pipeline.from_pretrained(
            type, 
            use_auth_token=HF_TOKEN
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
    

class RollingClusters:
    def __init__(self, match_threshold=0.3, new_threshold=0.55, max_history=10):
        self.match_threshold = match_threshold
        self.new_threshold = new_threshold
        self.max_history = max_history

        self.speaker_registry = {}
        self.speaker_count = 0

    def preprocess(self, embedding: np.ndarray):
        if embedding.shape[0] > 1:
            embedding = np.expand_dims(embedding, axis=0)

        embedding = embedding.astype(np.float32)
        norm = np.linalg.norm(embedding, axis=1, keepdims=True)
        return embedding / np.where(norm == 0, 1e-12, norm)
    
    def compute_centroids(self):
        speakers = list(self.speaker_registry.keys())
        centroids = []
        for speaker in speakers:
            avg_embedding = np.mean(self.speaker_registry[speaker], axis=0)
            centroids.append(self.preprocess(avg_embedding)[0])

        return speakers, np.array(centroids)
    
    def process_chunk(self, chunk_embedding: np.ndarray):
        
        if np.all(chunk_embedding == 0) or np.linalg.norm(chunk_embedding) < 1e-6:
            return None
        
        processed_embedding = self.preprocess(chunk_embedding)

        if not self.speaker_registry:
            speaker_id = f'SPEAKER_{self.speaker_count:02d}'
            self.speaker_registry[speaker_id] = [processed_embedding[0]]
            self.speaker_count += 1
            
            return speaker_id

        speaker_ids, centroids = self.compute_centroids()
        distances = cdist(processed_embedding, centroids, metric='cosine')[0]
        best_match_index = np.argmin(distances)
        min_dist = distances[best_match_index]
        best_speaker_id = speaker_ids[best_match_index]


        if min_dist <= self.match_threshold:
            self.speaker_registry[best_speaker_id].append(processed_embedding[0])
            
            if len(self.speaker_registry[best_speaker_id]) > self.max_history:
                self.speaker_registry[best_speaker_id].pop(0)

            return best_speaker_id

        elif min_dist >= self.new_threshold:

            new_speaker_id = f'SPEAKER_{self.speaker_count:02d}'
            self.speaker_registry[new_speaker_id] = [processed_embedding[0]]
            self.speaker_count += 1
            
            return new_speaker_id
        else: 

            return best_speaker_id


############################################################################
########### Modified version of the original from WhisperX #################
########### To support different input format for speakers #################
############################################################################

def assign_word_speakers(segments: list[dict], speaker_times: list[dict]):
    intervals = [(item['start'], item['end'], item['speaker']) for item in speaker_times]
    interval_tree = IntervalTree(intervals=intervals)

    # Iterate the list of transcription segments:
    for segment in segments:
        for word in segment['word_segments']:

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