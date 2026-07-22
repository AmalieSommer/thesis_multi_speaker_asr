import os
import librosa
import re
import io
from datasets import load_dataset, Audio, Dataset
from torch.utils.data import IterableDataset
from faster_whisper.vad import VadOptions, get_speech_timestamps
import soundfile as sf
from multi_speaker_asr.utils.utils import LOGGING_CONFIG
from multi_speaker_asr.utils.vad import collect_audio_chunks, get_timestamps
import logging
import logging.config
import numpy as np
from pathlib import Path
import itertools

logging.config.dictConfig(LOGGING_CONFIG)


CWD = os.getcwd()



class AudioDataset(IterableDataset):
    def __init__(self, metadata, mode: str = 'segments', transform = None):
        self.metadata = metadata
        self.mode = mode
        self.transform = transform

    def __iter__(self):
        if self.mode == 'segments':
            yield from self.__iter__segments()
        elif self.mode == 'recordings':
            yield from self.__iter__recordings()
        else:
            raise ValueError('Unknown mode value...')
        
    def __iter__segments(self):
        iter = 0
        for sample in self.metadata:
            if 'segment_duration' in sample.keys() and sample['segment_duration'] <= 1.0:
                # Skip audio segments that are too short...
                continue
            
            audio = self.load_audio(
                sample['segment']
            )
            if audio is None: 
                continue

            yield {
                'audio_id': sample['audio_id'],
                'segment_id': iter if 'segment_id' not in sample.keys() else sample['segment_id'],
                'text': sample['text'],
                'audio': audio
            }
            iter += 1
    
    def __iter__recordings(self):
        for sample in self.metadata:
            if 'segment_duration' in sample.keys() and sample['segment_duration'] <= 1.0:
                # Skip audio segments that are too short...
                continue

            audio_offset = sample['audio_offset']
            audio = self.load_audio(
                sample['audio'],
                start=sample['start'],
                end=sample['end'],
                offset=audio_offset
            )

            yield {
                'audio_id': sample['audio_id'],
                'segment_id': sample['segment_id'],
                'text': sample['text'],
                'audio': audio,
            }


    def load_audio(self, audio, start=None, end=None, offset=0, target_sr=16000):
        if not audio:
            raise ValueError('Missing audio file...')
        
        if isinstance(audio, dict):
            if 'bytes' in audio.keys():
                audio = io.BytesIO(audio['bytes'])
        audio, sr = sf.read(audio, dtype='float32')
        
        # Check for multiple channels and convert to mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if sr != target_sr:
            audio = librosa.resample(
                audio,
                orig_sr=sr,
                target_sr=target_sr,
            )
            audio = audio.astype("float32")
    
        if not start or not end:
            return audio
        
        start = int(start - offset)
        end = int(end - offset)

        audio = audio[(start * sr) : (end * sr)]
        return audio
    

    def collator(self, batch):
        audio_batch = [b['audio'] for b in batch]
        metadata = [{
            'audio_id': b['audio_id'],
            'segment_id': b['segment_id'],
            'text': b['text'],
            'start': b['start'],
            'end': b['end']
        } for b in batch]
        return audio_batch, metadata


class AudioData(IterableDataset):
    """
    Data wrapper class to load either local or Huggingface datasets. Perform preprocessing, resampling and formatting as preparation for model training and inference.
    """
    logger = logging.getLogger(name='AudioData')

    def __init__(
            self,  
            target_sr=16000, 
            max_segment_duration=30, 
            vad_filter=True, 
            clip_timestamps=False,
            batch_size=5
            ):
        super().__init__()
        self.clip_timestamps = clip_timestamps
        self.vad_filter = vad_filter
        self.target_sr = target_sr
        self.max_segment_duration = max_segment_duration
        self.ds = None
        self.batch_size = batch_size
        self.chunk_buffer = []
        
        

    def load(self, path):
        """
        Loads a dataset either from local or online resource.
        If the path does not exist in the DATA dict, then it is assumed local, and will be loaded manually.
        """
        self.ds = Dataset.from_csv(path_or_paths=path, split='test').to_iterable_dataset()
        self.id_to_audio = {item['id']: item['audio'] for item in self.ds}  

    def __iter__(self):
        current_audio = []
        current_metadata = []
        current_audio_id = None
        offset = 0

        for item in self.ds:
            if not item["audio"]:
                continue

            #offset = item.get("start", 0)

            for chunk in self.preprocess(sample=item):

                if current_audio_id is None:
                    current_audio_id = chunk["audio_id"]

                if chunk["audio_id"] != current_audio_id:
                    # Yield previous audio
                    yield {
                        "audio_id": current_audio_id,
                        "offset": offset,
                        "audio": current_audio,
                        "chunk_metadata": current_metadata,
                    }

                    # Start new audio
                    current_audio_id = chunk["audio_id"]
                    current_audio = []
                    current_metadata = []
                    offset = item.get("start", 0)

                # Always append the current chunk
                current_audio.append(chunk["audio"])
                current_metadata.append(chunk["chunk_metadata"])
                offset = item.get("start", 0)

        # Yield the last audio
        if current_audio:
            yield {
                "audio_id": current_audio_id,
                "offset": offset,
                "audio": current_audio,
                "chunk_metadata": current_metadata,
            }


    def collator_fn(self, batch):
        return batch


    def preprocess(self, sample):
        try:
            audio = sample['audio'] if 'audio' in sample.keys() else None

            if audio == None:
                raise Exception('Missong audio in dataset sample!')
            
            audio = sample['audio']['bytes'] if type(sample['audio']) == dict else sample['audio']
            audio = self.read_audio(audio=audio)

            offset, clip_timestamps, audio = get_timestamps(
                data_sample=sample,
                audio=audio,
                duration=audio.shape[0],
                clip_timestamps=self.clip_timestamps,
                vad_filter=self.vad_filter
            )
            
            yield from collect_audio_chunks(
                id=sample['id'],
                audio=audio,
                clip_timestamps=clip_timestamps,
                offset=offset
            )
        except Exception as e:
            self.logger.error('Failed in preprocess() with error... ', e)


    def read_audio(self, audio, target_sr=16000):
        try: 
            if isinstance(audio, bytes):
                audio_ = io.BytesIO(audio)
            else:
                audio_ = Path(CWD, audio)
            wav, sr = sf.read(audio_, dtype='float32')
            
            # Check for multiple channels and convert to mono
            if wav.ndim > 1:
                wav = wav.mean(axis=1)

            if sr != target_sr:
                wav = librosa.resample(
                    wav,
                    orig_sr=sr,
                    target_sr=target_sr,
                )
                wav = wav.astype("float32")
            return wav
        except Exception as e:
            print('Failed with exception: ', e)

    
    

class SegmentedData(AudioData):
    def __init__(
            self, 
            target_sr=16000, 
            max_segment_duration=30
            ):
        self.target_sr = target_sr
        self.max_segment_duration = max_segment_duration
        
    def load(self, path: str = 'CoRal-project/coral-v3', name: str = 'conversation', split: str = 'test'):
        # If the path is from an online resource, then load it using datasets built-in function:
            self.ds = load_dataset(
                path=path,
                name=name,
                split=split,
                streaming=True
            )
            # Ensure it does not decode audio using torchDecoder
            self.ds = self.ds.cast_column('audio', Audio(decode=False))
            self.ds = self.ds.rename_column('id_conversation', 'id')

    def collator(self, batch):
        return super().collator(batch)

    def __iter__(self):
        
        current_speaker = None
        current_segments = []
        current_duration = 0
        total_duration = 0
        current_audio = np.array([], dtype=np.float32)
        try:
            iter = 0
            for item in self.ds:
                if current_speaker is None:
                    current_speaker = item['id_speaker']
                audio = item['audio'] if 'audio' in item.keys() else None

                if audio is None:
                    raise Exception('Missong audio in dataset sample!')
                
                audio = item['audio']['bytes'] if type(item['audio']) == dict else item['audio']
                wav = self.read_audio(audio=audio)
                
                chunk_start = 0
                chunk_end = wav.shape[0]

                if item['id_speaker'] != current_speaker:
                    yield {
                        "batch_id": iter,
                        "audio": current_audio,
                        "chunk_metadata": {
                            'offset': total_duration / self.target_sr,
                            "duration": (chunk_end - chunk_start) / self.target_sr,
                            "segments": current_segments,
                        },
                    }

                    total_duration += current_duration
                    current_segments = []
                    current_segments.append({
                            'id': item['id'],
                            'id_speaker': item['id_speaker'],
                            'age': item['age'],
                            'gender': item['gender'],
                            'country_birth': item['country_birth'],
                            'dialect': item['dialect'],
                            'overlap': item['overlap'],
                            'text': item['text'],
                            'start': 0,
                            'end': wav.shape[0]
                        })

                    current_audio = wav
                    current_duration = chunk_end - chunk_start
                    current_speaker = item['id_speaker']
                else:
                    if (
                        current_duration + chunk_end - chunk_start
                        > self.max_segment_duration * self.target_sr
                    ):
    
                        yield {
                            "batch_id": iter,
                            "audio": current_audio,
                            "chunk_metadata": {
                                'offset': total_duration / self.target_sr,
                                "duration": current_duration / self.target_sr,
                                "segments": current_segments,
                            },
                        }
                        
                        total_duration += current_duration
                        current_segments = []
                        current_audio = wav
                        current_duration = chunk_end - chunk_start
                        current_speaker = item['id_speaker']

                        current_segments.append({
                            'id': item['id'],
                            'id_speaker': item['id_speaker'],
                            'age': item['age'],
                            'gender': item['gender'],
                            'country_birth': item['country_birth'],
                            'dialect': item['dialect'],
                            'overlap': item['overlap'],
                            'text': item['text'],
                            'start': 0,
                            'end': wav.shape[0]
                        })
                    else:
                        current_segments.append({
                            'id': item['id'],
                            'id_speaker': item['id_speaker'],
                            'age': item['age'],
                            'gender': item['gender'],
                            'country_birth': item['country_birth'],
                            'dialect': item['dialect'],
                            'overlap': item['overlap'],
                            'text': item['text'],
                            'start': 0,
                            'end': wav.shape[0]
                        })
                        current_audio = np.concatenate(
                            (current_audio, wav)
                        )

                        current_duration += chunk_end - chunk_start
                        current_speaker = item['id_speaker']

                iter += 1

            yield {
                "batch_id": iter,
                "audio": current_audio,
                "chunk_metadata": {
                    'offset': total_duration / self.target_sr,
                    "duration": (chunk_end - chunk_start) / self.target_sr,
                    "segments": current_segments,
                },
            }
        except Exception as e:
            self.logger.error('Failed in collator...')

"""
def cast(object: dict):
    return SingleSegment(
        start=object['start'],
        end=object['end'],
        text=object['text'],
        avg_logprob=object['avg_logprob']
    )
"""






