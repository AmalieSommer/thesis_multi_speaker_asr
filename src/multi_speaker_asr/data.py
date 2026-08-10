import os
import librosa
import re
import io
from datasets import load_dataset, Audio, Dataset
from torch.utils.data import IterableDataset
from faster_whisper.vad import VadOptions, get_speech_timestamps
from faster_whisper.audio import decode_audio, pad_or_trim
import soundfile as sf
from multi_speaker_asr.utils.utils import LOGGING_CONFIG
from multi_speaker_asr.utils.vad import collect_audio_chunks, get_timestamps
import logging
import logging.config
import numpy as np
from pathlib import Path
import itertools
import numpy as np
import torch
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
import torchaudio
import torchaudio.functional as F
from huggingface_hub import dataset_info
from huggingface_hub.errors import HFValidationError
from datasets import load_dataset, load_from_disk



logging.config.dictConfig(LOGGING_CONFIG)
CWD = os.getcwd()


COLUMN_MAPPING = {
    "audio": ["audio", "wav_path", "file", "filepath", "audio_filepath", 'path'],
    "sample_id": ["sample_id", "id", "utterance_id", "id_recording", 'id_conversation', 'meeting_id'],
}



class AudioDataset(IterableDataset):
    def __init__(self, data_config: dict, target_sr: int = 16000, max_segment_duration: int = 30):
        self.target_sr = target_sr
        self.max_duration = max_segment_duration
        self.metadata = self.load_data(**data_config)


    def _find_column(self, available_columns, possible_names):
        for name in possible_names:
            if name in available_columns:
                return name
        raise ValueError(
            f"None of {possible_names} found in {available_columns}"
        )
        

    def load_data(self, **data_config) -> IterableDataset:
        dataset_path = data_config['path']
        split = data_config['split']
        name = data_config['name']

        data_type, ext_type, path = validate_filepath(dataset_path)
        if ext_type == 'file':
            file_ext = path.suffix.lstrip('.')

            # Edge case for jsonl files needing a json builder:
            if file_ext == 'jsonl':
                file_ext = 'json'

            metadata_path = Path(dataset_path).resolve()
            self.metadata_path = metadata_path.parent   # Setting the base directory for the local data

            return load_dataset(file_ext, data_files=str(dataset_path), split=split, streaming=True)
        elif data_type == 'hub':
            # Read data from Huggingface
            data = load_dataset(path=dataset_path, name=name, split=split, streaming=True)        
            return data.cast_column('audio', Audio(decode=False))


    def __iter__(self):
        for sample in self.metadata:
            audio_column_name = self._find_column(list(sample.keys()), COLUMN_MAPPING['audio'])
            id_column_name = self._find_column(list(sample.keys()), COLUMN_MAPPING['sample_id'])

            sample['audio'] = sample.pop(audio_column_name)
            sample['sample_id'] = sample.pop(id_column_name)


            start = None if 'start' not in sample.keys() else sample['start']
            end = None if 'end' not in sample.keys() else sample['end']


            audio, sr = self.load_wav(
                sample['audio'],
                start=start,
                end=end
            )

            yield from self.search_cutoff_points(
                audio_np=audio,
                sr=sr,
                sample_info=sample,
                max_sec=self.max_duration
            )



    def load_wav(self, audio, start=None, end=None, offset=0, target_sr=16000):
        print(audio)
        if not isinstance(audio, str):
            if isinstance(audio, dict):
                audio = io.BytesIO(audio['bytes'])
        else:
            audio = self.metadata_path / audio


        wav, sr = torchaudio.load(audio)
        if sr != target_sr:
            wav = F.resample(waveform=wav, orig_freq=sr, new_freq=target_sr)
            sr = target_sr

        if wav.shape[0] > 1:
            wav = wav.mean(axis=0)

        wav = wav.squeeze(0).numpy()
        if not start or not end:
            return wav, sr
                
        start = int(start - offset)
        end = int(end - offset)

        return wav[(start * target_sr) : (end * target_sr)], sr
 

    def collator(self, batch):
        return batch
        

    def search_cutoff_points(self, audio_np, sample_info, sr: int = 16000, max_sec: int = 30, search_window_sec: int = 5, grace_sec: float = 0.5, min_remainder_sec: int = 10):
        """
            Takes an audio sample and checks if the length of the audio is within the maximum audio duration set. If shorter, then returns the full audio array.
            Otherwise, it uses the Silero VAD model to look for natural pauses in the audio. 
            In case the audio is much longer than the max_sec, e.g. tens of minutes, a search_window_range is defined. Meaning the hard cut-off point which is the 
            max_samples value, gets encapsulated by a search_window, such that the VAD model only searches for speech segments in the range: [max_samples - search_window : max_samples + search_window].

            Once the cut-off index has been determined, it checks if the duration of the remainder of the audio after the cut-off index, is less than a pre-determined 
            lower-bound on a minimum audio length. This is to minimize the risk of ASR hallucinations or infinite decoding loops, since that is proven more likely on 
            very short audio segments.
            So, if the remainder of audio is less than the minimum duration threshold, it concatenates the remainder to the audio segment. Meaning the maximum audio duration
            is max_sec + min_remainder_sec.

            Parameters:
                - audio_np\: Numpy array representing the audio file
                - sample_info\: A dict object containing metadata information on the sample
                - sr\: Integer representing the samplerate of the audio file
                - max_sec\: Integer representing the maximum duration, allowed to be yielded, of an audio sample
                - search_window_sec\: Integer representing the range of samples surrounding the max_samples to be passed to the VAD model
                - grace_sec\: Float representing the grace period after the hard cut-off point to allow as a possible cut-off index, since the aim is to choose the cut-off index closest to the max_sec
                - min_remainder_sec\: Integer representing the minimum duration of an audio sample allowed
        """

        max_samples = int(max_sec * sr)
        grace_samples = int(grace_sec * sr)
        min_remainder = int(min_remainder_sec * sr)

        chunks = []
        total_length = len(audio_np)
        current_start = 0
        vad_model = load_silero_vad()
        iter = 0
        while current_start < total_length:
            theoretical_end = current_start + max_samples
            iter += 1
            
            if theoretical_end >= total_length:
                yield {
                    'sample_id': sample_info['sample_id'],
                    'audio': audio_np[current_start:],
                    'samplerate': sr,
                    'start': current_start / self.target_sr,
                    'end': total_length / self.target_sr
                }
                break
                
            search_start = max(current_start, theoretical_end - int(search_window_sec * sr))
            search_end = min(total_length, theoretical_end + int(search_window_sec * sr))
            # Extract the search window for Silero VAD
            window_slice = audio_np[search_start:search_end]
            window_tensor = torch.from_numpy(window_slice).float()
            speech_intervals = get_speech_timestamps(window_tensor, vad_model, sampling_rate=sr)

            cut_index = self._find_cut_index(
                speech_intervals=speech_intervals,
                search_start=search_start,
                search_end=search_end,
                theoretical_end=theoretical_end,
                grace_samples=grace_samples
            )

            if cut_index > total_length:
                raise IndexError('Index out of bounds. Cut_index %i is greater than the length of the audio array')
            elif cut_index < 0:
                raise IndexError('Index is less than zero.')


            remaining = total_length - cut_index

            if remaining < min_remainder:
                yield {
                    'sample_id': sample_info['sample_id'],
                    'audio': audio_np[current_start:],
                    'samplerate': sr,
                    'start': len(audio_np[:current_start]) / self.target_sr,
                    'end': total_length / self.target_sr
                }
                break
            else:
                audio_chunk = audio_np[current_start:cut_index]
                yield {
                    'sample_id': sample_info['sample_id'],
                    'audio': audio_chunk,
                    'samplerate': sr,
                    'start': len(audio_np[:current_start]) / self.target_sr,
                    'end': (len(audio_np[:current_start]) + len(audio_chunk)) / self.target_sr
                }
                current_start = cut_index
        return chunks



    def _find_cut_index(self, speech_intervals: list[dict] | None, search_start: int, search_end: int, theoretical_end: int, grace_samples: float) -> int:
        """
            Given a list of time intervals for speech segments in speech_intervals, it looks through the search interval,
            defined by search_start and search_end, and saves the midpoint between each speech segment that is within the
            search index.
            Then it terates through all the collected candidate cut_indices, and checks if the speech ending point is within
            the grace-period given at the end of each speech segment in order to get as many segments close to the 
            desired audio duration. E.g. it would be preferable to cut 0.5 seconds after the theoretical_end, then at the
            midpoint; theoretical_end / 2.

            If no candiate is found near the theoretical_end (within the grace-period), it just returns the last cut_index 
            midpoint before reaching the theoretical_end value.

            Parameters:
                - speech_intervals\: List of {start, end} dict objects returned by VAD speech detection
                - search_start\: Integer representing the start of a search window
                - search_end\: Integer representing the end of a search window
                - theoretical_end\: Integer representing the desired length of each speech segment. Acts as the default if no valid cut_index is found
                - grace_samples\: Integer representing the number of samples around the theoretical_end which can be seen as valid cut_indices (a grace-period).
    
        """

        if not speech_intervals:
            return search_start

        candidates = []

        for i, segment in enumerate(speech_intervals):
            segment_end = search_start + segment["end"]

            if i + 1 < len(speech_intervals):
                temp = speech_intervals[i + 1]["start"]
                print(f'Next speech start: {temp}')
                gap_end = search_start + temp
                print(f'Search_start: {search_start}')
                print(f'Gap end: {gap_end}')
                print(f'Segment_end: {segment_end}')
            else:
                gap_end = search_end

            candidates.append({
                "speech_end": segment_end,
                "mid_point": (segment_end + gap_end) // 2,
            })
            
            print(candidates)

        for candidate in candidates:
            print(candidate)
            if theoretical_end <= candidate["speech_end"] <= theoretical_end + grace_samples:
                return candidate["mid_point"]

        before = [
            c for c in candidates
            if c["speech_end"] <= theoretical_end
        ]

        print(before)

        if before:
            return before[-1]["mid_point"]

        return theoretical_end
    


def validate_filepath(filepath: str) -> str | tuple[str, str, Path]:
    # First check if the filepath is to a local dataset:
    if filepath == None:
        raise ValueError('Filepath is None.')
    if not isinstance(filepath, str):
        raise ValueError('Filepath must be a string.')
    filepath = filepath.strip()
    if len(filepath) < 1:
        raise ValueError('Filepath is empty.')
    
    

    valid_ext = {
        '.csv', '.tsv', '.json', '.jsonl', '.parquet', '.arrow', '.txt', '.xml', '.gz'
    }

    path = Path(filepath)
    if path.exists():  
        print(path)
        valid_extensions = [ext for ext in path.suffixes if ext in valid_ext]
        print(valid_extensions)
        unique_exts = list(set(valid_extensions))
        if len(unique_exts) > 1:
            raise TypeError('The dataset files must have the same extension format')

        if path.is_dir():
            return 'local', 'dir', path.resolve()

        return 'local', 'file', path.resolve()

    # Next check if the filepath is to a Huggingface dataset:
    try:
        dataset_info(repo_id=filepath)
        return 'hub', 'repo', None
    except HFValidationError as e:
        pass

    # Finally, raise error if no file was found at either location
    raise FileNotFoundError('The file %s was not found either locally or on Huggingface', filepath)
