import logging
import os
import numpy as np
from einops import rearrange
from pyannote.core import Annotation
from diart.sinks import RTTMWriter
from pathlib import Path
import sys
import types
import torch
# ------ HACKY WORKAROUND FOR USING DIART WITHOUT INSTALLING PORTAUDIO FOR STREAMING, SINCE ONLINE STREAMING IS NOT USED ----------
sd = types.ModuleType("sounddevice")

class DummyInputStream:
    pass

sd.InputStream = DummyInputStream
sd.query_devices = lambda *args, **kwargs: []

sys.modules["sounddevice"] = sd
# ------------------------------------------------
from diart import SpeakerDiarization, SpeakerDiarizationConfig
from diart.inference import StreamingInference
from diart.sources import AudioSource


HF_TOKEN = os.getenv('HF_TOKEN')
log = logging.getLogger(__name__)


class NumpyArrAudioSource(AudioSource):
    """
    A custom AudioSource class to extend support of using Diart StreamingInference for preprocessed audio samples, i.e.
    audio files that already have been loaded to a numpy array with the correct samplerate.
    It also accepts a chunk_size value, which represents the number of seconds to process at a time.

    This class servers as a data stream emitting digestible audio chunks to an asynchronous data stream that is read by subscribed observables.
    """
    def __init__(
            self, 
            audio_arr: np.ndarray, 
            chunk_size: float = 0.5, 
            uri: str = 'local_path',
            sample_rate: int = 16000, 
            is_closed: bool = False
            ):
        super().__init__(uri, sample_rate)
        self.chunk_samples = int(chunk_size * sample_rate)
        self.is_closed = is_closed

        if not isinstance(audio_arr, np.ndarray):
            raise TypeError('The parameter audio_arr must be either a numpy array or a list. The found type was invalid.')

        if audio_arr.ndim == 1:
            # Should expand to (channels, samples) as Diart API expects
            self.audio = np.expand_dims(audio_arr, axis=0)
        else:
            self.audio = audio_arr
      
            

    def read(self) -> None:
        """
        This function reads the audio from the AudioSource, and changes it to the required format for the stream pipeline.
        Once it has the correct format, it iteratively emits one chunk at a time to the pipe through on_next()

        TODO: Could refactor the input reformatting to the Dataset class, such that the audio passed to the NumpyAudioSource is a list of chunks of the correct size already.
        """
        # Stream blocks
        log.debug('Reading the input stream...')
        # Split into blocks
        waveform = torch.tensor(self.audio)
        _, num_samples = waveform.shape
        chunks = rearrange(
            waveform.unfold(1, self.chunk_samples, self.chunk_samples),
            "channel chunk sample -> chunk channel sample",
        ).numpy()

        # Add last incomplete chunk with padding
        if num_samples % self.chunk_samples != 0:
            last_chunk = (
                waveform[:, chunks.shape[0] * self.chunk_samples :].unsqueeze(0).numpy()
            )
            diff_samples = self.chunk_samples - last_chunk.shape[-1]
            last_chunk = np.concatenate(
                [last_chunk, np.zeros((1, 1, diff_samples))], axis=-1
            )
            chunks = np.vstack([chunks, last_chunk])
            for chunk in chunks:
                try:
                    if self.is_closed:
                        break
                    
                    self.stream.on_next(chunk)
                except BaseException as e:
                    self.stream.on_error(e)
                    break
        self.stream.on_completed()
        self.close()


    def close(self) -> None:
        """
        Once the stream is detected to be closed, it manually forces a close on further audio emissions.
        """
        self.is_closed = True
        return super().close()


class SpeakerDiarizationPipeline:
    """
    It takes the parameters: SpeakerDiarizationConfig, and runs all audio samples using StreamingInference.
    """
    def __init__(self, config: SpeakerDiarizationConfig):
        self.config = config
        self.pipeline = SpeakerDiarization(
            config=config
        )


    def diarize(self, sample: NumpyArrAudioSource, output_path: str | Path) -> Annotation:
        """
        Setting up a streaming diarization pipeline on the given NumpyArrAudioSource. It attach an observer to write results continuously to an RTTM file.
        
        Args:
            sample (NumpyArrAudioSource): The sample source being pushed to the monitored pipeline
            output_path (str | Path): The path to the RTTM file containing the resulting diarization output 

        Returns:
            Annotation: Predictions from the speaker diarization pipeline
        """
        # ----------- Path validity check -------------
        if output_path is None:
            raise ValueError('Parameter output_path is None.')

        if not isinstance(output_path, Path):
            if isinstance(output_path, str):
                if len(output_path.strip()) == 0:
                    raise ValueError('Parameter output_path is an empty string.')
                elif output_path.isnumeric():
                    raise ValueError('Parameter output_path found only numeric characters, which is invalid.')
                else:
                    output_path = Path(output_path)
            else:
                raise TypeError('Parameter output_path must be a string or a Path. Received parameter type was: %s', type(output_path))

        if output_path.suffix != '.rttm':
            raise ValueError('The path must be to a .rttm file')

        if not output_path.is_file():
            raise ValueError('Parameter output_path is not a valid file.')

        
        # ------------- Running Streaming Inference -------------
        inference = StreamingInference(
            pipeline=self.pipeline, 
            source=sample, 
            do_plot=False)

        inference.attach_observers(RTTMWriter(uri=sample.uri, path=output_path))
        prediction = inference()
        return prediction


    def get_speaker_segments(self, result: Annotation) -> list[dict]:
        """
        Fetches the speaker labels and timestamps once the streaming diarization is complete. 
        Useful for a combined pipeline test to avoid additional I/O-instruction overhead. 

        Args:
            result (Annotation): The predictions from the speaker diarization pipeline
        
        Returns:
            list: A segments list containing dictionary objects with the format;
                {
                speaker,
                start (time),
                end (time),
                duration
                }
        """
        if result is None:
            raise ValueError('Parameter result is None.')
        
        speaker_segments = []
        try:
            for segment, _, speaker in result.itertracks(yield_label=True):
                log.info('Assigned speaker label: %s', speaker)

                speaker_segments.append({
                    'speaker': speaker,
                    'start': segment.start,
                    'end': segment.end,
                    'duration': segment.duration
                })
        except Exception as e:
            log.error('Failed with error: %s', e)

        return speaker_segments


"""
@dataclass
class DiarizationConfig:
    model: str = 'pyannote/speaker-diarization-3.0'
    hf_token: str = None
    device: str = 'cpu'

class SpeakerDiarizationPipeline:
    def __init__(self, config: DiarizationConfig):
        self.config = config
        self.cluster_registry = RollingClusters()

        self.pipeline = Pipeline.from_pretrained(
            checkpoint_path=config.model,
            use_auth_token=config.hf_token
        ).to(device=config.device)


    def diarize(
            self, 
            sample, 
            target_sr=16000,
            return_embeddings: bool = False,
            num_speakers: Optional[int] = None,
            min_speakers: Optional[int] = None,
            max_speakers: Optional[int] = None
            ) -> list[dict]:

        if not sample or 'audio' not in sample.keys():
            raise KeyError('Required audio key is missing from sample')

        idx_to_embedding = {}
        audio = sample['audio']
        wav = torch.tensor(audio).unsqueeze(0)   # To get the correct format of (channel, time) Tensor.

        speaker_segments = []
        with ProgressHook() as hook:
            result = self.pipeline({
                'waveform': wav,
                'sample_rate': target_sr
            },
            hook=hook,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            return_embeddings=return_embeddings
            )

        # Manually checking, since pyannote.apply() can return either Annotation or tuple[Annotation, Embeddings]
        if isinstance(result, tuple):
            diarization, embeddings = result
        else:
            diarization = result

        try:
            # Catches value errors thrown by e.g. zip, if the lists are of different length
            if return_embeddings:
                # It returns an output tuple[Annotation, Embedding]
                idx_to_embedding = {
                    speaker_label: emb
                    for speaker_label, emb in zip(diarization.labels(), embeddings)
                }
        except ValueError as e:
            log.error(e)

        try:
            for segment, _, speaker in diarization.itertracks(yield_label=True):
                print(f'Pyannote assigned speaker label: {speaker}')
                if return_embeddings:
                    speaker_emb = idx_to_embedding.get(speaker)
                    speaker = self.cluster_registry.process_chunk_ahc(chunk_embedding=speaker_emb)

                speaker_segments.append({
                    'speaker': speaker,
                    'start': segment.start,
                    'end': segment.end,
                    'duration': segment.duration
                })
        except ValueError as e:
            log.error(e)
        
        return {
            'segments': speaker_segments,
            'embeddings': embeddings if return_embeddings else None    # If return_embeddings=True, output is a tuple[Annotation, Embedding]
        }
    

class RollingClusters:
    def __init__(self, match_threshold=0.3, new_threshold=0.55, max_history=10, method='average', metric='cosine'):
        self.match_threshold = match_threshold
        self.new_threshold = new_threshold
        self.max_history = max_history
        self.method = Categorical([method])
        self.metric = metric

        self.threshold = Uniform(0.0, 2.0)  # assume unit-normalized embeddings

        self.speaker_registry = {}
        self.speaker_count = 0

    def preprocess(self, embedding: np.ndarray):
        if embedding.ndim == 1:
            embedding = np.expand_dims(embedding, axis=0)

        embedding = embedding.astype(np.float32)
        norm = np.linalg.norm(embedding, axis=1, keepdims=True)
        return embedding / np.where(norm == 0, 1e-12, norm)
    
    def compute_centroids(self):
        speakers = list(self.speaker_registry.keys())

        centroids = []
        for speaker in speakers:
            first_dim = self.speaker_registry[speaker][0].shape
            if not all(e.shape == first_dim for e in self.speaker_registry[speaker]):
                raise ValueError('All embeddings in a cluster must have the same dimensions.')

            avg_embedding = np.mean(self.speaker_registry[speaker], axis=0)
            print(f'Checking avg_embedding sum: {sum(avg_embedding)}')
            centroids.append(self.preprocess(avg_embedding)[0])

        return speakers, np.array(centroids)



    def process_chunk_ahc(self, chunk_embedding: np.ndarray):
        # 1. Gather all existing embeddings into a flat matrix while keeping track of speaker IDs
        flat_history = []
        index_to_speaker = []
        
        for speaker_id, emb_list in self.speaker_registry.items():
            for emb in emb_list:
                flat_history.append(emb)
                index_to_speaker.append(speaker_id)


        # If no history exists, register as the first speaker directly
        if not flat_history:
            speaker_id = f'speaker_{self.speaker_count:02d}'
            self.speaker_registry[speaker_id] = [chunk_embedding]
            self.speaker_count += 1
            return speaker_id

        # 2. Append incoming chunk to the end of the embedding pool
        candidate_index = len(flat_history)
        all_embeddings = np.vstack([flat_history, chunk_embedding])

        print(f'Length of the flat history: {len(flat_history)} and length of all_embeddings: {len(all_embeddings)}')
        print(f'Shape of embeddings vector: {all_embeddings.shape}')

        # 3. Compute AHC using Pyannote's metric & linkage scheme
        # Note: method='average', metric='cosine'
        Z = linkage(all_embeddings, method=self.method, metric=self.metric)
        
        # 4. Form flat clusters based on distance threshold (Pyannote typical range: 0.40 to 0.60)
        cluster_labels = fcluster(Z, self.threshold, criterion='distance') - 1

        candidate_label = cluster_labels[candidate_index]
        
        # 5. Check if candidate was grouped with any existing historical vectors
        historical_labels = cluster_labels[:candidate_index]
        matching_indices = np.where(historical_labels == candidate_label)[0]

        if len(matching_indices) > 0:
            print(f'A match is found!')
            # Match found! Map back to the existing speaker ID of the matched historical vector
            matched_speaker_id = index_to_speaker[matching_indices[0]]
            
            # Append candidate to matched speaker's pool
            self.speaker_registry[matched_speaker_id].append(chunk_embedding)
            if len(self.speaker_registry[matched_speaker_id]) > self.max_history:
                self.speaker_registry[matched_speaker_id].pop(0)
                
            return matched_speaker_id
        else:
            # No match found -> Form a brand new speaker cluster
            new_speaker_id = f'speaker_{self.speaker_count:02d}'
            self.speaker_registry[new_speaker_id] = [chunk_embedding]
            self.speaker_count += 1
            return new_speaker_id

    
    
    def process_chunk(self, chunk_embedding: np.ndarray):
        if chunk_embedding is None:
            raise ValueError('Cannot process chunk_embedding that is None')
        
        if np.all(chunk_embedding == 0) or np.linalg.norm(chunk_embedding) < 1e-6:
            return None
        
        processed_embedding = self.preprocess(chunk_embedding)

        if not self.speaker_registry:
            speaker_id = f'speaker_{self.speaker_count:02d}'
            self.speaker_registry[speaker_id] = [processed_embedding[0]]
            self.speaker_count += 1

            print(f'No speaker exists... Adding {speaker_id}. Total speakers: {self.speaker_count}')
            return speaker_id

        speaker_ids, centroids = self.compute_centroids()
        distances = cdist(processed_embedding, centroids, metric='cosine')[0]
        print(f'All distances: {distances}')
        best_match_index = np.argmin(distances)
        min_dist = distances[best_match_index]
        print(f'Minimum distance: {min_dist}')
        best_speaker_id = speaker_ids[best_match_index]

        if min_dist <= self.match_threshold:
            self.speaker_registry[best_speaker_id].append(processed_embedding[0])
            print(f'Best match already exists... Best match speaker is: {best_speaker_id}')
            
            if len(self.speaker_registry[best_speaker_id]) > self.max_history:
                self.speaker_registry[best_speaker_id].pop(0)

            return best_speaker_id

        elif min_dist >= self.new_threshold:

            new_speaker_id = f'speaker_{self.speaker_count:02d}'
            self.speaker_registry[new_speaker_id] = [processed_embedding[0]]
            self.speaker_count += 1
            
            return new_speaker_id
        else: 

            return best_speaker_id
"""