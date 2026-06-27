from transformers import pipeline
import numpy as np

from faster_whisper import WhisperModel
from faster_whisper import BatchedInferencePipeline
from faster_whisper.audio import pad_or_trim
from faster_whisper.vad import VadOptions
from faster_whisper.tokenizer import Tokenizer
from faster_whisper.transcribe import TranscriptionInfo, TranscriptionOptions, get_suppressed_tokens

from typing import BinaryIO, Iterable, List, Optional, Tuple, Union
from ..utils.utils import profile, LOGGING_CONFIG
import logging
import logging.config

logging.config.dictConfig(LOGGING_CONFIG)


class ASR:
    """
    Wrapper class for ASR models.

    It will contain the Whisper and Wav2Vec2 models, which will inherit basic functions such as unload()
    """

    pass

    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.pipeline = None


    def unload(self):
        self.model = None



class Whisper(ASR):
    logger = logging.getLogger(name='Whisper')

    def __init__(self, compute_type, cpu_threads, device='cpu', model='CoRal-project/roest-v3-whisper-1.5b'):
        super().__init__(model, device)
        self.load(
            compute_type=compute_type,
            cpu_threads=cpu_threads
        )
        model_memory = self.load.memory_stats[0]
        self.logger.info('Whisper Model Memory Stats...: Before load: %f, After load: %f, Delta: %f', model_memory['before'], model_memory['after'], model_memory['delta'])


    @profile
    def load(self, compute_type, cpu_threads):

        print(f'Loading model...')

        whisper_model = WhisperModel(
            model_size_or_path=self.model,
            device=self.device,
            compute_type=compute_type,
            cpu_threads=int(cpu_threads),
            num_workers=1
        )
        self.pipeline = BatchedInferencePipeline(
            model=whisper_model
        )

    def transcribe(
            self, 
            audio_chunks: list, 
            chunks_metadata: list,
            batch_size: int = 4,
            language: Optional[str] = 'da',
            task: str = "transcribe",
            log_progress: bool = False,
            beam_size: int = 5,
            best_of: int = 5,
            patience: float = 1,
            length_penalty: float = 1,
            repetition_penalty: float = 1,
            no_repeat_ngram_size: int = 0,
            temperature: Union[float, List[float], Tuple[float, ...]] = [
                0.0,
                0.2,
                0.4,
                0.6,
                0.8,
                1.0,
            ],
            compression_ratio_threshold: Optional[float] = 2.4,
            log_prob_threshold: Optional[float] = -1.0,
            no_speech_threshold: Optional[float] = 0.6,
            condition_on_previous_text: bool = True,
            prompt_reset_on_temperature: float = 0.5,
            initial_prompt: Optional[Union[str, Iterable[int]]] = None,
            prefix: Optional[str] = None,
            suppress_blank: bool = True,
            suppress_tokens: Optional[List[int]] = [-1],
            without_timestamps: bool = False,
            max_initial_timestamp: float = 1.0,
            word_timestamps: bool = False,
            prepend_punctuations: str = "\"'“¿([{-",
            append_punctuations: str = "\"'.。,，!！?？:：”)]}、",
            multilingual: bool = False,
            vad_filter: bool = False,
            vad_parameters: Optional[Union[dict, VadOptions]] = None,
            max_new_tokens: Optional[int] = None,
            chunk_length: Optional[int] = None,
            clip_timestamps: Union[str, List[float]] = "0",
            hallucination_silence_threshold: Optional[float] = None,
            hotwords: Optional[str] = None,
            language_detection_threshold: Optional[float] = 0.5,
            language_detection_segments: int = 1
            ):
        
        print('Model is multilingual...: ', self.pipeline.model.model.is_multilingual)
        print('Supported languages: ', self.pipeline.model.supported_languages)
        
        # Generate features from the audio chunks:
        features = (
            [self.pipeline.model.feature_extractor(chunk)[..., :-1] for chunk in audio_chunks]
            if audio_chunks else []
        )
        features = (
            np.stack([pad_or_trim(feature) for feature in features]) 
            if features else []
            )

        tokenizer = Tokenizer(
            tokenizer=self.pipeline.model.hf_tokenizer,
            multilingual=self.pipeline.model.model.is_multilingual,
            task=task,
            language=language
        )

        options = TranscriptionOptions(
            beam_size=beam_size,
            best_of=best_of,
            patience=patience,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            log_prob_threshold=log_prob_threshold,
            no_speech_threshold=no_speech_threshold,
            compression_ratio_threshold=compression_ratio_threshold,
            temperatures=(
                temperature[:1]
                if isinstance(temperature, (list, tuple))
                else [temperature]
            ),
            initial_prompt=initial_prompt,
            prefix=prefix,
            suppress_blank=suppress_blank,
            suppress_tokens=(
                get_suppressed_tokens(tokenizer, suppress_tokens)
                if suppress_tokens
                else suppress_tokens
            ),
            prepend_punctuations=prepend_punctuations,
            append_punctuations=append_punctuations,
            max_new_tokens=max_new_tokens,
            hotwords=hotwords,
            word_timestamps=word_timestamps,
            hallucination_silence_threshold=None,
            condition_on_previous_text=False,
            clip_timestamps=clip_timestamps,
            prompt_reset_on_temperature=0.5,
            multilingual=self.pipeline.model.model.is_multilingual,
            without_timestamps=without_timestamps,
            max_initial_timestamp=0.0,
        )

        info = TranscriptionInfo(
            language=language,
            language_probability=1.0,
            duration=sum([chunk.shape[0] for chunk in audio_chunks]),
            duration_after_vad=None,
            transcription_options=options,
            vad_options=vad_parameters,
            all_language_probs=1.0,
        )

        segments = self.pipeline._batched_segments_generator(
            features,
            tokenizer,
            chunks_metadata,
            batch_size,
            options,
            log_progress
        )
        return segments


class Wav2Vec2(ASR):
    def __init__(self, model='CoRal-project/roest-v3-wav2vec2-315m', device='cpu'):
        super().__init__(model, device)

    def load(self):
        self.pipeline = pipeline(
            task='automatic-speech-recognition',
            model=self.model,
            device=self.device
        )

