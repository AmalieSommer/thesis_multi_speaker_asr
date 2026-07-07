import itertools

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper import BatchedInferencePipeline
from faster_whisper.audio import pad_or_trim
from faster_whisper.vad import VadOptions, get_speech_timestamps, collect_chunks
from faster_whisper.tokenizer import Tokenizer
from faster_whisper.utils import format_timestamp
from faster_whisper.transcribe import TranscriptionInfo, TranscriptionOptions, get_suppressed_tokens
from ..utils.vad import VAD

from typing import Iterable, List, Optional, Tuple, Union
from ..utils.utils import profile, LOGGING_CONFIG
import logging
import logging.config

logging.config.dictConfig(LOGGING_CONFIG)


class WhisperPipeline(BatchedInferencePipeline):
    """
    Wrapper class for ASR models.

    It will contain the Whisper and Wav2Vec2 models, which will inherit basic functions such as unload()
    """
    logger = logging.getLogger(name='Whisper')

    def __init__(self, compute_type, cpu_threads, model, device='cpu'):
        super().__init__(device)
        self.ts_map = {} # For storing the ts_maps needing when/if recovering the original audio timeline
        self.device = device
        self.model = self.load(
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            model=model
        )
        model_memory = self.load.memory_stats[0]
        self.logger.info('Whisper Model Memory Stats...: Before load: %f, After load: %f, Delta: %f', model_memory['before'], model_memory['after'], model_memory['delta'])

    @profile
    def load(self, compute_type, cpu_threads, model):

        print(f'Loading model...')

        return WhisperModel(
            model_size_or_path=model,
            device=self.device,
            compute_type=compute_type,
            cpu_threads=int(cpu_threads),
            num_workers=1
        )

    def unload(self):
        self.model = None

    def transcribe(
            self, 
            audio_chunks,
            chunks_metadata,
            ids,
            language = None, 
            task = "transcribe", 
            log_progress = False, 
            beam_size = 5, 
            best_of = 5, 
            patience = 1, 
            length_penalty = 1, 
            repetition_penalty = 1, 
            no_repeat_ngram_size = 0, 
            temperature = [
                0.0,
                0.2,
                0.4,
                0.6,
                0.8,
                1.0,
            ], 
            compression_ratio_threshold = 2.4, 
            log_prob_threshold = -1, 
            no_speech_threshold = 0.6, 
            condition_on_previous_text = True, 
            prompt_reset_on_temperature = 0.5, 
            initial_prompt = None, 
            prefix = None, 
            suppress_blank = True, 
            suppress_tokens = [-1], 
            without_timestamps = True, 
            max_initial_timestamp = 1, 
            word_timestamps = False, 
            prepend_punctuations = "\"'“¿([{-", 
            append_punctuations = "\"'.。,，!！?？:：”)]}、", 
            multilingual = False, 
            vad_filter = True, 
            vad_parameters = None, 
            max_new_tokens = None, 
            chunk_length = None, 
            clip_timestamps = None, 
            clip_timestamps_provided: bool = False,
            hallucination_silence_threshold = None, 
            batch_size = 8, 
            hotwords = None, 
            language_detection_threshold = 0.5, 
            language_detection_segments = 1
            ):
        
        sampling_rate = self.model.feature_extractor.sampling_rate

        if multilingual and not self.model.model.is_multilingual:
            self.model.logger.warning(
                "The current model is English-only but the multilingual parameter is set to"
                "True; setting to False instead."
            )
            multilingual = False
        chunk_length = chunk_length or self.model.feature_extractor.chunk_length
      
        # If either vad was applied to the audio or the audio was clipped, it will restore the original timeline:
        if clip_timestamps_provided | vad_filter:
            # Create the mappings to map from the speech-only timeline to the original timeline.
            if len(self.ts_map) == 0:
                # Initialize and generate a new mapping:
                for key, group in itertools.groupby(clip_timestamps, lambda x: x['id']):
                    ts_mapping = VAD()
                    ts_mapping.build_mapping([{'start': x['start'], 'end': x['end']} for x in group])
                    self.ts_map[key] = ts_mapping
            else:
                items_to_keep = []
                for key, group in itertools.groupby(clip_timestamps, lambda x: x['id']):
                    group_ = [{'start': x['start'], 'end': x['end']} for x in group]
                    if key in self.ts_map.keys():
                        ts_mapping = self.ts_map.get(key)
                        ts_mapping.update_mapping(group_)
                        items_to_keep.append(key)
                    else:
                        # it needs to create a new timestamp mapping
                        ts_mapping = VAD()
                        ts_mapping.build_mapping(group_)
                        self.ts_map[key] = ts_mapping
                        items_to_keep.append(key)
                    
                self.ts_map = {k: v for k, v in self.ts_map.items() if k in items_to_keep}

        duration_after_processing = (
            sum((segment["end"] - segment["start"]) for segment in clip_timestamps)
            / sampling_rate
        )

        features = (
            [self.model.feature_extractor(chunk)[..., :-1] for chunk in audio_chunks]
            if duration_after_processing
            else []
        )

        all_language_probs = None
        # detecting the language if not provided
        if language is None:
            if not self.model.model.is_multilingual:
                language = "en"
                language_probability = 1
            else:
                (
                    language,
                    language_probability,
                    all_language_probs,
                ) = self.model.detect_language(
                    features=np.concatenate(
                        features
                        + [
                            np.full((self.model.model.n_mels, 1), -1.5, dtype="float32")
                        ],
                        axis=1,
                    ),  # add a dummy feature to account for empty audio
                    language_detection_segments=language_detection_segments,
                    language_detection_threshold=language_detection_threshold,
                )

                self.model.logger.info(
                    "Detected language '%s' with probability %.2f",
                    language,
                    language_probability,
                )
        else:
            if not self.model.model.is_multilingual and language != "en":
                self.model.logger.warning(
                    "The current model is English-only but the language parameter is set to '%s'; "
                    "using 'en' instead." % language
                )
                language = "en"

            language_probability = 1

        tokenizer = Tokenizer(
            self.model.hf_tokenizer,
            self.model.model.is_multilingual,
            task=task,
            language=language,
        )

        features = (
            np.stack([pad_or_trim(feature) for feature in features]) if features else []
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
            multilingual=multilingual,
            without_timestamps=without_timestamps,
            max_initial_timestamp=0.0,
        )

        info = TranscriptionInfo(
            language=language,
            language_probability=language_probability,
            duration=duration_after_processing,
            duration_after_vad=duration_after_processing,
            transcription_options=options,
            vad_options=vad_parameters,
            all_language_probs=all_language_probs,
        )

        segments = self._batched_segments_generator(
            features,
            tokenizer,
            chunks_metadata,
            batch_size,
            options,
            log_progress,
        )

        if clip_timestamps_provided | vad_filter:
            segments = self.restore_original_timeline(
                segments, ids
            )
        return segments, info
    

    def restore_original_timeline(self, segments, segment_ids):
        for segment in segments:
            if segment.words:
                segment_id = segment_ids[segment.id - 1]
                ts_map = self.ts_map[segment_id]
                words = []
                for word in segment.words:
                    # Ensure the word start and end times are resolved to the same chunk.
                    middle = (word.start + word.end) / 2
                    chunk_index = ts_map.get_chunk_index(middle)
                    word.start = ts_map.get_original_time(word.start, chunk_index)
                    word.end = ts_map.get_original_time(word.end, chunk_index)
                    words.append(word)

                segment.start = words[0].start
                segment.end = words[-1].end
                segment.words = words

            else:
                segment.start = ts_map.get_original_time(segment.start)
                segment.end = ts_map.get_original_time(segment.end, is_end=True)

            yield segment
