from tqdm import tqdm
from .data import AudioData, stream_audio, cast
import librosa
from torch.utils.data import DataLoader
from multi_speaker_asr.models.asr import ASR, Whisper
from .data import clean_transcription, read_bytes
from pyannote.audio.pipelines.utils.hook import ProgressHook
from multi_speaker_asr.models.diarization import Diarize
from multi_speaker_asr.models.alignment import Wav2Vec2
import torch
from multi_speaker_asr.utils.utils import LOGGING_CONFIG, profile
import gc
import time
import logging
import logging.config
import sys


logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(name='Evaluate')

def collator_fn(batch):
    # TODO!!!
    """Should ensure that it returns batch object of the same format, i.e. same parameter names and types"""
    return batch


def batched_inference(data: AudioData, model: ASR):
    """
    This assumes the dataset is pre-segmented into short chunks, and will process the chunks in batches.
    """
    batch_size = 2
    loader = DataLoader(
        dataset=data,
        batch_size=batch_size,
        num_workers=1,
        shuffle=False,
        collate_fn=collator_fn
    )
    before_alignment = []
    try: 
        
        for counter, batch in enumerate(tqdm(loader, total=(8440 // batch_size))):
            if len(batch) == 0:
                continue

            if counter == 4:
                break
            
            for sample in batch:
                # Processing time for RTF calculation per audio recording
                start_process_time = time.process_time()
                iterable_segments = []
                
                if isinstance(sample['audio'], bytes):
                    wav = read_bytes(sample['audio'])
                if isinstance(model, Whisper):
                    outputs, info = model.pipeline.transcribe(
                        audio=wav,
                        language='da'
                    )
                    
                    for out in outputs:
                        iterable_segments.append(cast(out))

                    end_process_time = time.process_time()
                    cpu_time = end_process_time - start_process_time
                    rtf_sample = cpu_time / info.duration # processing time divided by the actual audio duration
                
                logger.info('Batched Inference CPU Time: %f', cpu_time)
                logger.info('Batched Inference Real-Time Factor (RTF): %f', rtf_sample)

                before_alignment.append(
                    (sample['id'], sample['audio'], iterable_segments)
                )
            logger.info('Before Alignment List Memory...: %f', sys.getsizeof(before_alignment))
    except Exception as e:
        print(f'An error occurred...{e}')
    
    finally:
        model.unload()
        del model
        del loader
        del data
        gc.collect()

    return before_alignment



def streamed_inference(data: AudioData, model: ASR):
    """
    This assumes the dataset contains raw long-form audio recordings, and will therefore include streaming (using librosa) and process each audio stream sequentially.
    """
    batch_size = 1
    loader = DataLoader(
        dataset=data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator_fn,
        num_workers=0
    )
    iterable_segments = []
    before_alignment = []
    try:
        print('Starting inference evaluation...')
        for iter, batch in enumerate(tqdm(loader)):
            if len(batch) == 0:
                continue
            
            if iter == 1:
                break

            for sample in batch:
                start_process_time = time.process_time()
                stream = stream_audio(
                    audio=sample['audio']
                )
                audio_duratio = sample['end']
                inner_tqdm = tqdm(stream, total=int(audio_duratio / 30.0))

                running_duration = 0.0  # To keep a count of the duration of the amount of audio streams processed so far...
                for index, y in enumerate(inner_tqdm):

                    inner_tqdm.refresh() # To enure the progressbar is visible for the individual audio recordings...
                    if isinstance(model, Whisper):
                        outputs, info = model.pipeline.transcribe(
                        audio=y,
                        language='da',
                        batch_size=1,
                        word_timestamps=True
                        )
                        for out in outputs:
                        
                            if index > 0:
                                # Add the duration of the current running duration to start, and the duration of the current stream to the end as well as the running duration:
                                out.start += running_duration
                                out.end += running_duration

                            iterable_segments.append(cast(out))
                            running_duration += info.duration

                end_process_time = time.process_time()
                cpu_time = end_process_time - start_process_time
                rtf_sample = cpu_time / info.duration # processing time divided by the actual audio duration
                
                logger.info('Streamed Inference CPU Time: %f', cpu_time)
                logger.info('Streamed Inference Real-Time Factor (RTF): %f', rtf_sample)

                before_alignment.append(
                    (sample['id'], sample['audio'], iterable_segments)
                )

    except Exception as e:
        print(f'An error occurred...{e}')
    
    finally:
        model.unload()
        del model
        gc.collect()

        return before_alignment


def inference_streaming_diarize(data: AudioData, model: Diarize):
    batch_size = 2
    loader = DataLoader(
        dataset=data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator_fn,
        num_workers=1
    )
    results = []

    try:
        for iter, batch in enumerate(tqdm(loader)):

            if iter == 1:
                break

            for sample in batch:
                start_process_time = time.process_time()
                if isinstance(sample['audio'], bytes):
                    audio = read_bytes(sample['audio'])
                else:
                    audio, _ = librosa.load(sample['audio'], sr=data.target_sr)

                audio_time = librosa.get_duration(y=audio, sr=data.target_sr)
                wav = torch.tensor(audio).unsqueeze(0)   # To get the correct format of (channel, time) Tensor.
                print(f'Shape of tensor: {wav.shape}')

                speaker_segments = []
                with ProgressHook() as hook:
                    output = model.model(
                        {'waveform': wav, 'sample_rate': data.target_sr}, 
                        hook=hook,
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
                        #print(f"{segment.start:.2f} --> {segment.end:.2f} ({segment.duration:.2f}s) Speaker: {speaker}")
                    
                end_process_time = time.process_time()
                cpu_time = end_process_time - start_process_time
                rtf_sample = cpu_time / audio_time # processing time divided by the actual audio duration
            
                logger.info('Diarization Inference CPU Time: %f', cpu_time)
                logger.info('Diarization Inference Real-Time Factor (RTF): %f', rtf_sample)
                
                results.append({
                'id': sample['id'],
                'speaker_segments': speaker_segments
                })
    except Exception as e:
        print(f'Failed with error...: {e}')
    finally:
        model.unload()
        del model
        del loader
        gc.collect()

    return results


def align_transcripts(asr_output: list[tuple], config):
    try:
        alignment_pipeline = Wav2Vec2(config=config, device='cpu')
        after_alignment = []
        for (id, audio, segments) in asr_output:
            start_process_time = time.process_time()

            print(f'Running alignment for audio...: {id}')
            if isinstance(audio, bytes):
                wav = read_bytes(audio)
            else:
                wav, _ = librosa.load(path=audio, sr=16000)

            audio_time = librosa.get_duration(y=wav, sr=16000)
            transcription_result = alignment_pipeline.run_alignment(
                transcript=segments,
                audio=wav
            )

            end_process_time = time.process_time()
            cpu_time = end_process_time - start_process_time
            rtf_sample = cpu_time / audio_time # processing time divided by the actual audio duration

            logger.info('Aligning Transcripts CPU Time: %f', cpu_time)
            logger.info('Aligning Transcripts Real-Time Factor (RTF): %f', rtf_sample)

            after_alignment.append({
                'id': id,
                'transcript': transcription_result
            })
    except Exception as e:
        print(f'Failed with error: {e}')
    finally:
        alignment_pipeline.unload()
        del alignment_pipeline
        gc.collect()

        print('Finished aligning the transcripts...!')
    
    return after_alignment

"""
def eval_bert(bert, info):
    for item in info.values():
        encoded_pred_transcripts = bert.tokenizer(item['pred'], padding=True, truncation=True, return_tensors='pt')
        encoded_target_transcripts = bert.tokenizer(item['target'], padding=True, truncation=True, return_tensors='pt')

        pred_transcripts_embeddings = bert(
            input_ids=encoded_pred_transcripts.input_ids.to(bert.device),
            attention_mask=encoded_pred_transcripts.attention_mask.to(bert.device)
        )
        transcripts_embeddings = bert(
            input_ids=encoded_target_transcripts.input_ids.to(bert.device),
            attention_mask=encoded_target_transcripts.attention_mask.to(bert.device)
        )
        score = compute_cosine_sim(pred_transcripts_embeddings, transcripts_embeddings)
        item['semdist'] = score

    return info
    """