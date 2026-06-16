from tqdm import tqdm
import time
from memory_profiler import profile
import torch
from pyannote.audio.pipelines.utils.hook import ProgressHook
from .data import AudioData, stream_audio
from .models.asr import Whisper
import librosa
from torch.utils.data import DataLoader
import numpy as np
from faster_whisper.tokenizer import Tokenizer
from faster_whisper.audio import pad_or_trim
import os
from multi_speaker_asr.models.alignment import Wav2Vec2


def transcribe(audio_chunks, batchedModel: Whisper):
    features = (
        [batchedModel.model.model.feature_extractor(chunk)[..., :-1] for chunk in audio_chunks]
    )

    tokenizer = Tokenizer(
        batchedModel.model.model.hf_tokenizer,
        False,
        language='da'
    )

    features = (
        np.stack([pad_or_trim(feature) for feature in features] if features else [])
    )

    segments = batchedModel.model._batched_segments_generator(
        features=features,
        tokenizer=tokenizer,
        chunks_metadata=None,
        batch_size=4,
        options=None,
        log_progress=None
    )
    return segments


def collator(batch):
    return batch



def generate_chunks(audio, sr: int = 16000):
    curr_duration = 0.0 # a counter for duration up until the 30 seconds window limit
    total_duration = 0.0
    max_duration = 30.0


    current_audio = np.array([], dtype=np.float32)  # To hold the concatenated audio chunks into one audio array
    audio_chunks = []                               # To hold each ndarray audio chunk
    chunks_metadata = []                            # To hold dict objects with offset value corresponding to the duration of the previous chunk, duration of the current chunk and the index corresponding to the audio_chunk array index
    segments_metadata = []                          # To hold dict objects with audio index and the corresponding duration value

    for index, audio_item in enumerate(audio):
        # Either one audio item if it is within the block limits, or multiple small blocked segments:
        audio_duration = librosa.get_duration(y=audio_item, sr=sr)
        if audio_duration + curr_duration > max_duration:
            # Reset accumulation of audio chunks and metadata and add this to the next iteration of transcription chunks...
            audio_chunks.append(current_audio)
            total_duration += curr_duration
            chunks_metadata.append({
                'offset': total_duration,
                'duration': curr_duration,
                'segments': segments_metadata
            })

            # Reset values for the remaining parts of the audio and add the current audio chunk to this new iteration:
            current_audio = audio_item
            segments_metadata = []
            segments_metadata.append({
                'audio_index': index,
                'segment_duration': audio_duration
            })
            curr_duration = audio_duration

        else:
            # Add the next chunk of audio to the batch prepared for processing:
            current_audio = np.concatenate(
                (current_audio, audio_item)
            )
            curr_duration += audio_duration
            segments_metadata.append({
                'audio_index': index,
                'segment_duration': audio_duration
            })



def inference(dataset: AudioData, model: Wav2Vec2, pre_segmented: bool = False):
    loader = DataLoader(
        dataset=dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=collator
    )

    # Add carbon tracking:
    print('Starting inference...')
    try:
        results = []
        batches = []
        for batch in tqdm(loader):
            start_time = time.time()
            print(f'Start time...: {start_time}')

            # TODO: Test out implementation of combining segments into one input to the forward step            
            for sample in batch:
                audio = sample['audio']

                if not pre_segmented:
                    # The audio is long-form and should be streamed in short chunks into memory:
                    generator = stream_audio(audio=audio)

                    for chunk in generator['stream']:
                        output = model.run_pipeline(
                            input=chunk
                        )
                        batches.append(output)

                else:
                    # The audio is delivered in segments of short size and should be batched together:
                    wav, sr = librosa.load(path=audio, sr=dataset.target_sr)
                    wav_length = librosa.get_duration(y=wav, sr=sr)
                    output = model.run_pipeline(
                            input=chunk
                        )
                    batches.append(output)
                    
            processing_time = time.time() - start_time
            rtf = processing_time 
            results.append({
                'id': sample['id'],
                'segments': batches,
                'rtf': rtf
            })
            print(f'RTF: {rtf}')            
    except Exception as e:
        print(f'An error occurred...{e}')

    finally:
        model.unload()
        model = None
        
        return results



@profile
def inference_asr(dataset: AudioData, model: Whisper, pre_segmented: bool = False):

    loader = DataLoader(
        dataset=dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=collator
    )

    # Add carbon tracking:
    print('Starting inference...')
    try:
        results = []
        batches = []
        for batch in tqdm(loader):
            start_time = time.time()
            print(f'Start time...: {start_time}')

            # TODO: Test out implementation of combining segments into one input to the forward step            
            for sample in batch:
                audio = sample['audio']

                if not pre_segmented:
                    # The audio is long-form and should be streamed in short chunks into memory:
                    generator = stream_audio(audio=audio)

                    for chunk in generator['stream']:
                        segments, _ = model.model.transcribe(
                            audio=chunk,
                            language='da',
                            batch_size=4
                        )
                        for segment in segments:
                            item = {
                                'start': segment.start,
                                'end': segment.end,
                                'text': segment.text
                            }
                            batches.append(item)

                else:
                    # The audio is delivered in segments of short size and should be batched together:
                    wav, sr = librosa.load(path=audio, sr=dataset.target_sr)
                    wav_length = librosa.get_duration(y=wav, sr=sr)
                    segments, _ = model.model.transcribe(
                            audio=wav,
                            language='da',
                            chunk_length=wav_length,
                            batch_size=4
                        )
                    for segment in segments:
                        item = {
                            'start': segment.start,
                            'end': segment.end,
                            'text': segment.text
                        }
                        batches.append(item)


            processing_time = time.time() - start_time
            rtf = processing_time 
            results.append({
                'id': sample['id'],
                'segments': batches,
                'rtf': rtf
            })
            print(f'RTF: {rtf}')            
    except Exception as e:
        print(f'An error occurred...{e}')

    finally:
        model.unload()
        model = None
        
        return results

"""
@profile
def inference_asr(dataset: AudioData, model: Whisper):
    # Add carbon tracking:
    print('Starting inference...')
    try:
        results = []
        
        for sample in tqdm(dataset, total=55):
            start_time = time.time()
            print(f'Start time...: {start_time}')

            audio_path = sample['path']
            audio_stream = stream_audio(
                audio=audio_path
            )

            seg_list = [] # stores segments from the whole audio file, even though it is processed in streamed chunks
            for sample_stream in audio_stream:
                sample_duration = int(librosa.get_duration(y=sample_stream, sr=dataset.target_sr))
                print(f'Duration: {sample_duration}')

                
                segments, _ = model.model.transcribe(
                      audio=sample_stream,
                      batch_size=4,
                      language='da',
                      word_timestamps=True,
                      chunk_length=sample_duration
                )

                
                for segment in segments:
                    item = {
                        'start': segment.start,
                        'end': segment.end,
                        'text': segment.text
                    }
                    seg_list.append(item)
            
            # After it has streamed the entire audio, calculate the processing time:
            processing_time = time.time() - start_time
            rtf = processing_time 
            results.append({
                'id': sample['id'],
                'segments': seg_list,
                'rtf': rtf
            })
            print(f'RTF: {rtf}')            
    except Exception as e:
        print(f'An error occurred...{e}')

    finally:
        model.unload()
        model = None
        
        return results
    """

#@profile
def inference_diarize(loader, model, batch_size):

    try:
        results = []
        for batch in tqdm(loader):

            for sample in batch:
                start_time = time.time()
                print(f'Start time...: {start_time}')
                
                id = sample['id']
                # TODO: Could add config params for setting min and max speakers if known...
                print(type(model.model))

                wav = torch.tensor(sample['audio']).unsqueeze(0)   # To get the correct format of (channel, time) Tensor.
                print(f'Shape of tensor: {wav.shape}')


                with ProgressHook() as hook:
                    output = model.model(
                        {'waveform': wav, 'sample_rate': sample['sr']}, 
                        hook=hook
                    )
                
                processing_time = time.time() - start_time
                rtf = processing_time # currently just passes total processing time for full batch

                results.append({
                    'id': id,
                    'path': sample['path'],
                    'speaker_segments': output,
                    'rtf': rtf
                })

    finally:
        model.unload()
        model = None

        return results



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
