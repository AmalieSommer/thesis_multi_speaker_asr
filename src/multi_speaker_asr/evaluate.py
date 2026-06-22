from tqdm import tqdm
from .data import AudioData, stream_audio, cast
import librosa
from torch.utils.data import DataLoader
from jiwer import wer, cer
from multi_speaker_asr.models.asr import ASR, Whisper, Wav2Vec2
from .data import clean_transcription, read_bytes
from pyannote.audio.pipelines.utils.hook import ProgressHook
from multi_speaker_asr.models.diarization import Diarize
import torch
import soundfile as sf
import io
from collections.abc import Iterable
from whisperx.schema import SingleSegment

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
        num_workers=0,
        shuffle=False,
        collate_fn=collator_fn
    )
    results = []
    before_alignment = []
    try: 
        for counter, batch in enumerate(tqdm(loader, total=(8440 // batch_size))):
            if len(batch) == 0:
                continue

            if counter == 1:
                break

            for sample in batch:
                iterable_segments = []
                print(type(sample['audio']))
                
                if isinstance(sample['audio'], bytes):
                    wav = read_bytes(sample['audio'])

                if isinstance(model, Whisper):
                    outputs, info = model.pipeline.transcribe(
                        audio=wav,
                        language='da'
                    )
                    for out in outputs:
                        results.append({
                            'id': sample['id'],
                            'start': out.start,
                            'end': out.end,
                            'ref': sample['text'],
                            'hyp': out.text
                        })
                        iterable_segments.append(cast(out))

                elif isinstance(model, Wav2Vec2):
                    audio, _ = librosa.load(path=audio, sr=data.target_sr)
                    output = model.run_pipeline(
                        input=audio
                    )
                    clean_ref = clean_transcription(sample['text'])
                    clean_hyp = clean_transcription(out.text)
                    word_err_rate = wer(reference=clean_ref, hypothesis=clean_hyp)
                    char_err_rate = cer(reference=clean_ref, hypothesis=clean_hyp)
                    results.append({
                        'cer': char_err_rate,
                        'wer': word_err_rate,
                        'ref': sample['text'],
                        'hyp': output.text,
                        'id': sample['id']
                    })
                before_alignment.append(
                            (sample['id'], sample['audio'], iterable_segments)
                        )
    except Exception as e:
        print(f'An error occurred...{e}')
    
    finally:
        return results, before_alignment


def streamed_inference(data: AudioData, model: ASR):
    """
    This assumes the dataset contains raw long-form audio recordings, and will therefore include streaming (using librosa) and process each audio stream sequentially.
    """
    batch_size = 2
    loader = DataLoader(
        dataset=data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator_fn,
        num_workers=0
    )
    results = []
    iterable_segments = []
    before_alignment = []
    try:
        print('Starting inference evaluation...')
        for iter, batch in enumerate(tqdm(loader, total=8440 // batch_size)):
            if len(batch) == 0:
                continue
            
            if iter == 1:
                break

            for sample in batch:
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
                            start = out.start
                            end = out.end
                        
                            if index > 0:
                                # Add the duration of the current running duration to start, and the duration of the current stream to the end as well as the running duration:
                                start = out.start + running_duration
                                end = out.end + running_duration

                            results.append({
                                'id': sample['id'],
                                'start': start,
                                'end': end,
                                'hyp': out.text,
                                'words': out.words
                            })
                            iterable_segments.append(outputs)
                            running_duration += info.duration

                before_alignment.append(
                    (sample['id'], sample['audio'], iterable_segments)
                )

    except Exception as e:
        print(f'An error occurred...{e}')
    
    finally:
        return results, before_alignment


def inference_streaming_diarize(data: AudioData, model: Diarize):
    loader = DataLoader(
        dataset=data,
        batch_size=2,
        shuffle=False,
        collate_fn=collator_fn,
        num_workers=0
    )
    results = []

    try:
        for iter, batch in enumerate(tqdm(loader)):

            if iter == 1:
                break

            for sample in batch:
                """
                stream = stream_audio(
                    audio=sample['audio']
                )
                audio_duratio = sample['end']
                inner_tqdm = tqdm(stream, total=int(audio_duratio / 30.0))
                
                for index, y in enumerate(inner_tqdm):
                """
                if isinstance(sample['audio'], bytes):
                    audio = read_bytes(sample['audio'])
                else:
                    audio, _ = librosa.load(sample['audio'], sr=data.target_sr)
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
                        print(f"{segment.start:.2f} --> {segment.end:.2f} ({segment.duration:.2f}s) Speaker: {speaker}")
                    
                    
                    results.append({
                    'id': sample['id'],
                    'speaker_segments': speaker_segments
                })
    except Exception as e:
        print(f'Failed with error...: {e}')
    finally:
        return results


"""
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
    """