from tqdm import tqdm
from .data import AudioData, stream_audio
import librosa
from torch.utils.data import DataLoader
from jiwer import wer, cer
from multi_speaker_asr.models.asr import ASR, Whisper, Wav2Vec2
from .data import clean_transcription
from pyannote.audio.pipelines.utils.hook import ProgressHook
from multi_speaker_asr.models.diarization import Diarize
import torch

def collator_fn(batch):
    # TODO!!!
    """Should ensure that it returns batch object of the same format, i.e. same parameter names and types"""
    return batch


def batched_inference(data: AudioData, model: ASR):
    """
    This assumes the dataset is pre-segmented into short chunks, and will process the chunks in batches.
    """
    loader = DataLoader(
        dataset=data,
        batch_size=2,
        num_workers=1,
        shuffle=False,
        collate_fn=collator_fn
    )
    results = []
    try: 
        for batch in tqdm(loader):
            if len(batch) == 0:
                continue

            for sample in batch:
                audio = sample['audio']


                if isinstance(model, Whisper):
                    outputs, _ = model.pipeline.transcribe(
                        audio=audio,
                        language='da',
                        batch_size=1
                    )
                    for out in outputs:
                        clean_ref = clean_transcription(sample['text'])
                        clean_hyp = clean_transcription(out.text)
                        word_err_rate = wer(reference=clean_ref, hypothesis=clean_hyp)
                        char_err_rate = cer(reference=clean_ref, hypothesis=clean_hyp)

                        results.append({
                            'cer': char_err_rate,
                            'wer': word_err_rate,
                            'start': out.start,
                            'end': out.end,
                            'ref': sample['text'],
                            'hyp': out.text,
                            'id': sample['id']
                        })

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

    except Exception as e:
        print(f'An error occurred...{e}')
    
    finally:
        model.unload()
        model = None
        loader = None
        
        return results


def streamed_inference(data: AudioData, model: ASR):
    """
    This assumes the dataset contains raw long-form audio recordings, and will therefore include streaming (using librosa) and process each audio stream sequentially.
    """
    loader = DataLoader(
        dataset=data,
        batch_size=2,
        shuffle=False,
        collate_fn=collator_fn,
        num_workers=1
    )
    results = []

    try:
        print('Starting inference evaluation...')
        for batch in tqdm(loader, total=data.len_estimate):
            if len(batch) == 0:
                continue

            for sample in batch:
                stream = stream_audio(
                    audio=sample['audio']
                )
                audio_duratio = sample['end']
                inner_tqdm = tqdm(stream, total=int(audio_duratio / 30.0))
<<<<<<< HEAD
                for y in inner_tqdm:
                    #print('Iterating streamed block...')

                    inner_tqdm.refresh()    # To ensure the terminal gets refreshed at every inner-loop iteration...
                    
=======

                running_duration = 0.0  # To keep a count of the duration of the amount of audio streams processed so far...
                for index, y in enumerate(inner_tqdm):

                    inner_tqdm.refresh() # To enure the progressbar is visible for the individual audio recordings...
>>>>>>> d36b066bf05dd9c0183b9395ff96960f7f4edf6c
                    if isinstance(model, Whisper):
                        outputs, info = model.pipeline.transcribe(
                        audio=y,
                        language='da',
                        batch_size=1
                        )
                        for out in outputs:
                            start = out.start
                            end = out.end
                            if index > 0:
                                # Add the duration of the current running duration to start, and the duration of the current stream to the end as well as the running duration:
                                start = out.start + running_duration
                                end = out.end + running_duration

                            results.append({
                                'start': start,
                                'end': end,
                                'hyp': out.text,
                                'id': sample['id']
                            })
                            running_duration += info.duration
                            print(f'Duration so far is... {running_duration}sec.')

                    elif isinstance(model, Wav2Vec2):
                        output = model.run_pipeline(
                            input=y
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


    except Exception as e:
        print(f'An error occurred...{e}')
    
    finally:
        return results


def inference_streaming_diarize(data: AudioData, model: Diarize):
    loader = DataLoader(
        dataset=data,
        batch_size=2,
        shuffle=False,
        collate_fn=collator_fn,
        num_workers=1
    )
    results = []

    try:
        for batch in tqdm(loader):

            for sample in batch:
                stream = stream_audio(
                    audio=sample['audio']
                )
                audio_duratio = sample['end']
                inner_tqdm = tqdm(stream, total=int(audio_duratio / 30.0))

                for index, y in enumerate(inner_tqdm):
                    wav = torch.tensor(y)   # To get the correct format of (channel, time) Tensor.
                    print(f'Shape of tensor: {wav.shape}')


                    with ProgressHook() as hook:
                        output = model.model(
                            {'waveform': wav, 'sample_rate': data.target_sr}, 
                            hook=hook
                        )
                    
                    results.append({
                    'id': sample['id'],
                    'speaker_segments': output
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