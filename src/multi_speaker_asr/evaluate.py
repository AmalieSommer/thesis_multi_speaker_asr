from tqdm import tqdm
import time
from memory_profiler import profile
import torch
from pyannote.audio.pipelines.utils.hook import ProgressHook
from .data import AudioData, stream_audio
from .models.asr import Whisper


#@profile
def inference_asr(dataset: AudioData, model: Whisper):
    # Add carbon tracking:
    print('Starting inference...')
    try:
        results = []
        
        for sample in tqdm(dataset):
            start_time = time.time()
            print(f'Start time...: {start_time}')

            audio_path = sample['path']
            audio_stream = stream_audio(
                audio=audio_path
            )
            seg_list = [] # stores segments from the whole audio file, even though it is processed in streamed chunks
            for sample_stream in audio_stream:
                segments, _ = model.model.transcribe(
                      audio=sample_stream,
                      batch_size=8,
                      language='da',
                      word_timestamps=True
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
    except Exception as e:
        print(f'An error occurred...{e}')

    finally:
        model.unload()
        model = None
        
        return results


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
