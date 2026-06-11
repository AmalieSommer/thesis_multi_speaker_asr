from tqdm import tqdm
from carbontracker.tracker import CarbonTracker
import time
from memory_profiler import profile
import torch
from pyannote.audio.pipelines.utils.hook import ProgressHook



#@profile
def inference_asr(loader, model):
    
    # Add carbon tracking:
    tracker = CarbonTracker(epochs=loader.batch_size)
    tracker.epoch_start()
    try:
        results = []
        
        for batch in tqdm(loader):
            start_time = time.time()
            print(f'Start time...: {start_time}')

            for sample in batch: # Because the batching is done per audio sample for audio longer than 30 secconds, and batching beyond that does not make sense...
                """
                segments, _ = model.model.transcribe(audio=sample['audio'], 
                                                        batch_size=1,
                                                        language='en',
                                                        word_timestamps=True)
                """
                segments = model.pipeline(sample['wav'])
                seg_list = []
                for segment in segments:
                    item = {
                        'start': segment.start,
                        'end': segment.end,
                        'text': segment.text
                    }
                    seg_list.append(item)

                processing_time = time.time() - start_time
                rtf = processing_time # currently just passes total processing time for full batch

                results.append({
                    'id': sample['id'],
                    'path': sample['path'],
                    'segments': seg_list,
                    'rtf': rtf
                })
            
    finally:
        tracker.epoch_end()
        tracker.stop()
        model.unload()
        model = None
        
        return results


#@profile
def inference_diarize(loader, model, batch_size):

    tracker = CarbonTracker(epochs=loader.batch_size)
    tracker.epoch_start()
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
        tracker.epoch_end()
        tracker.stop()
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
