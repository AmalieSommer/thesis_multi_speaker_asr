import numpy as np
#from .utils.utils import compute_cosine_sim
from tqdm import tqdm
#from time import sleep
#from whisperx.alignment import align
import torch
from multi_speaker_asr.models.asr import Whisper
#from multi_speaker_asr.models.alignment import Wav2Vec2
import gc
#from itertools import islice
from multi_speaker_asr.data import Data
from carbontracker.tracker import CarbonTracker
#from multi_speaker_asr.models.diarization import Diarize
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


def collator_fn(batch):
    """To generate batches of arbitrary size for batched inference"""
    samples = []
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None

    audio = [torch.from_numpy(item["audio"]) for item in batch if item is not None]
    audio_t = pad_sequence(audio, batch_first=True)
    audio_n = []
    for i in range(len(audio_t)):
        num_arr = audio_t[i, :].numpy()
        audio_n.append(num_arr)

    for i, sample in enumerate(batch):
        if sample is None:
            continue
        res = {
            'id': sample['id'],
            'audio_path': sample['audio_path'],
            'audio': audio_n[i],
            'transcription': sample['transcription'],
            'speaker_id': sample["client_id"],
            'sentence_id': sample["sentence_id"]
        }
        samples.append(res)

    if len(samples) == 0:
        return None
    
    return samples



def inference_asr(model_size, compute_type, device, data_path, batch_size, cpu_threads):
    dataset = Data(path=data_path)
    dataset.load()
    model = Whisper(device=device)
    model.load(model_size=model_size, compute_type=compute_type, cpu_threads=cpu_threads)

    loader = DataLoader(
        dataset=dataset,
        batch_size=8,
        collate_fn=collator_fn
    )

    # Add carbon tracking:
    tracker = CarbonTracker(epochs=len(dataset))
    tracker.epoch_start()
    try:
        res_dict = {} # All transcription results to save in a jsonl file
        for batch in tqdm(loader):
            if batch is None or None in batch:
                continue

            for sample in batch:
                if sample is None:
                    continue

                id = sample['id'] # ID of audio file
                audio_arr = sample['audio']
                segments, _ = model.model.transcribe(audio=audio_arr, batch_size=batch_size)
                seg_list = []
                for segment in segments:
                    item = {
                        'start': segment.start,
                        'end': segment.end,
                        'text': segment.text
                    }
                    seg_list.append(item)
                res_dict[id] = {
                    'segments': seg_list
                }

    finally:
        tracker.epoch_end()
        tracker.stop()
        model.unload()
        dataset.delete_dataset()
        gc.collect()
        
        return res_dict


"""
def inference_align(alignConfig, datasetConfig, res_dict):
    dataset = Data()
    dataset.load_from_hf(config=datasetConfig)
    model = Wav2Vec2()
    model.load(config=alignConfig)
    
    limit = 4
    batch_iter = iterate_batch(dataset.dataset)
    try:
        model.model.eval()
        with torch.no_grad():
            aligned_dict = res_dict.copy()
            for sample in tqdm(islice(batch_iter, limit), total=limit):

                id = sample['id_conversation']
                obj = res_dict.get(id)

                asr_output = []
                for seg in obj['segments']:
                    asr_output.append(
                        SingleSegment({
                            'start': seg['start'],
                            'end': seg['end'],
                            'text': seg['text']
                        })
                    )

                bytes = sample['audio']['bytes']
                audio = convert_audio(bytes)
                res = align(
                    transcript=asr_output,
                    model=model.model.float(),
                    align_model_metadata=model.metadata,
                    audio=audio,
                    device=model.device
                )
                aligned_dict[id]['aligned'] = res['word_segments']
    finally:
        model.unload()
        dataset.delete_dataset()
        gc.collect()
    
        return aligned_dict


def inference_diarize(diarizeConfig, datasetConfig, res_dict):
    dataset = Data()
    dataset.load_from_hf(config=datasetConfig)
    diarize = Diarize()
    diarize.load(config=diarizeConfig)

    limit = 4
    batch_iter = iterate_batch(dataset.dataset)
    try:
        diarize.model.eval()
        with torch.no_grad():
            for sample in tqdm(islice(batch_iter, limit), total=limit):

                id = sample['id_conversation']
                obj = res_dict.get(id)

                bytes = sample['audio']['bytes']
                audio = convert_audio(bytes)
                diarization_res = diarize.model(
                    audio=audio,
                    min_speakers=1,
                    max_speakers=2,
                    return_embeddings=True
                )
                aligned_res = res_dict[id]['aligned']
                speaker_transcripts, speaker_embeddings = diarize.assign_wordlevel_speakers(
                    diarize_segments=diarization_res,
                    transcript=aligned_res
                )

    finally:
        diarize.unload()
        dataset.delete_dataset()
        gc.collect()


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