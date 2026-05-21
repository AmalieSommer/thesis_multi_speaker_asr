import torch

from torchmetrics.text import WordErrorRate, CharErrorRate
from utils.utils import compute_cosine_sim
from speechbrain.utils.metric_stats import MetricStats
import os
import tempfile
import json
import soundfile as sf
import io
from tqdm import tqdm
from time import sleep


def collator_fn(batch):
    item = batch[0]
    bytes = item["audio"]['bytes']
    audio_bytes = io.BytesIO(bytes)

    return {
        'audio': audio_bytes,
        'text': item['text'],
        'ids': item['id_conversation']
    }

def inference(whisper, ds):
    device = whisper.device

    wer = WordErrorRate()
    wer.to(device)
    cer = CharErrorRate()
    cer.to(device)

    info = []

    for item in tqdm(ds, total=8440):
        
        target_text = item['text']

        bytes = item["audio"]['bytes']
        audio_bytes = io.BytesIO(bytes)
        segments, _ = whisper.model.transcribe(audio_bytes, without_timestamps=True, language='da', vad_filter=True)
        all_segments = [i.text for i in segments]
        transcript = " ".join(all_segments)

        wer.update(preds=transcript, target=target_text)
        cer.update(preds=transcript, target=target_text)

        # Save predicted and actual transcripts for later check:
        info.append((item['id_conversation'], transcript, target_text))

        sleep(0.01)

    wer_final = wer.compute()
    cer_final = cer.compute()

    return {
        "wer": wer_final.item(),
        "cer": cer_final.item(),
        "info": info
    }

def eval_bert(bert, info):
    device = 'cpu'
    semdist = MetricStats(metric=compute_cosine_sim)
    for id, pred, target in info:

        encoded_pred_transcripts = bert.tokenizer(pred, padding=True, truncation=True, return_tensors='pt')
        encoded_target_transcripts = bert.tokenizer(target, padding=True, truncation=True, return_tensors='pt')

        pred_transcripts_embeddings = bert(
            input_ids=encoded_pred_transcripts.input_ids.to(device),
            attention_mask=encoded_pred_transcripts.attention_mask.to(device)
        )
        transcripts_embeddings = bert(
            input_ids=encoded_target_transcripts.input_ids.to(device),
            attention_mask=encoded_target_transcripts.attention_mask.to(device)
        )
        semdist.append(ids=id, pred_embeddings=pred_transcripts_embeddings, target_embeddings=transcripts_embeddings)
    semdist_avg = semdist.summarize()
    return semdist_avg
