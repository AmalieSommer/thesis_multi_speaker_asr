import torch

from torchmetrics.text import WordErrorRate, CharErrorRate
from .utils.utils import compute_cosine_sim, compute_cer, compute_wer
from speechbrain.utils.metric_stats import MetricStats
import os
import tempfile
import json
import soundfile as sf
import io
from tqdm import tqdm
from time import sleep


def inference(whisper, ds):
    info = []
    total = 5
    print('Before for loop...')
    for iter, item in enumerate(tqdm(ds, total=total)):
        if iter > total:
            break
        target_text = item['text']
        print(f'For loop: {target_text}')

        bytes = item["audio"]['bytes']
        audio_bytes = io.BytesIO(bytes)
        segments, _ = whisper.model.transcribe(audio_bytes, without_timestamps=True, language='da', vad_filter=True)
        all_segments = [i.text for i in segments]
        transcript = " ".join(all_segments)

        wer = compute_wer(transcript, target_text)
        cer = compute_cer(transcript, target_text)

        # Save predicted and actual transcripts for later check:
        info.append({
            'id_audio': item['id_conversation'],
            'id_speaker': item['id_speaker'],
            'age': item['age'],
            'gender': item['gender'],
            'pred': transcript,
            'target': target_text,
            'wer': wer,
            'cer': cer
        })

        sleep(0.01)

    print(info)
    sum_wer = sum(c['wer'] for c in info)
    sum_cer = sum(c['cer'] for c in info)

    return {
        "avg_wer": sum_wer / total,
        "avg_cer": sum_cer / total,
        "info": info
    }


def eval_bert(bert, info):
    for item in info:

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
