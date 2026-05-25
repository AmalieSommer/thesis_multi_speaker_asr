import numpy as np
from .utils.utils import compute_cosine_sim, compute_cer, compute_wer
import io
from tqdm import tqdm
from time import sleep
from whisperx.alignment import align
from whisperx.types import SingleSegment
import soundfile as sf
import torch
from multi_speaker_asr.models.asr import Whisper
from multi_speaker_asr.models.alignment import Wav2Vec2
import gc
from itertools import islice
from multi_speaker_asr.data import Data



def convert_audio(bytes):
    audio_bytes = io.BytesIO(bytes)
    wav, _ = sf.read(audio_bytes)
    audio = wav.astype(np.float32)
    return audio


def iterate_batch(batched_dataset):
    for batch in batched_dataset:
        rand_key = list(batch.keys())[0]
        batch_length = len(batch[rand_key])

        for i in range(batch_length):
            sample = {key: batch[key][i] for key in batch.keys()}
            yield sample


def inference_asr(asrConfig, dataConfig):
    dataset = Data()
    dataset.load_from_hf(config=dataConfig)
    model = Whisper()
    model.load(config=asrConfig)


    limit = 2
    batch_iter = iterate_batch(dataset.dataset)
    try:
        res_dict = {} # All transcription results to save in a jsonl file
        for sample in tqdm(islice(batch_iter, limit), total=limit):

            id = sample['id_conversation'] # ID of audio file
            bytes = sample['audio']['bytes']
            audio = convert_audio(bytes)
            output = model.model.transcribe(audio=audio, language='da')
            seg_list = []
            for segment in output['segments']:
                item = {
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': segment['text']
                }
                seg_list.append(item)
            res_dict[id] = {
                'segments': seg_list
            }
    finally:
        model.unload()
        dataset.delete_dataset()
        gc.collect()
        
        return res_dict



def inference_align(alignConfig, datasetConfig, res_dict):
    dataset = Data()
    dataset.load_from_hf(config=datasetConfig)
    model = Wav2Vec2()
    model.load(config=alignConfig)
    
    limit = 2
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

