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


def convert_audio(bytes):
    audio_bytes = io.BytesIO(bytes)
    wav, _ = sf.read(audio_bytes)
    audio = wav.astype(np.float32)
    return audio



def inference_asr(dataset, config):
    model = Whisper()
    model.load(config=config)

    iter = 2
    try:
        res_dict = {} # All transcription results to save in a jsonl file
        for i, sample in enumerate(tqdm(dataset, total=8440)):
            if i > iter:
                break
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
        # Remove model from memory...
        model.unload()
        gc.collect()
        
        return res_dict



def inference_align(dataset, config, res_dict):
    model = Wav2Vec2()
    model.load(config=config)
    iter = 2
    model.model.eval()
    with torch.no_grad():
        aligned_dict = res_dict.copy()
        for i, sample in enumerate(tqdm(dataset, total=8440)):
            if i > iter:
                break
            id = sample['id_conversation']
            obj = res_dict.get(id)
            print(obj)

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

    # Remove model from memory...
    model.unload()
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

