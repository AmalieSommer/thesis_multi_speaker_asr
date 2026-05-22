import torch
import os
import json
from multi_speaker_asr.data import Data
from multi_speaker_asr.models.asr import Whisper
from hydra import initialize, compose
from multi_speaker_asr.evaluate import inference, eval_bert
from multi_speaker_asr.models.bert import BERT
import ctranslate2
import gc

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print(f'Supported Compute Types: {ctranslate2.get_supported_compute_types(device)}')

with initialize(version_base=None, config_path='../configs'):
    config_data = compose(config_name='data')
    config_model = compose(config_name='whisper-base')

print(f'Data config file: {config_data}')
ds = Data()
ds.load_from_hf(config=config_data)

print(f'Model config file: {config_model}')
whisper = Whisper()
whisper.load(config=config_model)
print('before calling inference...')
out = inference(
    whisper=whisper,
    ds=ds.dataset
)

# UNLOAD WHISPER FROM MEMORY
del whisper
gc.collect()

# LOAD BERT INTO MEMORY
bert = BERT("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
info_updated = eval_bert(
    bert,
    out['info']
)

results = {
    'wer': out['avg_wer'],
    'cer': out['avg_cer'],
    'info': info_updated
}

file_path = 'src/results'
try:
    with open(os.path.join(file_path, 'experiment.json'), "w") as f:
        json.dump(results, f, indent=4)
    print(f"Successfully saved the results to file")
except Exception as e:
    print(f"Error saving results to file: {e}")

print("Result is the following: ")
print("WER: ", results["wer"], ", CER: ", results["cer"])
