import torch
import os
import json
from data import Data
from models.asr import Whisper
from hydra import initialize, compose
from evaluate import evaluate, inference
from models.bert import BERT

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")


bert = BERT("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
with initialize(version_base=None, config_path='..\\configs'):
    config_data = compose(config_name='data')
    config_model = compose(config_name='whisper-base')

print(f'Data config file: {config_data}')
ds = Data()
ds.load_from_hf(config=config_data)

print(f'Model config file: {config_model}')
whisper = Whisper()
whisper.load(config=config_model)

results = inference(
    whisper=whisper,
    dataset=ds.dataset,
    bert=bert
)

file_path = 'src\\results'
try:
    with open(os.path.join(file_path, 'experiment.json'), "w") as f:
        json.dump(results, f, indent=4)
    print(f"Successfully saved the results to file")
except Exception as e:
    print(f"Error saving results to file: {e}")

print("Result is the following: ")
print("WER: ", results["wer"], ", CER: ", results["cer"], "SemDist: ", results["semdist"])
