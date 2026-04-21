import torch
from multi_speaker_asr.evaluate import evaluate
from multi_speaker_asr.models.whisper import WhisperBase
from multi_speaker_asr.data import Data
from multi_speaker_asr.models.bert import BERT

import json
import argparse


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", type=str, required=True)
parser.add_argument("-data", "--data_path", type=str, required=True)
parser.add_argument("-metadata", "--metadata_path", type=str, required=True)
args = parser.parse_args()

data = "data/" + args.data_path

bert = BERT("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
whisper = WhisperBase("openai/whisper-small")
dataset = Data(
    local_data=True,
    data_path=data,
    metadata=args.metadata_path
    )
results = evaluate(
    whisper=whisper,
    dataset=dataset,
    bert=bert
)

file_path = "src/multi_speaker_asr/results/" + args.output
try:
    with open(file_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Successfully saved the results to file: {args.output}")
except Exception as e:
    print(f"Error saving results to file: {e}")

print("Result is the following: ")
print("Loss: ", results["loss"], ", WER: ", results["wer"], ", CER: ", results["cer"], "SemDist: ", results["semdist"])
