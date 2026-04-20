import torch
from multi_speaker_asr.evaluate import evaluate
from multi_speaker_asr.models.whisper import WhisperBase
from multi_speaker_asr.data import Data


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

whisper = WhisperBase("openai/whisper-tiny")
dataset = Data(
    local_data=True,
    data_path="data/lillelyd-main",
    metadata="manifest_test.jsonl"
    )
#dataset.load_hf("CoRal-project/coral-v3", "conversation", "test")
results = evaluate(
    model=whisper,
    processor=whisper.processor,
    device=whisper.device,
    dataset=dataset
)

print("Result is the following: ")
print("Loss: ", results["loss"], ", WER: ", results["wer"], ", CER: ", results["cer"], "SemDist: ", results["semdist"])
