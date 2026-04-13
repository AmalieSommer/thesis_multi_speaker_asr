from multi_speaker_asr.models.asr import WhisperPipeline
from multi_speaker_asr.data import Data
import torch
from transformers import pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
test_audio_path = "data/emotale_wav/EN_017_S_5.wav"

#whisper = WhisperPipeline(device=device)
#pipeline = whisper.load_pipeline()
#result = pipeline(test_audio_path)
#print(f"Result is: {result}")

transcriber = pipeline(
    task="automatic-speech-recognition",
    model="openai/whisper-tiny.en",
    torch_dtype=torch.float16,
    device=device)
result = transcriber(test_audio_path)

print("Result is the following: ")
print(result)