
# %% Prompt Whisper to output segment boundaries for each transcript
import os
import json
import pandas as pd
import string
from models.asr import Whisper
from hydra import initialize, compose
import librosa

# %%
path = ''

with open(path, 'r') as file:
    transcript = json.load(file)

line = transcript[0]['transcript']
norm_line = line.strip()
words = str.split(norm_line, sep=' ')
words = [word for word in words if word.strip()]
print(len(words))
print(words)
# %%
audio_segment_path = ''
wav, sr = librosa.load(audio_segment_path)
dur = librosa.get_duration(y=wav, sr=sr)
print(f'Audio duration: {dur}')

# Assuming about 90 words per 30 seconds
wps = 95 / 30 # to get words per second
words_per_segment = int(dur * wps)
print(f'Estimated words per segment: {words_per_segment}')

words_spoken = words[0:words_per_segment+1]
print(words_spoken)
utterance = ' '.join(words_spoken)
print(utterance)
# %%
# Initialize Whisper model:
with initialize(version_base=None, config_path='../configs'):
    config_asr = compose(config_name='whisper-base')

print(f'Model config file: {config_asr}')
whisper = Whisper()
whisper.load(config=config_asr)

segments, _ = whisper.model.transcribe(wav, without_timestamps=False, language='da', vad_filter=True)
for segment in segments:
    words = str.split(segment.text, ' ')
    print(f'Start: {segment.start}')
    print(f'End: {segment.end}')
    print(f'Text: {segment.text}')
    print(f'Number of words: {len(words)}')

# %%
