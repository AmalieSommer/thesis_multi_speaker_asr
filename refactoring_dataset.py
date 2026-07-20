import torchaudio
from silero_vad import load_silero_vad, get_speech_timestamps
import os
import pandas as pd
import json
import csv
import librosa
import soundfile as sf
import torch
import copy

model = load_silero_vad()

def split_long_audio(audio_path, max_duration_sec=10.0, sample_rate=16000):
    # 1. Load the VAD model and the audio file
    waveform, sr = librosa.load(audio_path)
    
    # Ensure standard sample rate for Wav2Vec2/Silero VAD
    if sr != sample_rate:
        waveform = librosa.resample(
                waveform,
                orig_sr=sr,
                target_sr=sample_rate,
            )

    # Silero expects a 1D tensor for single-channel processing
    waveform = torch.tensor(waveform).unsqueeze(0)
    audio_1d = waveform.mean(dim=0) 
    total_duration = len(audio_1d) / sample_rate
    
    # If it's already short enough, return it as a single chunk
    if total_duration <= max_duration_sec:
        return [{"start": 0.0, "end": total_duration, "waveform": waveform}]

    # 2. Get speech timestamps (in seconds)
    speech_timestamps = get_speech_timestamps(
        audio_1d, model, sampling_rate=sample_rate, return_seconds=True
    )
    
    chunks = []
    chunk_start = 0.0
    max_samples = int(max_duration_sec * sample_rate)
    
    # 3. Iterate through the gaps between speech to find natural cutting points
    for i in range(len(speech_timestamps) - 1):
        current_speech_end = speech_timestamps[i]['end']
        next_speech_start = speech_timestamps[i+1]['start']
        
        # The midpoint of the silence is the safest place to slice
        silence_midpoint = current_speech_end + (next_speech_start - current_speech_end) / 2
        
        # If adding the next segment would push us past our 10-second limit...
        if silence_midpoint - chunk_start > max_duration_sec:
            # Slice it right here at the midpoint of this silence
            start_sample = int(chunk_start * sample_rate)
            end_sample = int(silence_midpoint * sample_rate)
            
            chunks.append({
                "start": chunk_start,
                "end": silence_midpoint,
                "waveform": waveform[:, start_sample:end_sample]
            })
            chunk_start = silence_midpoint # Move the cursor up

    # 4. Append the final remaining piece of the audio
    final_start_sample = int(chunk_start * sample_rate)
    chunks.append({
        "start": chunk_start,
        "end": total_duration,
        "waveform": waveform[:, final_start_sample:]
    })
    
    # 5. Safety check: Hard-chop any chunk that STILL exceeds max duration 
    # (e.g., if someone spoke for 15 seconds straight without a single pause)
    final_chunks = []
    for chunk in chunks:
        chunk_len_sec = chunk["waveform"].shape[1] / sample_rate
        if chunk_len_sec > max_duration_sec:
            # Fallback: Force-slice into rigid 10-second blocks
            samples_per_chunk = max_samples
            for offset in range(0, chunk["waveform"].shape[1], samples_per_chunk):
                sub_wave = chunk["waveform"][:, offset:offset + samples_per_chunk]
                final_chunks.append({"waveform": sub_wave})
        else:
            final_chunks.append(chunk)

    return final_chunks

# --- Quick Test Execution ---
segments_path = 'L:\\Auditdata\\Wrist Angel - Video\\Amalie Sommer\\repo\\thesis_multi_speaker_asr\\data\\audio\\segments\\audio\\'
segments_dir = os.listdir(segments_path)

data = {}
with open('L:\\Auditdata\\Wrist Angel - Video\\Amalie Sommer\\repo\\thesis_multi_speaker_asr\\data\\audio\\segments\\metadata_new.csv', 'r', encoding='utf-8') as reader:
    dict_reader = csv.DictReader(reader)
    data = {row['segment_id']: row for row in dict_reader}

new_csv = 'L:\\Auditdata\\Wrist Angel - Video\\Amalie Sommer\\repo\\thesis_multi_speaker_asr\\data\\audio\\segments\\metadata_2.csv'
seg_dir = 'data\\audio\\segments\\audio'
with open(new_csv, 'a', encoding='utf-8') as writer:
    for audio in segments_dir:
        id = audio.split('.', 1)[0]
        info = data[id]
        path = segments_path + audio
        chunks = split_long_audio(path)
        if len(chunks) == 1:
            row = ",".join([str(item) for item in info.values()])
            writer.write(json.dumps(row) + '\n')
            writer.flush()
        else:
            for i, chunk in enumerate(chunks):
                new_segment = copy.deepcopy(info)
                id = info['segment_id']
                new_segment['segment_id'] = id + f'_{i}'
              
                filepath = seg_dir + f'\\{id}_{i}.wav'
                new_segment['segment'] = filepath
                new_segment['text'] = '""' + info['text'] + '""'
                wav = chunk['waveform'].squeeze(0).cpu().numpy()
                duration = len(wav) / 16000
                if duration < 1.0:
                    continue
                new_segment['segment_duration'] = duration
                row = ",".join([str(item) for item in new_segment.values()])
                writer.write(json.dumps(row) + '\n')
                writer.flush()

                print(f'Changed segment: {id}')
                
                sf.write(filepath, wav, 16000)

                

