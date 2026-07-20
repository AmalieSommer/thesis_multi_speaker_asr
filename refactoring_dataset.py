import torchaudio
from silero_vad import load_silero_vad, get_speech_timestamps

def split_long_audio(audio_path, max_duration_sec=10.0, sample_rate=16000):
    # 1. Load the VAD model and the audio file
    model = load_silero_vad()
    waveform, sr = torchaudio.load(audio_path)
    
    # Ensure standard sample rate for Wav2Vec2/Silero VAD
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
    
    # Silero expects a 1D tensor for single-channel processing
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
chunks = split_long_audio("/root/master_thesis/thesis_multi_speaker_asr/data/coral-v3-long-form-conversations/conv_1f2860dcc30248e710f1f39d128ea5ca.wav", max_duration_sec=10.0)
for idx, c in enumerate(chunks):
     print(f"Chunk {idx}: Shapes to {c['waveform'].shape}")
     torchaudio.save(f"split_output_{idx}.wav", c['waveform'], 16000)