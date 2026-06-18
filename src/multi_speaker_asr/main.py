import os
import json
from multi_speaker_asr.evaluate import streamed_inference, batched_inference
import torch
from tqdm import tqdm
from dotenv import load_dotenv
import yaml
from multi_speaker_asr.data import AudioData
from multi_speaker_asr.models.asr import Whisper
import argparse  


os.environ['OMP_NUM_THREADS'] = '4'

load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')


RESULT_PATH = '/zhome/28/9/151118/thesis/thesis_multi_speaker_asr/src/results'



tqdm.monitor_interval = 0 # Stops the tqdm from creating monitoring threads causing shutdown-race conditions...
# BECAUSE OF PYTORCH LOAD() CHANGE FOR PYTORCH>=2.6
original_torch_load = torch.load

# Modified function to always trust the download source, setting the weights_only flag to False
def trusted_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = trusted_torch_load


def fetch_data(filename):
    # Fetch transcripts from file:
    try:
        data = {}
        with open(os.path.join(RESULT_PATH, f'{filename}.jsonl'), "r") as file:
            for line in file:
                entry = json.loads(line)
                data.update(entry)
        print(f"Successfully loaded file.")
        return data
    except Exception as e:
        print(f"Error reading from file: {e}")


def save_data(result, filename):
    # Update json file with added aligned transcripts:
    try:
        with open(os.path.join(RESULT_PATH, f'{filename}.jsonl'), "w") as file:
            for key, value in result.items():
                json_line = json.dumps({key: value})
                file.write(json_line + '\n')
        print(f"Successfully updated the file")
    except Exception as e:
        print(f"Error updating the file: {e}")




def load_config():
    print('Loading config file...')
    parser = argparse.ArgumentParser(description='ASR Inference Runs')
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        return yaml.safe_load(file)



def load_data(path):
    print('Loading data...')
    data = AudioData(path=path)
    data.load()
    return data



def run_whisper_baseline_short_audio(config):
    data = load_data(path=config['data'])
    model = Whisper(
                compute_type=config['computetype'],
                cpu_threads=config['cputhreads'],
                device=config['device'],
                model=config['model']
            )
    asr_results = batched_inference(
        data=data,
        model=model
    )
    save_data(asr_results, f'whisper_baseline')


def run_whisper_baseline_streaming_audio(config):
    data = load_data(path=config['data'])
    model = Whisper(
                compute_type=config['computetype'],
                cpu_threads=config['cputhreads'],
                device=config['device'],
                model=config['model']
            )
    asr_results = streamed_inference(
        data=data,
        model=model
    )
    save_data(asr_results, f'whisper_baseline')




def main(config):
    #run_whisper_baseline_short_audio(config=config)
    run_whisper_baseline_streaming_audio(config=config)

    # NOTE FOR LATER:
    # WhisperX library is not compatible with current environment.
    # To run a separate phoneme wav2vec2 alignment model it would require running ASR and Wav2Vec2 on two separate subprocessess with their own uv environment.
    # Alternatively start with just running faster-whisper .transcribe() with word_timestamps=True and see how that works.

    # Call diarization model using Pyannote.audio modules:


if __name__=='__main__':
    print('Starting...')


    config = load_config()
    main(config=config)
