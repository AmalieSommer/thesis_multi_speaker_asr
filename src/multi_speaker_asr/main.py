import os
import json
from multi_speaker_asr.evaluate import inference_asr, inference_diarize
import torch
from tqdm import tqdm
from dotenv import load_dotenv
import yaml
from multi_speaker_asr.data import AudioData
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from multi_speaker_asr.models.asr import Whisper
from multi_speaker_asr.models.diarization import Diarize
import argparse
import numpy as np
import librosa

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




def collator_fn(batch):
    """
    It should use librosa.stream() to fetch only blocks of the full audio, and create batches based on a predefined length window (e.g. 30sec)
    So, 
    """
    print('Refactoring in collator...')
    cwd = os.getcwd()
    return batch


def load_data(path):
    print('Loading data...')
    data = AudioData(path=path)
    data.load()
    """
    loader = DataLoader(
        dataset=data,
        batch_size=8, # audio is long-form so keeping the data sample batch_sizes smaller
        collate_fn=collator_fn,
        num_workers=0
    )
    """
    return data


def main(config):

    model = Whisper(device=config['device'])
    model.load(
        model_size=config['model'],
        compute_type=config['computetype'],
        cpu_threads=config['cputhreads']
    )
    #model.load_model(name=config['model'])
    loader = load_data(config['data'])

    asr_results = inference_asr(
        dataset=loader,
        model=model
    )
    print(asr_results)

    #save_data(asr_results, filename=config['filename'])

    # NOTE FOR LATER:
    # WhisperX library is not compatible with current environment.
    # To run a separate phoneme wav2vec2 alignment model it would require running ASR and Wav2Vec2 on two separate subprocessess with their own uv environment.
    # Alternatively start with just running faster-whisper .transcribe() with word_timestamps=True and see how that works.

    # Call diarization model using Pyannote.audio modules:
    """"
    pipeline = Diarize()
    pipeline.load(token=HF_TOKEN)
    diarization_results = inference_diarize(
        loader=loader,
        model=pipeline,
        batch_size=config['threads']
    )
    
    diarize_filename = 'diarize_' + config['filename']
    save_data(diarization_results, filename=diarize_filename)
    """
    # TODO: Call function for aligning the asr output and speaker segments on timestamps and return one final batched result:

if __name__=='__main__':
    print('Starting...')
    config = load_config()
    main(config=config)
