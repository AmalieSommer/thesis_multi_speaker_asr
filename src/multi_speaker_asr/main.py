import os
import json
import ctranslate2
from multi_speaker_asr.evaluate import inference_asr
import torch
from tqdm import tqdm
import logging
from dotenv import load_dotenv
import yaml

load_dotenv()

tqdm.monitor_interval = 0 # Stops the tqdm from creating monitoring threads causing shutdown-race conditions...
# BECAUSE OF PYTORCH LOAD() CHANGE FOR PYTORCH>=2.6
# Gem den originale load-funktion
original_torch_load = torch.load

# Lav en modificeret udgave, der altid slår weights_only fra
def trusted_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
# Overskriv PyTorchs standardfunktion
torch.load = trusted_torch_load


#DATA_PATH = '/zhome/28/9/151118/thesis/thesis_multi_speaker_asr/data' # relative to the cwd...
#RESULT_PATH = '/zhome/28/9/151118/thesis/thesis_multi_speaker_asr/src/results'
DATA_PATH = '/root/master_thesis/thesis_multi_speaker_asr/data' # relative to the cwd...
RESULT_PATH = '/root/master_thesis/thesis_multi_speaker_asr/src/results'


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


def exp1(model_size, compute_type, device, batch_size, cpu_threads):
    """Run ASR inference on varying Whisper models under resource constraints"""
    res = inference_asr(
        model_size=model_size,
        compute_type=compute_type,
        device=device,
        data_path=DATA_PATH,
        batch_size=batch_size,
        cpu_threads=cpu_threads
    )
    return res


def exp2(filename):
    """Read output from selected Whisper models from exp1 and run inference on the full SA-ASR pipeline"""
    data_table = fetch_data(filename=filename)
    updated_res = inference_align(alignConfig=config_phoneme, datasetConfig=config_data, res_dict=data_table)
    save_data(updated_res, filename=filename)

    data_table = fetch_data(filename=filename)
    final_output = inference_diarize()
    #TODO Save the final results


import argparse
parser = argparse.ArgumentParser(description='ASR Inference Runs')
parser.add_argument('--config', type=str, required=True)


def load_config(filepath):
    with open(filepath, 'r') as file:
        return yaml.safe_load(file)


"""
parser.add_argument('--modelsize', type=str, required=True)
parser.add_argument('--device', type=str, required=True)
parser.add_argument('--computetype', type=str, required=True)
parser.add_argument('--batchsize',
                     type=int, 
                     required=False,
                      default=2, # same as in faster-whisper documentation 
                     help='Determines the batch size for inference. Defaults to 1. When on CPU keep low 1 to 2, if on GPU try ranges 8 to 16'
                     )
parser.add_argument('--filename', type=str, required=True) # The filename for the result
parser.add_argument('--threads', type=str, required=True, default=1, help='Should be equal to the number of CPU cores.')
"""

args = parser.parse_args()


if __name__=='__main__':

    config = load_config(args.config)
    result = exp1(
        model_size=config['modelsize'],
        compute_type=config['computetype'],
        device=config['device'],
        batch_size=int(config['batchsize']),
        cpu_threads=int(config['threads'])
    )

    save_data(result=result, filename=config['filename'])


"""
    model_size = args.modelsize
    device = args.device
    compute_type = args.computetype
    batch_size = args.batchsize
    filename = args.filename
    threads = args.threads

    print(f"Device: {device}")
    print(f'Supported Compute Types: {ctranslate2.get_supported_compute_types(device)}')
"""
    #res_filename = f'whisper_{model_size}_{compute_type}_{device}'
    
