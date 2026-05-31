import os
import json
import ctranslate2
from multi_speaker_asr.evaluate import inference_asr
import torch
from tqdm import tqdm
import logging


print(torch.cuda.is_available())
print(torch.backends.cudnn.version())


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

logging.basicConfig()
logging.getLogger("faster_whisper").setLevel(logging.DEBUG)

RESULTS_FILEPATH = 'src/results'
DATA_PATH = 'data/en/metadata.csv' # relative to the cwd...


def fetch_data(filename):
    # Fetch transcripts from file:
    try:
        data = {}
        with open(os.path.join(RESULTS_FILEPATH, f'{filename}.jsonl'), "r") as file:
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
        with open(os.path.join(RESULTS_FILEPATH, f'{filename}.jsonl'), "w") as file:
            for key, value in result.items():
                json_line = json.dumps({key: value})
                file.write(json_line + '\n')
        print(f"Successfully updated the file")
    except Exception as e:
        print(f"Error updating the file: {e}")


def exp1(filename, model_size, compute_type, device, batch_size):
    """Run ASR inference on varying Whisper models under resource constraints"""
    res = inference_asr(
        model_size=model_size,
        compute_type=compute_type,
        device=device,
        data_path=DATA_PATH,
        batch_size=batch_size
    )
    save_data(result=res, filename=filename)


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
parser.add_argument('--modelsize', type=str, required=True)
parser.add_argument('--device', type=str, required=True)
parser.add_argument('--computetype', type=str, required=True)
parser.add_argument('--batchsize', type=int, required=False, help='Determines the batch size for inference. Defaults to 1. When on CPU keep low 1 to 2, if on GPU try ranges 8 to 16', default=1)
args = parser.parse_args()

if __name__=='__main__':
    print(torch.__version__)
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))

    model_size = args.modelsize
    device = args.device
    compute_type = args.computetype
    batch_size = args.batchsize

    print(f"Device: {device}")
    print(f'Supported Compute Types: {ctranslate2.get_supported_compute_types(device)}')

    res_filename = f'whisper_{model_size}_{compute_type}_{device}'
    exp1(
        filename=res_filename,
        model_size=model_size,
        compute_type=compute_type,
        device=device,
        batch_size=batch_size
    )
