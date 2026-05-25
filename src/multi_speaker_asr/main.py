import os
import json
from multi_speaker_asr.data import Data
from multi_speaker_asr.models.asr import Whisper
from multi_speaker_asr.models.alignment import Wav2Vec2
from hydra import initialize, compose
from multi_speaker_asr.evaluate import inference, eval_bert, timestamp_alignment
from multi_speaker_asr.models.bert import BERT
import ctranslate2
import gc
from multi_speaker_asr.evaluate import inference_asr, inference_align
import torch


# BECAUSE OF PYTORCH LOAD() CHANGE FOR PYTORCH>=2.6
# Gem den originale load-funktion
original_torch_load = torch.load

# Lav en modificeret udgave, der altid slår weights_only fra
def trusted_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)

# Overskriv PyTorchs standardfunktion
torch.load = trusted_torch_load


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print(f'Supported Compute Types: {ctranslate2.get_supported_compute_types(device)}')

with initialize(version_base=None, config_path='../configs'):
    config_data = compose(config_name='data')
    config_asr = compose(config_name='whisper-base')
    config_phoneme = compose(config_name='wav2vec2')

print(f'Data config file: {config_data}')
print(f'Model config file: {config_asr}')


RESULTS_FILEPATH = 'src/results'


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



def main():

    filename = 'testing_sa_asr'
    ds = Data()
    ds.load_from_hf(config=config_data)

    res = inference_asr(ds.dataset, config=config_asr)
    save_data(result=res, filename=filename)


    data_table = fetch_data(filename=filename)
    updated_res = inference_align(dataset=ds.dataset, config=config_phoneme, res_dict=data_table)
    save_data(updated_res, filename=filename)

if __name__=='__main__':
    main()
    

    
    