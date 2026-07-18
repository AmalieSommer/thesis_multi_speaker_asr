from multi_speaker_asr.utils.utils import LOGGING_CONFIG
import os
from multi_speaker_asr.evaluate import (
    asr_inference,
    aligner_inference
    )
import torch
from tqdm import tqdm
from dotenv import load_dotenv
import yaml
import argparse
import logging
import logging.config
import timeit
from multi_speaker_asr.data import cast, AudioData
 




logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(name='Main')



load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')

tqdm.monitor_interval = 0 # Stops the tqdm from creating monitoring threads causing shutdown-race conditions...
# BECAUSE OF PYTORCH LOAD() CHANGE FOR PYTORCH>=2.6
original_torch_load = torch.load

# Modified function to always trust the download source, setting the weights_only flag to False
def trusted_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = trusted_torch_load


def load_config():
    """
    Loads a .yaml configuration file with argument params.
    """
    print('Loading config file...')
    parser = argparse.ArgumentParser(description='ASR Inference Runs')
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        return yaml.safe_load(file)

def run_asr(config):
    asr_inference(
        data_type=config['data'],
        vad_filter=config['vad_filter'],
        clip_timestamps=config['clip_timestamps'],
        batch_size=config['batchsize'],
        computetype=config['computetype'],
        cputhreads=config['cputhreads'],
        device=config['device'],
        model=config['model'],
        filename=config['output_file']
    )

def run_alignment(config):
    data = AudioData()
    data.load(path=config['data'])
    aligner_inference(
        data.id_to_audio,
        config['alignment_model'],
        config['align_output_filename'],
        config['asr_output_filename']
    )

if __name__=='__main__':
    config_file = load_config()
    run_alignment(config=config_file)
    #run_asr(config=config_file)
