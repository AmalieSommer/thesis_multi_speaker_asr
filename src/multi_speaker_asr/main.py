from multi_speaker_asr.utils.utils import LOGGING_CONFIG
import os
from multi_speaker_asr.evaluate import (
    asr_inference,
    aligner_inference,
    evaluate_inference,
    warmup
    )
import torch
from tqdm import tqdm
from dotenv import load_dotenv
import yaml
import argparse
import logging
import logging.config
import timeit
from multi_speaker_asr.data import AudioData, AudioDataset
from datasets import load_dataset, Dataset, Audio
from torch.utils.data import DataLoader
from multi_speaker_asr.models.asr import RoestASR
from optimum.quanto import qint8, qint4, qint2


 




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
        filename=config['asr_output_filename']
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


def exp_1(config):
    if config['dataset_location'] == 'remote': # The dataset is loaded from remote, e.g. Huggingface
        metadata = load_dataset('CoRal-project/coral-v3', 'conversation', split='test', streaming=True)
        metadata = metadata.cast_column('audio', Audio(decode=False))
        metadata = metadata.rename_column('id_conversation', 'audio_id')
        metadata = metadata.rename_column('audio', 'segment')
    elif config['dataset_location'] == 'local':
        metadata = Dataset.from_csv(config['metadata']).to_iterable_dataset()
    else:
        raise ValueError('Unknown dataset_mode passed...')

    dataset = AudioDataset(metadata=metadata, mode=config['dataset_mode'])
    loader = DataLoader(dataset=dataset, batch_size=config['batch_size'], shuffle=False, num_workers=1, collate_fn=dataset.collator)

    print(torch.__config__.show())

    model = RoestASR(model_type=config['model_type'], batch_size=config['batch_size'], backend=config['backend_type'])
    model.load(use_saved_model=config['use_saved_model'], compute_type=config['computetype'], local_models_dir=config['local_models_dir'])
   
    evaluate_inference(output_filepath=config['asr_filepath'], loader=loader, model=model, warmup=False)



def build_quantized_model(config):
    model = RoestASR(model_type=config['model_type'], batch_size=config['batch_size'], backend=config['backend_type'])
    model.load(local_models_dir=config['local_models_dir'], compute_type=config['computetype'])
    model.save_quantized_model(compute_type=config['computetype'], weights_q=qint4)



if __name__=='__main__':
    config_file = load_config()
    if config_file['build_model']:
        build_quantized_model(config=config_file)
    else:
        exp_1(config=config_file)
    
    print('Finished...!')
