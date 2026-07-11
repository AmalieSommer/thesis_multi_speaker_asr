from multi_speaker_asr.utils.utils import LOGGING_CONFIG
import os
from multi_speaker_asr.evaluate import (
    inference_asr,
    inference_asr_presegmented,
    inference_diarize,
    align_transcripts,
    generate_final_transcript
    )
import torch
from tqdm import tqdm
from dotenv import load_dotenv
import yaml
import argparse
import logging
import logging.config
import timeit
 




logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(name='Main')



load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')
RESULT_PATH = '/root/master_thesis/thesis_multi_speaker_asr/src/results'

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



def run_asr(config: yaml):
    _ = inference_asr_presegmented(
        batch_size=config['asr_batchsize'],
        computetype=config['computetype'],
        cputhreads=config['cputhreads'],
        device=config['device'],
        model=config['model'],
        filename=config['asr_output_filename']
        )

def run_pipeline(config: yaml):
    
    id_offset_map, id_segments_map = inference_asr(
        data_type=config['data'],
        audio_path=config['audio_path'],
        on_hpc=config['hpc'],
        vad_filter=config['vad_filter'],
        clip_timestamps=config['clip_timestamps'],
        batch_size=config['asr_batchsize'],
        computetype=config['computetype'],
        cputhreads=config['cputhreads'],
        device=config['device'],
        model=config['model'],
        filename=config['asr_output_filename']
        )


    if (id_offset_map is None) | (id_segments_map is None):
        print('Failed with error... inference_asr() returned None')
    
    align_status = align_transcripts(
        data_type=config['data'],
        audio_path=config['audio_path'],
        hpc=config['hpc'],
        vad_filter=False,
        clip_timestamps=config['clip_timestamps'],
        batch_size=config['align_batchsize'],
        align_filename=config['align_output_filename'],
        asr_filename=config['asr_output_filename'], 
        model_name=config['alignment_model'],
        id_offset_map=id_offset_map,
        id_segment_map=id_segments_map
        )
 

    diarize_status = inference_diarize(
        data_type=config['data'],
        audio_path=config['audio_path'],
        hpc=config['hpc'],
        vad_filter=False,
        clip_timestamps=config['clip_timestamps'],
        batch_size=config['diarize_batchsize'],
        diarize_filename=config['diarize_output_filename']
    )

    
    if (align_status == 'Success') & (diarize_status == 'Success'):
        generate_final_transcript(
            results_filename=config['final_output_filename'],
            align_filename=config['align_output_filename'],
            diarize_filename=config['diarize_output_filename']
        )
    else:
        logger.error('Failed with error... Diarization status: %s. Alignment status: %s', diarize_status, align_status)

    asr_memory = inference_asr.memory_stats[0]
    logger.info('Memory Stats during ASR inference...: Before load: %f, After load: %f, Delta: %f', asr_memory['before'], asr_memory['after'], asr_memory['delta'])

    w2v_memory = align_transcripts.memory_stats[0]
    logger.info('Memory Stats during timestamp alignment...: Before load: %f, After load: %f, Delta: %f', w2v_memory['before'], w2v_memory['after'], w2v_memory['delta'])

    diarize_memory = inference_diarize.memory_stats[0]
    logger.info('Memory Stats during diarization...: Before load: %f, After load: %f, Delta: %f', diarize_memory['before'], diarize_memory['after'], diarize_memory['delta'])



if __name__=='__main__':
    config_file = load_config()
    
    run_asr(config_file)
    #run_pipeline(config=config_file)

    
