import os
import json
from multi_speaker_asr.evaluate import (
    streamed_inference, 
    batched_inference, 
    inference_streaming_diarize,
    align_transcripts
    )
import torch
from tqdm import tqdm
from dotenv import load_dotenv
import yaml
from multi_speaker_asr.data import AudioData
from multi_speaker_asr.models.asr import Whisper
import argparse
from multi_speaker_asr.models.diarization import Diarize, assign_word_speakers
import itertools
import logging
import logging.config
from multi_speaker_asr.utils.utils import LOGGING_CONFIG

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(name='PipelineLogger')


os.environ['OMP_NUM_THREADS'] = '6'

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


def save_data(result: list[dict], filename: str):
    # Update json file with added aligned transcripts:
    try:
        with open(os.path.join(RESULT_PATH, f'{filename}.jsonl'), "w") as file:
            for item in result:
                json_line = json.dumps(item)
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



def load_data(path, hpc):
    print('Loading data...')
    data = AudioData(path=path, hpc=hpc)
    return data



def run_whisper_baseline_short_audio(config):
    data = load_data(path=config['data'], hpc=config['hpc'])
    
    model = Whisper(
                compute_type=config['computetype'],
                cpu_threads=config['cputhreads'],
                device=config['device'],
                model=config['model']
            )
    model_memory = model.model_memory
    logger.info('Whisper Model Memory Stats...: Before load: %f, After load: %f, Delta: %f', model_memory['before'], model_memory['after'], model_memory['delta'])
    

    _, before_alignment = batched_inference(
        data=data,
        model=model
    )

    memory = batched_inference.memory_stats[0]
    logger.info('Batched Inference Memory Stats... Before start: %f, After end: %f, Delta: %f', memory['before'], memory['after'], memory['delta'])
    return before_alignment


def run_whisper_baseline_streaming_audio(config):
    data = load_data(path=config['data'], hpc=config['hpc'])
    computetype = config['computetype']
    
    model = Whisper(
                compute_type=computetype,
                cpu_threads=config['cputhreads'],
                device=config['device'],
                model=config['model']
            )
    model_memory = model.model_memory
    logger.info('Whisper Model Memory Stats...: Before load: %f, After load: %f, Delta: %f', model_memory['before'], model_memory['after'], model_memory['delta'])
    
    _, before_alignment = streamed_inference(
        data=data,
        model=model
    )
    memory = streamed_inference.memory_stats[0]
    logger.info('Streamed Inference Memory Stats... Before start: %f, After end: %f, Delta: %f', memory['before'], memory['after'], memory['delta'])
    
    return before_alignment


def run_diarization_streaming(config):
    data = AudioData(path=config['data'], hpc=False)
    
    model = Diarize(token=HF_TOKEN)   # Default values are fine for now
    model_memory = model.model_memory
    logger.info('Diarization Model Memory Stats...: Before load: %f, After load: %f, Delta: %f', model_memory['before'], model_memory['after'], model_memory['delta'])
    
    res_diarize = inference_streaming_diarize(
        data=data,
        model=model
    )
    
    memory = inference_streaming_diarize.memory_stats[0]
    logger.info('Diarization Inference Memory Stats... Before start: %f, After end: %f, Delta: %f', memory['before'], memory['after'], memory['delta'])

    return res_diarize


def generate_final_transcript(aligned_results: list, rttm_results: list):
    final_transcripts = []
    for (aligned_tuple, diarize_tuple) in zip(aligned_results, rttm_results):
        if aligned_tuple['id'] == diarize_tuple['id']:
            segments = aligned_tuple['transcript']['segments']
            speaker_info = diarize_tuple['speaker_segments']
        
            final_transcripts.append(assign_word_speakers(
                aligned_tuple['id'],
                segments_list=segments,
                speaker_times=speaker_info
            ))
    
    return list(itertools.chain(*final_transcripts))


def main(config, long_form=False):
    filename = 'coral_baseline'

    if long_form:
        segments = run_whisper_baseline_streaming_audio(config=config)
    else:
        segments = run_whisper_baseline_short_audio(config=config)

    aligned_results = align_transcripts(segments, config)
    aligned_inference_mem = align_transcripts.memory_stats[0]
    logger.info('Diarization Inference Memory Stats... Before start: %f, After end: %f, Delta: %f', aligned_inference_mem['before'], aligned_inference_mem['after'], aligned_inference_mem['delta'])
    diarize_results = run_diarization_streaming(config=config)
    
    final_transcripts = generate_final_transcript(
        aligned_results=aligned_results,
        rttm_results=diarize_results
    )
    save_data(final_transcripts, filename)




if __name__=='__main__':
    print('Starting...')
    config = load_config()
    main(config=config, long_form=True)
