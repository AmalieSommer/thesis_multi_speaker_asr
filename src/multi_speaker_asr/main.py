from multi_speaker_asr.utils.utils import LOGGING_CONFIG
import os
from multi_speaker_asr.evaluate import (
    inference_asr,
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


def main(config: yaml):
    
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


if __name__=='__main__':
    config_file = load_config()
    main(config=config_file)

    asr_memory = inference_asr.memory_stats[0]
    logger.info('Memory Stats during ASR inference...: Before load: %f, After load: %f, Delta: %f', asr_memory['before'], asr_memory['after'], asr_memory['delta'])

    w2v_memory = align_transcripts.memory_stats[0]
    logger.info('Memory Stats during timestamp alignment...: Before load: %f, After load: %f, Delta: %f', w2v_memory['before'], w2v_memory['after'], w2v_memory['delta'])

    diarize_memory = inference_diarize.memory_stats[0]
    logger.info('Memory Stats during diarization...: Before load: %f, After load: %f, Delta: %f', diarize_memory['before'], diarize_memory['after'], diarize_memory['delta'])













"""
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


@profile
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




def run_whisper_baseline_short_audio(config, filename = 'asr_output.jsonl'):
    data = load_data(path=config['data'], hpc=config['hpc'])
    
    model = Whisper(
                compute_type=config['computetype'],
                cpu_threads=config['cputhreads'],
                device=config['device'],
                model=config['model']
            )
    
    result = batched_inference(
        data=data,
        model=model,
        asr_result_filename=filename
    )
    return result


def run_whisper_baseline_streaming_audio(config, filename = 'asr_output.jsonl'):
    data = load_data(path=config['data'], hpc=config['hpc'])
    computetype = config['computetype']
    
    model = Whisper(
                compute_type=computetype,
                cpu_threads=config['cputhreads'],
                device=config['device'],
                model=config['model']
            )
    result = streamed_inference(
        data=data,
        model=model,
        asr_result_filename=filename
    )
    
    return result


def run_diarization_streaming(config, output_filename):
    data = AudioData(path=config['data'], hpc=False)
    
    model = Diarize(token=HF_TOKEN)   # Default values are fine for now
    res_diarize = inference_streaming_diarize(
        data=data,
        model=model,
        output_filename=output_filename
    )
    
    return res_diarize


def generate_final_transcript(output_filename: str, final_transcript_filename: str):
    
    try:
        if not output_filename:
            logger.error('Failed to read empty filename.')
            return None
        
        queue = Queue()
        reader_process = Process(target=reader, args=(queue, output_filename))
        reader_process.start()

        with open(final_transcript_filename, 'w') as f:
            while True:
                segments_list = []
                segments, speaker_segments = queue.get()
                segments_list.append(segments)
                transcript = assign_word_speakers(
                    id=segments['id'],
                    segments_list=segments_list,
                    speaker_times=speaker_segments['speaker_segments']
                )
                json_line = json.dumps(transcript)
                f.write(json_line + '\n')
            
            # TODO: Once verified that the final file is correct, add functionality to remove the other intermediate files.
    except Exception as e:
        logger.error('Failed with error: %s', e)
    finally:
        reader_process.join()
        if reader_process.is_alive():
            reader_process.close()
        


def main(config, long_form=False):
    filename = 'coral_baseline.jsonl'
    asr_output_filename = 'asr_output.jsonl'

    if long_form:
        saved_results = run_whisper_baseline_streaming_audio(config=config, filename=asr_output_filename)
    else:
        saved_results = run_whisper_baseline_short_audio(config=config, filename=asr_output_filename)


    aligned_results = align_transcripts(asr_output_filename, config, alignment_result_filename='aligned_output.jsonl')
    diarize_results = run_diarization_streaming(config=config, output_filename='aligned_output.jsonl')

    if (aligned_results is not None) & (diarize_results is not None):
        generate_final_transcript('aligned_output.jsonl', filename)
    else:
        logger.error('Failed to generate transcript, because alignment or diarization returned None... Aligned result: %s, Diarization result: %s', aligned_results, diarize_results)

if __name__=='__main__':
    print('Starting...')
    
    config = load_config()
    config_mem = load_config.memory_stats[0]
    logger.info('Config Memory Stats...: Before load: %f, After load: %f, Delta: %f', config_mem['before'], config_mem['after'], config_mem['delta'])
    
    main(config=config, long_form=True)
"""
    