from multi_speaker_asr.data import AudioDataset
from multi_speaker_asr.models.asr import ASR
from multi_speaker_asr.models.alignment import Alignment, align_words_speakers
from multi_speaker_asr.models.diarization import SpeakerDiarizationPipeline, NumpyArrAudioSource, SpeakerDiarizationConfig
import json
from torch.utils.data import DataLoader
from multi_speaker_asr.utils.logging_config import get_logger
import os
import json
from multi_speaker_asr.utils.memory_tracking import MemoryTracker
import tempfile
from pathlib import Path
import numpy as np
import torch


# BECAUSE OF PYTORCH LOAD() CHANGE FOR PYTORCH>=2.6
original_torch_load = torch.load

# Modified function to always trust the download source, setting the weights_only flag to False
def trusted_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = trusted_torch_load

log = get_logger(__name__)


def pipeline_inference(
        transcription_filepath: str,
        data_config: dict,
        asr_config: dict,
        engine_config: dict,
        diarize_config: dict,
        align_config: dict = None
):
    # Start memory tracking:
    mem_tracker = MemoryTracker()
    mem_tracker.start()

    try:
        asr_output = asr_inference(
            results_filepath=tempfile.NamedTemporaryFile().name,
            engine_config=engine_config,
            data_config=data_config,
            asr_config=asr_config,
            align_config=align_config
        )
    except Exception:
        log.exception('ASR or Alignment module failed with error')

    try:
        diarization_output = diarization_inference(
            data_config=data_config,
            diarize_config=diarize_config,
            result_filepath=tempfile.NamedTemporaryFile().name
        )
    except Exception:
        log.exception('Diarization module failed with error')

    transcript = assign_words_speakers(diarize_output=diarization_output, asr_output=asr_output, filename=transcription_filepath)

    # Stop the memory tracker and log the memory use:
    avg_mem, peak_mem = mem_tracker.stop()
    log.info('ASR inference... Avg memory: %s, Peak memory: %s', avg_mem, peak_mem)

    return transcript


def asr_inference(
        results_filepath: str, 
        engine_config: dict, 
        data: dict | str, 
        asr_config: dict, 
        align_config: dict = None
        ) -> list[dict]:
    """
    A function for running inference of the asr module of the pipeline. 
    It creates a dataset, loader and the ASR model to be used, then process all batches while saving the results to a .jsonl file, as well as returns all results.

    Args:
        results_filepath (str): The filepath for saving the results
        backend (str): The given backend type you want to run the pipeline with, e.g. pytorch or onnx
        data_config (dict): A dictionary object for holding all relevant configuration parameters
        model_config (dict): A dictionary object for holding all relevant configuration parameters for instantiating the model
        batch_size (int): Number of samples in a batch
        timestamps (bool): A boolean value to indicate whether to return word-based timestamps or no timestamps at all
    
    Returns:
        list[dict]: A list of dictionary objects, one for each audio sample.
    """
    if results_filepath is None:
        raise ValueError('Filepath was None.')
    os.makedirs(os.path.dirname(results_filepath), exist_ok=True)
    if os.path.exists(results_filepath):
        log.info('Result already exists. Loading results from existing directory: %s', results_filepath)
        loaded_results = []
        with open(results_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    loaded_results.append(json.loads(line))
        return loaded_results

    log.info('Loading audio batch for ASR:')
    data = AudioDataset( # Depending on the type of the data parameter, it will call the appropriate init function.
        data,
        asr_config['target_sr'],
        asr_config['max_duration']
    )
    loader = DataLoader(
        dataset=data,
        batch_size=asr_config['batch_size'],
        collate_fn=data.collator
    )
    pipeline = ASR(
        asr_cfg=asr_config,
        engine_cfg=engine_config
    )

    # Start memory tracking:
    mem_tracker = MemoryTracker()
    mem_tracker.start()

    output_results = []
    with open(results_filepath, 'w', encoding="utf-8") as write_result, tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        count = 0
        for batch in loader:
            if not batch:
                continue
            try:
                ts = None
                if asr_config['timestamps']:
                    if pipeline.type == 'ctc':
                        ts = 'word'
                    elif pipeline.type == 'seq2seq':
                        ts = True
                log.info('Running transcription')
                prediction = pipeline.transcribe(audio_batch=batch['audio'], return_timestamps=ts)
            except Exception as e:
                log.error('Failed with error: %s', e)
                continue

            batch_result = [
                    {'sample_id': s_id, 'segments': pred} 
                    for s_id, pred in zip(batch['sample_id'], prediction)
                ]
            for item in batch_result:
                write_result.write(json.dumps(item, ensure_ascii=False) + '\n')
                output_results.append(item)

            # if the chosen asr model is not a ctc-based model, save the results for post-processing word-level timestamps
            if pipeline.engine.model_type == 'seq2seq' and asr_config['timestamps']:
                np.savez(
                    tmp_path / f'batch_{count}.npz',
                    **dict(zip(batch['sample_id'], batch['audio']))
                )

        # Stop the memory tracker and log the memory use:
        avg_mem, peak_mem = mem_tracker.stop()
        log.info('ASR inference... Avg memory: %s, Peak memory: %s', avg_mem, peak_mem)


        if pipeline.engine.model_type == 'seq2seq' and asr_config['timestamps']:
            output_results = alignment_inference(output_results, tmp_path, config=align_config)

    del pipeline
    del loader
    return output_results



def alignment_inference(asr_output: list, audio_dir: Path, config: dict) -> list[dict]:
    if asr_output is None or audio_dir is None or config is None:
        raise ValueError('A parameter is None.')

    # Map the list to a dictionary for O(1) look-up:
    asr_by_id = {
        item['sample_id']: item['segments']['words']
        for item in asr_output
    }
    # Start memory tracking:
    mem_tracker = MemoryTracker()
    mem_tracker.start()
    
    pipeline = Alignment(config)

    for batch_file in audio_dir.glob("batch_*.npz"):
        with np.load(batch_file) as samples:
            for sample_id in samples.files:
                audio = samples[sample_id]
                asr_res = asr_by_id[sample_id]

                input = pipeline.format_model_input(
                    asr_output=asr_res
                )
                aligned_res = pipeline.align(prediction=input, audio=audio)
                asr_by_id[sample_id] = aligned_res

    # Stop the memory tracker and log the memory use:
    avg_mem, peak_mem = mem_tracker.stop()
    log.info('ASR inference... Avg memory: %s, Peak memory: %s', avg_mem, peak_mem)

    del pipeline

    return [
        {'sample_id': sample_id, 'segments': pipeline.format_model_output(result)}
        for sample_id, result in asr_by_id.items()
    ]



def diarization_inference(data: str | dict, diarize_config: dict, result_filepath: str) -> list[dict]:
    log.info('Loading audio batch for ASR:')
    ds = AudioDataset(data, max_segment_duration=diarize_config['max_duration']) # Setting the maximum duration to a really high number, because Diart conducts streaming diarization looking at a few seconds at a time...
    loader = DataLoader(
        dataset=ds,
        batch_size=diarize_config['batch_size'],
        collate_fn=ds.collator
    )
    config = SpeakerDiarizationConfig(
        sample_rate=diarize_config['target_sr'],
        max_speakers=diarize_config['max_speakers'],
        device=diarize_config['device']
    )
    pipeline = SpeakerDiarizationPipeline(config=config)
    # Start memory tracking:
    mem_tracker = MemoryTracker()
    mem_tracker.start()

    output = []
    for sample in loader:
        if not sample:
            continue

        source = NumpyArrAudioSource(audio_arr=sample['audio'], uri=sample['sample_id'])
        try:
            annotation = pipeline.diarize(sample=source, output_path=result_filepath)
        except Exception as e:
            log.error('Failed with error: %s', e)

        result = pipeline.get_speaker_segments(result=annotation)
        output.append({
            'sample_id': sample['sample_id'],
            'segments': result
        })
    # Stop the memory tracker and log the memory use:
    avg_mem, peak_mem = mem_tracker.stop()
    log.info('Diarize inference... Avg memory: %s, Peak memory: %s', avg_mem, peak_mem)

    del pipeline
    del loader
    return output



def assign_words_speakers(diarize_output: list[dict], asr_output: list[dict], filename: str) -> list:
    filepath = Path(filename)
    # First transform the asr_output back to the necessary format:
    for item in asr_output:
        res_list = item['segments']
        first_item = res_list['words'][0]
        last_item = res_list['words'][-1]
        res_list['start'] = first_item['start']
        res_list['end'] = last_item['end']

        sorted_dict = dict(sorted(res_list.items(), key=lambda item: item[0]))

        item['segments'] = sorted_dict

    output = []
    with open(filepath, 'w', encoding="utf-8") as write_result:
        for sd, asr in zip(diarize_output, asr_output):
            aligned_res = align_words_speakers(sd_output=sd, asr_output=asr)

            write_result.write(json.dumps(aligned_res, ensure_ascii=False) + '\n')
            output.append(aligned_res)

    return output