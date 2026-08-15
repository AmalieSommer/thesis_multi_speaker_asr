import os
import torch
from dotenv import load_dotenv
import argparse
import logging
import hydra
from omegaconf import DictConfig
from hydra import compose, initialize_config_dir
from pathlib import Path
from multi_speaker_asr.inference import asr_inference, diarization_inference, pipeline_inference

log = logging.getLogger(__name__)

torch.set_num_threads(6)
torch.set_num_interop_threads(6)

load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')


@hydra.main(version_base=None, config_path='../../configs', config_name='config')
def hydra_main(config: DictConfig) -> None:
    """
    This is the function for running an experiment using predefined config files with Hydra, in order
    to be able to easily reproduce research results

    Args:
        config (DictConfig): The Hydra main configuration file containing default settings and the option to overwrite with other config files
    """
    if config.task == 'asr':
        asr_inference(
            results_filepath=config.output.asr_filepath,
            engine_config=config.engine,
            data_config=config.data,
            asr_config=config.asr,
            align_config=config.alignment
        )
    elif config.task == 'diarize':
        diarization_inference(
            data_config=config.data,
            diarize_config=config.diarization,
            result_filepath=config.output.diarize_filepath
        )
    elif config.task == 'pipeline':
        pipeline_inference(
            asr_filepath=config.output.asr_filepath,
            diarization_filepath=config.output.diarize_filepath,
            transcription_filepath=config.output.transcription_filepath,
            data_config=config.data,
            asr_config=config.asr,
            engine_config=config.engine,
            diarize_config=config.diarization,
            align_config=config.alignment
        )
    else:
        raise ValueError('No valid task was found...')



def build_parser():
    """
    Build the Argument Parser for receiving command-line input.
    It allows for users to run either the full pipeline, or individual modules, using the cmd with just
    an audio file and a filepath for the output.
    """
    parser = argparse.ArgumentParser(
        prog='multi_speaker_asr',
        description='Speaker Diarization and ASR Inference'
    )

    subparsers = parser.add_subparsers(dest='command', required=True)

    asr_parser = subparsers.add_parser('asr')
    asr_parser.add_argument('--audio', type=str, required=True)
    asr_parser.add_argument('--output-filepath', type=str, required=True)
    asr_parser.add_argument('--model', type=str, choices=['whisper', 'wav2vec2', 'parakeet'], help='Choice of ASR model (default is Whisper)', required=False)
    asr_parser.add_argument('--engine', type=str, choices=['torch', 'onnx', 'ct2'], help='The inference engine to run ASR (default is Torch)', required=False)
    asr_parser.set_defaults(fun=run_asr)

    diarization_parser = subparsers.add_parser('diarize')
    diarization_parser.add_argument('--audio', type=str, help='Path to the audio file', required=True)
    diarization_parser.add_argument('--output-filepath', type=str, help='Path for the diarization output', required=True)
    diarization_parser.set_defaults(fun=run_diarization)

    pipeline_parser = subparsers.add_parser('pipeline')
    pipeline_parser.add_argument('--audio', type=str, required=True)
    pipeline_parser.add_argument('--output-filepath', type=str, required=True)
    pipeline_parser.add_argument('--model', type=str, choices=['whisper', 'wav2vec2', 'parakeet'], help='Choice of ASR model (default is Whisper)', required=True)
    pipeline_parser.add_argument('--engine', type=str, choices=['torch', 'onnx', 'ct2'], help='The inference engine for running ASR (default is Torch)', required=True)
    pipeline_parser.set_defaults(fun=run_pipeline)

    return parser


def load_inference_configs(model: str = None, engine: str = None):
    config_path = Path('../../configs').resolve()
    alignment_model = 'whisperx' if model == 'whisper' else None
    with initialize_config_dir(version_base=None, config_dir=str(config_path)):
        config = compose(
            config_name='config',
            overrides=[
                f'asr={model}',
                f'engine={engine}',
                f'alignment={alignment_model}'
            ]
        )
    return config


def run_asr(audio, output_filepath, model, engine):
    cfg = load_inference_configs(model=model, engine=engine)
    result = asr_inference(
        results_filepath=output_filepath,
        engine_config=cfg.engine,
        asr_config=cfg.asr,
        align_config=cfg.alignment,
        data=audio
    )
    log.info('ASR Result: %s', result)


def run_diarization(audio, output_filepath):
    cfg = load_inference_configs()
    result = diarization_inference(
        data=audio,
        diarize_config=cfg.diarization,
        result_filepath=output_filepath
    )
    log.info('Diarization Result: %s', result)


def run_pipeline(audio, output_filepath, model, engine):
    cfg = load_inference_configs(model=model, engine=engine)
    pipeline_inference(
        asr_filepath='',
        diarization_filepath='',
        transcription_filepath=output_filepath,
        data_config=audio,
        asr_config=cfg.asr,
        engine_config=cfg.engine,
        diarize_config=cfg.diarization,
        align_config=cfg.alignment
    )

    pass


def cli_main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


