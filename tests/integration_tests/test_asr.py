import pytest
from jiwer import cer
import onnx
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq, ORTModelForCTC, pipeline
from optimum.onnxruntime.configuration import QuantType
import os
from multi_speaker_asr.models.asr import ASR
from multi_speaker_asr.models.engines import (
    PytorchEngine,
    OnnxEngine,
    CT2,
    BaseEngine
)
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import torchaudio
import torchaudio.functional as F
from huggingface_hub.errors import HFValidationError
from huggingface_hub import repo_exists
from optimum.exporters.tasks import TasksManager
from multi_speaker_asr.utils.utils import LOGGING_CONFIG
import logging
import logging.config
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(name='ASR')

# ---------------- FIXTURES -------------------
@pytest.fixture(scope='session')
def asr_model(request):
    model_path, model_name, model_type, backend = request.param
    return ASR(
        model_type=model_type,
        model_name=model_name,
        model_path=model_path,
        backend=backend
    )



# -------------- INTEGRATION TESTS ----------------
def read_audio_helper(audio: str, target_sr: int = 16000):
    wav, sr = torchaudio.load(audio)
    if sr != target_sr:
        wav = F.resample(waveform=wav, orig_freq=sr, new_freq=target_sr)

    if wav.shape[0] > 1:
        wav = wav.mean(axis=0)

    return wav.squeeze(0).numpy()


@pytest.mark.integration
@pytest.mark.parametrize('asr_model', [
    ('openai/whisper-tiny', 'whisper', 'seq2seq', 'torch'),
    ('openai/whisper-tiny', 'whisper', 'seq2seq', 'onnx')
], indirect=['asr_model'])
def test_save_torch_model(tmp_path, asr_model):
    d = tmp_path / "models"
    d.mkdir()

    asr_model.load()
    asr_model.engine.save_model(output_path=d)

    assert type(asr_model.engine) in (PytorchEngine, OnnxEngine)
    assert asr_model.engine.model is not None
    assert asr_model.engine.processor is not None
    assert len(os.listdir(path=d)) > 0


@pytest.mark.integration
def test_apply_optimization_ctc(tmp_path):
    model_dict = {
        'model_type': 'ctc',
        'model_name': 'wav2vec2',
        'backend': 'onnx',
        'model_path': 'CoRal-project/roest-v3-wav2vec2-315m'
    }
    model = ASR(**model_dict)
    opt_config = {
        'optimization_level': 1
    }

    d = tmp_path / "optimized_models"
    d.mkdir()

    transcript = 'The black sheet of paper is located up there besides the piece of timber'
    wav = read_audio_helper(audio='data/EmoTale-main/wav/EN_009_H_2.wav')
    
    model.load()
    output_path = model.engine.apply_optimizations(
        optimizations_config=opt_config,
        output_path=str(d)
    )
    logger.debug('BEFORE... The directory: %s contain the files: %s', output_path, os.listdir(output_path))
    model.save_model(output_path)
    model.save_processor(output_path)
    logger.debug('AFTER... The directory: %s contain the files: %s', output_path, os.listdir(output_path))

    assert any('.onnx' in file.suffix for file in output_path.iterdir() if file.is_file()) == True
    assert type(model.engine) == OnnxEngine
    assert d == output_path

    # Assess the quality of the predictions:
    pred = model.transcribe(wav, return_timestamps=False)
    logger.debug('First prediction was: %s', pred)
    assert cer(reference=transcript, hypothesis=pred[0]) < 0.8


    # Reload the model from the saved dir and use onnx API to check that the model is correct:
    logger.debug('Reloading the model at directory: %s', str(output_path))
    logger.debug('Model Dict BEFORE: %s', model_dict.items())

    with patch.object(OnnxEngine, 'is_exported', return_value=True):
        model_dict['model_path'] = str(output_path)
        logger.debug('Model Dict AFTER: %s', model_dict.items())
        model = ASR(**model_dict)
        model.load()
        pred = model.transcribe(wav, return_timestamps=False)
    
    assert cer(reference=transcript, hypothesis=pred[0]) < 0.8


@pytest.mark.integration
def test_apply_optimization_seq2seq(tmp_path):
    model_dict = {
            'model_type': 'seq2seq',
            'model_name': 'whisper',
            'backend': 'onnx',
            'model_path': 'openai/whisper-tiny'
        }
    model = ASR(**model_dict)
    opt_config = {
        'optimization_level': 1
    }

    d = tmp_path / "optimized_models"
    d.mkdir()

    transcript = 'The black sheet of paper is located up there besides the piece of timber'
    wav = read_audio_helper(audio='data/EmoTale-main/wav/EN_009_H_2.wav')
    
    model.load()
    output_path = model.engine.apply_optimizations(
        optimizations_config=opt_config,
        output_path=str(d)
    )
    logger.debug('BEFORE... The directory: %s contain the files: %s', output_path, os.listdir(output_path))
    model.save_model(output_path)
    model.save_processor(output_path)
    logger.debug('AFTER... The directory: %s contain the files: %s', output_path, os.listdir(output_path))

    assert any('.onnx' in file.suffix for file in output_path.iterdir() if file.is_file()) == True
    assert type(model.engine) == OnnxEngine
    assert d == output_path

    # Assess the quality of the predictions:
    pred = model.transcribe(wav, return_timestamps=False)
    logger.debug('First prediction was: %s', pred)
    assert cer(reference=transcript, hypothesis=pred[0]) < 0.8


    # Reload the model from the saved dir and use onnx API to check that the model is correct:
    logger.debug('Reloading the model at directory: %s', str(output_path))
    logger.debug('Model Dict BEFORE: %s', model_dict.items())

    with patch.object(OnnxEngine, 'is_exported', return_value=True):
        model_dict['model_path'] = str(output_path)
        logger.debug('Model Dict AFTER: %s', model_dict.items())
        model = ASR(**model_dict)
        model.load()
        pred = model.transcribe(wav, return_timestamps=False)
    
    assert cer(reference=transcript, hypothesis=pred[0]) < 0.8


@pytest.mark.integration
def test_apply_quantization_seq2seq(tmp_path):
    model_dict = {
        'model_type':   'seq2seq',
        'model_name':   'whisper',
        'backend':      'onnx',
        'model_path':   'openai/whisper-tiny'
    }
    model = ASR(**model_dict)
    d = tmp_path / 'quantized_models'
    d.mkdir()

    transcript = 'The black sheet of paper is located up there besides the piece of timber'
    wav = read_audio_helper(audio='data/EmoTale-main/wav/EN_009_H_2.wav')
    quant_config = {
        'is_static': False,
        'per_channel': True
    }
    model.load()
    output_path = model.engine.apply_quantization(
        quantization_type='dynamic',
        output_path=str(d),
        quant_config=quant_config
    )
    logger.debug('BEFORE... The directory: %s contain the files: %s', output_path, os.listdir(output_path))
    model.save_model(output_path)
    model.save_processor(output_path)
    logger.debug('AFTER... The directory: %s contain the files: %s', output_path, os.listdir(output_path))

    assert any('.onnx' in file.suffix for file in output_path.iterdir() if file.is_file()) == True
    assert type(model.engine) == OnnxEngine
    assert d == output_path

    # Assess the quality of the predictions:
    pred = model.transcribe(wav, return_timestamps=False)
    logger.debug('First prediction was: %s', pred)
    assert cer(reference=transcript, hypothesis=pred[0]) < 0.8


    # Reload the model from the saved dir and use onnx API to check that the model is correct:
    logger.debug('Reloading the model at directory: %s', str(output_path))
    logger.debug('Model Dict BEFORE: %s', model_dict.items())

    with patch.object(OnnxEngine, 'is_exported', return_value=True):
        model_dict['model_path'] = str(output_path)
        logger.debug('Model Dict AFTER: %s', model_dict.items())
        model = ASR(**model_dict)
        model.load()
        pred = model.transcribe(wav, return_timestamps=False)
    
    assert cer(reference=transcript, hypothesis=pred[0]) < 0.8


@pytest.mark.integration
def test_apply_quantization_ctc(tmp_path):
    model_dict = {
        'model_type':   'ctc',
        'model_name':   'wav2vec2',
        'backend':      'onnx',
        'model_path':   'CoRal-project/roest-v3-wav2vec2-315m'
    }
    model = ASR(**model_dict)
    d = tmp_path / 'quantized_models'
    d.mkdir()

    transcript = 'The black sheet of paper is located up there besides the piece of timber'
    wav = read_audio_helper(audio='data/EmoTale-main/wav/EN_009_H_2.wav')
    quant_config = {
        'is_static': False,
        'per_channel': True,
        'operators_to_quantize': ['MatMul']
    }
    model.load()
    output_path = model.engine.apply_quantization(
        quantization_type='dynamic',
        quant_config=quant_config,
        output_path=str(d)
    )
    logger.debug('BEFORE... The directory: %s contain the files: %s', output_path, os.listdir(output_path))
    model.save_model(output_path)
    model.save_processor(output_path)
    logger.debug('AFTER... The directory: %s contain the files: %s', output_path, os.listdir(output_path))

    assert any('.onnx' in file.suffix for file in output_path.iterdir() if file.is_file()) == True
    assert type(model.engine) == OnnxEngine
    assert d == output_path

    # Assess the quality of the predictions:
    pred = model.transcribe(wav, return_timestamps=False)
    logger.debug('First prediction was: %s', pred)
    assert cer(reference=transcript, hypothesis=pred[0]) < 0.8


    # Reload the model from the saved dir and use onnx API to check that the model is correct:
    logger.debug('Reloading the model at directory: %s', str(output_path))
    logger.debug('Model Dict BEFORE: %s', model_dict.items())

    with patch.object(OnnxEngine, 'is_exported', return_value=True):
        model_dict['model_path'] = str(output_path)
        logger.debug('Model Dict AFTER: %s', model_dict.items())
        model = ASR(**model_dict)
        model.load()
        pred = model.transcribe(wav, return_timestamps=False)
    
    assert cer(reference=transcript, hypothesis=pred[0]) < 0.8


@pytest.mark.integration
@pytest.mark.parametrize('asr_model', [
    ('openai/whisper-tiny', 'whisper', 'seq2seq', 'onnx')
], indirect=['asr_model'])
def test_generate_calibration_dataset(asr_model):
    long_audio_ds = {
        'path': '/root/master_thesis/thesis_multi_speaker_asr/data/amicorpus/metadata.csv',
        'name': None,
        'split': 'train'
    }
    asr_model.load()
    num_samples = 2
    ds = asr_model.engine.generate_calibration_dataset(ds_config=long_audio_ds, num_samples=num_samples)

    assert type(ds) == list
    assert len(ds) == 2
    assert all(type(sample) == dict for sample in ds)


@pytest.mark.integration
def test_apply_static_quantization_ctc(tmp_path):
    model_dict = {
        'model_type':   'ctc',
        'model_name':   'wav2vec2',
        'backend':      'onnx',
        'model_path':   'CoRal-project/roest-v3-wav2vec2-315m'
    }
    short_audio = {
    'path': '/root/master_thesis/thesis_multi_speaker_asr/data/lillelyd-main/lillelyd-main/manifest_test.jsonl',
    'name': None,
    'split': 'train'
}
    model = ASR(**model_dict)
    d = tmp_path / 'quantized_models'
    d.mkdir()

    quant_config = {
        'is_static': True,
        'per_channel': True,
        'operators_to_quantize': ['MatMul']
    }
    model.load()
    output_path = model.engine.apply_quantization(
        quantization_type='static',
        quant_config=quant_config,
        output_path=str(d),
        calibration_data_config=short_audio,
        calibration_num_samples=1
    )
    logger.debug('BEFORE... The directory: %s contain the files: %s', output_path, os.listdir(output_path))
    model.save_model(output_path)
    model.save_processor(output_path)
    logger.debug('AFTER... The directory: %s contain the files: %s', output_path, os.listdir(output_path))

    assert any('.onnx' in file.suffix for file in output_path.iterdir() if file.is_file()) == True
    assert type(model.engine) == OnnxEngine
    assert d == output_path

    # Reload the model from the saved dir and use onnx API to check that the model is correct:
    logger.debug('Reloading the model at directory: %s', str(output_path))
    logger.debug('Model Dict BEFORE: %s', model_dict.items())

    with patch.object(OnnxEngine, 'is_exported', return_value=True):
        model_dict['model_path'] = str(output_path)
        logger.debug('Model Dict AFTER: %s', model_dict.items())
        model = ASR(**model_dict)
        model.load()
    
        assert type(model.engine) is OnnxEngine



@pytest.mark.integration
@pytest.mark.parametrize('asr_model', [
    ('openai/whisper-tiny', 'whisper', 'seq2seq', 'onnx'),
    ('CoRal-project/roest-v3-wav2vec2-315m', 'wav2vec2', 'ctc', 'onnx'),
], indirect=['asr_model'])
def test_invalid_static_quantization(asr_model, tmp_path):
    asr_model.load()
    quant_config = {
        'is_static': True,
        'per_channel': True,
        'operators_to_quantize': ['MatMul']
    }
    d = tmp_path / 'quantized_models'
    d.mkdir()
    with pytest.raises((ValueError, TypeError)) as exec_info:
        output_path = asr_model.engine.apply_quantization(
            quantization_type='static',
            quant_config=quant_config,
            output_path=str(d),
            calibration_num_samples=2
        )
        assert output_path is None
    assert exec_info.type in (ValueError, TypeError)
    


@pytest.mark.parametrize('asr_model, timestamps', [
    (('openai/whisper-tiny', 'whisper', 'seq2seq', 'onnx'), False),
    (('CoRal-project/roest-v3-wav2vec2-315m', 'wav2vec2', 'ctc', 'onnx'), None),
    (('openai/whisper-tiny', 'whisper', 'seq2seq', 'torch'), False),
    (('CoRal-project/roest-v3-wav2vec2-315m', 'wav2vec2', 'ctc', 'torch'), None)
], indirect=['asr_model'])
def test_transcription_without_timestamps(asr_model, timestamps):
    test_wav = read_audio_helper(audio='/root/master_thesis/thesis_multi_speaker_asr/data/lillelyd-main/lillelyd-main/9859dab0/rec_9_boredom.flac')
    wav_ref = 'Det sorte ark papir er placeret deroppe ved siden af tømmerstykket'

    asr_model.load()
    result = asr_model.transcribe(audio_batch=test_wav, return_timestamps=timestamps)
    first_result = result[0]

    assert type(result) is list
    assert type(first_result) is list
    assert len(result) > 0
    assert len(first_result) > 0
    assert all(type(item) is dict for item in first_result)
    assert cer(reference=wav_ref, hypothesis=first_result[0]['text']) != None




@pytest.mark.parametrize('asr_model, timestamps', [
    (('openai/whisper-tiny', 'whisper', 'seq2seq', 'onnx'), True),
    (('CoRal-project/roest-v3-wav2vec2-315m', 'wav2vec2', 'ctc', 'onnx'), 'word'),
    (('openai/whisper-tiny', 'whisper', 'seq2seq', 'torch'), True),
    (('CoRal-project/roest-v3-wav2vec2-315m', 'wav2vec2', 'ctc', 'torch'), 'word')
], indirect=['asr_model'])
def test_transcription_with_timestamps(asr_model, timestamps):
    test_wav = read_audio_helper(audio='/root/master_thesis/thesis_multi_speaker_asr/data/lillelyd-main/lillelyd-main/9859dab0/rec_9_boredom.flac')
    wav_ref = 'Det sorte ark papir er placeret deroppe ved siden af tømmerstykket'

    asr_model.load()
    result = asr_model.transcribe(audio_batch=test_wav, return_timestamps=timestamps)
    first_result = result[0]

    assert type(result) is list
    assert type(first_result) is list
    assert len(result) > 0
    assert len(first_result) > 0
    assert all(type(item) is dict for item in first_result)
    assert 'start' in first_result[0].keys() and 'end' in first_result[0].keys() and 'text' in first_result[0].keys()
    assert cer(reference=wav_ref, hypothesis=first_result[0]['text']) != None