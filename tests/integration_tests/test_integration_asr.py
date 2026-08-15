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
import logging
from multi_speaker_asr.inference import asr_inference


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
    model.save_model(output_path)
    model.save_processor(output_path)

    assert any('.onnx' in file.suffix for file in output_path.iterdir() if file.is_file()) == True
    assert type(model.engine) == OnnxEngine
    assert d == output_path

    # Assess the quality of the predictions:
    pred = model.transcribe(wav, return_timestamps=False)
    assert cer(reference=transcript, hypothesis=pred[0]) < 0.8


    # Reload the model from the saved dir and use onnx API to check that the model is correct:

    with patch.object(OnnxEngine, 'is_exported', return_value=True):
        model_dict['model_path'] = str(output_path)
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
    model.save_model(output_path)
    model.save_processor(output_path)

    assert any('.onnx' in file.suffix for file in output_path.iterdir() if file.is_file()) == True
    assert type(model.engine) == OnnxEngine
    assert d == output_path

    # Assess the quality of the predictions:
    pred = model.transcribe(wav, return_timestamps=False)
    assert cer(reference=transcript, hypothesis=pred[0]) < 0.8


    # Reload the model from the saved dir and use onnx API to check that the model is correct:
    with patch.object(OnnxEngine, 'is_exported', return_value=True):
        model_dict['model_path'] = str(output_path)
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
    model.save_model(output_path)
    model.save_processor(output_path)

    assert any('.onnx' in file.suffix for file in output_path.iterdir() if file.is_file()) == True
    assert type(model.engine) == OnnxEngine
    assert d == output_path

    # Assess the quality of the predictions:
    pred = model.transcribe(wav, return_timestamps=False)
    assert cer(reference=transcript, hypothesis=pred[0]) < 0.8


    # Reload the model from the saved dir and use onnx API to check that the model is correct:
    with patch.object(OnnxEngine, 'is_exported', return_value=True):
        model_dict['model_path'] = str(output_path)
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
    model.save_model(output_path)
    model.save_processor(output_path)

    assert any('.onnx' in file.suffix for file in output_path.iterdir() if file.is_file()) == True
    assert type(model.engine) == OnnxEngine
    assert d == output_path

    # Assess the quality of the predictions:
    pred = model.transcribe(wav, return_timestamps=False)
    assert cer(reference=transcript, hypothesis=pred[0]) < 0.8


    # Reload the model from the saved dir and use onnx API to check that the model is correct:
    with patch.object(OnnxEngine, 'is_exported', return_value=True):
        model_dict['model_path'] = str(output_path)
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
    model.save_model(output_path)
    model.save_processor(output_path)

    assert any('.onnx' in file.suffix for file in output_path.iterdir() if file.is_file()) == True
    assert type(model.engine) == OnnxEngine
    assert d == output_path

    # Reload the model from the saved dir and use onnx API to check that the model is correct:
    with patch.object(OnnxEngine, 'is_exported', return_value=True):
        model_dict['model_path'] = str(output_path)
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



# -------------- Pipeline Testing ------------------------
@pytest.mark.integration
@pytest.mark.parametrize('model_config', [
    ({
        'model_path': 'openai/whisper-tiny',
        'model_type': 'seq2seq',
        'model_name': 'whisper',
        'device': 'cpu'
    }),
    ({
        'model_path': 'CoRal-project/roest-v3-wav2vec2-315m',
        'model_type': 'ctc',
        'model_name': 'wav2vec2',
        'device': 'cpu'
    })
])
def test_torch_asr_pipeline_timestamps(model_config, tmp_path):
    output_filepath = tmp_path / "test_results.jsonl"
    data_config = {
        'path': '/root/master_thesis/thesis_multi_speaker_asr/data/lillelyd-main/lillelyd-main/test_manifest.jsonl',
        'split': 'train',
        'name': None
    }
    align_config = {
        'language_code': 'da',
        'device': 'cpu',
        'model_name': 'CoRal-project/roest-v3-wav2vec2-315m',
        'model_dir': None
    }

    result = asr_inference(
        results_filepath=output_filepath, 
        backend='torch', 
        data_config=data_config, 
        model_config=model_config, 
        timestamps=True,
        align_config=align_config
        )

    assert result is not None
    assert isinstance(result, list)
    assert type(result[0]) == dict
    assert len(list(result[0].keys())) > 0
    assert 'sample_id' in result[0].keys() and 'result' in result[0].keys()


@pytest.mark.parametrize('model_config', [
    ({
        'model_path': '/root/master_thesis/thesis_multi_speaker_asr/src/multi_speaker_asr/models/saved_models/coral_whisper_onnx/',
        'model_type': 'seq2seq',
        'model_name': 'whisper',
        'device': 'cpu'
    }),
    ({
        'model_path': '/root/master_thesis/thesis_multi_speaker_asr/src/multi_speaker_asr/models/saved_models/coral_wav2vec2_onnx/',
        'model_type': 'ctc',
        'model_name': 'wav2vec2',
        'device': 'cpu'
    })
])
def test_onnx_asr_pipeline_timestamps(model_config, tmp_path):
    output_filepath = tmp_path / "test_results.jsonl"
    data_config = {
        'path': '/root/master_thesis/thesis_multi_speaker_asr/data/lillelyd-main/lillelyd-main/test_manifest.jsonl',
        'split': 'train',
        'name': None
    }
    align_config = {
        'language_code': 'da',
        'device': 'cpu',
        'model_name': '/root/master_thesis/thesis_multi_speaker_asr/src/multi_speaker_asr/models/saved_models/coral_wav2vec2',
        'model_dir': None
    }

    result = asr_inference(
        results_filepath=output_filepath, 
        backend='onnx', 
        data_config=data_config, 
        model_config=model_config, 
        timestamps=True,
        align_config=align_config
        )

    assert result is not None
    assert isinstance(result, list)
    assert type(result[0]) == dict
    assert len(list(result[0].keys())) > 0
    assert 'sample_id' in result[0].keys() and 'result' in result[0].keys()
    chunks = result[0]['result']['chunks'][0]
    assert 'start' in chunks.keys()
    assert 'end' in chunks.keys()
    assert 'word' in chunks.keys()




@pytest.mark.parametrize('model_config', [
    ({
        'model_path': '/root/master_thesis/thesis_multi_speaker_asr/src/multi_speaker_asr/models/saved_models/coral_whisper_onnx/',
        'model_type': 'seq2seq',
        'model_name': 'whisper',
        'device': 'cpu'
    }),
    ({
        'model_path': '/root/master_thesis/thesis_multi_speaker_asr/src/multi_speaker_asr/models/saved_models/coral_wav2vec2_onnx/',
        'model_type': 'ctc',
        'model_name': 'wav2vec2',
        'device': 'cpu'
    })
])
def test_onnx_asr_pipeline_no_timestamps(model_config, tmp_path):
    output_filepath = tmp_path / "test_results.jsonl"
    data_config = {
        'path': '/root/master_thesis/thesis_multi_speaker_asr/data/lillelyd-main/lillelyd-main/test_manifest.jsonl',
        'split': 'train',
        'name': None
    }

    result = asr_inference(
        results_filepath=output_filepath, 
        backend='onnx', 
        data_config=data_config, 
        model_config=model_config, 
        timestamps=False,
        align_config=None
        )

    assert result is not None
    assert isinstance(result, list)
    assert type(result[0]) == dict
    assert len(list(result[0].keys())) > 0
    assert 'sample_id' in result[0].keys() and 'result' in result[0].keys()