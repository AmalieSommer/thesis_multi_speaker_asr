import pytest
from unittest.mock import MagicMock, Mock
from multi_speaker_asr.evaluate import asr_inference


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