import pytest
from pathlib import Path
from multi_speaker_asr.models.diarization import SpeakerDiarizationConfig
from multi_speaker_asr.evaluate import diarization_inference, asr_inference, assign_words_speakers
import os

HF_TOKEN = os.getenv('HF_TOKEN')

@pytest.mark.integration
def test_diarize_inference():
    output_path = '/root/master_thesis/thesis_multi_speaker_asr/output_file.rttm'
    data_config = {
        'path': '/root/master_thesis/thesis_multi_speaker_asr/data/coral-v3-long-form-conversations/test/coral_metadata.csv',
        'split': 'train',
        'name': None
    }
    diarize_config = {
        'max_speakers': 2,
        'duration': 1,
        'hf_token': HF_TOKEN
    }
    
    res = diarization_inference(
        data_config=data_config,
        diarize_config=diarize_config,
        result_filepath=output_path
    )

    assert isinstance(res, list)
    assert len(res) > 0
    assert 'sample_id' in res[0].keys()
    assert 'result' in res[0].keys()
    if isinstance(output_path, str):
        assert Path.exists(Path(output_path))
    else:
        assert Path.exists(output_path)


@pytest.mark.parametrize('model_config', [
    ({
        'model_path': '/root/master_thesis/thesis_multi_speaker_asr/src/multi_speaker_asr/models/saved_models/coral_wav2vec2',
        'model_type': 'ctc',
        'model_name': 'wav2vec2',
        'device': 'cpu'
    })
])
def test_assign_words_speakers(model_config, tmp_path):
    output_asr_filepath = tmp_path / "test_results.jsonl"
    data_config = {
        'path': '/root/master_thesis/thesis_multi_speaker_asr/data/lillelyd-main/lillelyd-main/test_manifest.jsonl',
        'split': 'train',
        'name': None
    }

    asr_result = asr_inference(
        results_filepath=output_asr_filepath, 
        backend='torch', 
        data_config=data_config, 
        model_config=model_config, 
        timestamps=True,
        align_config=None
        )

    output_diarize_path = '/root/master_thesis/thesis_multi_speaker_asr/output_file.rttm'
    diarize_config = {
        'max_speakers': 2,
        'duration': 1,
        'hf_token': HF_TOKEN
    }
    
    diarize_res = diarization_inference(
        data_config=data_config,
        diarize_config=diarize_config,
        result_filepath=output_diarize_path
    )

    filename = tmp_path / 'test_words_speakers.jsonl'
    final_res = assign_words_speakers(diarize_output=diarize_res, asr_output=asr_result, filename=filename)
    assert isinstance(final_res, list)
    assert len(final_res) > 0
    assert 'sample_id' in final_res[0].keys()
    assert 'segments' in final_res[0].keys()