import pytest
from pathlib import Path
from multi_speaker_asr.models.diarization import SpeakerDiarizationConfig
from multi_speaker_asr.evaluate import diarization_inference
import os

HF_TOKEN = os.getenv('HF_TOKEN')


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