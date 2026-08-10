import pytest
from multi_speaker_asr.data import AudioDataset, validate_filepath
from unittest.mock import MagicMock, patch, Mock
from pathlib import Path
import json
import numpy as np
import os
from huggingface_hub.errors import HFValidationError
from datasets import Dataset

# --- FIXTURES ---
@pytest.fixture
def mock_hf_dataset():
    mock_ds = MagicMock()
    mock_ds.rename_columns.return_value = mock_ds
    mock_ds.__getitem__.return_value = mock_ds
    mock_ds.cast_column.return_value = mock_ds
    return mock_ds


# --- UNIT TESTS ---
@patch('multi_speaker_asr.data.validate_filepath', return_value=('local', 'file', Path("metadata.json")))
def test_load_wav(_, tmp_path):
    test_metadata = [{
        'sample_id': 'test_file_1',
        'audio': 'data/EmoTale-main/wav/EN_017_S_5.wav'
    }]

    
    json_path = Path(os.path.join(tmp_path, "metadata.json"))
    json_path.write_text(json.dumps(test_metadata))

    print(json_path)
    data = AudioDataset(data_path=json_path, mode='segments')
    sample = next(iter(data))
    assert sample['sample_id'] == 'test_file_1'
    assert isinstance(sample['audio'], np.ndarray)
    assert sample['samplerate'] == 16000
    assert round(sample['audio'].shape[0] / sample['samplerate']) == 2.0
    

@pytest.mark.parametrize('file_extension', [
    'txt',
    'csv',
    'tsv',
    'json',
    'sonl',
    'parquet',
    'arrow',
    'xml',
    'gz'
])
def test_validate_filepath_local(tmp_path, file_extension):
    temp_path = tmp_path / f'temp_data.{file_extension}'
    temp_path.write_text("{}")
    file_type, path_type, path = validate_filepath(str(temp_path))

    assert file_type == 'local'
    assert path_type == 'file'
    assert path == temp_path


def test_valid_filepath_hf():
    hf_repo = 'CoRal-project/coral-v3'
    file_type, path_type, path = validate_filepath(hf_repo)

    assert file_type == 'hub'
    assert path_type == 'repo'
    assert path == None


@pytest.mark.parametrize('filepath', [
    '    ',
    None,
    '',
    '---',
    '#123   -    123#',
    2008
])
def test_invalid_filepath(filepath):
    with pytest.raises(Exception) as exec_info:
        file_type, path_type, path = validate_filepath(filepath)

    assert exec_info.type in (ValueError, HFValidationError, TypeError, FileNotFoundError)


@patch("multi_speaker_asr.data.load_dataset")
@patch("multi_speaker_asr.data.validate_filepath")
def test_load_data_local(mock_validate, mock_load_dataset):
    mock_validate.return_value = ("local", "file", Path("metadata.json"))
    mock_dataset = Mock()
    mock_load_dataset.return_value = mock_dataset

    dataset = AudioDataset(data_path='metadata.json', split='train')
    mock_load_dataset.assert_called_once_with(
        "json",
        data_files="metadata.json",
        split="train",
        streaming=True,
    )
    assert dataset.metadata is mock_dataset



@patch("multi_speaker_asr.data.load_dataset")
@patch("multi_speaker_asr.data.validate_filepath")
def test_load_data_hub(mock_validate, mock_load_dataset):
    mock_validate.return_value = ("hub", "repo", None)

    mock_dataset = Mock()
    mock_load_dataset.return_value = mock_dataset

    dataset = AudioDataset(data_path='CoRal-project/coral-v3', split='test')

    mock_load_dataset.assert_called_once_with(
        'CoRal-project/coral-v3',
        split='test',
        streaming=True,
    )
    mock_dataset.cast_column.assert_called_once()


@patch.object(AudioDataset, 'load_data')
def test_iter_segments(mock_load_data):
    data = Dataset.from_list([{
        "sample_id": "id1",
        "audio": "dummy.wav"
    }])
    mock_load_data.return_value = data

    dataset = AudioDataset(...)
    dataset.load_wav = Mock(
        return_value=(np.zeros(16000, dtype=np.float32), 16000)
    )
    dataset.search_cutoff_points = Mock(
        return_value=iter([
            {
                "sample_id": "id1",
                "audio": np.zeros(16000, dtype=np.float32),
                "samplerate": 16000,
            }
        ])
    )

    sample = next(iter(dataset))

    assert sample["sample_id"] == "id1"
    assert sample["samplerate"] == 16000


@patch.object(AudioDataset, 'load_data')
def test_find_cut_index_short_audio(mock_load_data):
    data = Dataset.from_list([{
            "sample_id": "id1",
            "audio": "dummy.wav"
        }])
    mock_load_data.return_value = data
    dataset = AudioDataset(...)
    cut = dataset._find_cut_index(
            speech_intervals=[],
            search_start=25 * 16000,
            search_end=35 * 16000,
            theoretical_end=30,
            grace_samples=0.5 * 16000,
        )
    assert cut == 25 * 16000


@patch.object(AudioDataset, 'load_data')
def test_find_cut_index_speech_timestamps_outside_grace(mock_load_data):
    data = Dataset.from_list([{
            "sample_id": "id1",
            "audio": "dummy.wav"
        }])
    mock_load_data.return_value = data
    dataset = AudioDataset(...)

    speech_intervals = [
        {
            'start': 1.0 * 16000,
            'end': 1.8 * 16000
        },
        {
            'start': 5.2 * 16000,
            'end': 7.4 * 16000
        }
    ]
    cut = dataset._find_cut_index(
            speech_intervals=speech_intervals,
            search_start=25 * 16000,
            search_end=35 * 16000,
            theoretical_end=30 * 16000,
            grace_samples=0.5 * 16000,
        )
    assert int(cut) == int(28.5 * 16000) # Should select the mid-point of the gap after the first speech segment, since the ending point for the speech of the second segment is greater than the 0.5 grace period after the hard cut-off point.


@patch.object(AudioDataset, 'load_data')
def test_find_cut_index_speech_timestamps_inside_grace(mock_load_data):
    data = Dataset.from_list([{
            "sample_id": "id1",
            "audio": "dummy.wav"
        }])
    mock_load_data.return_value = data
    dataset = AudioDataset(...)

    speech_intervals = [
        {
            'start': 1.0 * 16000,
            'end': 1.8 * 16000
        },
        {
            'start': 3.2 * 16000,
            'end': 5.4 * 16000
        }
    ]
    cut = dataset._find_cut_index(
            speech_intervals=speech_intervals,
            search_start=25 * 16000,
            search_end=35 * 16000,
            theoretical_end=30 * 16000,
            grace_samples=0.5 * 16000,
        )
    assert int(cut) == int(32.7 * 16000) # Should select the mid-point of the gap after the last speech segment, since the ending point for the speech of the last segment is within the 0.5 grace period after the hard cut-off point.



@patch("multi_speaker_asr.data.AudioDataset._find_cut_index")
@patch.object(AudioDataset, 'load_data')
def test_search_cutoff_point_cut_index_out_of_bounds(mock_load_data, mock_cut_index):
    data = Dataset.from_list([{
            "sample_id": "id1",
            "audio": "dummy.wav"
        }])
    mock_load_data.return_value = data
    dataset = AudioDataset(...)
    audio = np.zeros(40 * 16000, dtype=np.float32)
    mock_cut_index.return_value = 41 * 16000

    with pytest.raises(IndexError) as exec_info:
        result = list(
            dataset.search_cutoff_points(
                audio_np=audio,
                sample_info={"sample_id": "id1"},
                sr=16000,
                max_sec=30,
            )
        )

    assert exec_info.type is IndexError

    