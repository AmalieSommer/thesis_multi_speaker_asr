import pytest
from multi_speaker_asr.data import AudioDataset



local_dataset_path_short_audio = {
    'path': '/root/master_thesis/thesis_multi_speaker_asr/data/lillelyd-main/lillelyd-main/manifest_test.jsonl',
    'name': None,
    'split': 'train'
}
local_dataset_path_long_audio = {
    'path': '/root/master_thesis/thesis_multi_speaker_asr/data/amicorpus/metadata.csv',
    'name': None,
    'split': 'train'
}
test_remote_data = {
    'path': 'CoRal-project/coral-v3',
    'name': 'conversation',
    'split': 'test'
}



@pytest.mark.integration
def test_local_dataset_iteration_short_audio():
    ds = AudioDataset(data_config=local_dataset_path_short_audio)

    it = iter(ds)
    for _ in range(10):
        sample = next(it)

        assert set(sample) == {
            'audio',
            'samplerate',
            'sample_id',
            'start',
            'end'
        }

        assert sample["samplerate"] == 16000
        assert sample["audio"].ndim == 1
        assert len(sample["audio"]) > 0



@pytest.mark.integration
def test_local_dataset_iteration_long_audio():
    ds = AudioDataset(data_config=local_dataset_path_long_audio, max_segment_duration=20)

    it = iter(ds)
    for _ in range(5):
        sample = next(it)

        assert set(sample) == {
            'audio',
            'samplerate',
            'sample_id',
            'start',
            'end'
        }

        assert sample["samplerate"] == 16000
        assert sample["audio"].ndim == 1
        assert len(sample["audio"]) > 0
        assert len(sample['audio']) <= 30 * 16000



@pytest.mark.integration
def test_remote_dataset_iteration():
    ds = AudioDataset(data_config=test_remote_data)

    for _, sample in zip(range(10), ds):
        assert sample["audio"].ndim == 1
        assert sample["samplerate"] == 16000




