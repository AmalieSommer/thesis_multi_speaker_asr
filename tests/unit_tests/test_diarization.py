from multi_speaker_asr.models.diarization import SpeakerDiarizationPipeline, RollingClusters, DiarizationConfig
from unittest.mock import MagicMock, patch, Mock
import numpy as np
import pytest
import torch


# ------ FIXTURES -------
@pytest.fixture(scope='function')
def diarization(request):
    with patch("multi_speaker_asr.models.diarization.Pipeline.from_pretrained") as mock_from_pretrained:
        mock_pipeline = MagicMock()
        mock_from_pretrained.return_value = mock_pipeline
        mock_pipeline.to.return_value = mock_pipeline
        return SpeakerDiarizationPipeline(config=request.param), mock_pipeline


@pytest.fixture(scope='function')
def rc():
    return RollingClusters()


# ------- UNIT TESTS --------
# ------ ROLLING CLUSTERS -------
def test_rolling_cluster_init():
    rc = RollingClusters()

    assert rc.match_threshold == 0.3
    assert rc.new_threshold == 0.55
    assert rc.max_history == 10
    assert rc.speaker_registry == {}
    assert rc.speaker_count == 0


def test_rc_preprocess(rc):
    embedding = np.array([3, 4])
    processed_emb = rc.preprocess(embedding=embedding)

    assert np.isclose(np.linalg.norm(processed_emb), 1.0)



def test_rc_embedding_error(rc):
    embedding = np.zeros((1, 192))
    processed_emb = rc.preprocess(embedding=embedding)

    assert not np.isnan(processed_emb).any()


@pytest.mark.parametrize('centroids', [
    ('SPEAKER_00', [np.array([1,2,3]), np.array([1,2,3]), np.array([1,2,3])]),
    ('SPEAKER_00', [np.array([0,4,1,2,3]), np.array([1,1,1,2,3])])
])
def test_compute_centroids(centroids, rc):
    rc.speaker_registry[centroids[0]] = centroids[1]
    speaker_ids, centroids_updated = rc.compute_centroids()

    assert speaker_ids == ['SPEAKER_00']
    assert len(centroids_updated) == 1


@pytest.mark.parametrize('centroids', [
    {
        'SPEAKER_00': [np.array([1,2,3,4]), np.array([1,2,3]), np.array([1,2,3])],
        'SPEAKER_01': [np.array([0,4,1,2,3]), np.array([1,1,1,2,3])]
    },
    {
        'SPEAKER_00': [np.array([1,2,3,4]), np.array([1,2,3]), np.array([1,2,3])],
        'SPEAKER_01': [np.array([0,4,1,2,3]), np.array([1,1,1,2,3])]
    }
])
def test_compute_centroids_error(centroids, rc):
    rc.speaker_registry = centroids
    with pytest.raises((ValueError)) as exec_info:
        speaker_ids, centroids_updated = rc.compute_centroids()
    assert exec_info.type == ValueError


def test_zero_embedding(rc):
    zero_emb = np.zeros(16,)
    processed_chunk = rc.process_chunk(zero_emb)
    assert processed_chunk == None


def test_empty_speaker_registry(rc):
    rand_emb = np.random.rand(16,)

    assert len(rc.speaker_registry.items()) == 0
    assert rc.speaker_count == 0

    processed_chunk = rc.process_chunk(rand_emb)

    assert rc.speaker_count == 1
    assert len(rc.speaker_registry.items()) == 1
    assert processed_chunk == 'SPEAKER_00'


def test_inconsistent_embedding_size(rc):
    speaker_registry = {
        "SPEAKER_00": [
            np.ones(192),
            np.ones(256)   # inconsistent dimension
        ]
    }
    rc.speaker_registry = speaker_registry
    with pytest.raises((ValueError)) as exec_info:
        rc.process_chunk(np.ones(256,))
    assert exec_info.type == ValueError


def test_matching_chunks(rc):
    embedding = np.array([1.0, 2.0, 3.0])
    existing_emb = rc.preprocess(embedding=np.array([1.1, 2.1, 3.1]))
    rc.speaker_registry = {
        "SPEAKER_00": [existing_emb]
    }
    rc.speaker_count = 1

    speaker = rc.process_chunk(embedding)

    assert speaker == "SPEAKER_00"
    assert len(rc.speaker_registry["SPEAKER_00"]) == 2


def test_process_chunk_max_history(rc):
    rc.max_history = 2
    embedding = np.array([1.0, 0.0, 0.0])

    rc.speaker_registry = {
        "SPEAKER_00": [
            embedding.copy(),
            embedding.copy()
        ]
    }
    rc.speaker_count = 1
    rc.process_chunk(embedding)

    assert len(rc.speaker_registry["SPEAKER_00"]) == 2


# ------- SPEAKER DIARIZATION PIPELINE ---------
@patch("multi_speaker_asr.models.diarization.Pipeline.from_pretrained")
def test_init_loads_pipeline(mock_from_pretrained,):
    mock_pipeline = MagicMock()
    mock_from_pretrained.return_value = mock_pipeline

    config = DiarizationConfig(model='test_model', hf_token='test_token')
    SpeakerDiarizationPipeline(config)

    mock_from_pretrained.assert_called_once_with(
        checkpoint_path='test_model',
        use_auth_token='test_token'
    )
    mock_pipeline.to.assert_called_once_with(device='cpu')


@pytest.mark.parametrize('diarization', [DiarizationConfig(model='test_mode', hf_token='test_token')], indirect=True)
def test_diarization_result_format(diarization):
    sample = {
        'audio': np.random.randn(16000)
    }

    diarization_pipeline = diarization[0]
    mock_pipeline = diarization[1]

    mock_output = MagicMock()
    mock_pipeline.return_value = mock_output

    # Fake segments
    segment1 = MagicMock(start=0.0, end=1.2, duration=1.2)
    segment2 = MagicMock(start=1.2, end=2.0, duration=0.8)

    mock_output.speaker_diarization.itertracks.return_value = [
        (segment1, None, 'SPEAKER_00'),
        (segment2, None, 'SPEAKER_01'),
    ]

    result = diarization_pipeline.diarize(sample)
    assert result['segments'] == [
        {
            'speaker': 'SPEAKER_00',
            'start': 0.0,
            'end': 1.2,
            'duration': 1.2,
        },
        {
            'speaker': 'SPEAKER_01',
            'start': 1.2,
            'end': 2.0,
            'duration': 0.8,
        },
    ]

@pytest.mark.parametrize('diarization', [DiarizationConfig(model='test_mode', hf_token='test_token')], indirect=True)
def test_setting_speaker_constraints(diarization):
    sample = {
        'audio': np.random.randn(16000)
    }

    diarization_pipeline = diarization[0]
    mock_pipeline = diarization[1]

    _ = diarization_pipeline.diarize(
        sample=sample,
        num_speakers=2,
        min_speakers=2,
        max_speakers=3
    )

    mock_pipeline.assert_called_once()
    args, kwargs = mock_pipeline.call_args

    print(args)
    print(kwargs)

    assert kwargs['num_speakers'] == 2
    assert kwargs['min_speakers'] == 2
    assert kwargs['max_speakers'] == 3

    assert args[0]['samplerate'] == 16000
    assert 'waveform' in args[0].keys()
    assert args[0]['waveform'].shape == torch.Size([1, 16000])


@pytest.mark.parametrize('diarization', [DiarizationConfig(model='test_mode', hf_token='test_token')], indirect=True)
def test_no_speaker_detected(diarization):
    diarization_pipeline = diarization[0]
    mock_pipeline = diarization[1]

    mock_output = MagicMock()
    mock_pipeline.return_value = mock_output
    mock_output.speaker_diarization.itertracks.return_value = []
    mock_output.embeddings = None

    sample = {
            'audio': np.random.randn(16000)
        }
    result = diarization_pipeline.diarize(sample)
    assert result['segments'] == []
    assert result['embeddings'] == None
    mock_pipeline.assert_called_once()
    mock_output.speaker_diarization.itertracks.assert_called_once_with(
        yield_label=True
    )


@pytest.mark.parametrize('diarization', [DiarizationConfig(model='test_mode', hf_token='test_token')], indirect=True)
def test_no_valid_audio_input(diarization):
    diarization_pipeline, _ = diarization
    with pytest.raises(KeyError) as exec_info:
        sample = {}
        diarization_pipeline.diarize(sample=sample)
    assert exec_info.type == KeyError


@pytest.mark.parametrize('diarization', [DiarizationConfig(model='test_mode', hf_token='test_token')], indirect=True)
def test_diarize_requests_embeddings(diarization):
    diarization_pipeline = diarization[0]
    mock_pipeline = diarization[1]

    mock_output = MagicMock()
    mock_pipeline.return_value = mock_output

    # Fake segments
    segment1 = MagicMock(start=0.0, end=1.2, duration=1.2)
    segment2 = MagicMock(start=1.2, end=2.0, duration=0.8)

    mock_output.speaker_diarization.itertracks.return_value = [
        (segment1, None, 'SPEAKER_00'),
        (segment2, None, 'SPEAKER_01'),
    ]

    sample = {
        'audio': np.random.randn(16000)
    }    
    result = diarization_pipeline.diarize(sample, return_embeddings=True)
    mock_pipeline.assert_called_once()

    _, kwargs = mock_pipeline.call_args
    assert kwargs["return_embeddings"] is True
    assert result["embeddings"] is not None