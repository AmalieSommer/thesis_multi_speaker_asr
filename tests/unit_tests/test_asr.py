import pytest
from multi_speaker_asr.models.asr import ASR
from multi_speaker_asr.models.engines import (
    PytorchEngine,
    OnnxEngine,
    CT2,
    BaseEngine
)
from unittest.mock import Mock
import torchaudio
import torchaudio.functional as F
from huggingface_hub.errors import HFValidationError


# --- FIXTURES ---
@pytest.fixture(scope='function')
def model(request):
    model_type, model_name, model_path, backend = request.param
    return ASR(
        model_type=model_type,
        model_name=model_name,
        model_path=model_path,
        backend=backend
    )

@pytest.fixture(scope='function')
def engine(request):
    engine_type, model_path, model_name, model_type = request.param

    match type(engine_type):
        case BaseEngine.__class__:
            yield BaseEngine(model_path=model_path, model_name=model_name, model_type=model_type)
        case PytorchEngine.__class__:
            yield PytorchEngine(model_path=model_path, model_name=model_name, model_type=model_type)
        case OnnxEngine.__class__:
            yield OnnxEngine(model_path=model_path, model_name=model_name, model_type=model_type)
        case _:
            yield None


@pytest.mark.parametrize('engine_type, model_path, model_name, model_type', [
    (BaseEngine, None, 'wav2vec2', 'ctc'),
    (PytorchEngine, '', 'wav2vec2', 'ctc'),
    (OnnxEngine, '         ', 'whisper', 'seq2seq'),
    (BaseEngine, 9054, 'whisper', 'seq2seq'),
    (PytorchEngine, [], 'wav2vec2', 'ctc'),
    (OnnxEngine, {'testing': 123}, 'whisper', 'seq2seq')
])
def test_invalid_engine_initialization(engine_type, model_path, model_name, model_type):
    with pytest.raises(ValueError) as exec_info:
        engine_type(
            model_type=model_type,
            model_path=model_path,
            model_name=model_name
        )
    assert exec_info.type is ValueError
    

@pytest.mark.parametrize('engine', [
    (BaseEngine, 'openai/whisper-tiny', 'whisper', 'seq2seq'),
    (PytorchEngine, 'openai/whisper-tiny', 'whisper', 'seq2seq'),
    (OnnxEngine, 'openai/whisper-tiny', 'whisper', 'seq2seq'),
    (CT2, 'openai/whisper-tiny', 'whisper', 'seq2seq')
], indirect=['engine'])
def test_engine_initialization(engine):
    model_path = engine._get_model_path()
    assert model_path is not None
    assert len(model_path) > 0


# --- UNIT TESTS ---
@pytest.mark.parametrize('model_type, model_name, model_path, backend', [
    ('seq2seq', 'whisper', 'openai/whisper-tiny', None),
    ('seq2seq', 'whisper', '   /test-', 'torch'),
    ('seq2seq', 'whisper', 'openai/whisper-tiny', ''),
    ('ctc', 'wav2vec2', 'CoRal-project/roest-v3-wav2vec2-315m', '    ')
])
def test_model_initialization(model_type, model_name, model_path, backend):
    with pytest.raises((ValueError, HFValidationError)) as exec_info:
        model = ASR(
            model_type=model_type,
            model_name=model_name,
            model_path=model_path,
            backend=backend
        )
        assert model is not None, 'Model failed to initialize'
        assert model._get_engine() is None, 'Engine should be empty'
    assert exec_info.type is ValueError


@pytest.mark.parametrize('model_type, model_name, model_path, backend', [
    ('seq2seq', 'whisper', 'openai/whisper-tiny', 'torch'),
    ('seq2seq', 'whisper', 'openai/whisper-tiny', 'onnx'),
    ('seq2seq', 'whisper', 'pluttodk/roest-v3-whisper-1.5b-ct2', 'ct2') # Currently the only Huggingface seq2seq model which is supported by faster-whisper ct2
])
def test_model_engine_loading(model_type, model_name, model_path, backend):
    model = ASR(
        model_type=model_type,
        model_path=model_path,
        model_name=model_name,
        backend=backend
    )
    model.load()
    if model.backend == 'torch':
        assert isinstance(model.engine, PytorchEngine)
    elif model.backend == 'onnx':
        assert isinstance(model.engine, OnnxEngine)
    elif model.backend == 'ct2':
        assert isinstance(model.engine, CT2)
    else:
        assert isinstance(model.engine, BaseEngine)


def read_audio_helper(audio: str, target_sr: int = 16000):
    wav, sr = torchaudio.load(audio)
    if sr != target_sr:
        wav = F.resample(waveform=wav, orig_freq=sr, new_freq=target_sr)

    if wav.shape[0] > 1:
        wav = wav.mean(axis=0)

    return wav.squeeze(0).numpy()


@pytest.mark.parametrize('engine_type, model_type',
    [
        (PytorchEngine, "seq2seq"),
        (OnnxEngine, "seq2seq"),
        (PytorchEngine, "ctc"),
        (OnnxEngine, "ctc"),
    ])
def test_transcribe_calls_pipeline(engine_type, model_type):
    test_audio = read_audio_helper(
        audio="data/lillelyd-main/lillelyd-main/9be5af06/rec_0_neutral.flac"
    )

    pipe = Mock()
    pipe.return_value = [{"text": "dugen ligger på køleskabet"}]

    engine_ = engine_type.__new__(engine_type)
    engine_.model_type = model_type
    engine_.model = pipe

    result = engine_.transcribe(test_audio, return_timestamps=False)

    pipe.assert_called_once_with(test_audio)
    assert result[0] == "dugen ligger på køleskabet"

