import pytest
import pandas as pd
from pathlib import Path
import os
from multi_speaker_asr.models.diarization import NumpyArrAudioSource, SpeakerDiarizationPipeline, SpeakerDiarizationConfig
import torch
import io
import torchaudio
import torchaudio.functional as F
from pyannote.core import Annotation


original_torch_load = torch.load
def trusted_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = trusted_torch_load


HF_TOKEN = os.getenv('HF_TOKEN')


# -------- FIXTURES ---------
@pytest.fixture(scope='session')
def diarization(request):
    return SpeakerDiarizationPipeline(config=request.param)

@pytest.fixture(scope="session")
def diarization_dataset(request):
    print(request.param)
    print(request.param[0])
    path = Path(request.param[0]).parent / 'data' / request.param[1]
    print(path)

    if not path.exists():
        pytest.fail(
            "Dataset missing. Download it before running integration tests."
        )
    if request.param[2] == 'parquet':
        return pd.read_parquet(path)
    elif request.param[2] == 'csv':
        return pd.read_csv(path)



def helper_load_audio(audio, samplerate: int = 16000):
    if isinstance(audio, bytes):
        audio = io.BytesIO(audio)
    wav, sr = torchaudio.load(audio)
    if sr != samplerate:
        wav = F.resample(waveform=wav, orig_freq=sr, new_freq=samplerate)
        sr = samplerate

    if wav.shape[0] > 1:
        wav = wav.mean(axis=0)
    wav = wav.squeeze(0).numpy()

    return wav, sr

# ----------- INTEGRATION TESTS -------------
@pytest.mark.parametrize('diarization', [SpeakerDiarizationConfig(max_speakers=2, duration=1)], indirect=True)
def test_verify_rttm_filepath(diarization):

    output_path = '/root/master_thesis/thesis_multi_speaker_asr/output_file.rttm'
    test_path = '/root/master_thesis/thesis_multi_speaker_asr/data/coral-v3-long-form-conversations/conv_1f2860dcc30248e710f1f39d128ea5ca.wav'
    wav, sr = helper_load_audio(test_path)

    audio = NumpyArrAudioSource(
        audio_arr=wav,
        uri='conv_1f2860dcc30248e710f1f39d128ea5ca.wav'
        )
    result = diarization.diarize(
        sample=audio,
        output_path=output_path
    )

    assert isinstance(result, Annotation)


@pytest.mark.parametrize('diarization', [SpeakerDiarizationConfig(max_speakers=2, duration=1)], indirect=True)
def test_verify_error_catch_missing_audio(diarization):

    output_path = '/root/master_thesis/thesis_multi_speaker_asr/output_file.rttm'
    test_path = '/root/master_thesis/thesis_multi_speaker_asr/data/coral-v3-long-form-conversations/conv_1f2860dcc30248e710f1f39d128ea5ca.wav'
    wav, sr = helper_load_audio(test_path)

    with pytest.raises(TypeError) as exec_info:
        audio = NumpyArrAudioSource(
            audio_arr=[wav],
            uri='conv_1f2860dcc30248e710f1f39d128ea5ca.wav'
            )
        result = diarization.diarize(
            sample=audio,
            output_path=output_path
        )

        assert isinstance(result, None)
    assert exec_info.type is TypeError


@pytest.mark.parametrize('output_filepath', [
    '/root/master_thesis/thesis_multi_speaker_asr/output_file.rttm',
    Path('/root/master_thesis/thesis_multi_speaker_asr/output_file.rttm')
])
@pytest.mark.parametrize('diarization', [SpeakerDiarizationConfig(max_speakers=2, duration=1)], indirect=True)
def test_verify_valid_output_filepath(diarization, output_filepath):

    output_path = output_filepath
    test_path = '/root/master_thesis/thesis_multi_speaker_asr/data/coral-v3-long-form-conversations/conv_1f2860dcc30248e710f1f39d128ea5ca.wav'
    wav, sr = helper_load_audio(test_path)

    audio = NumpyArrAudioSource(
        audio_arr=wav,
        uri='conv_1f2860dcc30248e710f1f39d128ea5ca.wav'
        )
    result = diarization.diarize(
        sample=audio,
        output_path=output_path
    )

    assert isinstance(result, Annotation)

    if isinstance(output_path, str):
        assert Path.exists(Path(output_path))
    else:
        assert Path.exists(output_path)


@pytest.mark.parametrize('output_filepath', [
    None,
    123456,
    ['.txt'],
    12.002
])
@pytest.mark.parametrize('diarization', [SpeakerDiarizationConfig(max_speakers=2, duration=1)], indirect=True)
def test_catch_invalid_filepath_type(diarization, output_filepath):

    output_path = output_filepath
    test_path = '/root/master_thesis/thesis_multi_speaker_asr/data/coral-v3-long-form-conversations/conv_1f2860dcc30248e710f1f39d128ea5ca.wav'
    wav, sr = helper_load_audio(test_path)

    with pytest.raises((TypeError, ValueError)) as exec_info:
        audio = NumpyArrAudioSource(
            audio_arr=[wav],
            uri='conv_1f2860dcc30248e710f1f39d128ea5ca.wav'
            )
        result = diarization.diarize(
            sample=audio,
            output_path=output_path
        )

        assert isinstance(result, None)
    assert exec_info.type is TypeError


@pytest.mark.parametrize('output_filepath', [
    None,
    '/root/master_thesis/thesis_multi_speaker_asr/output_file.txt',
    '12345',
    '',
    '               ',
    Path('hjur4215')
])
@pytest.mark.parametrize('diarization', [SpeakerDiarizationConfig(max_speakers=2, duration=1)], indirect=True)
def test_catch_invalid_output_filepath(diarization, output_filepath):

    output_path = output_filepath
    test_path = '/root/master_thesis/thesis_multi_speaker_asr/data/coral-v3-long-form-conversations/conv_1f2860dcc30248e710f1f39d128ea5ca.wav'
    wav, sr = helper_load_audio(test_path)

    with pytest.raises((TypeError, ValueError)) as exec_info:
        audio = NumpyArrAudioSource(
            audio_arr=wav,
            uri='conv_1f2860dcc30248e710f1f39d128ea5ca.wav'
            )
        result = diarization.diarize(
            sample=audio,
            output_path=output_path
        )

        assert isinstance(result, None)
    assert exec_info.type is ValueError


@pytest.mark.parametrize('diarization', [SpeakerDiarizationConfig(max_speakers=2, duration=1)], indirect=True)
def test_verify_annotation_output(diarization):

    output_path = '/root/master_thesis/thesis_multi_speaker_asr/output_file.rttm'
    test_path = '/root/master_thesis/thesis_multi_speaker_asr/data/coral-v3-long-form-conversations/conv_1f2860dcc30248e710f1f39d128ea5ca.wav'
    wav, sr = helper_load_audio(test_path)

    audio = NumpyArrAudioSource(
        audio_arr=wav,
        uri='conv_1f2860dcc30248e710f1f39d128ea5ca.wav'
        )
    result = diarization.diarize(
        sample=audio,
        output_path=output_path
    )

    assert isinstance(result, Annotation)

    reformatted_res = diarization.get_speaker_segments(result)
    assert isinstance(reformatted_res, dict)
    assert 'segments' in reformatted_res.keys()
    assert len(reformatted_res.get('segments')) > 0
