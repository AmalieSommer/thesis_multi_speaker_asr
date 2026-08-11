from multi_speaker_asr.models.alignment import Alignment
import pytest
from transformers import Wav2Vec2ForCTC
import torchaudio
import torchaudio.functional as F
from whisperx.types import AlignedTranscriptionResult, SingleSegment



@pytest.mark.parametrize('dict_config', [
    ({
        'language_code': 'da',
        'device': 'cpu',
        'model_name': 'CoRal-project/roest-v3-wav2vec2-315m',
        'model_dir': None
    }),
    ({
        'language_code': 'da',
        'model_name': 'CoRal-project/roest-v3-wav2vec2-315m',
        'model_dir': None
    })
])
def test_align_init(dict_config):
    align = Alignment(align_config=dict_config)
    assert align.device == 'cpu'
    assert align.metadata != None
    assert align.model != None 
    assert 'language' in list(align.metadata.keys()) and 'dictionary' in list(align.metadata.keys()) and 'type' in list(align.metadata.keys())
    assert type(align.model) is Wav2Vec2ForCTC


@pytest.mark.parametrize('dict_config', [
    ({
        'device': 'cpu',
        'model_name': 'CoRal-project/roest-v3-wav2vec2-315m',
        'model_dir': None
    }),
    ({
        'language_code': None,
        'model_name': 'CoRal-project/roest-v3-wav2vec2-315m',
        'model_dir': None
    }),
    ({
        'language_code': 1234,
        'model_name': 'CoRal-project/roest-v3-wav2vec2-315m',
        'model_dir': None
    }),
    ({
        'language_code': 'da',
        'model_name': None,
        'model_dir': None
    }),
    ({
        'language_code': None,
        'model_name': 1234,
        'model_dir': None
    }),
    ({
        'language_code': 'da',
        'model_name': '            ',
        'model_dir': None
    }),
    ({
        'language_code': 'da',
        'model_dir': None
    })
])
def test_invalid_align_initialization(dict_config):
    with pytest.raises((ValueError, TypeError)) as exec_info:
        align = Alignment(align_config=dict_config)

    assert exec_info.type in (ValueError, TypeError)


def read_audio_helper(audio: str, target_sr: int = 16000):
    wav, sr = torchaudio.load(audio)
    if sr != target_sr:
        wav = F.resample(waveform=wav, orig_freq=sr, new_freq=target_sr)

    if wav.shape[0] > 1:
        wav = wav.mean(axis=0)

    return wav.squeeze(0).numpy()


@pytest.mark.parametrize('config', [
    ({
        'language_code': 'da',
        'device': 'cpu',
        'model_name': 'CoRal-project/roest-v3-wav2vec2-315m',
        'model_dir': None
    })
])
def test_align_asr_output(config):
    audio= 'data/lillelyd-main/lillelyd-main/9be5af06/rec_0_neutral.flac'
    pred = {
        'start': 0.0,
        'end': 2.8,
        'text': 'Dugen ligger på køleskabet'
    }
    align = Alignment(align_config=config)
    wav = read_audio_helper(audio=audio)

    new_format = align.format_model_input(asr_output=[pred])
    assert isinstance(new_format, list)

    result = align.align(prediction=new_format, audio=wav)['word_segments']
    assert isinstance(result, list)
    assert all("word" in res.keys() for res in result)
    assert all("start" in res.keys() for res in result)
    assert all("end" in res.keys() for res in result)

    assert all(isinstance(res["word"], str) for res in result) 
    assert all(isinstance(res["start"], (int, float)) for res in result)
    assert all(isinstance(res["end"], (int, float)) for res in result)

    assert all(res["start"] <= res['end'] for res in result)



@pytest.mark.parametrize('config', [
    ({
        'language_code': 'da',
        'device': 'cpu',
        'model_name': 'CoRal-project/roest-v3-wav2vec2-315m',
        'model_dir': None
    })
])
def test_assign_speakers_words(config):
    audio= 'data/lillelyd-main/lillelyd-main/9be5af06/rec_0_neutral.flac'
    pred = {
        'start': 0.0,
        'end': 2.8,
        'text': 'Dugen ligger på køleskabet'
    }
    sd_pred = [
        {'start': 0.0, 'end': 0.8, 'speaker': 'SPEAKER_00'},
        {'start': 0.8, 'end': 1.2, 'speaker': 'SPEAKER_00'},
        {'start': 1.2, 'end': 2.0, 'speaker': 'SPEAKER_01'},
        {'start': 2.0, 'end': 2.3, 'speaker': 'SPEAKER_00'},
        {'start': 2.3, 'end': 2.8, 'speaker': 'SPEAKER_01'},
    ]
    align = Alignment(align_config=config)
    wav = read_audio_helper(audio=audio)

    new_format = align.format_model_input(asr_output=[pred])
    assert isinstance(new_format, list)

    result = align.align(prediction=new_format, audio=wav)
    speakers_words = align.align_words_speakers(sd_output=sd_pred, asr_output=result)

    result = result['word_segments']
    assert isinstance(result, list)
    assert all("word" in res.keys() for res in result)
    assert all("start" in res.keys() for res in result)
    assert all("end" in res.keys() for res in result)

    assert all(isinstance(res["word"], str) for res in result) 
    assert all(isinstance(res["start"], (int, float)) for res in result)
    assert all(isinstance(res["end"], (int, float)) for res in result)

    assert all(res["start"] <= res['end'] for res in result)

    assert isinstance(speakers_words, dict)
    assert 'segments' in speakers_words.keys()
    assert len(speakers_words['segments']) > 0
    assert all(isinstance(item, dict) and 'speaker' in item.keys() for item in speakers_words['segments'])


    