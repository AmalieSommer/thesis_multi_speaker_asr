import torch
import os
import json
from multi_speaker_asr.data import Data
from multi_speaker_asr.models.asr import Whisper
from multi_speaker_asr.models.alignment import Wav2Vec2
from hydra import initialize, compose
from multi_speaker_asr.evaluate import inference, eval_bert, timestamp_alignment
from multi_speaker_asr.models.bert import BERT
import ctranslate2
import gc
import tempfile


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print(f'Supported Compute Types: {ctranslate2.get_supported_compute_types(device)}')

with initialize(version_base=None, config_path='../configs'):
    config_data = compose(config_name='data')
    config_asr = compose(config_name='whisper-base')
    config_phoneme = compose(config_name='wav2vec2')

print(f'Data config file: {config_data}')
print(f'Model config file: {config_asr}')


RESULTS_FILEPATH = 'src/results'


def run_asr_inference(filename):
    ds = Data()
    ds.load_from_hf(config=config_data)

    whisper = Whisper()
    whisper.load(config=config_asr)
    out = inference(
        whisper=whisper,
        ds=ds.dataset
    )

    # UNLOAD WHISPER FROM MEMORY
    del whisper
    gc.collect()

    # LOAD BERT INTO MEMORY
    bert = BERT("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    info_updated = eval_bert(
        bert,
        out['info']
    )

    results = {
        'wer': out['avg_wer'],
        'cer': out['avg_cer'],
        'info': info_updated
    }

    try:
        with open(os.path.join(RESULTS_FILEPATH, f'{filename}.json'), "w") as f:
            json.dump(results, f, indent=4)
        print(f"Successfully saved the results to file")
    except Exception as e:
        print(f"Error saving results to file: {e}")

    print("Result is the following: ")
    print("WER: ", results["wer"], ", CER: ", results["cer"])

    # UNLOAD BERT FROM MEMORY
    del bert
    gc.collect()


def run_timestamp_alignment(filename):
    # LOAD PHONEME MODEL INTO MEMORY
    phoneme = Wav2Vec2()
    phoneme.load(config=config_phoneme)

    # Fetch transcripts from file:
    try:
        with open(os.path.join(RESULTS_FILEPATH, f'{filename}.json'), "r") as f:
            data = json.load(f)
        print(f"Successfully loaded file.")
    except Exception as e:
        print(f"Error reading from file: {e}")

    for item in data['info']:

        result = timestamp_alignment(
            model=phoneme,
            info=item
        )

        item['aligned_segments'] = result['segments']

    # Update json file with added aligned transcripts:
    try:
        with open(os.path.join(RESULTS_FILEPATH, f'{filename}.json'), "w") as f:
            json.dumps(data, f, indent=4)
        print(f"Successfully updated the file")
    except Exception as e:
        print(f"Error updating the file: {e}")


if __name__=='__main__':
    temp_dir = tempfile.TemporaryDirectory()
    try:
        filename = 'testing_sa_asr'
        run_asr_inference(filename=filename)
        run_timestamp_alignment(filename=filename)
    
    finally:
        temp_dir.cleanup()
