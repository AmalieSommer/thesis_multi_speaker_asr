from torchmetrics.text import WordErrorRate, CharErrorRate
from torch.nn.functional import cosine_similarity
from difflib import SequenceMatcher
import re
import psutil
import os
from num2words import num2words
from jiwer import wer, cer
import json
import pickle


LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        },
    },
    'handlers': {
        'file': {
            'level': 'DEBUG',
            'class': 'logging.FileHandler',
            'filename': '/zhome/28/9/151118/thesis/thesis_multi_speaker_asr/results/torch_engine/fp32/performance_wav2vec.log',
            'formatter': 'default',
        },
        'stdout': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'default',
        },
    },
    'loggers': {
        'ASR': {
            'handlers': ['file', 'stdout'],
            'level': 'DEBUG',
            'propagate': True,
                },
        'Evaluate': {
            'handlers': ['file', 'stdout'],
            'level': 'DEBUG',
            'propagate': True,
        },
        'Wav2Vec2': {
            'handlers': ['file', 'stdout'],
            'level': 'DEBUG',
            'propagate': True,
        },
        'Diarization': {
            'handlers': ['file', 'stdout'],
            'level': 'DEBUG',
            'propagate': True,
        },
        'AudioData': {
            'handlers': ['file', 'stdout'],
            'level': 'DEBUG',
            'propagate': True,
        },
        'Whisper': {
            'handlers': ['file', 'stdout'],
            'level': 'DEBUG',
            'propagate': True,
        },
        'Main': {
            'handlers': ['file', 'stdout'],
            'level': 'DEBUG',
            'propagate': True,
        },
        'Engine': {
            'handlers': ['file', 'stdout'],
            'level': 'DEBUG',
            'propagate': True,
        }
    },
}


# USING PSUTIL FOR MEMORY PROFILING OF INDIVIDUAL FUNCTIONS
def process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss

# decorator function
def profile(func):
    def wrapper(*args, **kwargs):

        mem_before = process_memory()
        result = func(*args, **kwargs)
        mem_after = process_memory()
        record = {
            "function": func.__name__,
            "before": mem_before / (1e+6),                  # Converting bytes to MB
            "after": mem_after / (1e+6),                    # Converting bytes to MB
            "delta": (mem_after - mem_before)  / (1e+6),    # Converting bytes to MB
        }

        wrapper.memory_stats.append(record)
        return result

    wrapper.memory_stats = []
    return wrapper



def compute_wer(pred, target):
    wer = WordErrorRate()
    pred = normalize(pred)
    target = normalize(target)
    return wer(pred, target).item()


def compute_cer(pred, target):
    cer = CharErrorRate()
    pred = normalize(pred)
    target = normalize(target)
    return cer(pred, target).item()

def normalize(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]','', text)
    return text

def compute_cosine_sim(pred_embeddings, target_embeddings):
    sim = cosine_similarity(pred_embeddings, target_embeddings).item()
    return 1 - sim


def compute_ember(pred_emb, target_emb):
    """Will use SequenceMatcher and return the opcodes containing the replacement words"""
    seq = SequenceMatcher(None, a=pred_emb, b=target_emb)
    substitutions = seq.get_opcodes()
    return substitutions


def clean_transcription(sentence: str):
    """
    Function to preprocess the ground truth and predicted transcripts before computing the performance using WER, CER etc...
    Should standardize the text to lowercase, no punctuations or special characters.
    It should also map all occurrences of numbers to textual representations using library function.
    """
    sentence = str.lower(sentence)
    sentence = re.sub(r'-(?!\d)', '', sentence)             # Remove - that are not followed by a number
    sentence = re.sub(r'(?<!\d)\.|\.?(?!\d)', '', sentence) # Remove . that are not enclosed by two numbers
    sentence = re.sub(r'[^\w\s.-]', '', sentence)           # Remove all punctuation except for the - and .
    sentence = re.sub(' +', ' ', sentence)                  # Replacing all duplicate spaces with single space.
    
    sentence_copy = str(sentence)

    for s in sentence.split():
        try: 
            num = float(s)
            word_rep = str(num2words(number=num))
            sentence_copy = sentence_copy.replace(s, word_rep)
        except ValueError as e:
            continue

    return sentence_copy


def save_asr_results(asr_output, asr_metadata, output_file):
    with open(output_file, 'a') as writer:
        results = []
        for data in asr_metadata:
            ref_text = data['text']
            transcripts = [asr_output[batch_info['ref_indices']] for batch_info in data['audio_batch_info']]
            words = []
            for info, transcript in zip(data['audio_batch_info'], transcripts):
                words.append([{'word': word['word'], 'start': word['start'] + info['start'], 'end': word['end'] + info['start']} for word in transcript])
            segment = ' '.join([word['word'] for row in words for word in row])

            wer_ = wer(reference=clean_transcription(ref_text), hypothesis=clean_transcription(segment))
            cer_ = cer(reference=clean_transcription(ref_text), hypothesis=clean_transcription(segment))

            results.append({
                'audio_id': data['audio_id'],
                'segment_id': data['segment_id'],
                'ref': ref_text,
                'seg_start': data['start'],
                'seg_end': data['end'],
                'wer': wer_,
                'cer': cer_,      
                'hyp': segment
            })
        writer.write(json.dumps(results) + '\n')
        writer.flush()


def save_logits(output, metadata, filepath):
    file = os.path.join(filepath, 'logits.pkl')
    with open(file, 'ab') as f:
        logits = {
            "logits": output.cpu(),
            "metadata": metadata,
        }
        pickle.dump(logits, f)