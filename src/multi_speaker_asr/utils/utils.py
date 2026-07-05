from torchmetrics.text import WordErrorRate, CharErrorRate
from torch.nn.functional import cosine_similarity
from difflib import SequenceMatcher
import re
import psutil
import os

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
            'filename': 'data/results/int8/cpu_threads_8/pipeline_performance.log',
            'formatter': 'default',
        },
        'stdout': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'default',
        },
    },
    'loggers': {
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