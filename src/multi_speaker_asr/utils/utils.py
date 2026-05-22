from torchmetrics.text import WordErrorRate, CharErrorRate
from torch.nn.functional import cosine_similarity
from difflib import SequenceMatcher
import re


import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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