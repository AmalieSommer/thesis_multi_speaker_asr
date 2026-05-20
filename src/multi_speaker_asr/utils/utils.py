from torchmetrics.text import WordErrorRate, CharErrorRate
from torch.nn.functional import cosine_similarity
from difflib import SequenceMatcher


import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def compute_wer(pred, target):
    wer = WordErrorRate()
    res = wer(pred, target)
    return wer(pred, target)


def compute_cer(pred, target):
    cer = CharErrorRate()
    return cer(pred, target)

def compute_cosine_sim(pred_embeddings, target_embeddings):
    return 1 - cosine_similarity(pred_embeddings, target_embeddings)


def compute_ember(pred_emb, target_emb):
    """Will use SequenceMatcher and return the opcodes containing the replacement words"""
    seq = SequenceMatcher(None, a=pred_emb, b=target_emb)
    substitutions = seq.get_opcodes()
    return substitutions

