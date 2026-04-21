from torchmetrics.text import WordErrorRate, CharErrorRate
from sentence_transformers import SentenceTransformer
from torch.nn.functional import cosine_similarity


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


#TODO: Move BERT to a model class!
#bert_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def compute_cosine_sim(pred_embeddings, target_embeddings):
    return 1 - cosine_similarity(pred_embeddings, target_embeddings)


def compute_ember(pred_emb, target_emb):

    return None

