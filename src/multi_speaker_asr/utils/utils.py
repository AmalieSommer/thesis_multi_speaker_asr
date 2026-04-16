import torch.nn as nn
from torchmetrics.text import WordErrorRate, CharErrorRate
import io
import librosa


import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def compute_wer(pred, target):
    wer = WordErrorRate()
    return wer(pred, target)


def compute_cer(pred, target):
    cer = CharErrorRate()
    return cer(pred, target)




"""
class Metrics(nn.Module):
    

    def __init__(self, metrics_dict=None):
        
    
        self.metrics = metrics_dict or {"wer": WordErrorRate()}

    
    def semDist(self, sentence, transcript):
        return


    def embER(self, transcript, embedding):
   

    def _calc_custom_metric(self, metric_name, predictions, transcripts, embeddings):
        if metric_name == "semDist":
            return self.semDist(predictions, transcripts)
        else:
            return self.embER(transcripts, embeddings)

    
    def update(self, predictions, transcripts, embeddings=None):
   

        for metric_type, metric_func in self.metrics.items():
            if metric_type in ["semDist", "embER"]:
                result = self._calc_custom_metric(metric_type, predictions, transcripts, embeddings)
            else:
                result = metric_func.update()

        return result


    def compute(self):


        return
"""