from omegaconf import DictConfig
import torch.nn as nn
import torch
import transformers
from transformers import AutoTokenizer, AutoModel

class BERT(nn.Module):
    """BERT Model for creating embeddings to be used with semantic evaluations (e.g. SemDist)"""
    def __init__(self, model_name, pooling="cls", device='cpu'):
        super().__init__()
        self.device = device
        self.model_name = model_name
        self.pooling = pooling
        self.load() # load model when initialized...


    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)


    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device)
        )

        hidden_state = outputs.last_hidden_state

        if self.pooling == "cls":
            return hidden_state[:, 0]

