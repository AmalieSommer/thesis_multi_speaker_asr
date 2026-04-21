from omegaconf import DictConfig
import torch.nn as nn
import torch
import transformers
from transformers import AutoTokenizer, AutoModel

class BERT(nn.Module):

    def __init__(self, model_name, pooling="cls"):
        super().__init__()
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"
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
            # using semdist...
            return hidden_state[:, 0]
        elif self.pooling == "mean":
            # using ember...
            mask = attention_mask.unsqueeze(-1).to(self.device)
            return (hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            raise ValueError("Invalid pooling type...")

