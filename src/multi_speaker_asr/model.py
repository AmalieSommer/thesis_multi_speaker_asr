from torch import nn
import torch

from transformers import pipeline

class Model(nn.Module):
    """
    The chosen ASR model based on Whisper-large-v3, with the baseline and finetuned versions
    """
    MODELS_DICT = {
        "base": "CoRal-project/roest-v3-whisper-1.5b",
        "finetuned": ""
    }

    def __init__(self, model: str = "base"):
        super().__init__()

        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)

if __name__ == "__main__":
    model = Model()
    x = torch.rand(1)
    print(f"Output shape of model: {model(x).shape}")
