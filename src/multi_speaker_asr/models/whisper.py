from omegaconf import DictConfig
import torch.nn as nn
import torch
import transformers
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

class WhisperBase(nn.Module):
    """Base pretrained Whisper-based model loaded from Huggingface."""
    
    def __init__(self, config: DictConfig):
        """Initialize the model with the appropriate config file"""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.processor: AutoProcessor
        self.model: AutoModelForSpeechSeq2Seq

    
    def load_processor(self):
        """Load the pretrained processor into the AutoProcessor"""
        processor = self.config.model_name
        self.processor = AutoProcessor.from_pretrained(processor)

    def load_model(self):
        """Load the pretrained STT model from Huggingface"""
        model = self.config.model_name
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model)
    

    def save_model(self):
        """Locally saves the weights of the previously loaded models weights"""


class WhisperZeroShot(WhisperBase):
    """Whisper-based model to use for zero-shot evaluations"""

    def __init__(self, config: DictConfig):
        WhisperBase.__init__(self, config=config)

    
    @torch.no_grad()
    def evaluate(self):
        """Run zero-shot evaluation on the base model for batch testing"""
        return
    

    def inference(self):
        """For running single example inference tests"""
        return
    

class WhisperLoRA(WhisperZeroShot):
    """Whisper-based model with an added low-rank adaptation for domain fine-tuning"""

    def __init__(self, config: DictConfig):
        WhisperZeroShot.__init__(self, config=config)

    #TODO: Add additional class functions necessary for fine-tuning using standard LoRA.

    

class WhisperAdaLoRA(WhisperLoRA):
    """Whisper-based model with an the PEFT method, AdaLoRA, for domain fine-tuning"""

    def __init__(self, config: DictConfig):
        WhisperLoRA.__init__(self, config=config)

    #TODO: Add the class functions necessary to fine-tune using the AdaLoRA method, but OMIT ANY REPEATING FUNCTIONS INHERITED FROM WhisperLoRA class!

