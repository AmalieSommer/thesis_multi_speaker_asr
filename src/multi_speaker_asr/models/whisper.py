from omegaconf import DictConfig
import torch.nn as nn
import torch
import transformers
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq


import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class WhisperBase(nn.Module):
    """Base pretrained Whisper-based model loaded from Huggingface."""
    
    def __init__(self, model_name):
        """Initialize the model with the appropriate config file"""
        super().__init__()

        #self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        
        self.load_processor()
        self.load_model()
    
    
    def load_processor(self):
        """Load the pretrained processor into the AutoProcessor"""
        self.processor = AutoProcessor.from_pretrained(
            self.model_name
            )

    def load_model(self):
        """Load the pretrained STT model from Huggingface"""
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_name
            ).to(self.device)


    def generate(self, input_features, attention_mask):
        return self.model.generate(
                input_features=input_features,
                attention_mask=attention_mask,
                task="transcribe",
                language="da",
                )


    def forward(self, input_features, labels=None):
        """Defines the models forward pass"""
        outputs = self.model(
            input_features=input_features, 
            labels=labels,
            output_hidden_states=True # to ensure it also works with computing EmbER metric
            )
        
        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
            "encoder_last_hidden_state": outputs.encoder_last_hidden_state,
            "decoder_last_hidden_state": outputs.decoder_hidden_states[-1]
        }
    


class WhisperLoRA(WhisperBase):
    """Whisper-based model with an added low-rank adaptation for domain fine-tuning"""

    def __init__(self, config: DictConfig):
        WhisperBase.__init__(self, config=config)

    #TODO: Add additional class functions necessary for fine-tuning using standard LoRA.

    

class WhisperAdaLoRA(WhisperLoRA):
    """Whisper-based model with an the PEFT method, AdaLoRA, for domain fine-tuning"""

    def __init__(self, config: DictConfig):
        WhisperLoRA.__init__(self, config=config)

    #TODO: Add the class functions necessary to fine-tune using the AdaLoRA method, but OMIT ANY REPEATING FUNCTIONS INHERITED FROM WhisperLoRA class!

