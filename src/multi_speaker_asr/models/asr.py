from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration, Pipeline
import torch

class WhisperPipeline:
    """
    A wrapper class for the ASR models, including Whisper-based models both base and finetuned versions from Huggingface.
    """
    MODELS_DICT = {
        "test": "openai/whisper-tiny", # for initial testing purposes
        "base": "CoRal-project/roest-v3-whisper-1.5b",
        "lora": "", # standard lora finetuning of base model for child-domain
    }

    def __init__(self, model: str = "test", device: str = "cpu"):
        if model not in self.MODELS_DICT:
            raise ValueError(f"Unknown model, {model}, passed.")
        
        self.model_id = self.MODELS_DICT[model]

        if isinstance(device, torch.device):
            self.device = device
        elif isinstance(device, str):
            self.device = torch.device(device)

        self.pipeline = self.load_pipeline()
    

    def load_pipeline(self):
        """
        Loads the pipeline, specified by the passed model selection, from Huggingface transformers.
        """
        processor = WhisperProcessor.from_pretrained(self.model_id)
        model = WhisperForConditionalGeneration.from_pretrained(self.model_id).to(self.device)

        result = pipeline(
            "automatic-speech-recognition",
            model=model,
            chunk_length_s=30,
            device=self.device,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor
        )
        return result

