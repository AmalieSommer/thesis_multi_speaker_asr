from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from transformers import pipeline, AutoModelForSpeechSeq2Seq, Wav2Vec2Processor, AutoModelForCTC, AutoProcessor, WhisperForConditionalGeneration, BitsAndBytesConfig
from faster_whisper import WhisperModel
import numpy as np
import torch

class BaseEngine:
    def __init__(self, model_path: str, device: str = 'cpu', sr: int = 16000):
        self.model_path = model_path
        self.device = device
        self.sr = sr

        self.processor = self.load_processor()
        self.model = self.load_model()


    def load_processor(self):
        return AutoProcessor.from_pretrained(self.model_path)


    def load_model(self):
        return AutoModelForSpeechSeq2Seq.from_pretrained(self.model_path)


    def transcribe(self, audio):
        if not isinstance(audio, np.array):
            raise TypeError('Audio must be a numpy array')
        
        # Run audio through processor to get features and generate token ids
        input_features = self.processor(audio, sampling_rate=self.sr, return_tensors='pt').input_features
        pred_ids = self.model.generate(input_features)

        # Decode token ids back to text
        transcription = self.processor.batch_decode(pred_ids, skip_special_tokens=True)
        return transcription
    

class PytorchEngine(BaseEngine):
    def __init__(self, model_path, device = 'cpu', model_type: str = 'whisper'):
        self.model_type = model_type
        super().__init__(model_path, device)

    def load_model(self, quantized: bool = False):
        if self.model_type == 'whisper':
            quantization_config = BitsAndBytesConfig(load_in_8bit=quantized)
            return WhisperForConditionalGeneration.from_pretrained(
                pretrained_model_name_or_path=self.model_path,
                quantization_config=quantization_config
                )
        elif self.model_type == 'wav2vec2':
            return AutoModelForCTC.from_pretrained("CoRal-project/roest-v3-wav2vec2-315m")
        
    def load_processor(self):
        if self.model_type == 'wav2vec2':
            return Wav2Vec2Processor.from_pretrained("CoRal-project/roest-v3-wav2vec2-315m")
        else: 
            return super().load_processor() 

    def transcribe(self, audio, language='da'):
        if self.model_type == 'whisper':
            # Run audio through processor to get features and generate token ids
            input_features = self.processor(audio, sampling_rate=self.sr, return_tensors='pt', padding=True).input_features
            pred_ids = self.model.generate(input_features)

            # Decode token ids back to text
            transcription = self.processor.batch_decode(pred_ids, skip_special_tokens=True)
        elif self.model_type == 'wav2vec2':
            # Run audio through processor to get features and generate token ids
            input_features = self.processor(audio, sampling_rate=self.sr, return_tensors='pt', padding=True)
            with torch.no_grad():
                logits = self.model(input_features.input_values).logits
            pred_ids = torch.argmax(logits, dim=-1)

            # Decode token ids back to text
            transcription = self.processor.batch_decode(pred_ids)
        else:
            raise ValueError('Unknown model type...')
        
        return transcription


class OnnxEngine(BaseEngine):
    def __init__(self, model_path, device = 'cpu'):
        super().__init__(model_path, device)

    def load_model(self):
        return ORTModelForSpeechSeq2Seq.from_pretrained(model_id=self.model_path)


class CT2(BaseEngine):
    def __init__(self, model_path, device = 'cpu', compute_type: str = 'fp32', cpu_threads: int = 6):
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        super().__init__(model_path, device)

    def load_model(self):
        self.model = WhisperModel(
            model_size_or_path=self.model_path, 
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=self.cpu_threads
        )

    def transcribe(self, audio, language: str = 'da'):
        if not isinstance(audio, list):
            audio = [audio]

        outputs = []
        for i, sample in enumerate(audio):
            segments, _ = self.pipeline.transcribe(
                audio=sample,
                language=language,
                log_progress=True
            )
            outputs.append([{'start': seg.start, 'end': seg.end, 'text': seg.text} for seg in segments])

        return outputs

