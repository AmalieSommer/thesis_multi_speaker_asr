from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from transformers import (
    pipeline, 
    AutoModelForSpeechSeq2Seq, 
    Wav2Vec2Processor, 
    AutoModelForCTC, 
    AutoProcessor, 
    WhisperForConditionalGeneration, 
    WhisperProcessor, 
    WhisperConfig, 
    AutoConfig
    )
import os
from pyctcdecode import build_ctcdecoder
from optimum.quanto import requantize
from safetensors.torch import save_file, load_file
from faster_whisper import WhisperModel
import numpy as np
import torch
import json
from ..utils.utils import profile, LOGGING_CONFIG
import logging
import logging.config
from pathlib import Path

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(name='Engine')

class BaseEngine:
    def __init__(self, model_path: str, device: str = 'cpu', sr: int = 16000, language: str = 'da', task: str = 'transcribe'):
        self.model_path = model_path
        self.device = device
        self.sr = sr
        self.language = language
        self.task = task

        self.processor = self.load_processor()
        processor_memory = self.load_processor.memory_stats[0]
        logger.info('Type: %s. Memory Stats of Processor...: Before load: %f, After load: %f, Delta: %f', self.model_type, processor_memory['before'], processor_memory['after'], processor_memory['delta'])

        self.model = self.load_model()
        model_memory = self.load_model.memory_stats[0]
        logger.info('Type: %s. Memory Stats of Model...: Before load: %f, After load: %f, Delta: %f', self.model_type, model_memory['before'], model_memory['after'], model_memory['delta'])

    @profile
    def load_processor(self):
        return AutoProcessor.from_pretrained(self.model_path)

    @profile
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
    def __init__(
            self, 
            model_path, 
            device = 'cpu', 
            model_type: str = 'whisper', 
            use_saved_model: bool = False,
            local_models_dir: str = None,
            cpu_threads: int = 4,
            compute_type: str = 'float32'
            ):
        torch.set_num_threads(cpu_threads) # To limit amount of context switching...
        self.model_type = model_type
        self.compute_type = compute_type
        self.use_saved_model = use_saved_model
        self.local_models_dir = local_models_dir
        super().__init__(model_path, device)


    @profile
    def load_model(self):
        if self.use_saved_model:
            if self.local_models_dir is None:
                raise ValueError('Missing path to local model...')
            
            if self.model_type == 'whisper':
                config = WhisperConfig.from_pretrained(self.local_models_dir)
                model = WhisperForConditionalGeneration(config=config)
            elif self.model_type == 'wav2vec2':
                config = AutoConfig.from_pretrained(self.local_models_dir)
                model = AutoModelForCTC.from_config(config=config)

            dir_path = os.path.join(self.local_models_dir,self.compute_type)
            state_dict = load_file(os.path.join(dir_path, 'model.safetensors'))
            with open(os.path.join(dir_path, 'quantization_map.json'), 'r') as f:
                quantization_map = json.load(f)
            requantize(model=model, state_dict=state_dict, quantization_map=quantization_map, device=self.device)
        else:
            fun_args = {'pretrained_model_name_or_path': self.model_path, 'device_map': self.device}
            # Else load the standard model:          
            if self.model_type == 'whisper':
                model = WhisperForConditionalGeneration.from_pretrained(**fun_args)
            elif self.model_type == 'wav2vec2':
                model = AutoModelForCTC.from_pretrained(**fun_args)

        return model
    
        
    @profile
    def load_processor(self):
        if self.use_saved_model:
            if self.local_models_dir is None:
                raise ValueError('Missing path to local processor...')
            path = self.local_models_dir
        else:
            path = self.model_path

        if self.model_type == 'whisper':
            return WhisperProcessor.from_pretrained(path)
        elif self.model_type == 'wav2vec2':
            processor = Wav2Vec2Processor.from_pretrained(path)
            print(processor.tokenizer.get_vocab())
            vocab = processor.tokenizer.get_vocab()
            sorted_vocab = [k for k, _ in sorted(vocab.items(), key=lambda x: x[1])]
            self.ctc_decoder = build_ctcdecoder(
                labels=sorted_vocab
            )
            return processor
        else: 
            return super().load_processor()
            


    def transcribe(self, audio, language='da'):
        if self.model_type == 'whisper':
            # Run audio through processor to get features and generate token ids
            inputs = self.processor(audio, sampling_rate=self.sr, return_tensors='pt', return_attention_mask=True)
            with torch.no_grad():
                pred_ids = self.model.generate(
                    inputs.input_features, 
                    attention_mask=inputs.attention_mask,
                    task='transcribe',
                    language=language
                    )
            # Decode token ids back to text
            transcription = self.processor.batch_decode(pred_ids, skip_special_tokens=True)
        elif self.model_type == 'wav2vec2':
            # Run audio through processor to get features and generate token ids
            input_features = self.processor(audio, sampling_rate=self.sr, return_tensors='pt', padding=True)
            with torch.no_grad():
                logits = self.model(input_features.input_values).logits
            texts = []
            for logit in logits:
                logit = logit.cpu().numpy()
                text = self.ctc_decoder.decode(logits=logit)
                texts.append(text)
            transcription = texts
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

