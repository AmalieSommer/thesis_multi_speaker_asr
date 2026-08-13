import optimum.onnxruntime as opt_onnx
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq, ORTModelForCTC, ORTOptimizer, ORTQuantizer
from transformers import (
    pipeline, 
    AutoModelForSpeechSeq2Seq, 
    AutoModelForCTC, 
    AutoProcessor, 
    WhisperProcessor,
    AutoTokenizer,
    AutoFeatureExtractor,
    Wav2Vec2ProcessorWithLM,
    AutomaticSpeechRecognitionPipeline
    )
from datasets import Dataset
from multi_speaker_asr.data import AudioDataset
import ctranslate2.converters.transformers as ct2_transformers
from optimum.exporters.tasks import TasksManager
from optimum.onnxruntime.configuration import OptimizationConfig, AutoQuantizationConfig, AutoCalibrationConfig
from huggingface_hub import hf_hub_download, list_repo_files
from faster_whisper.tokenizer import Tokenizer
import os
from huggingface_hub import repo_exists
from ctranslate2.models import Wav2Vec2
from pyctcdecode import build_ctcdecoder
from faster_whisper import WhisperModel
from faster_whisper.audio import pad_or_trim
import numpy as np
import torch
from multi_speaker_asr.utils.logging_config import get_logger
from pathlib import Path
from pywhispercpp.model import Model
import onnxruntime as ort
from multi_speaker_asr.utils.utils import get_config_type


HF_TOKEN = os.getenv('HF_TOKEN')
log = get_logger(__name__)


class BaseEngine:
    def __init__(
            self, 
            model_path: str, 
            model_type: str,
            model_name: str,
            device: str = 'cpu', 
            sr: int = 16000, 
            language: str = 'da', 
            task: str = 'automatic-speech-recognition',
            compute_type: str = 'int8', 
            cpu_threads: int = 6
            ):

        self.model_name = model_name
        self.model_type = model_type
        self.device = device
        self.sr = sr
        self.language = language
        self.task = task
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        if not isinstance(model_path, str) or not model_path.strip():
            raise ValueError('Parameter model_path must be a non-empty string.')
        self.model_path = model_path
       

        self.processor = self.load_processor()
        self.model = self.load_model()

    def load_processor(self) -> AutoProcessor:
        return AutoProcessor.from_pretrained(self.model_path)


    def load_model(self) -> AutoModelForSpeechSeq2Seq:
        return AutoModelForSpeechSeq2Seq.from_pretrained(self.model_path)


    def save_model(self, output_path: str) -> None:
        """
        Save a model locally on disk at the specified output_path.
        
        Args:
            output_path (str): The local path to store the model
        """
        write_path = Path(output_path)
        if os.path.isfile(write_path) and os.path.exists(write_path):
            raise ValueError('Cannot override existing filepath.')
        
        self.model.save_pretrained(output_path)
        self.processor.save_pretrained(output_path)


    def transcribe(self, audio, return_timestamps: bool):
        if not isinstance(audio, np.array):
            raise TypeError('Audio must be a numpy array')

        with torch.no_grad():
            # Run audio through processor to get features and generate token ids
            input_features = self.processor(audio, sampling_rate=self.sr, return_tensors='pt').input_features
            pred_ids = self.model.generate(input_features)

            # Decode token ids back to text
            output = self.processor.batch_decode(pred_ids, skip_special_tokens=True)

        if not return_timestamps:
            return output
        
        transcription = []
        for i, segment in enumerate(output):
            end = len(audio[i]) / self.sr
            transcription.append([{
                'start': item['timestamp'][0],
                'end': end if item['timestamp'][1] == None else item['timestamp'][1],
                'text': item['text']
            } for item in segment['words']])
        
        log.debug('Model output: %s', transcription)
        return transcription


    def format_output(self, output: list, audio: list, return_timestamps: bool | str) -> list:
        if output is None:
            raise ValueError('Prediction is None')
        if audio is None:
            raise ValueError('Audio is None')

        if not return_timestamps:
            log.debug('MODEL TYPE (%s).... Model output: %s', self.model_type, output)
            return [{'text': out['text']} for out in output]

        transcription = []
        for i, segment in enumerate(output):
            end = len(audio[i]) / self.sr
            transcription.append({
                'text': segment.get('text', ''),
                'words': [
                    {
                        'start': item['timestamp'][0],
                        'end': end if item['timestamp'][1] is None else item['timestamp'][1],
                        'word': item['text']
                    } for item in segment['chunks']
                ]
            })
        
        #log.debug('MODEL TYPE (%s).... Model output: %s', self.model_type, transcription)
        return transcription
        


    def _get_model_path(self):
        return self.model_path


    def has_language_model(self) -> bool:
        """
        Helper function to determine whether to load the default processor with an LM or a custom one without the
        LM, but with a standard CTC BeamSearch Decoder.

        Returns:
            bool: An indicator for whether a CTC model has an LM
        """
        required_files = {
            'language_model',
            'language_model.bin',
            'language_model.arpa'
        }
        filepath = Path(self.model_path)
        if filepath.exists():
            has_lm = any((filepath / f).exists() for f in required_files)
        else:
            hf_files = list_repo_files(self.model_path)
            has_lm = any('language_model' in f for f in hf_files)
        return True # Temporary...
        return has_lm


    def validate_filepath(self) -> str:
        """
        Given a filepath it will check whether the filepath is valid to either a local model or to a remote directory
        on Huggingface containing a valid model.
        """
        filepath = self.model_path
        if filepath == None:
            raise ValueError('Filepath is None.')
        if not isinstance(filepath, str):
            raise ValueError('Filepath must be a string.')
        filepath = filepath.strip()
        if len(filepath) < 1:
            raise ValueError('Filepath is empty.')
        
        path = Path(filepath)
        log.debug('Validating filepath: %s', filepath)
        if path.exists():
            log.debug('Path exists!!!')
            return 'local'

        # Next check if the filepath is to a Huggingface model directory:
        if repo_exists(repo_id=filepath):
            return 'hub'

        # Finally, raise error if no file was found at either location
        raise FileNotFoundError('The file %s was not found either locally or on Huggingface', filepath)
        

class PytorchEngine(BaseEngine):
    def __init__(self, config: dict):
        super().__init__(**config)


    def load_model(self) -> AutomaticSpeechRecognitionPipeline:
        model = None
        match self.model_type:
            case 'seq2seq':
                model = AutoModelForSpeechSeq2Seq.from_pretrained(pretrained_model_name_or_path=self.model_path)
            case 'ctc':
                log.debug('Loading model: %s', self.model_type)
                model = AutoModelForCTC.from_pretrained(pretrained_model_name_or_path=self.model_path)
            case _:
                raise ValueError('Parameter: model_type value is unknown. Pass in either ctc or seq2seq.')
            
        return  pipeline(
                    task=self.task,
                    model=model,
                    language=self.language,
                    dtype=torch.float32,
                    tokenizer=self.processor.tokenizer,
                    feature_extractor=self.processor.feature_extractor
                )
    
        
    def load_processor(self) -> (AutoProcessor | Wav2Vec2ProcessorWithLM):
        match self.model_type:
            case 'seq2seq':
                return AutoProcessor.from_pretrained(pretrained_model_name_or_path=self.model_path)
            case 'ctc':
                if not self.has_language_model():
                    return AutoProcessor.from_pretrained(self.model_path)

                tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=self.model_path)
                feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_model_name_or_path=self.model_path)
                vocab = tokenizer.get_vocab()
                sorted_vocab = sorted((id, token) for token, id in vocab.items())
                labels = [token for _, token in sorted_vocab]
                decoder = build_ctcdecoder(labels)

                return Wav2Vec2ProcessorWithLM(
                    feature_extractor=feature_extractor,
                    tokenizer=tokenizer,
                    decoder=decoder
                )
            

    def transcribe(self, audio, return_timestamps: bool | str) -> list[list[str | dict]]:
        """
        The function call for transcribing a list of audio array(s). Depending on the model type, i.e. CTC (e.g. Wav2Vec2) or Seq2Seq (e.g. Whisper),
        it can also predict timestamps either at the word-level for CTC-based models or at the segment level for Seq2Seq-based models.

        Args:
            audio (list): A list of np.ndarrays representing the loaded audio files
            return_timestamps (bool | str): A value to determine whether to predict timestamps and what type to predict given the model. For transformer models, set to True, and for phoneme models set to \'word\' or \'char\'.
        Returns:
            list[list[str | dict]]: It returns a list of lists, where each sublist correspond to the predictions for a single of the audio arrays passed to the function.
        """
        if self.model_type not in ("seq2seq", "ctc"):
            raise ValueError("Unknown model type...")
        
        with torch.no_grad():
            output = self.model(audio, return_timestamps=return_timestamps)
        return self.format_output(output=output, audio=audio, return_timestamps=return_timestamps)


class OnnxEngine(BaseEngine):
    def __init__(self, config: dict):
        super().__init__(**config)


    def is_exported(self) -> bool:
        """
        Will check if the model directory contains .onnx files or not, to determine whether to include export=True
        in the .from_pretrained() call.
        """
        path_location = self.validate_filepath()
        log.debug('Validated filepath returned: %s', path_location)
        if path_location == 'local':
            path = Path(self.model_path)
            if any(item for item in path.glob('*.onnx')):
                log.debug('An onnx file was found...')
                return True

        elif path_location == 'hub':
            if any(suffix == '.onnx' for suffix in list_repo_files(repo_id=self.model_path)):
                return True
        return False
                    

    def load_model(self) -> AutomaticSpeechRecognitionPipeline:
        supported_models = list(TasksManager.get_supported_model_type_for_task(task=self.task, exporter='onnx'))
        if self.model_name in supported_models:

            # Check whether the model_path is valid and if it leads to a local or a remote model repo:
            export = False if self.is_exported() else True
            log.debug('Export: %s', export)
            model = None
            input_name_attr = None
            match self.model_type:
                case 'seq2seq':
                    model = ORTModelForSpeechSeq2Seq.from_pretrained(self.model_path, export=export)
                    input_name_attr = 'input_features'
                case 'ctc':
                    model = ORTModelForCTC.from_pretrained(self.model_path, export=export)
                    input_name_attr = 'input_values'
                case _:
                    raise ValueError('Parameter: model_type value is unknown. Pass in either ctc or seq2seq.')

            if not hasattr(model, 'main_input_name'):
                model.main_input_name = getattr(model.config, 'main_input_name', input_name_attr)
            return  opt_onnx.pipeline(
                        task=self.task,
                        model=model,
                        language=self.language,
                        dtype=torch.float32,
                        tokenizer=self.processor.tokenizer,
                        feature_extractor=self.processor.feature_extractor
                    )
        else:
            # The model with model_name is not supported by transformer.onnx api for model export...
            raise ValueError('The model %s is not supported for the task %s. Please select a different task or different model.', self.model_name, self.task)

        

    def load_processor(self) -> (AutoProcessor | Wav2Vec2ProcessorWithLM):
        match self.model_type:
            case 'seq2seq':
                return AutoProcessor.from_pretrained(pretrained_model_name_or_path=self.model_path)
            case 'ctc':
                if not self.has_language_model(): # TODO CHANGE IT FROM DETECTING WHETHER AN LM IS PRESENT TO A FLAG INSTEAD!
                    return AutoProcessor.from_pretrained(self.model_path)

                tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=self.model_path)
                feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_model_name_or_path=self.model_path)
                vocab = tokenizer.get_vocab()
                sorted_vocab = sorted((id, token) for token, id in vocab.items())
                labels = [token for _, token in sorted_vocab]
                decoder = build_ctcdecoder(labels)

                return Wav2Vec2ProcessorWithLM(
                    feature_extractor=feature_extractor,
                    tokenizer=tokenizer,
                    decoder=decoder
                )


    def apply_optimizations(self, optimizations_config: dict, output_path: str) -> str:
        """
        Uses Optimum OnnxRuntime optimization API to apply a series of selected optimization strategies to the currently loaded model.
        It saves the optimized model to the chosen directory. If one wants to use the optimized model afterwards, simply pass the output_path 
        into the .from_pretrained() function.

        Args:
            optimizations_config (dict): A dictionary of valid optimization parameters for the OptimizationConfig object.
            output_path (str): A valid string representation for a path to a local directory for saving the optimized model

        Returns:
            str: The path to the local directory storing the optimized model
        """
        if self.model is None:
            raise ValueError('Failed because model is None')
        if optimizations_config is None or output_path is None:
            raise ValueError('Parameter cannot be None.')
        if not isinstance(optimizations_config, dict) or len(optimizations_config.keys()) == 0:
            raise TypeError('Parameter quant_config must be a non-empty dictionary.')
        if not isinstance(output_path, str) or len(output_path.strip()) == 0:
            raise TypeError('Parameter output_path must be a non-empty string.')
        try: 
            optimizer = ORTOptimizer.from_pretrained(model_or_path=self.model.model)
        except NotImplementedError as e:
            log.info(e)

            # Optimization not supported by Optimum... Applying optimization using OnnxRuntime
            exported_onnx_path = Path(self.model.model.model_save_dir) / "model.onnx"
            session_options = ort.SessionOptions()
            optimization_level = optimizations_config.get('optimization_level')
            if optimization_level == 1:
                session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            elif optimization_level == 2:
                session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

            opt_filepath = Path(output_path) / 'model.onnx'
            session_options.optimized_model_filepath = str(opt_filepath)
            _ = ort.InferenceSession(path_or_bytes=str(exported_onnx_path), sess_options=session_options)
            return Path(os.path.dirname(opt_filepath))

        configuration = OptimizationConfig(**optimizations_config)
        return optimizer.optimize(save_dir=output_path, optimization_config=configuration)


    def apply_quantization(
            self, 
            quantization_type: str, 
            quant_config: dict, 
            output_path: str, 
            calibration_data_config: dict = None,
            calibration_num_samples: int = 50
            ) -> Path:
        """
        Uses Optimum[OnnxRuntime] API for applying either dynamic or static quantization.
        It will apply quantization according to the valid parameters specified in the quantization configuration.

        A note from the library is that only dynamic quantization is supported for Seq2Seq models.

        Args:
            quantization_type (str): Should be either \'dynamic\' or \'static\' to represent the type of quantization to perform
            quant_config (dict): A dictionary object containing the valid configuration parameters to apply
            output_path (str): Path to the directory for saving the model

        Returns:
            Path: The path of the resulting quantized model.
        """
        # ------------- Valid input check -------------
        if quantization_type is None or quant_config is None or output_path is None:
            raise ValueError('Parameter cannot be None.')
        if not isinstance(quantization_type, str) or len(quantization_type.strip()) == 0:
            raise TypeError('Parameter quantization_type must be a non-empty string.')
        if not isinstance(quant_config, dict) or len(quant_config.keys()) == 0:
            raise TypeError('Parameter quant_config must be a non-empty dictionary.')
        if not isinstance(output_path, str) or len(output_path.strip()) == 0:
            raise TypeError('Parameter output_path must be a non-empty string.')

        if self.model is None:
            raise ValueError('Failed because model is currently None')
        
        # Define the quantization strategy:
        match quantization_type:
            case 'dynamic':
                # Apply dynamic quantization:
                dynamic_quant_config = get_config_type(quant_config=quant_config)
                if self.model_type == 'seq2seq':
                    # Iterate over each .onnx file and apply the quantization strategy
                    model_dir = self.model.model.model_save_dir
                    onnx_models = list(Path(model_dir).glob(pattern='*.onnx'))
                    # Create a quantizer for each onnx file
                    quantizers = [
                        ORTQuantizer.from_pretrained(model_or_path=model_dir, file_name=onnx_model.name)
                        for onnx_model in onnx_models
                    ]
                    # Apply the quantization config to each onnx model
                    quant_modelpath = Path('')
                    for quant in quantizers:
                        quant_modelpath = quant.quantize(
                            quantization_config=dynamic_quant_config,
                            save_dir=output_path
                        )
                    return quant_modelpath
                else:
                    # Quantize just the single .onnx file
                    quantizer = ORTQuantizer.from_pretrained(model_or_path=self.model.model)
                    return quantizer.quantize(
                        quantization_config=dynamic_quant_config,
                        save_dir=output_path
                    )
            
            case 'static':
                static_quant_config = get_config_type(quant_config=quant_config)

                if self.model_type == 'seq2seq':
                    raise TypeError('Optimum-ONNX currently do not provide static quantization support for Seq2Seq models.')
                if calibration_data_config is None:
                    raise ValueError('Parameter calibration_data is None')
                
                quantizer = ORTQuantizer.from_pretrained(model_or_path=self.model.model)
                calib_ds = Dataset.from_list(self.generate_calibration_dataset(ds_config=calibration_data_config, num_samples=calibration_num_samples))
                calibration_config = AutoCalibrationConfig.minmax(calib_ds)
                ranges = quantizer.fit(
                    dataset=calib_ds,
                    calibration_config=calibration_config,
                    operators_to_quantize=static_quant_config.operators_to_quantize
                )
                return quantizer.quantize(
                    save_dir=output_path,
                    calibration_tensors_range=ranges,
                    quantization_config=static_quant_config
                )
            case _:
                raise ValueError('The quantization_type must be either \'dynamic\' or \'static\', but found %s', quantization_type)
            

    def generate_calibration_dataset(self, ds_config: dict, num_samples: int = 50) -> list[dict]:
        """
        A custom function for generating the calibration dataset to be used for Post-Training Quantization (PTQ).

        Args:
            ds_config (dict): A dictionary with configuration parameters to initialize the dataset
            num_samples (int): The number of samples to include in the calibration dataset
        
        Returns:
            list: A list of audio numpy arrays
        """
        ds = AudioDataset(data_config=ds_config)

        count = 0
        result = []
        for sample in ds:
            if count >= num_samples:
                return result

            result.append({
                'input_values': sample['audio']
            })
            count += 1
        return result
        

    def transcribe(self, audio, return_timestamps: bool | str) -> list[list[str | dict]]:
        """
        Engine function for transcribing a single audio file or a batch of audio files.

        Args:
            audio (list): A list of either a single audio array or multiple audio arrays.
            return_timestamps (bool | str): Depending on whether the model is e.g. Whisper, it should be given a value of True, or if the model is e.g. Wav2Vec2, it should have the value \'word\' or \'char\'.

        Returns:
            list[list[str | dict]]: It returns a list of lists, where each sublist correspond to the predictions for a single of the audio arrays passed to the function.
        """
        if self.model_type not in ("seq2seq", "ctc"):
            raise ValueError("Unknown model type...")

        with torch.no_grad():
            output = self.model(audio, return_timestamps=return_timestamps)
        return self.format_output(output=output, audio=audio, return_timestamps=return_timestamps)
        


class CT2(BaseEngine):
    def __init__(self, config: dict):
        super().__init__(**config)

        supported_architectures = sorted(list(ct2_transformers._SUPPORTED_MODELS.keys()))
        if config['model_name'] not in supported_architectures:
            raise ValueError('The model: %s is not currently supported by the CTranslate2 library', config['model_name'])


        if not isinstance(config['model_path'], str) or not config['model_path'].strip():
            raise ValueError('Parameter model_path must be a non-empty string.')

        if (not repo_exists(repo_id=config['model_path'])) and (not os.path.exists(path=config['model_path'])):
            raise FileNotFoundError('The model_path %s was not found on Huggingface or locally. Please verify that the model exists either locally or remotely.') 

        self.tokenizer = self.load_tokenizer()

        

    def load_model(self) -> (WhisperModel | Wav2Vec2):
        if str.lower(self.model_name) == 'whisper':
            return WhisperModel(
                model_size_or_path=self.model_path, 
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads
            )
        
        if str.lower(self.model_name) == 'wav2vec2':
            return Wav2Vec2(self.model_path)

        # TODO: Add support for additional ASR models
        raise ValueError('Currently only support CTranslate2 engines for Whisper and Wav2Vec2 models.')


    def load_processor(self):
        processor = WhisperProcessor.from_pretrained(self.model_path)
        return processor


    def load_tokenizer(self):
        tokenizer = Tokenizer(
            self.model.hf_tokenizer,
            self.model.model.is_multilingual,
            task=self.task,
            language=self.language,
        )
        return tokenizer


    def timestamps_helper(
            self,
            tokenizer: Tokenizer,
            tokens: list[int],
            time_precision: float = 0.02
    ):
        segments = []
        current_start = None
        text_tokens = []

        for token in tokens:
            if token >= tokenizer.timestamp_begin:
                timestamp = (token - tokenizer.timestamp_begin) * time_precision

                if current_start is None:
                    current_start = timestamp
                else:
                    if text_tokens:
                        segments.append(
                            {
                                "start": current_start,
                                "end": timestamp,
                                "text": tokenizer.decode(text_tokens).strip(),
                            }
                        )
                    current_start = timestamp
                    text_tokens = []
            else:
                text_tokens.append(token)

        return segments


    def transcribe(self, audio_batch, return_timestamps: bool, chunk_length: int = 30):
        if not isinstance(audio_batch, (list, np.ndarray)):
            audio_batch = [audio_batch]


        features_list = []
        previous_tokens = []
        for audio in audio_batch:
            feature = self.model.feature_extractor(audio, chunk_length=chunk_length)
            features_list.append(pad_or_trim(feature))
        batched_features = np.stack(features_list)

        
        
        prompt = self.model.get_prompt(
            tokenizer=self.tokenizer,
            previous_tokens=previous_tokens,
            without_timestamps=True
        )
        prompts = [prompt] * len(audio_batch)
        
        
        encoder_output = self.model.encode(batched_features)
        results = self.model.model.generate(
            encoder_output,
            prompts
        )
        
        transcriptions = []
        for result in results:
            tokens = result.sequences_ids[0]

            if return_timestamps:
                segments = self.timestamps_helper(tokenizer=self.tokenizer, tokens=tokens)
                transcriptions.append({
                    'text': ''.join([item['text'] for item in segments]),
                    'segments': segments
                })
            else:
                text = self.tokenizer.decode(tokens).strip()
                transcriptions.append(text)
        return transcriptions


class WhisperCPP(BaseEngine):
    def __init__(self, config: dict):

        if config['model_type'] != 'whisper':
            raise ValueError('Model_type must be a whisper model when using the Whisper.cpp engine...')
        
        super().__init__(**config)


    def load_model(self):
        if self.compute_type == 'fp32':
            model_name = 'ggml-roest-v3-model.bin'
        elif self.compute_type == 'int8':
            model_name = 'ggml-roest-v3-q8_0.bin'
        elif self.compute_type == 'int5':
            model_name = 'ggml-roest-v3-q5_0.bin'
        elif self.compute_type == 'int4':
            model_name = 'ggml-roest-v3-q4_0.bin'
        else:
            raise ValueError(f'Compute_type {self.compute_type} is not supported... The supported configurations are: fp32, int4, int5 and int8...')

        # Download from Huggingface Repository:
        model_path = hf_hub_download(
            repo_id='AmalieSommer/whisper.cpp-roest-v3-1.5b',
            filename=model_name,
            token=HF_TOKEN
        )

        return Model(
            model=model_path,
            n_threads=self.cpu_threads
        )

    def load_processor(self):
        pass
        
    def transcribe(self, audio):
        if not isinstance(audio, (list, tuple)):
            return self.model.transcribe(audio, extract_probability=True)

        # A batch of multiple audio files have been passed, and must be processed sequentially, because current Whisper.cpp does not support batched inference
        batch = []
        for a in audio:
            res = self.model.transcribe(
                a,
                temperature=0.0,
                print_realtime=True,
                beamsize=1,
                best_of=1,
                lang='da'
            )
            for item in res:
                batch.append({
                    'start': item.t0,
                    'end': item.t1,
                    'text': item.text,
                    'probability': item.probability
                })
        return batch