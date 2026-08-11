from torch.nn.functional import cosine_similarity
import re
from num2words import num2words
from jiwer import wer, cer
import json
import platform
import torch
from optimum.onnxruntime.configuration import (
    AutoQuantizationConfig,
    QuantizationConfig,
)



def get_config_type(quant_config: dict) -> (AutoQuantizationConfig | QuantizationConfig):
    """
    Reads what type of CPU instruction set extensions are supported by the current computer hardware,
    and based on that information decides what specific CPU vector instructions to use for the quantization configuration.

    Args:
        quant_config (dict): An object of valid configuration parameters to use when creating the QuantizationConfig object.
    Returns:
        QuantizationConfig: The type of quantization configuration with the correctly configured vector instruction for what is compatible with current hardware.
    """
    architecture = platform.machine().lower()

    # For ARM Architecture
    if 'arm' in architecture or 'aarch64' in architecture:
        return AutoQuantizationConfig.arm64(**quant_config)

    # For IBM PowerPC 64-bit Little Endian
    if "ppc64le" in architecture or "powerpc" in architecture:
        return AutoQuantizationConfig.ppc64le(**quant_config)

    # For x86-64 Architecture
    capabilities = torch.backends.cpu.get_cpu_capability()
    match capabilities:
        case 'AVX512':
            return AutoQuantizationConfig.avx512(**quant_config)
        case 'AVX2':
            return AutoQuantizationConfig.avx2(**quant_config)
        case _:
            # DEFAULT architecture detected...
            return QuantizationConfig(**quant_config)



def normalize(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]','', text)
    return text


def compute_cosine_sim(pred_embeddings, target_embeddings):
    sim = cosine_similarity(pred_embeddings, target_embeddings).item()
    return 1 - sim


def clean_transcription(sentence: str):
    """
    Function to preprocess the ground truth and predicted transcripts before computing the performance using WER, CER etc...
    Should standardize the text to lowercase, no punctuations or special characters.
    It should also map all occurrences of numbers to textual representations using library function.
    """
    sentence = str.lower(sentence)
    sentence = re.sub(r'-(?!\d)', '', sentence)             # Remove - that are not followed by a number
    sentence = re.sub(r'(?<!\d)\.|\.?(?!\d)', '', sentence) # Remove . that are not enclosed by two numbers
    sentence = re.sub(r'[^\w\s.-]', '', sentence)           # Remove all punctuation except for the - and .
    sentence = re.sub(' +', ' ', sentence)                  # Replacing all duplicate spaces with single space.
    
    sentence_copy = str(sentence)

    for s in sentence.split():
        try: 
            num = float(s)
            word_rep = str(num2words(number=num))
            sentence_copy = sentence_copy.replace(s, word_rep)
        except ValueError as e:
            continue

    return sentence_copy


def save_asr_results(asr_output, asr_metadata, output_file):
    with open(output_file, 'a') as writer:
        results = []
        for data in asr_metadata:
            ref_text = data['text']
            transcripts = [asr_output[batch_info['ref_indices']] for batch_info in data['audio_batch_info']]
            for transcript in transcripts:

                for seg in transcript:
                    wer_ = wer(reference=clean_transcription(ref_text), hypothesis=clean_transcription(seg['text']))
                    cer_ = cer(reference=clean_transcription(ref_text), hypothesis=clean_transcription(seg['text']))

                    results.append({
                        'audio_id': data['audio_id'],
                        'segment_id': data['segment_id'],
                        'ref': ref_text,
                        'seg_start': data['start'],
                        'seg_end': data['end'],
                        'wer': wer_,
                        'cer': cer_,      
                        'hyp': seg['text']
                    })
        writer.write(json.dumps(results) + '\n')
        writer.flush()

