from tqdm import tqdm
from .data import AudioData, stream_audio, cast
import librosa
from torch.utils.data import DataLoader
from multi_speaker_asr.models.asr import ASR, Whisper
from .data import clean_transcription, read_bytes
from pyannote.audio.pipelines.utils.hook import ProgressHook
from multi_speaker_asr.models.diarization import Diarize
from multi_speaker_asr.models.alignment import Wav2Vec2
import torch
from multi_speaker_asr.utils.utils import LOGGING_CONFIG, profile
import gc
import time
import logging
import logging.config
from multiprocessing import Process, Queue
import json
from base64 import b64decode, b64encode
import os
import psutil


logging.config.dictConfig(LOGGING_CONFIG)
logging.getLogger("faster_whisper").setLevel(logging.DEBUG)

logger = logging.getLogger(name='Evaluate')

def collator_fn(batch):
    # TODO!!!
    """Should ensure that it returns batch object of the same format, i.e. same parameter names and types"""
    return batch


def writer(queue: Queue, output_file: str):
    with open(output_file, "w") as f:
        while True:
            item = queue.get()

            if item is None:
                break
            f.write(json.dumps(item) + "\n")
            f.flush()


def updater(queue: Queue, output_file: str):
    try:
        # First read the file into memory:
        with open(output_file, 'r') as f:
            records = [json.loads(line) for line in f]

        while True:
            item = queue.get()

            if item is None:
                break
            for record in records:
                if record[0]['id'] == item['id']:
                    record.append(item)

        with open(output_file, 'w') as f:
            for record in records:
                f.write(json.dumps(record) + '\n')
    except Exception as e:
        logger.error('Failed with error: %s', e)



def reader(queue: Queue, input_file: str):
    try:
        with open(input_file, 'r') as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                record = json.loads(line)
                queue.put(record)
    except Exception as e:
        logger.error('Process failed with error: %s', e)
    finally:
        print('Finished reading the entire file...')
        queue.put(None) # For terminating the process



def batched_inference(data: AudioData, model: ASR, asr_result_filename: str):
    """
    This assumes the dataset is pre-segmented into short chunks, and will process the chunks in batches.
    """
    results_queue = Queue()
    write_results_process = Process(target=writer, args=(results_queue, asr_result_filename))
    write_results_process.start()

    batch_size = 2
    loader = DataLoader(
        dataset=data,
        batch_size=batch_size,
        num_workers=1,
        shuffle=False,
        collate_fn=collator_fn
    )
    try: 
        
        for counter, batch in enumerate(tqdm(loader, total=(8440 // batch_size))):
            if len(batch) == 0:
                continue

            if counter == 4:
                break
            
            for sample in batch:
                # Processing time for RTF calculation per audio recording
                start_process_time = time.process_time()
                iterable_segments = []
                
                if isinstance(sample['audio'], bytes):
                    wav = read_bytes(sample['audio'])
                if isinstance(model, Whisper):
                    outputs, info = model.pipeline.transcribe(
                        audio=wav,
                        language='da'
                    )
                    
                    for out in outputs:
                        iterable_segments.append({
                            'start': out.start,
                            'end': out.end,
                            'text': out.text,
                            'avg_logprob': out.avg_logprob
                        })
                        #iterable_segments.append(cast(out))

                    end_process_time = time.process_time()
                    cpu_time = end_process_time - start_process_time
                    rtf_sample = cpu_time / info.duration # processing time divided by the actual audio duration
                
                logger.info('Batched Inference CPU Time: %f', cpu_time)
                logger.info('Batched Inference Real-Time Factor (RTF): %f', rtf_sample)
        
                item = {
                        'id': sample['id'],
                        'duration': info.duration,
                        'audio': {
                            'bytes': b64encode(sample['audio']).decode('ascii') 
                        },
                        'segments': iterable_segments
                    }
                results_queue.put(item)
    except Exception as e:
        print(f'An error occurred...{e}')
        return None
    
    finally:
        results_queue.put(None) # To signal the process to terminate upon exit.
        write_results_process.join()
        if not write_results_process.is_alive():
            write_results_process.close()

        model.unload()
        del model
        del loader
        del data
        gc.collect()

    return 'Success'



def streamed_inference(data: AudioData, model: ASR, asr_result_filename: str):
    """
    This assumes the dataset contains raw long-form audio recordings, and will therefore include streaming (using librosa) and process each audio stream sequentially.
    """
    results_queue = Queue()
    write_results_process = Process(target=writer, args=(results_queue, asr_result_filename))
    write_results_process.start()

    batch_size = 2
    loader = DataLoader(
        dataset=data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator_fn,
        num_workers=1
    )
    iterable_segments = []
    before_alignment = []
    try:
        print('Starting inference evaluation...')
        for _, batch in enumerate(loader):
            if len(batch) == 0:
                continue

            for sample in batch:
                start_process_time = time.process_time()
                """
                stream = stream_audio(
                    audio=sample['audio']
                )
                audio_duratio = sample['end']
                inner_tqdm = tqdm(stream, total=int(audio_duratio / 30.0))

                running_duration = 0.0  # To keep a count of the duration of the amount of audio streams processed so far...
                for index, y in enumerate(stream):

                    inner_tqdm.refresh() # To enure the progressbar is visible for the individual audio recordings...
                    """
                if isinstance(model, Whisper):
                    before_mem = psutil.Process().memory_info().rss / (1e+6)
                    print('Before loading the full audio array: ', before_mem)
                    
                    wav, _ = librosa.load(sample['audio'], sr=16000)

                    after_mem = psutil.Process().memory_info().rss / (1e+6)
                    print('After loading the full audio array: ', after_mem)

                    delta_mem = after_mem - before_mem
                    print('Delta memory: ', delta_mem)

                    outputs, info = model.pipeline.transcribe(
                    audio=wav,
                    language='da',
                    word_timestamps=True,
                    log_progress=True,
                    chunk_length=15,
                    batch_size=4
                    )
                    for out in outputs:
                        iterable_segments.append({
                            'start': out.start,
                            'end': out.end,
                            'text': out.text,
                            'avg_logprob': out.avg_logprob
                        })

                end_process_time = time.process_time()
                cpu_time = end_process_time - start_process_time
                rtf_sample = cpu_time / info.duration # processing time divided by the actual audio duration
                
                #logger.info('Streamed Inference CPU Time: %f', cpu_time)
                #logger.info('Streamed Inference Real-Time Factor (RTF): %f', rtf_sample)
                item = {
                        'id': sample['id'],
                        'duration': info.duration,
                        'cpu_time': cpu_time,
                        'rtf': rtf_sample,
                        'audio': {
                            'bytes': b64encode(sample['audio']).decode('ascii') 
                        },
                        'segments': iterable_segments
                    }
                results_queue.put(item)
                """
                before_alignment.append(
                    (sample['id'], sample['audio'], iterable_segments)
                )
                """

    except Exception as e:
        print(f'An error occurred...{e}')
        return None
    
    finally:
        results_queue.put(None) # To signal the process to terminate upon exit.
        write_results_process.join()
        if not write_results_process.is_alive():
            write_results_process.close()

        model.unload()
        del model
        gc.collect()

        return 'Success'


def inference_streaming_diarize(data: AudioData, model: Diarize, output_filename: str):
    batch_size = 2
    loader = DataLoader(
        dataset=data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator_fn,
        num_workers=1
    )

    queue = Queue()
    write_process = Process(target=updater, args=(queue, output_filename))
    write_process.start()

    try:
        for iter, batch in enumerate(tqdm(loader)):

            if iter == 4:
                break

            for sample in batch:
                start_process_time = time.process_time()
                if isinstance(sample['audio'], bytes):
                    audio = read_bytes(sample['audio'])
                else:
                    audio, _ = librosa.load(sample['audio'], sr=data.target_sr)

                audio_time = librosa.get_duration(y=audio, sr=data.target_sr)
                wav = torch.tensor(audio).unsqueeze(0)   # To get the correct format of (channel, time) Tensor.

                speaker_segments = []
                with ProgressHook() as hook:
                    output = model.model(
                        {'waveform': wav, 'sample_rate': data.target_sr}, 
                        hook=hook,
                        min_speakers=1,
                        max_speakers=2
                    )
                    for segment, _, speaker in output.speaker_diarization.itertracks(yield_label=True):
                        speaker_segments.append({
                            'speaker': speaker,
                            'start': segment.start,
                            'end': segment.end,
                            'duration': segment.duration
                        })
                    
                end_process_time = time.process_time()
                cpu_time = end_process_time - start_process_time
                rtf_sample = cpu_time / audio_time # processing time divided by the actual audio duration
            
                item = {
                    'id': sample['id'],
                    'speaker_segments': speaker_segments
                    }
                queue.put(item)

                logger.info('Diarization Inference CPU Time: %f', cpu_time)
                logger.info('Diarization Inference Real-Time Factor (RTF): %f', rtf_sample)
    except Exception as e:
        print(f'Failed with error...: {e}')
        return None
    finally:
        queue.put(None) # To signal the process to terminate upon exit.
        write_process.join()
        if not write_process.is_alive():
            write_process.close()

        model.unload()
        del model
        del loader
        gc.collect()

    return 'Success'


def align_transcripts(asr_output_file: str, config, alignment_result_filename: str):
    try:
        print(f'Running alignment...')
        start_process_time = time.process_time()
        alignment_pipeline = Wav2Vec2(config=config, device='cpu')

        reader_queue = Queue()
        reader_process = Process(target=reader, args=(reader_queue, asr_output_file))
        reader_process.start()

        with open(alignment_result_filename, "w") as f:
            while True:
                asr_sample = reader_queue.get()

                if asr_sample is None:
                    break
                
                audio = b64decode(asr_sample['audio']['bytes']) if 'bytes' in asr_sample['audio'].keys() else asr_sample['audio']['path']
                
                if isinstance(audio, bytes):
                    wav = read_bytes(audio)
                else:
                    wav, _ = librosa.load(path=audio, sr=16000)

                segments = [cast(item) for item in asr_sample['segments']]
                transcription_result = alignment_pipeline.run_alignment(
                    transcript=segments,
                    audio=wav
                )
                
                end_process_time = time.process_time()
                cpu_time = end_process_time - start_process_time
                rtf_sample = cpu_time / asr_sample['duration'] # processing time divided by the actual audio duration

                logger.info('Aligning Transcripts CPU Time: %f', cpu_time)
                logger.info('Aligning Transcripts Real-Time Factor (RTF): %f', rtf_sample)

                transformed_result = [{'id': asr_sample['id'], 
                                    'start': item['start'], 
                                    'end': item['end'],
                                    'text': item['text'],
                                    'avg_logprob': item['avg_logprob'],
                                    'words': [{'word': obj['word'],
                                                'start': obj['start'],
                                                'end': obj['end'],
                                                'score': obj['score']} for obj in item['words']]} for item in transcription_result['segments']]
                
                
                f.write(json.dumps(transformed_result) + "\n")

    except Exception as e:
        print(f'Failed with error: {e}')
        return None
    finally:
        alignment_pipeline.unload()
        del alignment_pipeline

        reader_process.join()
        if reader_process.is_alive():
            reader_process.close()
        
        
        gc.collect()
        print('Finished aligning the transcripts...!')
    
    return 'Success'

"""
def eval_bert(bert, info):
    for item in info.values():
        encoded_pred_transcripts = bert.tokenizer(item['pred'], padding=True, truncation=True, return_tensors='pt')
        encoded_target_transcripts = bert.tokenizer(item['target'], padding=True, truncation=True, return_tensors='pt')

        pred_transcripts_embeddings = bert(
            input_ids=encoded_pred_transcripts.input_ids.to(bert.device),
            attention_mask=encoded_pred_transcripts.attention_mask.to(bert.device)
        )
        transcripts_embeddings = bert(
            input_ids=encoded_target_transcripts.input_ids.to(bert.device),
            attention_mask=encoded_target_transcripts.attention_mask.to(bert.device)
        )
        score = compute_cosine_sim(pred_transcripts_embeddings, transcripts_embeddings)
        item['semdist'] = score

    return info
    """