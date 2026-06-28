from tqdm import tqdm
from .data import cast, AudioData, chunk_batch
import librosa
from torch.utils.data import DataLoader
from multi_speaker_asr.models.asr import ASR, Whisper
from faster_whisper.transcribe import restore_speech_timestamps
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
import numpy as np



logging.config.dictConfig(LOGGING_CONFIG)
logging.getLogger("faster_whisper").setLevel(logging.DEBUG)

logger = logging.getLogger(name='Evaluate')



def result_to_offset(resultQueue: Queue, offsetQueue: Queue, output_file: str):
    with open(output_file, "w") as f:
        while True:
            item = resultQueue.get()

            if item is None:
                break
            f.write(json.dumps(item) + "\n")
            pos = f.tell()
            offsetQueue.put((item['id'], pos))


def offset_to_result(resultQueue: Queue, offsetQueue: Queue, output_file: str):
    with open(output_file, "w") as f:
        while True:
            offset = offsetQueue.get()

            if offset is None:
                break
            f.seek(offset)
            line = f.readline()
            result = json.loads(line)
            resultQueue.put(result)



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


def inference_asr(loader, config):

    with torch.inference_mode():
        pipeline = Whisper(
            compute_type=config['computetype'],
            cpu_threads=config['cputhreads'],
            device=config['device'],
            model=config['model']
        )

        resultsQueue = Queue()
        offsetQueue = Queue()
        write_results_process = Process(target=result_to_offset, args=(resultsQueue, offsetQueue, config['asr_output_filename']))
        write_results_process.start()

        id_offset_map ={}

        try:
                
            start_process_time = time.process_time()
            for batch in tqdm(loader):              
                (audio_chunks, 
                chunks_metadata, 
                clip_timestamps) = chunk_batch(batch=batch)

                segments, info = pipeline.transcribe(
                    audio_chunks=audio_chunks,
                    chunks_metadata=chunks_metadata,
                    clip_timestamps=clip_timestamps
                )

                segments = restore_speech_timestamps(
                    segments=segments, 
                    speech_chunks=clip_timestamps,
                    sampling_rate=16000
                    )
                iter_segments = []
                for segment in segments:
                    print(f'Start: {segment.start}, End: {segment.end}, Text: {segment.text}')
                    obj = {
                        'start': segment.start,
                        'end': segment.end,
                        'text': segment.text,
                        'avg_logprob': segment.avg_logprob
                    }
                    iter_segments.append(obj)
                    if config['segmented']:
                        id = batch[segment.id]['id']
                        item = {
                            'id': id,
                            'segments': iter_segments
                        }
                        resultsQueue.put(item)
                        (curr_id, curr_pos) = offsetQueue.get()
                        id_offset_map[curr_id] = curr_pos

                        iter_segments = []

                    if not config['segmented']:
                        id = batch[segment.id]
                        item = {
                            'id': id,
                            'segments': iter_segments
                        }
                        resultsQueue.put(item)
                        (curr_id, curr_pos) = offsetQueue.get()
                        id_offset_map[curr_id] = curr_pos

                    print(id_offset_map)

                end_process_time = time.process_time()
                cpu_time = end_process_time - start_process_time

                logger.info('CPU Time is %f', cpu_time)

        except Exception as e:
            logger.error('Failed with error: ', e)
            return None

        finally:
            resultsQueue.put(None) # To signal the process to terminate upon exit.
            write_results_process.join()
            if not write_results_process.is_alive():
                write_results_process.close()

            model.unload()
            del model
            del loader
            del data
            gc.collect()

            print(id_offset_map.keys())

    return 'Success', id_offset_map


"""
def batched_inference(data: AudioData, model: ASR, asr_result_filename: str):
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
    """

"""
def streamed_inference(data: AudioData, model: ASR, asr_result_filename: str):
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
        """

def inference_diarize(data: AudioData, config):
 
    pipeline = Diarize()   # Default values are fine for now

    queue = Queue()
    write_process = Process(target=updater, args=(queue, config['align_output_filename']))
    write_process.start()

    try:
        for _, sample in enumerate(tqdm(data)):
                start_process_time = time.process_time()

                
                with ProgressHook() as hook:
                    speaker_segments, audio_time = pipeline.diarize(sample=sample, hook=hook)
                    
                end_process_time = time.process_time()
                cpu_time = end_process_time - start_process_time
                rtf_sample = cpu_time / audio_time # processing time divided by the actual audio duration
            
                item = {
                    'id': sample['id'],
                    'cpu_time': cpu_time,
                    'rtf': rtf_sample,
                    'speaker_segments': speaker_segments
                    }
                queue.put(item)
                
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


def align_transcripts(data: AudioData, config, id_offset_map: dict):
    writer_queue = Queue()
    writer_process = Process(target=reader, args=(writer_queue, config['align_output_filename']))
    writer_process.start()

    segments_queue = Queue()
    offset_queue = Queue()
    reader_process = Process(target=offset_to_result, args=(segments_queue, offset_queue, config['asr_output_filename']))
    reader_process.start()
    
    try:
        print(f'Running alignment...')
        start_process_time = time.process_time()

        pipeline = Wav2Vec2(config=config, device='cpu')
        for sample in data:
            audio = sample['audio']
            id = sample['id']
            
            if isinstance(audio, bytes):
                wav = AudioData.read_audio(audio)
            else:
                wav, _ = librosa.load(path=audio, sr=16000)

            offset = id_offset_map.get(id)
            if offset == None:
                continue

            offset_queue.put(offset)
            asr_sample = segments_queue.get()

            segments = [cast(item) for item in asr_sample['segments']]
            transcription_result = pipeline.run_alignment(
                transcript=segments,
                audio=wav
            )
            
            end_process_time = time.process_time()
            cpu_time = end_process_time - start_process_time
            rtf_sample = cpu_time / asr_sample['duration'] # processing time divided by the actual audio duration


            transformed_result = [{
                                'id': id, 
                                'cpu_time': cpu_time,
                                'rtf': rtf_sample,   
                                'start': item['start'], 
                                'end': item['end'],
                                'text': item['text'],
                                'avg_logprob': item['avg_logprob'],
                                'words': [{'word': obj['word'],
                                            'start': obj['start'],
                                            'end': obj['end'],
                                            'score': obj['score']} for obj in item['words']]} for item in transcription_result['segments']]
            
            writer_queue.put(transformed_result)

    except Exception as e:
        print(f'Failed with error: {e}')
        return None
    finally:
        alignment_pipeline.unload()
        del alignment_pipeline

        writer_queue.put(None)
        offset_queue.put(None)
        
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