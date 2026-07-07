from .data import cast, AudioData
from multi_speaker_asr.models.asr import WhisperPipeline
from faster_whisper.transcribe import restore_speech_timestamps
from pyannote.audio.pipelines.utils.hook import ProgressHook
from multi_speaker_asr.models.diarization import Diarize, assign_word_speakers
from multi_speaker_asr.models.alignment import Wav2Vec2
import torch
from multi_speaker_asr.utils.utils import LOGGING_CONFIG, profile
import gc
import time
import logging
import logging.config
from multiprocessing import Process, Queue
import json
from torch.utils.data import DataLoader
import timeit
from faster_whisper.transcribe import SpeechTimestampsMap
import itertools
from .utils.vad import VAD




logging.config.dictConfig(LOGGING_CONFIG)
logging.getLogger("faster_whisper").setLevel(logging.DEBUG)

logger = logging.getLogger(name='Evaluate')



def result_to_offset(resultQueue: Queue, offsetQueue: Queue, output_file: str):
    try:
        with open(output_file, "w") as f:
            while True:
                item = resultQueue.get()

                if item is None:
                    break

                pos = f.tell()
                f.write(json.dumps(item) + "\n")
                
                offsetQueue.put((item['segment_id'], pos))
                f.flush()
    except Exception as e:
        logger.error('Failed in result_to_offset() with error: %s', e)


def offset_to_result(resultQueue: Queue, offsetQueue: Queue, output_file: str):
    try:
        with open(output_file, "r") as f:
            while True:
                offset = offsetQueue.get()

                if offset is None:
                    break
                f.seek(offset)
                line = f.readline()
                result = json.loads(line)
                resultQueue.put(result)
    except Exception as e:
        logger.error('Failed in offset_to_result() with error: %s', e)



def writer(queue: Queue, output_file: str):
    try:
        with open(output_file, "w") as f:
            while True:
                item = queue.get()

                if item is None:
                    break
                f.write(json.dumps(item) + "\n")
                f.flush()
    except Exception as e:
        logger.error('Failed in writer() with error: %s', e)


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
                if record[0]['audio_id'] == item['audio_id']:
                    record.append(item)
            
        with open(output_file, 'w') as f:
            for record in records:
                f.write(json.dumps(record) + '\n')
    except Exception as e:
        logger.error('Failed in updater() with error: %s', e)



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
        logger.error('Failed in reader() with error: %s', e)
    finally:
        logger.info('Finished reading the file: %s', input_file)
        queue.put(None) # For terminating the process

def read_multiples(alignQueue: Queue, diarizeQueue: Queue, align_file: str, diarize_file: str):
    try:
        with open(align_file, 'r') as a_file, open(diarize_file, 'r') as d_file:
            for a, d in zip(a_file, d_file):
                a = a.strip()
                d = d.strip()

                if not a:
                    continue
                a_record = json.loads(a)
                alignQueue.put(a_record)
                if not d:
                    continue
                d_record = json.loads(d)
                diarizeQueue.put(d_record)

    except Exception as e:
        logger.error('Process failed with error: %s', e)
    finally:
        print('Finished reading the entire file...')
        alignQueue.put(None)
        diarizeQueue.put(None)


def fetch_dataloader(
        data_type: str,
        audio_path: str,
        on_hpc: bool,
        vad_filter: bool,
        clip_timestamps: bool,
        batch_size: int,
        is_asr: bool
    ):
    
    data = AudioData(
        path=data_type,
        audio_path=audio_path,
        hpc=on_hpc,
        vad_filter=vad_filter,
        clip_timestamps=clip_timestamps,
        is_asr=is_asr
    )

    return DataLoader(
        dataset=data,
        shuffle=False,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=data.collator_fn
    )


@profile
def inference_asr(
                data_type: str,
                audio_path: str,
                on_hpc: bool,
                vad_filter: bool,
                clip_timestamps: bool,
                batch_size: int, 
                computetype='int8', 
                cputhreads=4, 
                device='cpu', 
                model='pluttodk/roest-v3-whisper-1.5b-ct2', 
                filename='asr_output_int8.jsonl'
            ):
    loader = fetch_dataloader(
        data_type=data_type,
        audio_path=audio_path,
        on_hpc=on_hpc,
        vad_filter=vad_filter,
        clip_timestamps=clip_timestamps,
        batch_size=batch_size,
        is_asr=True
    )

    # TODO: Add a flag or some indication that it should handle chunked audio or not using e.g. clip_timestamps. Requires some refactoring to make it work for both segmented audio (i.e. CoRal) and other audio types.
    with torch.inference_mode():
        pipeline = WhisperPipeline(
            compute_type=computetype,
            cpu_threads=cputhreads,
            model=model,
            device=device
        )
        
        resultsQueue = Queue()
        offsetQueue = Queue()
        write_results_process = Process(target=result_to_offset, args=(resultsQueue, offsetQueue, filename))
        write_results_process.start()

        id_offset_map ={}
        id_segments_map = {}
        try:
            t1 = timeit.default_timer()
            start_process_time = time.process_time()
            for batch in loader:

                audio_chunks = [chunks['audio'] for chunks in batch]
                metadata = [chunks['chunk_metadata'] for chunks in batch]
                original_timeline = list(itertools.chain.from_iterable([segmentsList['segments'] for segmentsList in metadata]))

                segments, _ = pipeline.transcribe(
                    audio_chunks=audio_chunks,
                    chunks_metadata=metadata,
                    ids=[item['audio_id'] for item in batch],
                    clip_timestamps=original_timeline,
                    clip_timestamps_provided=clip_timestamps,
                    vad_filter=vad_filter,
                    batch_size=batch_size,
                    log_progress=True,
                    word_timestamps=True
                    )
                
                for segment in segments:
                    seg_id_idx = segment.id - 1
                    obj = {
                        'audio_id': batch[seg_id_idx]['audio_id'], # Note, segment.id is not zero-indexed
                        'segment_id': batch[seg_id_idx]['segment_id'],
                        'start': segment.start,
                        'end': segment.end,
                        'text': segment.text,
                        'avg_logprob': segment.avg_logprob
                    }

                    resultsQueue.put(obj)
                    (curr_id, curr_pos) = offsetQueue.get()
                    id_offset_map[curr_id] = curr_pos

                    seg_id = batch[seg_id_idx]['segment_id']
                    seg_ids = id_segments_map.get(batch[seg_id_idx]['audio_id'], [])
                    seg_ids.append(seg_id)
                    id_segments_map[batch[seg_id_idx]['audio_id']] = seg_ids

        except Exception as e:
            logger.error('Failed with error: ', e)
        finally:
            resultsQueue.put(None) # To signal the process to terminate upon exit.
            write_results_process.join()
            if not write_results_process.is_alive:
                pipeline.unload()
                del pipeline
                del loader
            
            end_process_time = time.process_time()
            cpu_time = end_process_time - start_process_time
            t2 = timeit.default_timer()
            walltime = t2 - t1

            logger.info('Walltime is %f', walltime)
            logger.info('CPU Time is %f', cpu_time)

            gc.collect()

    return id_offset_map, id_segments_map

@profile
def align_transcripts(
        data_type: str,
        audio_path: str,
        hpc: bool,
        vad_filter: bool,
        clip_timestamps: bool,
        batch_size: int,
        align_filename: str,
        asr_filename: str,
        model_name: str,
        id_offset_map: dict,
        id_segment_map: dict,

        ):
    
    loader = fetch_dataloader(
        data_type=data_type,
        audio_path=audio_path,
        on_hpc=hpc,
        vad_filter=vad_filter,
        clip_timestamps=clip_timestamps,
        batch_size=batch_size,
        is_asr=False
    )

    writer_queue = Queue()
    writer_process = Process(target=writer, args=(writer_queue, align_filename))
    writer_process.start()

    segments_queue = Queue()
    offset_queue = Queue()
    reader_process = Process(target=offset_to_result, args=(segments_queue, offset_queue, asr_filename))
    reader_process.start()
    
    pipeline = Wav2Vec2(model_name=model_name, device='cpu')
    print(f'Running alignment...')
    
    
    try:
        t1 = timeit.default_timer()
        start_process_time = time.process_time()

        for batch in loader:
            for sample in batch:
                print(type(sample))
                print(sample)

                audio = sample['audio']
                audio_id = sample['audio_id']

                print('In alignment...')
                print(f'Shape of audio: {audio.shape}')
                

                seg_ids = id_segment_map.get(audio_id, None)
                print(f'Segment ids for audio {audio_id}: {seg_ids}')
                if seg_ids is None:
                    raise Exception('Audio id %s had an empty list of segments', audio_id)
                
                segments = []
                audio_offset = 0
                for seg in seg_ids:
                    offset = id_offset_map.get(seg)
                    if offset == None:
                        print('Offset was None!')
                        continue
                    
                    offset_queue.put(offset)
                    asr_sample = segments_queue.get()
                    audio_offset = sample['offset']
                    segments.append(cast(asr_sample))


                transcription_result = pipeline.run_alignment(
                    transcript=segments,
                    audio=audio
                )
                transformed_result = {
                                    'audio_id': audio_id,
                                    'offset': audio_offset, # if the audio is clipped it will be larger than zero, and will be added to the timestamps at the end of the pipeline
                                    'words': [{'word': obj['word'],
                                                'start': obj['start'],
                                                'end': obj['end'],
                                                'score': obj['score']} for obj in transcription_result['word_segments']]}
                
                writer_queue.put(transformed_result)

    except Exception as e:
        logger.error('Failed with error: %s', e)
    finally:
        writer_queue.put(None)
        offset_queue.put(None)
        writer_process.join()
        reader_process.join()
        if (not writer_process.is_alive()) & (not reader_process.is_alive()):
            pipeline.unload()
            del pipeline
            del loader
        
        t2 = timeit.default_timer()
        walltime = t2 - t1

        end_process_time = time.process_time()
        cpu_time = end_process_time - start_process_time
        
        logger.info('CPU Time is %f', cpu_time)
        logger.info('Walltime is %f', walltime)

        gc.collect()
    
    return 'Success'

@profile
def inference_diarize(
        data_type: str,
        audio_path: str,
        hpc: str,
        vad_filter: bool,
        clip_timestamps: bool,
        batch_size: int,
        diarize_filename: str,
        min_speakers: int = 1, 
        max_speakers: int = 4
    ):

    loader = fetch_dataloader(
        data_type=data_type,
        audio_path=audio_path,
        on_hpc=hpc,
        vad_filter=vad_filter,
        clip_timestamps=clip_timestamps,
        batch_size=batch_size,
        is_asr=False
    )
    
 
    pipeline = Diarize()   # Default values are fine for now

    writerQueue = Queue()
    write_process = Process(target=writer, args=(writerQueue, diarize_filename))
    write_process.start()

    readerQueue = Queue()
    reader_process = Process(target=reader, args=(readerQueue, diarize_filename))
    reader_process.start()

    try:
        t1 = timeit.default_timer()
        start_process_time = time.process_time()
        for batch in loader:
                for sample in batch:
                    audio = sample['audio']
                    wav = torch.tensor(audio).unsqueeze(0)   # To get the correct format of (channel, time) Tensor.

                    speaker_segments = []
                    with ProgressHook() as hook:
                        output = pipeline.model({
                            'waveform': wav,
                            'sample_rate': 16000
                        },
                        hook=hook,
                        min_speakers=min_speakers,
                        max_speakers=max_speakers
                        )

                    for segment, _, speaker in output.speaker_diarization.itertracks(yield_label=True):
                        speaker_segments.append({
                            'speaker': speaker,
                            'start': segment.start,
                            'end': segment.end,
                            'duration': segment.duration
                        })
                        
                    
                
                    item = {
                        'audio_id': sample['audio_id'],
                        'offset': sample['offset'],
                        'speaker_segments': speaker_segments
                        }
                    writerQueue.put(item)
                
    except Exception as e:
        logger.error('Failed with error...: %s', e)
    finally:
        writerQueue.put(None) # To signal the process to terminate upon exit.
        write_process.join()
        if not write_process.is_alive():
            pipeline.unload()
            del pipeline
            del loader


        end_process_time = time.process_time()
        cpu_time = end_process_time - start_process_time

        t2 = timeit.default_timer()
        walltime = t2 - t1
        
        logger.info('CPU Time is %f', cpu_time)
        logger.info('Walltime is %f', walltime)
        gc.collect()

    return 'Success'

def generate_final_transcript(
        results_filename: str,
        align_filename: str,
        diarize_filename: str
        ):
    
    try:
        writerQueue = Queue()
        write_process = Process(target=writer, args=(writerQueue, results_filename))
        write_process.start()

        read_align_queue = Queue()
        read_diarize_queue = Queue()
        read_process = Process(target=read_multiples, args=(read_align_queue, read_diarize_queue, align_filename, diarize_filename))
        read_process.start()

        while True:
            if not results_filename:
                logger.error('Failed in generate_final_transcript(). Missing results_filename!')
                break

            segments_list = []

            alignment = read_align_queue.get()
            diarization = read_diarize_queue.get()
            if (alignment is None) & (diarization is None):
                break

            segments_list.append(alignment)
            transcript = assign_word_speakers(
                segments=segments_list,
                speaker_times=diarization['speaker_segments']
            )
            for t in transcript:
                t_offset = t['offset']
                words_timeline = t['words']
                restored_timeline = list(map(lambda x: {
                    'word': x['word'],
                    'start': x['start'] + t_offset,
                    'end': x['end'] + t_offset,
                    'score': x['score'],
                    'speaker': x['speaker']
                }, words_timeline))
                t['words'] = restored_timeline
            writerQueue.put(transcript)
            
        # TODO: Once verified that the final file is correct, add functionality to remove the other intermediate files.
    except Exception as e:
        logger.error('Failed with error: %s', e)
    finally:
        writerQueue.put(None)
        write_process.join()
        read_process.join()



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