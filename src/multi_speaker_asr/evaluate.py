from multi_speaker_asr.data import cast, AudioData, SegmentedData
from multi_speaker_asr.models.asr import WhisperPipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from multi_speaker_asr.models.diarization import Diarize, assign_word_speakers, RollingClusters
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
import itertools
from .utils.vad import collect_word_chunks
from jiwer import wer, cer
from whisperx.schema import SingleSegment
import psutil, os
from time import perf_counter, process_time
import soundfile as sf
from scipy.spatial.distance import cdist
import numpy as np
import librosa
from carbontracker.tracker import CarbonTracker



logging.config.dictConfig(LOGGING_CONFIG)
logging.getLogger("faster_whisper").setLevel(logging.DEBUG)

logger = logging.getLogger(name='Evaluate')
proc = psutil.Process(os.getpid())


def fetch_dataloader(
        data_type: str,
        audio_path: str,
        vad_filter: bool,
        clip_timestamps: bool,
        batch_size: int,
    ):
    
    data = AudioData(
        vad_filter=vad_filter,
        clip_timestamps=clip_timestamps,
    )

    return DataLoader(
        dataset=data,
        shuffle=False,
        batch_size=batch_size,
        num_workers=0
        )


@profile
def inference_asr_presegmented(
                batch_size: int, 
                computetype='int8', 
                cputhreads=4, 
                device='cpu', 
                model='pluttodk/roest-v3-whisper-1.5b-ct2', 
                filename='asr_output_int8.jsonl'
            ):
    data = SegmentedData()
    data.load()
    loader = DataLoader(
        dataset=data,
        shuffle=False,
        batch_size=batch_size,
        num_workers=1,
        collate_fn=data.collator
    )
    max_epochs = 4
    tracker = CarbonTracker(epochs=max_epochs)

    with torch.inference_mode():
        pipeline = WhisperPipeline(
            compute_type=computetype,
            cpu_threads=cputhreads,
            model=model,
            device=device
        )

        try:
            with open(filename, 'a') as file:
                for epoch in range(max_epochs):
                    batch_num = 0
                    tracker.epoch_start()

                    for batch in loader:

                        audio_chunks = [chunks['audio'] for chunks in batch]
                        metadata = [chunks['chunk_metadata'] for chunks in batch]
                        original_timeline = list(itertools.chain.from_iterable([segmentsList['segments'] for segmentsList in metadata]))

                        epoch_start = perf_counter()
                        segments, _ = pipeline.transcribe(
                            audio_chunks=audio_chunks,
                            chunks_metadata=metadata,
                            ids=[],
                            clip_timestamps=original_timeline,
                            clip_timestamps_provided=False,
                            vad_filter=False,
                            batch_size=8,
                            log_progress=True,
                            word_timestamps=True
                            )
                        hypothesis = [segment.text for segment in segments] # To initialize and run the lazy loading implementation...
                        epoch_stop = perf_counter() - epoch_start
                        
                        chunks = [[segment['id'] for segment in segments_list['segments']] for segments_list in metadata]
                        batch_duration = sum([data['duration'] for data in metadata])

                        logger.critical('Epoch: %i, Batch: %i Walltime: %f Batch Duration: %f', epoch, batch_num, epoch_stop, batch_duration)
                        result = {
                            'epoch': epoch,
                            'batch_id': batch_num,
                            'hypothesis': hypothesis,
                            'segment_ids': chunks,
                            'walltime': epoch_stop,
                            'audio_duration': batch_duration
                        }
                        file.write(json.dumps(result) + '\n')
                        file.flush()
                        batch_num += 1

                    tracker.epoch_end()

        except Exception as e:
            logger.error('Failed with error: ', e)
            tracker.stop()
        finally:
            pipeline.unload()
            del pipeline
            del loader

            gc.collect()

MAX_EPOCHS = 5


def asr_inference(
        data_type: str,
        vad_filter: bool,
        clip_timestamps: bool,
        batch_size: int, 
        computetype='int8', 
        cputhreads=4, 
        device='cpu', 
        model='pluttodk/roest-v3-whisper-1.5b-ct2',
        filename='asr_output_int8.jsonl'
        ):
    data = AudioData(
            vad_filter=vad_filter,
            clip_timestamps=clip_timestamps,
        )
    data.load(path=data_type)
    asr_tracker = CarbonTracker(MAX_EPOCHS)

    try:
        with torch.inference_mode():
            pipeline = WhisperPipeline(
                                compute_type=computetype,
                                cpu_threads=cputhreads,
                                model=model,
                                device=device
                            )
            with open(filename, 'a') as file:
                for epoch in range(MAX_EPOCHS):
                    loader = DataLoader(
                                dataset=data,
                                shuffle=False,
                                batch_size=batch_size,
                                num_workers=0,
                                collate_fn=data.collator
                            )
                    batch_num = 0
                    asr_tracker.epoch_start()

                    for batch in loader: # ASR module...
                        asr_walltime_start = time.perf_counter()
                        asr_cputime_start = time.process_time()

                        for k, v in batch.items():
                            metadata = v['chunk_metadata']
                            orig_timeline = list(itertools.chain.from_iterable([segmentsList['segments'] for segmentsList in metadata]))
                        
                            segments, _ = pipeline.transcribe(
                                audio_chunks=v['audio'],
                                chunks_metadata=metadata,
                                clip_timestamps=orig_timeline,
                                clip_timestamps_provided=clip_timestamps,
                                vad_filter=vad_filter,
                                batch_size=batch_size,
                                log_progress=True,
                                word_timestamps=False
                            )
                            result = {'epoch': epoch, 'audio_id': k, 'segments': [{'segment_id': s.id, 'start': s.start, 'end': s.end, 'text': s.text} for s in segments]}
                            file.write(json.dumps(result) + '\n')
                            file.flush()
                        #result = {'epoch': epoch,'batch_id': batch_num, 'segments': [{'audio_id': batch[segment.id - 1]['audio_id'], 'clip_offset': batch[segment.id - 1]['offset'], 'words': segment.words} for segment in segments]}
                        asr_walltime = perf_counter() - asr_walltime_start
                        asr_cputime = process_time() - asr_cputime_start

                        logger.info(
                                "ASR Module... Epoch: %i Batch: %i RSS: %.2f GB",
                                epoch,
                                batch_num,
                                proc.memory_info().rss / (1024**3)
                            )
                        """
                        result['wall_time'] = asr_walltime
                        result['cpu_time'] = asr_cputime
                        result['segments'] = [{
                                                'audio_id': items['audio_id'],  
                                                'clip_offset': items['clip_offset'],
                                                'words': [{'start': word.start, 'end': word.end, 'word': word.word} for word in items['words']]} for items in result['segments']]
                        """
                        logger.info('Epoch: %i, Batch: %i Walltime: %f CPU time: %f', epoch, batch_num, asr_walltime, asr_cputime)
                        batch_num += 1

                    asr_tracker.epoch_end()

    except Exception as e:
        logger.error('Failed with error: %s', e)
    finally:
        pipeline.unload()
        del pipeline
        del loader
        gc.collect()


def aligner_inference(
        data_type: str,
        vad_filter: bool = False,
        clip_timestamps: bool = False,
        model='CoRal-project/roest-v3-wav2vec2-315m', 
        write_file='align_output_longer_int8.jsonl',
        read_file='asr_output_longer_int8.jsonl'
    ):
    data = AudioData(
            vad_filter=vad_filter,
            clip_timestamps=clip_timestamps,
        )
    data.load(path=data_type)

    try:
        batch_num = 0
        chunk_size = 10

        pipeline = Wav2Vec2(model_name=model, device='cpu')
        tracker = CarbonTracker(MAX_EPOCHS)
        asr_results_lookup = {}
        with open(read_file, 'r') as reader:
            for line in reader:
                out = json.loads(line)
                out_id = out['audio_id']
                asr_results_lookup[out_id] = out['segments']

        with torch.inference_mode():
            for epoch in range(MAX_EPOCHS):
                tracker.epoch_start()

                current_audiofile = None
                prev_chunk_duration = 0
                with open(write_file, 'a') as writer:
                    for sample in data.ds:

                        # Fetch the info from ASR output:
                        audio_file = sample['audio_id']
                        asr_output = asr_results_lookup.get(audio_file)
                        if not asr_output: continue


                        audio_path = data.id_to_audio.get(audio_file)
                        if audio_path is None:
                            continue
                            
                        for idx, segment in enumerate(sample['segments']):
                            words = segment['words']
                            if not words:
                                continue
                            
                            segment_end = max([word['end'] for word in words])
                            for chunk_offset, audio_chunk in fetch_audio_chunk(audio_path=audio_path, chunk_size=chunk_size, overlap=0.1, clip_offset=(clip_offset + prev_chunk_duration)):
                                chunk_offset -= clip_offset
                                if chunk_offset >= segment_end:
                                    break
                        
                                active_words_with_indices = [
                                    (idx, w) for idx, w in enumerate(words)
                                    if chunk_offset <= w['start'] < (chunk_offset + chunk_size)
                                ]
                                if not active_words_with_indices:
                                    continue
                                chunk_words = [item[1] for item in active_words_with_indices]
                                transcript = [SingleSegment(start=min([w['start'] for w in chunk_words]) - chunk_offset, 
                                                                end=max([w['end'] for w in chunk_words]) - chunk_offset, 
                                                                text=' '.join([w['word'] for w in chunk_words]))]
                                alignment_walltime_start = perf_counter()
                                alignment_cputime_start = process_time()
                                transcription_result = pipeline.run_alignment(
                                                            transcript=transcript,
                                                            audio=audio_chunk
                                                        )
                                alignment_walltime = perf_counter() - alignment_walltime_start
                                alignment_cputime = process_time() - alignment_cputime_start
                                logger.critical('Epoch: %i, Batch: %i Walltime: %f CPU time: %f', epoch, batch_num, alignment_walltime, alignment_cputime)
                                logger.info(
                                    "Alignment Module... Epoch: %i Batch: %i RSS: %.2f GB",
                                    epoch,
                                    batch_num,
                                    proc.memory_info().rss / (1024**3)
                                )
                                res = {
                                    'epoch': epoch,
                                    'seg_idx': idx, 
                                    'audio_id': audio_file,
                                    'segments': [{
                                        'start': (item['start'] + chunk_offset + clip_offset),
                                        'end': (item['end'] + chunk_offset + clip_offset),
                                        'text': item['text']} for item in transcription_result['segments']],
                                    'word_segments': [{'word': item['word'], 'start': (item['start'] + chunk_offset + clip_offset), 'end': (item['end'] + chunk_offset + clip_offset), 'score': item['score']} for item in transcription_result['word_segments']]
                                }
                                writer.write(json.dumps(res) + '\n')
                                writer.flush()

                                batch_num += 1
                                prev_chunk_duration = segment_end

        tracker.epoch_end()
    except Exception as e:
        logger.error('Failed with error: %s', e)
    finally:
        pipeline.unload()
        del pipeline
        gc.collect()
    

    


def fetch_audio_chunk(audio_path, chunk_size, overlap, clip_offset=0, target_sr=16000):
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    with sf.SoundFile(audio_path) as f:
        orig_sr = f.samplerate

        chunk_samples_orig = int(chunk_size * orig_sr)
        step_samples = int((chunk_size - overlap) * orig_sr)
    
        f.seek(int(clip_offset * orig_sr)) # Setting the offset to start with, if the audio was clipped, e.g. set to start 5 seconds into the audio
        
        while True:
            start = f.tell()
            chunk = f.read(chunk_samples_orig, dtype="float32")

            if chunk.ndim > 1:
                chunk = chunk.mean(axis=1)

            if orig_sr != target_sr:
                chunk = librosa.resample(
                    chunk,
                    orig_sr=orig_sr,
                    target_sr=target_sr,
                )
                chunk = chunk.astype("float32")
                chunk_samples_target = int(chunk_size * target_sr)

            if len(chunk) == 0:
                break

            yield start / orig_sr, chunk

            if len(chunk) < chunk_samples_target: # It compares with the re-sampled number of chunk samples using the 16kHz samplerate only here!
                break

            f.seek(start + step_samples)



def inference_full_pipeline(
        data_type: str,
        vad_filter: bool,
        clip_timestamps: bool,
        batch_size: int, 
        align_model: str,
        computetype='int8', 
        cputhreads=4, 
        device='cpu', 
        asr_model='pluttodk/roest-v3-whisper-1.5b-ct2', 
        asr_filename='asr_output_int8.jsonl',
        align_filename='data/coral-v3-long-form-conversations/test/align_output_longer_int8.jsonl',
        diarize_filename='data/coral-v3-long-form-conversations/test/diarize_output_coral_segments.jsonl',
        results_filename='data/coral-v3-long-form-conversations/test/output_longer_int8.jsonl'
        ):

        
        data = AudioData(
            vad_filter=vad_filter,
            clip_timestamps=clip_timestamps,
        )
        data.load(path=data_type)
        max_epochs = 4
        asr_tracker = CarbonTracker(max_epochs)
        align_tracker = CarbonTracker(max_epochs)
        diarize_tracker = CarbonTracker(max_epochs)
        pipeline_tracker = CarbonTracker(max_epochs)
        
        with torch.inference_mode():
            for epoch in range(5):
                pipeline_walltime_start = perf_counter()
                pipeline_cputime_start = process_time()
                pipeline_tracker.epoch_start()
                try:
                    loader = DataLoader(
                                dataset=data,
                                shuffle=False,
                                batch_size=batch_size,
                                num_workers=1,
                                collate_fn=data.collator
                            )
                    pipeline = WhisperPipeline(
                                compute_type=computetype,
                                cpu_threads=cputhreads,
                                model=asr_model,
                                device=device
                            )
                    batch_num = 0
                    asr_tracker.epoch_start()

                    with open(asr_filename, 'a') as file:
                        for batch in loader: # ASR module...
                            asr_walltime_start = time.perf_counter()
                            asr_cputime_start = time.process_time()
                            
                            segments = pipeline.run_whisper(
                                batch=batch,
                                batch_size=batch_size,
                                vad_filter=vad_filter,
                                clip_timestamps=clip_timestamps
                            )
                            
                            result = {'epoch': epoch,'batch_id': batch_num, 'segments': [{'audio_id': batch[segment.id - 1]['audio_id'], 'clip_offset': batch[segment.id - 1]['offset'], 'words': segment.words} for segment in segments]}
                            asr_walltime = perf_counter() - asr_walltime_start
                            asr_cputime = process_time() - asr_cputime_start

                            logger.info(
                                    "ASR Module... Epoch: %i Batch: %i RSS: %.2f GB",
                                    epoch,
                                    batch_num,
                                    proc.memory_info().rss / (1024**3)
                                )

                            result['wall_time'] = asr_walltime
                            result['cpu_time'] = asr_cputime
                            result['segments'] = [{
                                                    'audio_id': items['audio_id'],  
                                                    'clip_offset': items['clip_offset'],
                                                    'words': [{'start': word.start, 'end': word.end, 'word': word.word} for word in items['words']]} for items in result['segments']]
                            logger.critical('Epoch: %i, Batch: %i Walltime: %f CPU time: %f', epoch, batch_num, asr_walltime, asr_cputime)
                            batch_num += 1
                            
                            file.write(json.dumps(result) + '\n')
                            file.flush()

                    asr_tracker.epoch_end()

                except Exception as e:
                    logger.error('Failed with error: %s', e)
                finally:
                    pipeline.unload()
                    del pipeline
                    del loader
                    gc.collect()

                try:
                    pipeline = Wav2Vec2(model_name=align_model, device='cpu')

                    align_tracker.epoch_start()
                    align(pipeline, align_filename, asr_filename, data, epoch)
                    align_tracker.epoch_end()
                except Exception as e:
                    logger.error('Failed with error: %s', e)
                finally:
                    pipeline.unload()
                    del pipeline
                    gc.collect()


                try:
                    cluster_registry = RollingClusters()
                    pipeline = Diarize()   # Default values are fine for now
                    pipeline.load()
                    iter = 0

                    diarize_tracker.epoch_start()
                    with open(diarize_filename, 'a') as file:
                        for sample in data.ds:
                            # Split into subsegments of size 5 seconds with a small overlap of e.g. 0.5-1 second.
                            chunk_size = 10
                            overlap = 0.5

                            clip_start = sample['start']
                            clip_end = sample['end']

                            
                            for chunk_offset, audio_chunk in fetch_audio_chunk(audio_path=sample['audio'], chunk_size=chunk_size, overlap=overlap, clip_offset=clip_start):
                                chunk_offset -= clip_start
                                if chunk_offset >= (clip_end - clip_start):
                                    break
                                
                                speaker_segments = []
                                output = diarize(batch_num=batch_num, audio_chunk=audio_chunk, pipeline=pipeline, epoch=epoch)

                                idx_to_speaker = {
                                    i: cluster_registry.process_chunk(emb)
                                    for i, emb in enumerate(output.speaker_embeddings)
                                }
                                speaker_to_idx = {
                                    label: i
                                    for i, label in enumerate(output.speaker_diarization.labels())
                                }
                            
                                for segment, track, speaker in output.speaker_diarization.itertracks(yield_label=True):
                                    print(f'Speaker: {speaker}... Start: {segment.start}, End: {segment.end}... Track: {track}')
                                    idx = speaker_to_idx[speaker]
                                    speaker_label = idx_to_speaker[idx]

                                    if speaker_label is None:
                                        logger.error('Segment contains no speech. Skip!')
                                        continue
                                    else:
                                        speaker_segments.append({
                                            'speaker': speaker_label,
                                            'start': (segment.start + chunk_offset),
                                            'end': (segment.end + chunk_offset),
                                            'duration': segment.duration
                                        })

                                item = {
                                    'epoch': epoch,
                                    'idx': iter,
                                    'audio_id': sample['id'],
                                    'offset': chunk_offset,
                                    'speaker_segments': speaker_segments
                                    }
                                iter += 1
                                file.write(json.dumps(item) + '\n')
                                file.flush()

                    diarize_tracker.epoch_end()

                except Exception as e:
                    logger.error('Failed during diarization with error: %s', e)
                finally:
                    pipeline.unload()
                    del pipeline
                    gc.collect()

                
                try:
                    with open(align_filename, 'r') as reader_alignment, open(diarize_filename, 'r') as reader_diarize, open(results_filename, 'a') as writer:
                        for align_line, diarize_line in zip(reader_alignment, reader_diarize):
                            alignment = json.loads(align_line)
                            diarization = json.loads(diarize_line)

                            segments_list = []

                            segments_list.append(alignment)
                            transcript = assign_word_speakers(
                                segments=segments_list,
                                speaker_times=diarization['speaker_segments']
                            )
                            transcript['epoch'] = epoch
                            writer.write(json.dumps(transcript))
                            writer.flush()
                except Exception as e:
                    logger.error('Failed during combining the results to a final transcript with error: %s', e)
                

                # At the end of each epoch, log the compute resources used in total during the single epoch...
                pipeline_tracker.epoch_end()

                pipeline_walltime = perf_counter() - pipeline_walltime_start
                pipeline_cputime = process_time() - pipeline_cputime_start
                logger.critical('Epoch: %i, Batch: %i Walltime: %f CPU time: %f', epoch, batch_num, pipeline_walltime, pipeline_cputime)
                logger.info(
                    "Full Pipeline... Epoch: %i Batch: %i RSS: %.2f GB",
                    epoch,
                    batch_num,
                    proc.memory_info().rss / (1024**3)
                )

def align(pipeline, write_file, read_file, data, epoch):
    
    batch_num = 0
    chunk_size = 10

    current_audiofile = None
    prev_chunk_duration = 0
    with open(read_file, 'r') as reader, open(write_file, 'a') as writer:
        for line in reader:
            asr_output = json.loads(line)

            if asr_output['epoch'] != epoch:
                continue

            for audio_file, segments in itertools.groupby(asr_output['segments'], key=lambda x: x['audio_id']):
                if current_audiofile is None: 
                    current_audiofile = audio_file
                elif current_audiofile != audio_file:
                    prev_chunk_duration = 0
                    current_audiofile = audio_file

                segments = list(segments)
                clip_offset = segments[0]['clip_offset'] # just take the first element...

                audio_path = data.id_to_audio.get(audio_file)
                if audio_path is None:
                    continue
                
                for idx, segment in enumerate(segments):
                    words = segment['words']
                    if not words:
                        continue
                    
                    segment_end = max([word['end'] for word in words])
                    for chunk_offset, audio_chunk in fetch_audio_chunk(audio_path=audio_path, chunk_size=chunk_size, overlap=0.1, clip_offset=(clip_offset + prev_chunk_duration)):
                        chunk_offset -= clip_offset
                        if chunk_offset >= segment_end:
                            break
                 
                        active_words_with_indices = [
                            (idx, w) for idx, w in enumerate(words)
                            if chunk_offset <= w['start'] < (chunk_offset + chunk_size)
                        ]
                        if not active_words_with_indices:
                            continue
                        chunk_words = [item[1] for item in active_words_with_indices]
                        transcript = [SingleSegment(start=min([w['start'] for w in chunk_words]) - chunk_offset, 
                                                        end=max([w['end'] for w in chunk_words]) - chunk_offset, 
                                                        text=' '.join([w['word'] for w in chunk_words]))]
                        alignment_walltime_start = perf_counter()
                        alignment_cputime_start = process_time()
                        transcription_result = pipeline.run_alignment(
                                                    transcript=transcript,
                                                    audio=audio_chunk
                                                )
                        alignment_walltime = perf_counter() - alignment_walltime_start
                        alignment_cputime = process_time() - alignment_cputime_start
                        logger.critical('Epoch: %i, Batch: %i Walltime: %f CPU time: %f', epoch, batch_num, alignment_walltime, alignment_cputime)
                        logger.info(
                            "Alignment Module... Epoch: %i Batch: %i RSS: %.2f GB",
                            epoch,
                            batch_num,
                            proc.memory_info().rss / (1024**3)
                        )
                        res = {
                            'epoch': epoch,
                            'seg_idx': idx, 
                            'audio_id': audio_file,
                            'segments': [{
                                'start': (item['start'] + chunk_offset + clip_offset),
                                'end': (item['end'] + chunk_offset + clip_offset),
                                'text': item['text']} for item in transcription_result['segments']],
                            'word_segments': [{'word': item['word'], 'start': (item['start'] + chunk_offset + clip_offset), 'end': (item['end'] + chunk_offset + clip_offset), 'score': item['score']} for item in transcription_result['word_segments']]
                        }
                        writer.write(json.dumps(res) + '\n')
                        writer.flush()

                        batch_num += 1
                    prev_chunk_duration = segment_end
                   
                    """
                        for start_idx, end_idx in pipeline.get_chunk_generator(words=words, offset=chunk_offset):
                            if start_idx is None or end_idx is None:
                                continue
                            
                            local_start = active_words_with_indices[start_idx][0]
                            local_end = active_words_with_indices[end_idx][0]
                            
                            start = words[local_start]['start']
                            end = words[local_end]['end']
                            
                            
                            start = words[start_idx]['start']
                            end = words[end_idx]['end']

                            transcript = [SingleSegment(start=start - chunk_offset, 
                                                        end=end - chunk_offset, 
                                                        text=' '.join([word['word'] for word in words[start_idx : (end_idx + 1)]]))]
                
                            alignment_walltime_start = perf_counter()
                            alignment_cputime_start = process_time()
                            transcription_result = pipeline.run_alignment(
                                                        transcript=transcript,
                                                        audio=audio_chunk
                                                    )
                            alignment_walltime = perf_counter() - alignment_walltime_start
                            alignment_cputime = process_time() - alignment_cputime_start
                            logger.critical('Epoch: %i, Batch: %i Walltime: %f CPU time: %f', epoch, batch_num, alignment_walltime, alignment_cputime)
                            logger.info(
                                "Alignment Module... Epoch: %i Batch: %i RSS: %.2f GB",
                                epoch,
                                batch_num,
                                proc.memory_info().rss / (1024**3)
                            )
                            res = {
                                'epoch': epoch,
                                'seg_idx': idx, 
                                'audio_id': audio_file,
                                'segments': [{
                                    'start': (item['start'] + chunk_offset + clip_offset),
                                    'end': (item['end'] + chunk_offset + clip_offset),
                                    'text': item['text']} for item in transcription_result['segments']],
                                'word_segments': [{'word': item['word'], 'start': (item['start'] + chunk_offset + clip_offset), 'end': (item['end'] + chunk_offset + clip_offset), 'score': item['score']} for item in transcription_result['word_segments']]
                            }
                            writer.write(json.dumps(res) + '\n')
                            writer.flush()

                            batch_num += 1
                            """

def diarize(
        audio_chunk,
        epoch,
        batch_num,
        pipeline,
        max_speakers: int = 4
    ):
    
    wav = torch.tensor(audio_chunk).unsqueeze(0)   # To get the correct format of (channel, time) Tensor.
    diarization_walltime_start = perf_counter()
    diarization_cputime_start = process_time()
    with ProgressHook() as hook:
        output = pipeline.model({
                'waveform': wav,
                'sample_rate': 16000
            },
            hook=hook,
            max_speakers=max_speakers
        )
        diarization_walltime = perf_counter() - diarization_walltime_start
        diarization_cputime = process_time() - diarization_cputime_start
        logger.critical('Epoch: %i, Batch: %i Walltime: %f CPU time: %f', epoch, batch_num, diarization_walltime, diarization_cputime)
        logger.info(
            "Diarization Module... Epoch: %i Batch: %i RSS: %.2f GB",
            epoch,
            batch_num,
            proc.memory_info().rss / (1024**3)
        )

        return output

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
        vad_filter=vad_filter,
        clip_timestamps=clip_timestamps,
        batch_size=batch_size,
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
                    batch_size=8,
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

                audio = sample['audio']
                audio_id = sample['audio_id']


                seg_ids = id_segment_map.get(audio_id, None)
                if seg_ids is None:
                    raise Exception('Audio id %s had an empty list of segments', audio_id)
                
                segments = []
                audio_offset = 0
                for seg in seg_ids:
                    offset = id_offset_map.get(seg)
                    if offset == None:
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