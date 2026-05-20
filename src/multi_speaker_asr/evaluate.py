import torch
from torch.utils.data import DataLoader
from torchmetrics.text import WordErrorRate, CharErrorRate
from utils.utils import compute_cosine_sim
from speechbrain.utils.metric_stats import MetricStats
import os
import tempfile
import json
import soundfile as sf
import io


def collator_fn(batch):
    audio = []
    text = []
    ids = []
    for item in batch:
        bytes = item["audio"]['bytes']
        audio_bytes = io.BytesIO(bytes)
        audio.append(audio_bytes)
        text.append(item['text'])
        ids.append(item['id_conversation'])

    return {
        'audio': audio,
        'text': text,
        'ids': ids
    }

def inference(whisper, dataset, bert):
    device = whisper.device

    wer = WordErrorRate()
    wer.to(device)
    cer = CharErrorRate()
    cer.to(device)

    all_predictions = []
    all_transcripts = []

    semdist = MetricStats(metric=compute_cosine_sim)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        collate_fn=collator_fn
    )

    for batch in dataloader:
        target_text = batch['text'][0]
        segments, _ = whisper.model.transcribe(batch['audio'][0], without_timestamps=True)
        segments = [item.text for item in segments]
        transcript = " ".join(segments)
        print(f'Pred: {transcript}, Target: {target_text}')

        wer.update(preds=transcript, target=target_text)
        cer.update(preds=transcript, target=target_text)

        encoded_pred_transcripts = bert.tokenizer(transcript, padding=True, truncation=True, return_tensors='pt')
        encoded_target_transcripts = bert.tokenizer(target_text, padding=True, truncation=True, return_tensors='pt')

        pred_transcripts_embeddings = bert(
            input_ids=encoded_pred_transcripts.input_ids.to(device),
            attention_mask=encoded_pred_transcripts.attention_mask.to(device)
        )
        transcripts_embeddings = bert(
            input_ids=encoded_target_transcripts.input_ids.to(device),
            attention_mask=encoded_target_transcripts.attention_mask.to(device)
        )
        semdist.append(ids=batch['ids'], pred_embeddings=pred_transcripts_embeddings, target_embeddings=transcripts_embeddings)

        # Save predicted and actual transcripts for later check:
        all_predictions.append(transcript)
        all_transcripts.append(target_text)


    wer_final = wer.compute()
    cer_final = cer.compute()
    semdist_avg = semdist.summarize()

    return {
        "wer": wer_final.item(),
        "cer": cer_final.item(),
        "semdist": semdist_avg,
        "predictions": all_predictions,
        "ground_truths": all_transcripts 
    }


def evaluate(whisper, dataset, bert):
    whisper.eval()

    device = whisper.device

    wer = WordErrorRate()
    wer.to(device)
    cer = CharErrorRate()
    cer.to(device)

    semdist = MetricStats(metric=compute_cosine_sim)
    collator_fn = Collator(whisper.processor)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=6,
        collate_fn=collator_fn
    )

    print("Dataloader created...")

    total_loss = 0
    all_predictions = []
    all_transcripts = []

    iter = 0

    with torch.no_grad():
        print("Starting batch evaluation...")
        for batch in dataloader:
            iter += 1

            print("Running batch iteration: ", iter)

            input_features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            ids = batch["ids"] # for calculating semdist metric using speechbrain library

            outputs = whisper(
                input_features=input_features, 
                labels=labels,
            )

            loss = outputs["loss"].item()
            total_loss += loss # Saves cross-entropy loss

            print("Batch loss: ", loss)

            pred_ids = whisper.generate(input_features=input_features, attention_mask=attention_mask)
            pred_transcripts = whisper.processor.batch_decode(
                pred_ids,
                skip_special_tokens=True
            )

            labels[labels == -100] = whisper.processor.tokenizer.pad_token_id
            transcripts = whisper.processor.batch_decode(
                labels,
                skip_special_tokens=True
            )


            with tempfile.TemporaryDirectory() as tempDir:
                path = os.path.join(tempDir, "whisper_output.jsonl")
                with open(path, "w") as f:
                    
                    for i in range(len(pred_transcripts)):
                        data = {
                            "id": ids[i],
                            "prediction": pred_transcripts[i],
                            "transcripts": transcripts[i]
                        }
                        json_str = json.dumps(data, indent=4)
                        f.write(json_str)

            wer.update(preds=pred_transcripts, target=transcripts)
            cer.update(preds=pred_transcripts, target=transcripts)

            encoded_pred_transcripts = bert.tokenizer(pred_transcripts, padding=True, truncation=True, return_tensors='pt')
            encoded_target_transcripts = bert.tokenizer(transcripts, padding=True, truncation=True, return_tensors='pt')

            pred_transcripts_embeddings = bert(
                input_ids=encoded_pred_transcripts.input_ids.to(device),
                attention_mask=encoded_pred_transcripts.attention_mask.to(device)
            )
            transcripts_embeddings = bert(
                input_ids=encoded_target_transcripts.input_ids.to(device),
                attention_mask=encoded_target_transcripts.attention_mask.to(device)
            )
            #print("AFTER RUNNING BERT AND GETTING THE SENTENCE EMBEDDINGS...")
            semdist.append(ids=ids, pred_embeddings=pred_transcripts_embeddings, target_embeddings=transcripts_embeddings)
            #print("AFTER CALCULATING SEMDIST METRIC...")

            # Save predicted and actual transcripts for later check:
            all_predictions.append(pred_transcripts)
            all_transcripts.append(transcripts)


    avg_loss = total_loss / len(dataloader)
    wer_final = wer.compute()
    cer_final = cer.compute()
    semdist_avg = semdist.summarize()

    return {
        "loss": avg_loss,
        "wer": wer_final.item(),
        "cer": cer_final.item(),
        "semdist": semdist_avg,
        "predictions": all_predictions,
        "ground_truths": all_transcripts 
    }

def compute_eval(bert, dataset):
    bert.eval()

    device = bert.device
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=24,
        shuffle=False,
        collate_fn=lambda item: item
    )

    with torch.no_grad():
        for batch in dataloader:
            ids = batch["ids"]
            pred_transcripts = batch["predictions"]
            transcripts = batch["transcripts"]

    return

def compute_transcripts(whisper, dataset):
    whisper.eval()

    device = whisper.device
    collator_fn = Collator(whisper.processor)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=4,
        collate_fn=collator_fn
    )

    print("Dataloader created...")

    total_loss = 0
    iter = 0

    with torch.no_grad():
        print("Starting batch evaluation...")
        for batch in dataloader:
            iter += 1

            print("Running batch iteration: ", iter)

            input_features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            ids = batch["ids"] # for calculating semdist metric using speechbrain library

            outputs = whisper(
                input_features=input_features, 
                labels=labels,
            )

            loss = outputs["loss"].item()
            total_loss += loss # Saves cross-entropy loss

            print("Batch loss: ", loss)

            pred_ids = whisper.generate(input_features=input_features, attention_mask=attention_mask)
            pred_transcripts = whisper.processor.batch_decode(
                pred_ids,
                skip_special_tokens=True
            )

            labels[labels == -100] = whisper.processor.tokenizer.pad_token_id
            transcripts = whisper.processor.batch_decode(
                labels,
                skip_special_tokens=True
            )


            with tempfile.TemporaryDirectory() as tempDir:
                path = os.path.join(tempDir, "whisper_output.jsonl")
                with open(path, "w") as f:
                    
                    for i in range(len(pred_transcripts)):
                        data = {
                            "ids": ids[i],
                            "predictions": pred_transcripts[i],
                            "transcripts": transcripts[i]
                        }
                        json_str = json.dumps(data, indent=4)
                        f.write(json_str)

        return path