import torch
from multi_speaker_asr.collator import Collator
from multi_speaker_asr.data import Data
from torch.utils.data import DataLoader
from torchmetrics.text import WordErrorRate, CharErrorRate
from multi_speaker_asr.utils.utils import compute_cosine_sim
from speechbrain.utils.metric_stats import MetricStats


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
        batch_size=4,
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

            #print("AFTER RUNNING GENERATE FUNCTION...")
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
