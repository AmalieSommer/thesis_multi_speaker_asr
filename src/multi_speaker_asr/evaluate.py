import torch
from multi_speaker_asr.collator import Collator
from multi_speaker_asr.data import Data
from torch.utils.data import DataLoader
from torchmetrics.text import WordErrorRate, CharErrorRate
from multi_speaker_asr.utils.utils import compute_cosine_sim
from speechbrain.utils.metric_stats import MetricStats


def evaluate(model, processor, device, dataset):
    model.eval()

    wer = WordErrorRate()
    wer.to(device)
    cer = CharErrorRate()
    cer.to(device)

    semdist = MetricStats(metric=compute_cosine_sim)

    collator_fn = Collator(processor)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=24,
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

            outputs = model(
                input_features,
                labels
            )

            loss = outputs["loss"]
            total_loss += loss # Saves cross-entropy loss

            print("Batch loss: ", loss)

            pred_ids = model.model.generate(
                input_features=input_features,
                attention_mask=attention_mask,
                task="transcribe",
                language="da"
                )
            pred_transcripts = processor.batch_decode(
                pred_ids,
                skip_special_tokens=True
            )

            labels[labels == -100] = processor.tokenizer.pad_token_id
            transcripts = processor.batch_decode(
                labels,
                skip_special_tokens=True
            )

            wer.update(preds=pred_transcripts, target=transcripts)
            cer.update(preds=pred_transcripts, target=transcripts)

            semdist.append(ids=ids, preds=pred_transcripts, targets=transcripts)

            # Save predicted and actual transcripts for later check:
            all_predictions.append(pred_transcripts)
            all_transcripts.append(transcripts)


    avg_loss = total_loss / len(dataloader)
    wer_final = wer.compute()
    cer_final = cer.compute()
    semdist_avg = semdist.summarize()

    return {
        "loss": avg_loss.item(),
        "wer": wer_final.item(),
        "cer": cer_final.item(),
        "semdist": semdist_avg,
        "predictions": all_predictions,
        "ground_truths": all_transcripts 
    }
