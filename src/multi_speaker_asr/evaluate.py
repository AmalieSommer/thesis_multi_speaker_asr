import torch
from multi_speaker_asr.collator import Collator
from multi_speaker_asr.data import Data
from torch.utils.data import DataLoader
from torchmetrics.text import WordErrorRate, CharErrorRate

def evaluate(model, processor, device, dataset):
    model.eval()

    wer = WordErrorRate()
    wer.to(device)
    cer = CharErrorRate()
    cer.to(device)

    collator_fn = Collator(processor)
    """
    dataset = Data(
        data_path="data/lillelyd-main",
        metadata="manifest_test.jsonl"
    )
    """
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=8,
        collate_fn=collator_fn
    )

    total_loss = 0
    all_predictions = []
    all_transcripts = []

    with torch.no_grad():
        for batch in dataloader:
            input_features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_features,
                labels
            )

            total_loss += outputs["loss"] # Saves cross-entropy loss

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

            # Save predicted and actual transcripts for later check:
            all_predictions.append(pred_transcripts)
            all_transcripts.append(transcripts)


    avg_loss = total_loss / len(dataloader)
    wer_final = wer.compute()
    cer_final = cer.compute()

    return {
        "loss": avg_loss.item(),
        "wer": wer_final.item(),
        "cer": cer_final.item(),
        "predictions": all_predictions,
        "ground_truths": all_transcripts 
    }
