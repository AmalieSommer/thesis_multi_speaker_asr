class Collator:
    
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):

        audio = [item["audio"].squeeze(0).numpy() for item in batch]
        transcript = [item["text"] for item in batch]

        inputs = self.processor(
            audio,
            padding="max_length",
            return_tensors="pt",
            return_attention_mask=True,
            sampling_rate=16000
        )

        labels = self.processor(
            text=transcript,
            return_tensors="pt",
            padding=True
        ).input_ids

        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "input_features": inputs.input_features,
            "labels": labels,
            "attention_mask": inputs.attention_mask 
        }