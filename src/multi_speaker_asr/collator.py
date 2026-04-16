

class Collator:
    
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        audio = [item["audio"] for item in batch]
        transcript = [item["text"] for item in batch]

        inputs = self.processor(
            audio,
            padding=True,
            return_tensors="pt",
            sampling_rate=16000
        ).input_features

        labels = self.processor(
            transcript,
            return_tensors="pt",
            padding=True
        ).input_ids

        return {
            "input_features": inputs,
            "labels": labels
        }