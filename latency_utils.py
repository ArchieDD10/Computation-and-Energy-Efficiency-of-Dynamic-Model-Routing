import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ModelWrapper:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(DEVICE)
        self.model.eval()

    def infer(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(DEVICE)

        start = time.time()
        with torch.no_grad():
            outputs = self.model(**inputs)
        end = time.time()

        latency_ms = (end - start) * 1000

        probs = torch.softmax(outputs.logits, dim=-1)
        confidence = torch.max(probs).item()

        return latency_ms, confidence