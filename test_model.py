import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

text = "This movie was absolutely amazing."

inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

logits = outputs.logits
probs = torch.softmax(logits, dim=1)

print("Logits:", logits)
print("Probabilities:", probs)
print("Confidence:", torch.max(probs))
def get_confidence(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    confidence = torch.max(probs).item()
    return confidence
