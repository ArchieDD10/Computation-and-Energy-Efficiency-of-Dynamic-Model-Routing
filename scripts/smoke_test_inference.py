import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAMES = [
    "distilbert-base-uncased-finetuned-sst-2-english",
    "textattack/bert-base-uncased-SST-2",
    "siebert/sentiment-roberta-large-english"
]

TEXTS = [
    "This movie was amazing.",
    "The service was terrible and slow.",
    "I need help scheduling my homework and studying."
]

def fGetConfidence(oModel, oTokenizer, sText):
    oInputs = oTokenizer(sText, return_tensors="pt", truncation=True)
    with torch.no_grad():
        oOutputs = oModel(**oInputs)
    tProbs = torch.softmax(oOutputs.logits, dim=1)
    dConfidence = torch.max(tProbs).item()
    return dConfidence, tProbs.squeeze().tolist()

def main():
    for sModelName in MODEL_NAMES:
        print(f"\n=== {sModelName} ===")
        oTokenizer = AutoTokenizer.from_pretrained(sModelName)
        oModel = AutoModelForSequenceClassification.from_pretrained(sModelName)
        oModel.eval()

        for sText in TEXTS:
            dConf, aProbs = fGetConfidence(oModel, oTokenizer, sText)
            print(f"Text: {sText}")
            print(f"Confidence: {dConf:.4f}")
            print(f"Probs: {aProbs}")

if __name__ == "__main__":
    main()
