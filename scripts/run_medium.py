from datasets import load_dataset
from routing import medium
import pandas as pd

dataset = load_dataset("glue", "sst2", split="validation")

records = []

for example in dataset:
    text = example["sentence"]
    true_label = example["label"]

    results = medium(text)[0]
    best = max(results, key=lambda x: x["score"])
    pred_label_str = best["label"].upper()

    if pred_label_str in ["POSITIVE", "LABEL_1"]:
        pred = 1
    elif pred_label_str in ["NEGATIVE", "LABEL_0"]:
        pred = 0
    else:
        pred = 0

    records.append({
        "true_label": true_label,
        "pred_label": pred
    })

pd.DataFrame(records).to_csv("results/medium_results.csv", index=False)

print("DONE MEDIUM")