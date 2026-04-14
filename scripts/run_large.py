from datasets import load_dataset
from routing import large
import pandas as pd

dataset = load_dataset("glue", "sst2", split="validation")

records = []

for example in dataset:
    text = example["sentence"]
    label = example["label"]

    results = large(text)[0]
    best = max(results, key=lambda x: x["score"])
    pred = 1 if best["label"] == "POSITIVE" else 0

    records.append({
        "true_label": label,
        "pred_label": pred
    })

pd.DataFrame(records).to_csv("results/large_results.csv", index=False)

print("DONE LARGE")