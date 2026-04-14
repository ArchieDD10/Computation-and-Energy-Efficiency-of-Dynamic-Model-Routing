from datasets import load_dataset
from routing import small
import pandas as pd

dataset = load_dataset("glue", "sst2", split="validation")

records = []

for example in dataset:
    text = example["sentence"]
    label = example["label"]
    results = small(text)[0]  # list of predictions
    best = max(results, key=lambda x: x["score"])
    pred = 1 if best["label"] == "POSITIVE" else 0

    records.append({
        "true_label": label,
        "pred_label": pred
    })

pd.DataFrame(records).to_csv("results/small_results.csv", index=False)

print("DONE SMALL")