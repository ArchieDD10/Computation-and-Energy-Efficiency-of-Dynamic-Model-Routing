from datasets import load_dataset
from routing import EscalationRouter, small, medium, large
import pandas as pd

dataset = load_dataset("glue", "sst2", split="validation")

router = EscalationRouter(small, medium, large, tau_small=0.80, tau_med=0.85)

records = []

for example in dataset:
    text = example["sentence"]
    label = example["label"]  # 0 = negative, 1 = positive
    
    result = router.route(text)
    
    records.append({
        "text": text,
        "true_label": label,
        "pred_label": 1 if result["final"]["label"] == "POSITIVE" else 0,
        "chosen_model": result["final"]["chosen_model"],
        "confidence": result["final"]["confidence"],
        "num_models_used": len(result["decisions"]),
        "total_latency_ms": sum(d["latency_ms"] for d in result["decisions"]),
    })

df = pd.DataFrame(records)
df.to_csv("results/baseline_results.csv", index=False)

print("Done.")
