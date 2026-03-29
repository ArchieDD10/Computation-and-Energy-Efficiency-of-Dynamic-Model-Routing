import pandas as pd

df = pd.read_csv("results/baseline_results.csv")

accuracy = (df.true_label == df.pred_label).mean()
avg_latency = df.total_latency_ms.mean()
model_usage = df.chosen_model.value_counts(normalize=True)

print("Accuracy:", accuracy)
print("Average latency (ms):", avg_latency)
print("\nModel usage distribution:")
print(model_usage)