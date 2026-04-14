import pandas as pd

def acc(df):
    return (df.true_label == df.pred_label).mean()

small = pd.read_csv("results/small_results.csv")
medium = pd.read_csv("results/medium_results.csv")
large = pd.read_csv("results/large_results.csv")
router = pd.read_csv("results/baseline_results.csv")

print("Small:", acc(small))
print("Medium:", acc(medium))
print("Large:", acc(large))
print("Router:", acc(router))