import os
from huggingface_hub import snapshot_download

MODELS = [
    "distilbert-base-uncased-finetuned-sst-2-english",
    "textattack/bert-base-uncased-SST-2",
    "siebert/sentiment-roberta-large-english",
]

def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_dir = os.path.join(repo_root, "local_models")
    os.makedirs(local_dir, exist_ok=True)

    for m in MODELS:
        target = os.path.join(local_dir, m.replace("/", "__"))
        print(f"Downloading {m} -> {target}")
        snapshot_download(
            repo_id=m,
            local_dir=target,
            local_dir_use_symlinks=False,
        )
        print("Done.")

if __name__ == "__main__":
    main()