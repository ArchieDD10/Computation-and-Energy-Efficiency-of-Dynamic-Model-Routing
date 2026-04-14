import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def get_device():
    """Auto-detect best available device"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


DEVICE = get_device()
print(f"Device set to use {DEVICE}")


def normalize_label(sLabel: str) -> str:
    s = sLabel.upper()
    if s in ["LABEL_1", "POSITIVE"]:
        return "POSITIVE"
    if s in ["LABEL_0", "NEGATIVE"]:
        return "NEGATIVE"
    return sLabel


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL = os.path.join(REPO_ROOT, "local_models")

small_path = os.path.join(LOCAL, "distilbert-base-uncased-finetuned-sst-2-english")
medium_path = os.path.join(LOCAL, "textattack__bert-base-uncased-SST-2")
large_path = os.path.join(LOCAL, "siebert__sentiment-roberta-large-english")


def make_local_pipe(model_dir: str, device: str = None):
    if device is None:
        device = DEVICE
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    return pipeline(
        "text-classification",
        model=mdl,
        tokenizer=tok,
        top_k=None,
        device=device,
        truncation=True,
        max_length = 128
    )


small = make_local_pipe(small_path, device=DEVICE)
medium = make_local_pipe(medium_path, device=DEVICE)
large = make_local_pipe(large_path, device=DEVICE)


class EscalationRouter:
    def __init__(self, small_pipe, medium_pipe, large_pipe, tau_small: float = 0.80, tau_med: float = 0.85):
        self.tau_small = tau_small
        self.tau_med = tau_med
        self.small = small_pipe
        self.medium = medium_pipe
        self.large = large_pipe
    def _predict_with_conf(self, pipe, text: str) -> dict:
        t0 = time.perf_counter()
        scores = pipe(text)[0]  # list of {label, score}
        latency_ms = (time.perf_counter() - t0) * 1000.0

        best = max(scores, key=lambda d: d["score"])
        best_label = normalize_label(best["label"])

        return {
            "label": best_label,
            "confidence": float(best["score"]),
            "all_scores": scores,
            "latency_ms": float(latency_ms),
        }

    def route(self, text: str) -> dict:
        record = {"text": text, "decisions": [], "final": None}

        # small
        r1 = self._predict_with_conf(self.small, text)
        record["decisions"].append({"model": "small", "tau": self.tau_small, **r1})
        if r1["confidence"] >= self.tau_small:
            record["final"] = {
                "chosen_model": "small",
                "label": r1["label"],
                "confidence": r1["confidence"],
            }
            return record

        # medium
        r2 = self._predict_with_conf(self.medium, text)
        record["decisions"].append({"model": "medium", "tau": self.tau_med, **r2})
        if r2["confidence"] >= self.tau_med:
            record["final"] = {
                "chosen_model": "medium",
                "label": r2["label"],
                "confidence": r2["confidence"],
            }
            return record

        # large fallback
        r3 = self._predict_with_conf(self.large, text)
        record["decisions"].append({"model": "large", "tau": None, **r3})
        record["final"] = {
            "chosen_model": "large",
            "label": r3["label"],
            "confidence": r3["confidence"],
        }
        return record


if __name__ == "__main__":
    print("START routing.py")

    router = EscalationRouter(small, medium, large, tau_small=0.90, tau_med=0.95)    
    text = "The movie was not bad."
    result = router.route(text)

    print("FINAL:", result["final"])
    print("DECISIONS:")
    for d in result["decisions"]:
        print(d["model"], d["label"], d["confidence"], f"{d['latency_ms']:.1f}ms")

    total_ms = sum(d["latency_ms"] for d in result["decisions"])
    print("TOTAL_MS:", round(total_ms, 1))
    print("NUM_MODELS_USED:", len(result["decisions"]))
    print("END routing.py")
