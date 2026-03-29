import os
import time
import json
import math
from typing import Dict, Any, List

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Force local/offline behavior (no network calls)
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def normalize_label(sLabel: str) -> str:
    s = sLabel.upper()
    if s in ["LABEL_1", "POSITIVE"]:
        return "POSITIVE"
    if s in ["LABEL_0", "NEGATIVE"]:
        return "NEGATIVE"
    return sLabel


def folder_size_bytes(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def softmax_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=-1)


def entropy(probs: torch.Tensor) -> float:
    # entropy = -sum(p * log(p))
    p = probs.clamp(min=1e-12)
    return float(-(p * torch.log(p)).sum().item())


def top2_margin(probs: torch.Tensor) -> float:
    vals, _ = torch.topk(probs, k=min(2, probs.numel()))
    if vals.numel() < 2:
        return 0.0
    return float((vals[0] - vals[1]).item())


def device_sync(device: str) -> None:
    # Proper sync for accurate timing
    if device == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def measure_latency_ms(model, inputs: Dict[str, torch.Tensor], device: str, runs: int = 10, warmup: int = 3) -> float:
    model.eval()

    # warmup (not timed)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(**inputs)
        device_sync(device)

    # timed runs
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(**inputs)
        device_sync(device)
    t1 = time.perf_counter()

    return (t1 - t0) * 1000.0 / runs


def inspect_one_model(local_path: str, text: str, device: str = "mps") -> Dict[str, Any]:
    if not os.path.isdir(local_path):
        raise FileNotFoundError(f"Local model folder not found: {local_path}")

    tok = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(local_path, local_files_only=True)

    # Move to device
    if device == "mps" and torch.backends.mps.is_available():
        mdl = mdl.to("mps")
        tensor_device = "mps"
    elif device.startswith("cuda") and torch.cuda.is_available():
        mdl = mdl.to("cuda")
        tensor_device = "cuda"
    else:
        tensor_device = "cpu"

    inputs = tok(text, return_tensors="pt", truncation=True)
    inputs = {k: v.to(tensor_device) for k, v in inputs.items()}

    # Single forward for logits/probs/confidence
    mdl.eval()
    with torch.no_grad():
        out = mdl(**inputs)
    logits = out.logits[0].detach().to("cpu")
    probs = softmax_probs(logits)

    conf_val, pred_idx = torch.max(probs, dim=-1)

    # label mapping
    raw_label = None
    if hasattr(mdl.config, "id2label") and isinstance(mdl.config.id2label, dict) and len(mdl.config.id2label) > 0:
        raw_label = mdl.config.id2label.get(int(pred_idx), str(int(pred_idx)))
    else:
        raw_label = str(int(pred_idx))
    norm_label = normalize_label(str(raw_label))

    # latency
    avg_ms = measure_latency_ms(mdl, inputs, device=tensor_device, runs=10, warmup=3)

    # params + config
    total_params = sum(p.numel() for p in mdl.parameters())
    trainable_params = sum(p.numel() for p in mdl.parameters() if p.requires_grad)

    # config fields differ across architectures; pull safely
    cfg = mdl.config
    cfg_layers = getattr(cfg, "num_hidden_layers", None)
    cfg_hidden = getattr(cfg, "hidden_size", None)
    cfg_labels = getattr(cfg, "num_labels", None)
    cfg_arch = getattr(cfg, "architectures", None)

    disk_bytes = folder_size_bytes(local_path)

    # nice view lists
    logits_list = [float(x) for x in logits.tolist()]
    probs_list = [float(x) for x in probs.tolist()]

    return {
        "local_path": local_path,
        "device_used": tensor_device,
        "text": text,
        "pred_index": int(pred_idx),
        "pred_label_raw": raw_label,
        "pred_label_normalized": norm_label,
        "confidence_maxprob": float(conf_val),
        "entropy": entropy(probs),
        "margin_top1_top2": top2_margin(probs),
        "logits": logits_list,
        "probs": probs_list,
        "avg_latency_ms": float(avg_ms),
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "config": {
            "num_hidden_layers": cfg_layers,
            "hidden_size": cfg_hidden,
            "num_labels": cfg_labels,
            "architectures": cfg_arch,
            "model_type": getattr(cfg, "model_type", None),
        },
        "disk_size_mb": round(disk_bytes / (1024 * 1024), 2),
    }


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_models = os.path.join(repo_root, "local_models")

    models = [
        os.path.join(local_models, "distilbert-base-uncased-finetuned-sst-2-english"),
        os.path.join(local_models, "textattack__bert-base-uncased-SST-2"),
        os.path.join(local_models, "siebert__sentiment-roberta-large-english"),
    ]

    text = "The movie was not bad."

    results: List[Dict[str, Any]] = []
    for p in models:
        print("\n==============================")
        print("Inspecting:", p)
        r = inspect_one_model(p, text, device="mps")
        results.append(r)

        print("label:", r["pred_label_normalized"], "| conf:", r["confidence_maxprob"])
        print("entropy:", r["entropy"], "| margin:", r["margin_top1_top2"])
        print("avg_latency_ms:", r["avg_latency_ms"], "| params:", r["total_params"], "| disk_mb:", r["disk_size_mb"])
        print("logits:", r["logits"])
        print("probs :", r["probs"])

    out_path = os.path.join(repo_root, "results", "model_stats.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved:", out_path)


if __name__ == "__main__":
    main()