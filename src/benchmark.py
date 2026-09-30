from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image

from src.config import CHECKPOINT_DIR, METRIC_DIR, PROCESSED_DIR, load_train_config
from src.data.dataset import build_transform
from src.models.networks import build_model, load_checkpoint

CLASS_NAMES = ["dry", "wet", "recyclable", "e-waste"]


def benchmark(checkpoint_path: Path, kind: str, repeats: int = 30) -> dict:
    config = load_train_config()
    manifest_path = PROCESSED_DIR / "manifest.csv"
    import pandas as pd

    df = pd.read_csv(manifest_path)
    test_row = df[df["split"] == "test"].iloc[0]
    image = Image.open(test_row["path"]).convert("RGB")
    tensor = build_transform(config.image_size, train=False, augmentation="none")(image).unsqueeze(0).to(config.device)
    model = build_model(kind, len(CLASS_NAMES), pretrained=False, freeze_backbone=(kind == "transfer")).to(config.device)
    load_checkpoint(model, str(checkpoint_path), config.device)
    model.eval()

    with torch.no_grad():
        for _ in range(5):
            _ = model(tensor)
        if config.device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            _ = model(tensor)
        if config.device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    result = {
        "model_kind": kind,
        "device": config.device,
        "repeats": repeats,
        "total_seconds": elapsed,
        "milliseconds_per_image": (elapsed / repeats) * 1000.0,
        "images_per_second": repeats / elapsed if elapsed > 0 else None,
        "note": "Single-image forward-pass benchmark; includes model inference only, not camera capture/network latency.",
    }
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    with (METRIC_DIR / "benchmark.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["baseline", "transfer"], default="transfer")
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    benchmark(CHECKPOINT_DIR / f"{args.model}_best.pth", args.model, args.repeats)


if __name__ == "__main__":
    main()
