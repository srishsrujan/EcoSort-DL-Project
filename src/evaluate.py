from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from src.config import CHECKPOINT_DIR, FIGURE_DIR, METRIC_DIR, PROCESSED_DIR, ensure_dirs, load_train_config
from src.data.dataset import ManifestImageDataset
from src.models.networks import build_model, load_checkpoint
from src.utils import atomic_write_json, seed_everything

CLASS_NAMES = ["dry", "wet", "recyclable", "e-waste"]


def save_confusion_matrix(cm: np.ndarray, name: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, interpolation="nearest")
    ax.set_title(f"{name.title()} confusion matrix")
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=35, ha="right")
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{name}_confusion_matrix.png", dpi=160)
    plt.close(fig)


def evaluate_model(kind: str, config, augmentation: str = "none", artifact_name: str | None = None) -> dict:
    ensure_dirs()
    seed_everything(config.seed)
    manifest_path = PROCESSED_DIR / "manifest.csv"
    artifact_name = artifact_name or kind
    checkpoint_path = CHECKPOINT_DIR / f"{artifact_name}_best.pth"
    if not manifest_path.exists():
        raise FileNotFoundError("Missing dataset manifest. Run dataset preparation first.")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    ds = ManifestImageDataset(manifest_path, "test", config.image_size, "none")
    loader = DataLoader(
        ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.device == "cuda",
    )
    model = build_model(kind, num_classes=len(CLASS_NAMES), pretrained=False, freeze_backbone=(kind == "transfer")).to(config.device)
    payload = load_checkpoint(model, str(checkpoint_path), config.device)
    model.eval()

    all_targets: list[int] = []
    all_preds: list[int] = []
    all_probs: list[list[float]] = []
    all_paths: list[str] = []

    offset = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(config.device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            batch_size = images.size(0)
            batch_paths = ds.manifest.iloc[offset:offset + batch_size]["path"].tolist()
            offset += batch_size
            all_targets.extend(targets.tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
            all_paths.extend(batch_paths)

    report_dict = classification_report(
        all_targets,
        all_preds,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(len(CLASS_NAMES))))
    save_confusion_matrix(cm, artifact_name)

    predictions = []
    for path, target, pred, probs in zip(all_paths, all_targets, all_preds, all_probs):
        predictions.append(
            {
                "path": path,
                "true_class": CLASS_NAMES[target],
                "predicted_class": CLASS_NAMES[pred],
                "correct": bool(target == pred),
                "confidence": float(max(probs)),
                "top_1": CLASS_NAMES[int(np.argmax(probs))],
                "top_2": CLASS_NAMES[int(np.argsort(probs)[-2])],
                "top_3": CLASS_NAMES[int(np.argsort(probs)[-3])],
            }
        )
    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv(METRIC_DIR / f"{artifact_name}_predictions.csv", index=False)

    metrics = {
        "model_kind": kind,
        "artifact_name": artifact_name,
        "checkpoint": str(checkpoint_path),
        "test_accuracy": float(np.mean(np.array(all_targets) == np.array(all_preds))),
        "macro_precision": float(precision_score(all_targets, all_preds, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(all_targets, all_preds, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(all_targets, all_preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(all_targets, all_preds, average="weighted", zero_division=0)),
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "num_test_images": len(all_targets),
        "augmentation_used_for_training": augmentation,
        "checkpoint_payload": {k: v for k, v in payload.items() if k != "state_dict"},
    }
    atomic_write_json(METRIC_DIR / f"{artifact_name}_metrics.json", metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["baseline", "transfer"], default="baseline")
    args = parser.parse_args()
    config = load_train_config()
    evaluate_model(args.model, config)


if __name__ == "__main__":
    main()
