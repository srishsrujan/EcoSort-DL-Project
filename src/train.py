from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn, optim
from torch.utils.data import DataLoader

from src.config import CHECKPOINT_DIR, FIGURE_DIR, METRIC_DIR, PROCESSED_DIR, ensure_dirs, load_train_config
from src.data.dataset import ManifestImageDataset
from src.models.networks import build_model
from src.utils import atomic_write_json, seed_everything

CLASS_NAMES = ["dry", "wet", "recyclable", "e-waste"]


def class_weights_from_manifest(manifest_path: Path) -> torch.Tensor:
    df = pd.read_csv(manifest_path)
    train_counts = df[df["split"] == "train"]["target_class"].value_counts()
    total = train_counts.sum()
    weights = []
    for name in CLASS_NAMES:
        count = max(int(train_counts.get(name, 1)), 1)
        weights.append(total / (len(CLASS_NAMES) * count))
    return torch.tensor(weights, dtype=torch.float32)


def epoch_pass(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    all_targets: list[int] = []
    all_preds: list[int] = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        if training:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        all_targets.extend(targets.detach().cpu().tolist())
        all_preds.extend(logits.argmax(dim=1).detach().cpu().tolist())

    mean_loss = total_loss / len(loader.dataset)
    accuracy = float(np.mean(np.array(all_targets) == np.array(all_preds)))
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
    return mean_loss, accuracy, macro_f1


def plot_history(history: list[dict], name: str) -> None:
    df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(df["epoch"], df["train_loss"], label="train")
    axes[0].plot(df["epoch"], df["val_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(df["epoch"], df["train_accuracy"], label="train")
    axes[1].plot(df["epoch"], df["val_accuracy"], label="validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{name}_training_curves.png", dpi=160)
    plt.close(fig)


def train_model(kind: str, augmentation: str, pretrained: bool, config, manifest_path: Path, artifact_name: str | None = None) -> dict:
    seed_everything(config.seed)
    device = config.device
    ensure_dirs()

    train_ds = ManifestImageDataset(manifest_path, "train", config.image_size, augmentation)
    val_ds = ManifestImageDataset(manifest_path, "val", config.image_size, "none")
    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": device == "cuda",
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    model = build_model(
        kind,
        num_classes=len(CLASS_NAMES),
        pretrained=pretrained,
        freeze_backbone=(kind == "transfer"),
    ).to(device)
    weights = class_weights_from_manifest(manifest_path).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    lr = config.lr_baseline if kind == "baseline" else config.lr_transfer
    optimizer = optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=lr,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

    best_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    history: list[dict] = []
    artifact_name = artifact_name or kind
    checkpoint_path = CHECKPOINT_DIR / f"{artifact_name}_best.pth"

    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc, train_f1 = epoch_pass(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc, val_f1 = epoch_pass(model, val_loader, criterion, device)
        scheduler.step(val_f1)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "train_macro_f1": train_f1,
            "val_macro_f1": val_f1,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(json.dumps(row))

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_kind": kind,
                    "class_names": CLASS_NAMES,
                    "image_size": config.image_size,
                    "augmentation": augmentation,
                    "best_val_macro_f1": best_f1,
                    "epoch": epoch,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs > config.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    history_path = METRIC_DIR / f"{artifact_name}_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    plot_history(history, artifact_name)

    result = {
        "model_kind": kind,
        "artifact_name": artifact_name,
        "augmentation": augmentation,
        "pretrained": bool(pretrained),
        "device": device,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "checkpoint": str(checkpoint_path),
        "history": history,
    }
    atomic_write_json(METRIC_DIR / f"{artifact_name}_train_summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["baseline", "transfer"], default="baseline")
    parser.add_argument("--augmentation", choices=["none", "standard", "strong"], default="standard")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--no-pretrained", action="store_true", help="Smoke-test option; not a transfer-learning experiment.")
    parser.add_argument("--artifact-name", type=str, default=None)
    args = parser.parse_args()

    config = load_train_config()
    if args.epochs is not None:
        config = config.__class__(**{**config.__dict__, "epochs": args.epochs})
    if args.image_size is not None:
        config = config.__class__(**{**config.__dict__, "image_size": args.image_size})
    if args.batch_size is not None:
        config = config.__class__(**{**config.__dict__, "batch_size": args.batch_size})
    config.validate()
    manifest_path = PROCESSED_DIR / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError("Run python -m src.data.prepare_dataset first.")
    train_model(args.model, args.augmentation, not args.no_pretrained, config, manifest_path, args.artifact_name)


if __name__ == "__main__":
    main()
