from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACT_DIR = ROOT_DIR / "artifacts"
CHECKPOINT_DIR = ARTIFACT_DIR / "checkpoints"
METRIC_DIR = ARTIFACT_DIR / "metrics"
FIGURE_DIR = ARTIFACT_DIR / "figures"
REPORT_DIR = ROOT_DIR / "reports"
CONFIG_DIR = ROOT_DIR / "configs"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CLASSES = ["dry", "wet", "recyclable", "e-waste"]


def ensure_dirs() -> None:
    for path in (
        RAW_DIR,
        PROCESSED_DIR,
        CHECKPOINT_DIR,
        METRIC_DIR,
        FIGURE_DIR,
        REPORT_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 2
    epochs: int = 8
    lr_baseline: float = 1e-3
    lr_transfer: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 2
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    test_fraction: float = 0.15

    @property
    def device(self) -> str:
        forced = os.getenv("ECOSORT_DEVICE")
        if forced:
            return forced
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def validate(self) -> None:
        if abs(self.train_fraction + self.val_fraction + self.test_fraction - 1.0) > 1e-6:
            raise ValueError("Train/validation/test fractions must sum to 1.0")
        if self.image_size < 64:
            raise ValueError("image_size must be at least 64")
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")


def load_train_config(path: Path | None = None) -> TrainConfig:
    path = path or (CONFIG_DIR / "training_config.json")
    raw = load_json(path)
    return TrainConfig(
        seed=int(raw.get("seed", 42)),
        image_size=int(raw.get("image_size", 224)),
        batch_size=int(raw.get("batch_size", 32)),
        num_workers=int(raw.get("num_workers", 2)),
        epochs=int(raw.get("epochs", 8)),
        lr_baseline=float(raw.get("learning_rate_baseline", 1e-3)),
        lr_transfer=float(raw.get("learning_rate_transfer", 3e-4)),
        weight_decay=float(raw.get("weight_decay", 1e-4)),
        patience=int(raw.get("patience", 2)),
        train_fraction=float(raw.get("train_fraction", 0.70)),
        val_fraction=float(raw.get("val_fraction", 0.15)),
        test_fraction=float(raw.get("test_fraction", 0.15)),
    )
