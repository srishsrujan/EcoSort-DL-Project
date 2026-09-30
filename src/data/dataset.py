from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.config import DEFAULT_CLASSES


class ManifestImageDataset(Dataset):
    def __init__(self, manifest_path: Path, split: str, image_size: int = 224, augmentation: str = "standard"):
        self.manifest = pd.read_csv(manifest_path)
        self.manifest = self.manifest[self.manifest["split"] == split].reset_index(drop=True)
        if self.manifest.empty:
            raise ValueError(f"No rows for split={split}")
        self.class_names = DEFAULT_CLASSES
        self.class_to_idx = {name: index for index, name in enumerate(self.class_names)}
        self.transform = build_transform(image_size, train=(split == "train"), augmentation=augmentation)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int):
        row = self.manifest.iloc[index]
        path = Path(row["path"])
        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Failed to read image: {path}") from exc
        image = self.transform(image)
        label = self.class_to_idx[str(row["target_class"])]
        return image, label


def build_transform(image_size: int, train: bool, augmentation: str) -> transforms.Compose:
    common = [
        transforms.Resize((image_size, image_size)),
    ]
    if train and augmentation == "standard":
        common += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03),
        ]
    elif train and augmentation == "strong":
        common += [
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05),
            transforms.RandomPerspective(distortion_scale=0.15, p=0.25),
        ]
    common += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    return transforms.Compose(common)
