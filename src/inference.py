from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from src.data.dataset import build_transform
from src.models.networks import build_model, load_checkpoint


@dataclass
class Prediction:
    label: str
    confidence: float
    alternatives: list[tuple[str, float]]


class WasteClassifier:
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        payload = torch.load(checkpoint_path, map_location=device)
        self.class_names = payload.get("class_names", ["dry", "wet", "recyclable", "e-waste"])
        self.image_size = int(payload.get("image_size", 224))
        self.model_kind = payload.get("model_kind", "transfer")
        self.device = device
        self.model = build_model(
            self.model_kind,
            len(self.class_names),
            pretrained=False,
            freeze_backbone=(self.model_kind == "transfer"),
        ).to(device)
        load_checkpoint(self.model, checkpoint_path, device)
        self.model.eval()
        self.transform = build_transform(self.image_size, train=False, augmentation="none")

    def predict(self, image: Image.Image, top_k: int = 3) -> Prediction:
        image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probabilities = torch.softmax(self.model(tensor), dim=1)[0].cpu().numpy()
        order = np.argsort(probabilities)[::-1][: max(1, top_k)]
        top = [(self.class_names[int(i)], float(probabilities[int(i)])) for i in order]
        return Prediction(label=top[0][0], confidence=top[0][1], alternatives=top[1:])
