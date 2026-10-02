from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt

from src.config import CHECKPOINT_DIR, FIGURE_DIR, PROCESSED_DIR, load_train_config
from src.data.dataset import build_transform
from src.models.networks import build_model, load_checkpoint

CLASS_NAMES = ["dry", "wet", "recyclable", "e-waste"]


def _target_layer(model, kind: str):
    if kind == "transfer":
        return model.backbone.features[-1]
    return model.features[-3]


def gradcam(model, input_tensor: torch.Tensor, target_index: int, kind: str, device: str) -> np.ndarray:
    model.eval()
    activations = []
    gradients = []
    layer = _target_layer(model, kind)

    def forward_hook(_, __, output):
        activations.append(output.detach())

    def backward_hook(_, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    h1 = layer.register_forward_hook(forward_hook)
    h2 = layer.register_full_backward_hook(backward_hook)
    try:
        model.zero_grad(set_to_none=True)
        output = model(input_tensor)
        score = output[0, target_index]
        score.backward()
        acts = activations[0]
        grads = gradients[0]
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1).squeeze(0)
        cam = torch.relu(cam)
        cam -= cam.min()
        cam /= cam.max().clamp_min(1e-8)
        return cam.cpu().numpy()
    finally:
        h1.remove()
        h2.remove()


def overlay_heatmap(image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    width, height = image.size
    heatmap = Image.fromarray(np.uint8(cam * 255), mode="L").resize((width, height))
    cmap = plt.get_cmap("jet")
    heat_rgb = (cmap(np.asarray(heatmap) / 255.0)[..., :3] * 255).astype(np.uint8)
    heat_image = Image.fromarray(heat_rgb, mode="RGB")
    enhanced = ImageEnhance.Contrast(image).enhance(1.1)
    return Image.blend(enhanced, heat_image, alpha)


def explain_image(image: Image.Image, model, kind: str, image_size: int, device: str, target_index: int) -> Image.Image:
    original = image.convert("RGB")
    transform = build_transform(image_size, train=False, augmentation="none")
    tensor = transform(original).unsqueeze(0).to(device)
    tensor.requires_grad_(True)
    cam = gradcam(model, tensor, target_index, kind, device)
    return overlay_heatmap(original, cam)


def generate_sample() -> None:
    config = load_train_config()
    manifest = Path(PROCESSED_DIR) / "manifest.csv"
    import pandas as pd
    df = pd.read_csv(manifest)
    row = df[df["split"] == "test"].iloc[0]
    image = Image.open(row["path"]).convert("RGB")
    model = build_model("transfer", len(CLASS_NAMES), pretrained=False, freeze_backbone=True).to(config.device)
    load_checkpoint(model, str(CHECKPOINT_DIR / "transfer_best.pth"), config.device)
    transform = build_transform(config.image_size, train=False, augmentation="none")
    tensor = transform(image).unsqueeze(0).to(config.device)
    with torch.no_grad():
        pred = int(model(tensor).argmax(dim=1).item())
    result = explain_image(image, model, "transfer", config.image_size, config.device, pred)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    result.save(FIGURE_DIR / f"gradcam_{row['dataset_index']}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-sample", action="store_true")
    args = parser.parse_args()
    if args.generate_sample:
        generate_sample()
