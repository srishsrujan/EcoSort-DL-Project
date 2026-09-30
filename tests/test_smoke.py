from pathlib import Path

import torch

from src.config import DEFAULT_CLASSES, ROOT_DIR, load_train_config
from src.data.dataset import build_transform
from src.models.networks import build_model


def test_project_paths_exist():
    assert (ROOT_DIR / "README.md").exists()
    assert (ROOT_DIR / "configs" / "class_mapping.json").exists()


def test_config_is_valid():
    config = load_train_config()
    config.validate()
    assert config.image_size >= 64
    assert config.train_fraction == 0.70


def test_baseline_forward_shape():
    model = build_model("baseline", num_classes=len(DEFAULT_CLASSES), pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    assert tuple(output.shape) == (2, len(DEFAULT_CLASSES))


def test_transfer_forward_shape_without_download():
    model = build_model("transfer", num_classes=len(DEFAULT_CLASSES), pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    assert tuple(output.shape) == (2, len(DEFAULT_CLASSES))


def test_transform_shape():
    from PIL import Image
    image = Image.new("RGB", (320, 240), "white")
    tensor = build_transform(128, train=False, augmentation="none")(image)
    assert tuple(tensor.shape) == (3, 128, 128)
