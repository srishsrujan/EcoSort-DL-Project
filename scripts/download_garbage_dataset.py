from __future__ import annotations

import shutil
from pathlib import Path

DATASET_HANDLE = "sumn2u/garbage-classification-v2"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DOWNLOAD_DIR = PROJECT_ROOT / "data" / "_kaggle_garbage_download"
EXPECTED_CLASSES = {
    "battery",
    "biological",
    "cardboard",
    "clothes",
    "glass",
    "metal",
    "paper",
    "plastic",
    "shoes",
    "trash",
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def count_images(folder: Path) -> int:
    return sum(1 for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def find_class_dirs(root: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for p in root.rglob("*"):
        if p.is_dir() and p.name.lower() in EXPECTED_CLASSES and count_images(p) > 0:
            found.setdefault(p.name.lower(), p)
    return found


def main() -> None:
    try:
        import kagglehub
    except ImportError as exc:
        raise SystemExit(
            "kagglehub is not installed. Run: python -m pip install -U kagglehub"
        ) from exc

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading public Kaggle dataset: {DATASET_HANDLE}")
    dataset_path = Path(
        kagglehub.dataset_download(
            DATASET_HANDLE,
            output_dir=str(DOWNLOAD_DIR),
        )
    )
    print(f"Dataset downloaded to: {dataset_path}")

    class_dirs = find_class_dirs(dataset_path)
    missing = sorted(EXPECTED_CLASSES - set(class_dirs))
    if missing:
        raise SystemExit(
            "Could not find these expected class folders after download: "
            + ", ".join(missing)
            + f"\nInspect: {dataset_path}"
        )

    copied = {}
    for class_name, source_dir in sorted(class_dirs.items()):
        destination = RAW_DIR / class_name
        if destination.exists() and any(destination.iterdir()):
            print(f"Skipping non-empty existing folder: {destination}")
            continue
        shutil.copytree(source_dir, destination, dirs_exist_ok=True)
        copied[class_name] = count_images(destination)

    print("\nEcoSort raw dataset folders:")
    total = 0
    for class_name in sorted(EXPECTED_CLASSES):
        count = count_images(RAW_DIR / class_name)
        total += count
        print(f"  {class_name:10s} {count:5d} images")
    print(f"  {'TOTAL':10s} {total:5d} images")
    print(f"\nRaw dataset ready at: {RAW_DIR}")
    print("Next command: python -m src.data.prepare_dataset")


if __name__ == "__main__":
    main()
