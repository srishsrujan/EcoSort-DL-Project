from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import CONFIG_DIR, IMAGE_EXTENSIONS, PROCESSED_DIR, RAW_DIR, ensure_dirs
from src.utils import sha256_file


def load_mapping(path: Path) -> tuple[list[str], dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    classes = payload["target_classes"]
    mapping = {str(k).strip().lower(): str(v).strip() for k, v in payload["source_to_target"].items()}
    for value in mapping.values():
        if value not in classes:
            raise ValueError(f"Mapping points to unknown target class: {value}")
    return classes, mapping


def discover_images(raw_dir: Path, mapping: dict[str, str], classes: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for source_dir in sorted(raw_dir.iterdir()):
        if not source_dir.is_dir() or source_dir.name.startswith("."):
            continue
        source_name = source_dir.name.strip().lower()
        target = source_name if source_name in classes else mapping.get(source_name)
        if target is None:
            continue
        for path in source_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                try:
                    file_hash = sha256_file(path)
                    rows.append(
                        {
                            "path": str(path.resolve()),
                            "source_class": source_dir.name,
                            "target_class": target,
                            "sha256": file_hash,
                        }
                    )
                except OSError as exc:
                    print(f"Skipping unreadable file {path}: {exc}")
    if not rows:
        raise FileNotFoundError(
            f"No supported images found in {raw_dir}. Add class folders and update configs/class_mapping.json."
        )
    return pd.DataFrame(rows)


def split_manifest(df: pd.DataFrame, classes: list[str], seed: int, train_frac: float, val_frac: float, test_frac: float) -> pd.DataFrame:
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("Train/val/test fractions must sum to 1.0")

    duplicate_groups = df.groupby("sha256")["target_class"].nunique()
    conflicting = duplicate_groups[duplicate_groups > 1]
    if len(conflicting):
        raise ValueError(
            "At least one exact duplicate image has conflicting labels. Fix the raw dataset before splitting. "
            f"Conflicting hash count: {len(conflicting)}"
        )

    before = len(df)
    df = df.drop_duplicates(subset=["sha256"], keep="first").copy()
    df["duplicate_removed"] = False
    removed = before - len(df)

    counts = df["target_class"].value_counts()
    missing = [c for c in classes if c not in counts.index]
    if missing:
        raise ValueError(f"Missing target classes in dataset: {missing}. Each class needs labelled images.")
    if (counts < 6).any():
        raise ValueError(
            "Each class needs at least 6 unique images to create stratified train/val/test splits. "
            f"Current counts: {counts.to_dict()}"
        )

    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - train_frac),
        random_state=seed,
        stratify=df["target_class"],
    )
    val_share_of_temp = val_frac / (val_frac + test_frac)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_share_of_temp),
        random_state=seed,
        stratify=temp_df["target_class"],
    )

    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")
    manifest = pd.concat([train_df, val_df, test_df], ignore_index=True)
    manifest = manifest.sort_values(["split", "target_class", "path"]).reset_index(drop=True)
    manifest["dataset_index"] = range(len(manifest))
    manifest.attrs["duplicates_removed"] = removed
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare an auditable EcoSort dataset manifest.")
    parser.add_argument("--mapping", type=Path, default=CONFIG_DIR / "class_mapping.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    args = parser.parse_args()

    ensure_dirs()
    classes, mapping = load_mapping(args.mapping)
    raw_df = discover_images(RAW_DIR, mapping, classes)
    manifest = split_manifest(raw_df, classes, args.seed, args.train, args.val, args.test)
    out_path = PROCESSED_DIR / "manifest.csv"
    manifest.to_csv(out_path, index=False)

    summary = {
        "source_image_count": int(len(raw_df)),
        "unique_image_count": int(len(manifest)),
        "exact_duplicates_removed": int(manifest.attrs.get("duplicates_removed", 0)),
        "classes": classes,
        "class_counts": manifest["target_class"].value_counts().sort_index().to_dict(),
        "split_counts": manifest["split"].value_counts().to_dict(),
        "split_class_counts": manifest.groupby(["split", "target_class"]).size().unstack(fill_value=0).to_dict(),
        "seed": args.seed,
        "mapping_file": str(args.mapping),
    }
    with (PROCESSED_DIR / "dataset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
