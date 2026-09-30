from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from src.benchmark import benchmark
from src.config import CHECKPOINT_DIR, METRIC_DIR, PROCESSED_DIR, REPORT_DIR, load_train_config, ensure_dirs
from src.data.prepare_dataset import load_mapping, discover_images, split_manifest
from src.evaluate import evaluate_model
from src.reporting import generate_failure_log, generate_reports
from src.train import train_model
from src.utils import atomic_write_json, seed_everything
from src.config import CONFIG_DIR, RAW_DIR


def prepare() -> dict:
    config = load_train_config()
    classes, mapping = load_mapping(CONFIG_DIR / "class_mapping.json")
    raw_df = discover_images(RAW_DIR, mapping, classes)
    manifest = split_manifest(
        raw_df,
        classes,
        config.seed,
        config.train_fraction,
        config.val_fraction,
        config.test_fraction,
    )
    manifest_path = PROCESSED_DIR / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    summary = {
        "source_image_count": int(len(raw_df)),
        "unique_image_count": int(len(manifest)),
        "exact_duplicates_removed": int(manifest.attrs.get("duplicates_removed", 0)),
        "classes": classes,
        "class_counts": manifest["target_class"].value_counts().sort_index().to_dict(),
        "split_counts": manifest["split"].value_counts().to_dict(),
        "split_class_counts": manifest.groupby(["split", "target_class"]).size().unstack(fill_value=0).to_dict(),
        "seed": config.seed,
    }
    atomic_write_json(PROCESSED_DIR / "dataset_summary.json", summary)
    return summary


def choose_variant(summary: dict[str, dict]) -> tuple[str, str]:
    # Validation metrics are used for the model/augmentation selection decision.
    best_name = max(summary, key=lambda name: summary[name]["best_val_macro_f1"])
    record = summary[best_name]
    return record["model_kind"], record["augmentation"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete EcoSort AI evidence pipeline.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--skip-ablation", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    config = load_train_config()
    if args.epochs is not None:
        config = config.__class__(**{**config.__dict__, "epochs": args.epochs})
    if args.image_size is not None:
        config = config.__class__(**{**config.__dict__, "image_size": args.image_size})
    if args.batch_size is not None:
        config = config.__class__(**{**config.__dict__, "batch_size": args.batch_size})
    config.validate()
    seed_everything(config.seed)

    print("[1/7] Preparing dataset and exact-duplicate leakage check...")
    dataset_summary = prepare()
    manifest_path = PROCESSED_DIR / "manifest.csv"

    print("[2/7] Training baseline CNN...")
    baseline_train = train_model(
        "baseline", "standard", False, config, manifest_path, artifact_name="baseline"
    )
    baseline_test = evaluate_model("baseline", config, "standard", artifact_name="baseline")

    print("[3/7] Training transfer-learning baseline...")
    transfer_train = train_model(
        "transfer", "standard", True, config, manifest_path, artifact_name="transfer"
    )
    transfer_test = evaluate_model("transfer", config, "standard", artifact_name="transfer")

    variant_records = {
        "baseline_standard": baseline_train,
        "transfer_standard": transfer_train,
    }

    ablation_values = {
        "transfer_standard": float(transfer_train["best_val_macro_f1"]),
    }

    if not args.skip_ablation:
        print("[4/7] Running augmentation ablation (transfer model)...")
        transfer_strong = train_model(
            "transfer", "strong", True, config, manifest_path, artifact_name="transfer_strong"
        )
        ablation_values["transfer_strong"] = float(transfer_strong["best_val_macro_f1"])
        variant_records["transfer_strong"] = transfer_strong
        strong_test = evaluate_model("transfer", config, "strong", artifact_name="transfer_strong")
        variant_records["transfer_strong_test"] = strong_test
    else:
        print("[4/7] Skipping augmentation ablation by request.")

    atomic_write_json(METRIC_DIR / "augmentation_ablation.json", ablation_values)

    selected_key = max(variant_records.keys(), key=lambda name: variant_records[name].get("best_val_macro_f1", -1))
    selected_record = variant_records[selected_key]
    selected_model_kind = selected_record["model_kind"]
    selected_augmentation = selected_record["augmentation"]
    selected_artifact = selected_record["artifact_name"]

    # The baseline/transfer model comparison is based on validation macro-F1. The final test metrics remain held out.
    selected_checkpoint = CHECKPOINT_DIR / f"{selected_artifact}_best.pth"
    final_checkpoint = CHECKPOINT_DIR / "final_model.pth"
    shutil.copy2(selected_checkpoint, final_checkpoint)

    final_meta = {
        "selected_artifact": selected_artifact,
        "model_kind": selected_model_kind,
        "augmentation": selected_augmentation,
        "selection_metric": "validation_macro_f1",
        "selection_value": float(selected_record["best_val_macro_f1"]),
        "checkpoint": str(final_checkpoint),
        "classes": ["dry", "wet", "recyclable", "e-waste"],
        "image_size": config.image_size,
        "seed": config.seed,
    }
    atomic_write_json(METRIC_DIR / "final_model_meta.json", final_meta)

    # Reuse the selected model's held-out test evaluation; do not tune on the test set.
    print("[5/7] Finalizing the held-out test artifacts...")
    if selected_artifact == "baseline":
        final_test = baseline_test
    elif selected_artifact == "transfer":
        final_test = transfer_test
    elif selected_artifact == "transfer_strong":
        final_test = variant_records["transfer_strong_test"]
    else:
        raise RuntimeError(f"Unknown selected artifact: {selected_artifact}")
    atomic_write_json(METRIC_DIR / "final_metrics.json", final_test)
    selected_predictions = METRIC_DIR / f"{selected_artifact}_predictions.csv"
    shutil.copy2(selected_predictions, METRIC_DIR / "predictions.csv")
    generate_failure_log(selected_predictions)

    print("[6/7] Measuring inference speed...")
    benchmark(CHECKPOINT_DIR / "final_model.pth", selected_model_kind, repeats=30)

    print("[7/7] Generating report...")
    generate_reports(selected_model_kind, selected_augmentation, PROCESSED_DIR / "dataset_summary.json")

    pipeline_summary = {
        "dataset": dataset_summary,
        "baseline_test": baseline_test,
        "transfer_test": transfer_test,
        "final_test": final_test,
        "selection": final_meta,
        "ablation": ablation_values,
    }
    atomic_write_json(METRIC_DIR / "pipeline_summary.json", pipeline_summary)
    print(json.dumps({"status": "complete", "selection": final_meta}, indent=2))


if __name__ == "__main__":
    main()
