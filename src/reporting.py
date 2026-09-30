from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import CHECKPOINT_DIR, METRIC_DIR, REPORT_DIR
from src.utils import atomic_write_json


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_report(selected_model: str, selected_augmentation: str, dataset_summary: dict, baseline: dict, transfer: dict, final_test: dict, ablation: dict | None, benchmark: dict | None) -> str:
    lines = [
        "# EcoSort AI Experiment Report",
        "",
        "> This report is generated from real pipeline outputs. It intentionally does not invent missing phone-test evidence.",
        "",
        "## 1. Objective",
        "",
        "Classify waste images into dry, wet, recyclable, and e-waste using a baseline CNN and a transfer-learning model, compare them on a held-out test set, inspect failures, and deploy the selected model behind a Streamlit interface.",
        "",
        "## 2. Dataset and leakage control",
        "",
        f"- Unique images used: **{dataset_summary.get('unique_image_count', 'N/A')}**",
        f"- Exact duplicates removed before splitting: **{dataset_summary.get('exact_duplicates_removed', 'N/A')}**",
        f"- Class counts: `{dataset_summary.get('class_counts', {})}`",
        f"- Split counts: `{dataset_summary.get('split_counts', {})}`",
        "- Exact duplicate leakage was checked by SHA-256 file hash before stratified splitting.",
        "- Dataset provenance must be completed in `reports/dataset_provenance.md` using the actual source used.",
        "",
        "## 3. Models",
        "",
        "### Baseline CNN",
        "A compact four-block convolutional network provides a transparent reference point and a useful compute/performance baseline.",
        "",
        "### Transfer learning",
        "MobileNetV3-Small uses ImageNet-pretrained features with a four-class output head. The backbone is frozen during this project version so the comparison is fast and deployment-friendly.",
        "",
        "## 4. Test-set comparison",
        "",
        "| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |",
        "|---|---:|---:|---:|---:|",
        f"| Baseline CNN | {baseline.get('test_accuracy', float('nan')):.4f} | {baseline.get('macro_precision', float('nan')):.4f} | {baseline.get('macro_recall', float('nan')):.4f} | {baseline.get('macro_f1', float('nan')):.4f} |",
        f"| Transfer MobileNetV3-Small | {transfer.get('test_accuracy', float('nan')):.4f} | {transfer.get('macro_precision', float('nan')):.4f} | {transfer.get('macro_recall', float('nan')):.4f} | {transfer.get('macro_f1', float('nan')):.4f} |",
        "",
        f"The final checkpoint selected by validation performance is **{selected_model}** with augmentation **{selected_augmentation}**. Test-set results are reported only after the selection decision.",
        "",
        "### Final selected checkpoint — held-out test set",
        "",
        f"Accuracy: **{final_test.get('test_accuracy', float('nan')):.4f}**; Macro precision: **{final_test.get('macro_precision', float('nan')):.4f}**; Macro recall: **{final_test.get('macro_recall', float('nan')):.4f}**; Macro F1: **{final_test.get('macro_f1', float('nan')):.4f}**.",
        "",
        "## 5. Augmentation experiment",
        "",
    ]
    if ablation:
        lines.append("| Variant | Best validation macro F1 |")
        lines.append("|---|---:|")
        for name, value in ablation.items():
            lines.append(f"| {name} | {value:.4f} |")
        lines.append("")
        lines.append("The table reports measured validation macro F1. Use the same held-out test split only for the final comparison.")
    else:
        lines += ["Ablation results were not generated. Run the pipeline without `--skip-ablation`.", ""]

    lines += [
        "## 6. Error analysis",
        "",
        "The file `artifacts/metrics/final_misclassified_examples.csv` lists at least the available misclassified test examples. Inspect at least 20 examples for the final submission and document visual causes such as occlusion, unusual backgrounds, reflections, mixed waste, or ambiguous labels.",
        "",
        "## 7. Explainability",
        "",
        "Grad-CAM is implemented in `src/gradcam.py`. The Streamlit app can show a heatmap over the uploaded/camera image for the predicted class.",
        "",
        "## 8. Practical inference",
        "",
    ]
    if benchmark:
        lines += [
            f"- Device: **{benchmark.get('device')}**",
            f"- Measured inference: **{benchmark.get('milliseconds_per_image', float('nan')):.2f} ms/image**",
            f"- Throughput: **{benchmark.get('images_per_second', float('nan')):.2f} images/s**",
            "",
        ]
    else:
        lines += ["Benchmark output is missing. Run `python -m src.benchmark --model transfer` after training.", ""]

    lines += [
        "## 9. Phone-image test protocol",
        "",
        "Collect at least 10–20 real phone photos across different lighting, backgrounds, object angles, and partially visible objects. Save them under a local test folder and record true class, prediction, confidence, and failure reason. Do not mix these phone images into the training set.",
        "",
        "## 10. Reproducibility",
        "",
        "The repository includes a fixed seed, configuration files, environment setup scripts, tests, and generated artifact paths. Run `pytest -q` and `python -m compileall app src tests` before the final demo.",
    ]
    return "\n".join(lines) + "\n"


def generate_reports(selected_model: str, selected_augmentation: str, dataset_summary_path: Path) -> None:
    dataset_summary = _load_json(dataset_summary_path)
    baseline = _load_json(METRIC_DIR / "baseline_metrics.json")
    transfer = _load_json(METRIC_DIR / "transfer_metrics.json")
    final_test = _load_json(METRIC_DIR / "final_metrics.json")
    ablation_path = METRIC_DIR / "augmentation_ablation.json"
    benchmark_path = METRIC_DIR / "benchmark.json"
    ablation = _load_json(ablation_path) if ablation_path.exists() else None
    benchmark = _load_json(benchmark_path) if benchmark_path.exists() else None
    report = build_report(selected_model, selected_augmentation, dataset_summary, baseline, transfer, final_test, ablation, benchmark)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "experiment_report.md").write_text(report, encoding="utf-8")


def generate_failure_log(predictions_path: Path, top_n: int = 20) -> None:
    df = pd.read_csv(predictions_path)
    errors = df[df["correct"] == False].copy()  # noqa: E712
    errors = errors.sort_values("confidence", ascending=False).head(top_n)
    out_csv = METRIC_DIR / "final_misclassified_examples.csv"
    errors.to_csv(out_csv, index=False)
    lines = [
        "# Failure Log",
        "",
        "Fill the reason column after visually inspecting each listed example. Suggested categories: `occlusion`, `ambiguous class`, `lighting`, `background`, `object too small`, `reflection`, `mixed waste`, `label noise`, `camera blur`, `other`.",
        "",
        "| Image | True class | Predicted | Confidence | Reason |",
        "|---|---|---|---:|---|",
    ]
    for _, row in errors.iterrows():
        lines.append(
            f"| `{Path(row['path']).name}` | {row['true_class']} | {row['predicted_class']} | {row['confidence']:.3f} | TODO |")
    (REPORT_DIR / "failure_log.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
