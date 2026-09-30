# 5–8 Minute Demo Script

## 0:00–0:45 — Problem

“EcoSort AI classifies a waste photo into dry, wet, recyclable, or e-waste. The goal is not only prediction, but an auditable deep-learning workflow with data leakage control, model comparison, error analysis, and explainability.”

## 0:45–1:45 — Dataset and preparation

Show `data/raw/`, `configs/class_mapping.json`, and `artifacts/metrics/dataset_summary.json`.

Explain:
- where the images came from;
- why the mapping is needed;
- exact-duplicate SHA-256 checks;
- stratified 70/15/15 split.

## 1:45–3:00 — Model comparison

Show the baseline CNN and MobileNetV3-Small results in `reports/experiment_report.md`.

Explain that validation macro-F1 was used for selection and the test set was held out until the final comparison.

## 3:00–4:00 — Augmentation and failure analysis

Show `augmentation_ablation.json`, confusion matrices and at least 20 misclassified examples.

Pick 2–3 failures and explain the visible reason: lighting, background, occlusion, ambiguous material, etc.

## 4:00–6:00 — Live Streamlit demo

1. Upload a clean waste image.
2. Show the class, confidence and top alternatives.
3. Open Grad-CAM.
4. Take a camera photo.
5. Use a partially visible object and discuss confidence/failure behavior.

## 6:00–7:00 — Practical performance

Show `benchmark.json` and state the measured device and inference time.

## 7:00–8:00 — Engineering and reproducibility

Show `requirements.txt`, Windows setup script, tests, report, architecture diagram and Git history.

End with one sentence: “The system is a deployable demonstrator; ambiguous or low-confidence predictions should be reviewed by a human.”
