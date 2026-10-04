# EcoSort AI Experiment Report

> This report is generated from real pipeline outputs. It intentionally does not invent missing phone-test evidence.

## 1. Objective

Classify waste images into dry, wet, recyclable, and e-waste using a baseline CNN and a transfer-learning model, compare them on a held-out test set, inspect failures, and deploy the selected model behind a Streamlit interface.

## 2. Dataset and leakage control

- Unique images used: **12259**
- Exact duplicates removed before splitting: **0**
- Class counts: `{'dry': 6541, 'e-waste': 756, 'recyclable': 4263, 'wet': 699}`
- Split counts: `{'train': 8581, 'test': 1839, 'val': 1839}`
- Exact duplicate leakage was checked by SHA-256 file hash before stratified splitting.
- Dataset provenance must be completed in `reports/dataset_provenance.md` using the actual source used.

## 3. Models

### Baseline CNN
A compact four-block convolutional network provides a transparent reference point and a useful compute/performance baseline.

### Transfer learning
MobileNetV3-Small uses ImageNet-pretrained features with a four-class output head. The backbone is frozen during this project version so the comparison is fast and deployment-friendly.

## 4. Test-set comparison

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| Baseline CNN | 0.6090 | 0.6059 | 0.6350 | 0.6091 |
| Transfer MobileNetV3-Small | 0.9396 | 0.9202 | 0.9452 | 0.9322 |

The final checkpoint selected by validation performance is **transfer** with augmentation **standard**. Test-set results are reported only after the selection decision.

### Final selected checkpoint — held-out test set

Accuracy: **0.9396**; Macro precision: **0.9202**; Macro recall: **0.9452**; Macro F1: **0.9322**.

## 5. Augmentation experiment

| Variant | Best validation macro F1 |
|---|---:|
| transfer_standard | 0.9264 |
| transfer_strong | 0.9188 |

The table reports measured validation macro F1. Use the same held-out test split only for the final comparison.
## 6. Error analysis

The file `artifacts/metrics/final_misclassified_examples.csv` lists at least the available misclassified test examples. Inspect at least 20 examples for the final submission and document visual causes such as occlusion, unusual backgrounds, reflections, mixed waste, or ambiguous labels.

## 7. Explainability

Grad-CAM is implemented in `src/gradcam.py`. The Streamlit app can show a heatmap over the uploaded/camera image for the predicted class.

## 8. Practical inference

- Device: **cpu**
- Measured inference: **12.47 ms/image**
- Throughput: **80.21 images/s**

## 9. Phone-image test protocol

Collect at least 10–20 real phone photos across different lighting, backgrounds, object angles, and partially visible objects. Save them under a local test folder and record true class, prediction, confidence, and failure reason. Do not mix these phone images into the training set.

## 10. Reproducibility

The repository includes a fixed seed, configuration files, environment setup scripts, tests, and generated artifact paths. Run `pytest -q` and `python -m compileall app src tests` before the final demo.
