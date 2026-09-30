# EcoSort AI — Deep Learning Waste Classifier

A Streamlit-ready deep-learning system that classifies uploaded/camera waste photos into **dry**, **wet**, **recyclable**, or **e-waste**, returns confidence scores and top alternatives, and can visualize Grad-CAM explanations.

> **Important:** The uploaded brief requires a real labelled waste dataset and evidence from real images. The source package in this ZIP contains the complete training, evaluation, reporting, testing, Grad-CAM, and Streamlit deployment pipeline, but it does **not** fabricate trained metrics or a final checkpoint when no labelled waste dataset was supplied with the request. Put your real dataset into `data/raw/`, run the one-command pipeline, and the package will generate the submission evidence.

## What is included

- Baseline CNN and MobileNetV3-Small transfer-learning model.
- Exact-duplicate detection using image hashes before splitting.
- Stratified train/validation/test split.
- Configurable data augmentation experiments.
- Training/validation loss and accuracy plots.
- Class-wise precision, recall, F1 and confusion matrices.
- Saved best checkpoint with reproducible configuration.
- Top-k inference with confidence score.
- Grad-CAM heatmaps for the final transfer-learning model.
- Streamlit upload and webcam demo.
- Automated tests and a failure log template.
- Architecture and design-decision documentation.
- AI-usage statement template.
- Windows PowerShell and `.bat` environment setup scripts.

## Expected target classes

The application expects exactly these four final classes:

1. `dry`
2. `wet`
3. `recyclable`
4. `e-waste`

Your raw dataset can use any source class names. Use `configs/class_mapping.json` to map source folders into the four final classes. The mapping in that file is intentionally an **example**; replace it with the mapping supported by your actual dataset and project definition.

## Recommended folder layout

The easiest input format is one folder per source class:

```text
data/raw/
├── cardboard/
├── glass/
├── metal/
├── paper/
├── plastic/
├── battery/
├── biological/
└── ...
```

Images may be `.jpg`, `.jpeg`, `.png`, `.webp`, or `.bmp`.

The pipeline maps these folders using `configs/class_mapping.json`, removes exact duplicate images, creates stratified splits, then trains and evaluates both models.

## Fast start — Windows 11

### 1. Install Python

Use Python 3.11 for the most predictable PyTorch/Streamlit environment on Windows.

### 2. Create the virtual environment

PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\\scripts\\setup_windows.ps1
.\\.venv\\Scripts\\Activate.ps1
```

Command Prompt:

```bat
scripts\\setup_windows.bat
.venv\\Scripts\\activate.bat
```

### 3. Put the dataset in `data/raw/`

Then edit `configs/class_mapping.json` so every source folder is mapped to one of the four target classes.

### 4. Run the complete training/evaluation pipeline

```powershell
python -m src.run_pipeline
```

For a faster CPU-friendly smoke experiment:

```powershell
python -m src.run_pipeline --epochs 2 --image-size 160 --batch-size 16
```

For the full project, use the defaults and a GPU when available.

### 5. Run the Streamlit app

```powershell
streamlit run app/app.py
```

The app will open on `http://localhost:8501`.

## Streamlit deployment

Commit the repository to GitHub **after training** so the final checkpoint is available. For a typical Streamlit deployment:

1. Push the repository to GitHub.
2. Keep `requirements.txt` at the repository root.
3. Keep `app/app.py` as the Streamlit entry point.
4. Ensure `artifacts/checkpoints/final_model.pth` and `artifacts/metrics/final_model_meta.json` are present.
5. In Streamlit Community Cloud, set the main file to `app/app.py`.

Large model files may exceed ordinary Git hosting limits. If that happens, use Git LFS or a model artifact store and change the model path in `src/config.py`.

## One-command evidence generation

The pipeline generates:

```text
artifacts/
├── checkpoints/
│   ├── baseline_best.pth
│   ├── transfer_best.pth
│   └── final_model.pth
├── metrics/
│   ├── baseline_metrics.json
│   ├── transfer_metrics.json
│   ├── final_model_meta.json
│   ├── benchmark.json
│   └── predictions.csv
└── figures/
    ├── baseline_training_curves.png
    ├── transfer_training_curves.png
    ├── baseline_confusion_matrix.png
    ├── transfer_confusion_matrix.png
    └── gradcam_*.png

reports/
├── experiment_report.md
├── failure_log.md
└── demo_script.md
```

## What counts as valid evidence

The script does not invent evidence. After a real run, use the generated outputs to show:

- Per-class precision, recall and F1.
- Confusion matrix.
- A model-choice comparison using the same held-out test split.
- Predictions on at least 20 misclassified test images.
- Real phone photos with different lighting, backgrounds, angles, and partially visible objects.
- Augmentation ablation results.
- Practical inference speed from `benchmark.json`.

The included report template explicitly marks evidence that still needs real-world phone tests.

## Dataset provenance

Record the exact dataset name, URL, license, download date, preprocessing, and any additional images in `reports/dataset_provenance.md` before final submission. The repository includes a starter provenance template; do not claim a dataset source you did not actually use.

## Candidate public datasets

The project was designed to support common waste-image datasets. During preparation, publicly visible examples included TrashNet-style six-class data and datasets containing e-waste categories. See `docs/DATASET_SOURCES.md` for references and licensing notes. Your final report must cite the dataset you actually used.

## Important modeling note

“Dry”, “wet”, “recyclable”, and “e-waste” are project labels rather than universal image-taxonomy standards. The class mapping must therefore be documented as a project decision, and ambiguous items should be included in the failure analysis rather than silently forced into a class.

## Testing

Run:

```powershell
pytest -q
```

Also run a syntax check:

```powershell
python -m compileall app src tests
```

## Git history

The package is initialized as a Git repository with staged milestone commits. Use `git log --oneline --decorate` to review the history after extracting the ZIP.

## License

The application code is MIT-licensed in `LICENSE`. Dataset licenses remain separate and must be followed according to the actual source used.
