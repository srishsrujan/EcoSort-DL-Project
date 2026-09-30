# One-Day Execution Runbook

This package is organized so the remaining work is the **real-data run and evidence collection**, not project scaffolding.

## 1. Setup — 10–20 min

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_windows.ps1
.\.venv\Scripts\Activate.ps1
pytest -q
python -m compileall app src tests
```

## 2. Dataset — 20–40 min

Put the labelled images in `data/raw/` and update `configs/class_mapping.json`.

Run:

```powershell
python -m src.data.prepare_dataset
```

Open `data/processed/dataset_summary.json` and confirm all four classes are represented.

## 3. Train and generate evidence — depends on dataset/GPU

Full run:

```powershell
python -m src.run_pipeline
```

Emergency CPU-conscious run:

```powershell
python -m src.run_pipeline --epochs 4 --image-size 160 --batch-size 16 --skip-ablation
```

The emergency command is a time-saving fallback. It should not be presented as equivalent to a careful full experiment.

## 4. Local Streamlit test — 10 min

```powershell
streamlit run app/app.py
```

Test:

- one clean object;
- one partially visible object;
- one image with clutter/background;
- one phone-camera image.

Record prediction and confidence.

## 5. Failure analysis — 30–60 min

Open `artifacts/metrics/predictions.csv` and `reports/failure_log.md`.

Inspect at least 20 wrong test examples and write a concrete reason for each. Do not write “model failed” as the reason.

## 6. Phone test evidence — 30–60 min

Collect separate real phone images under different lighting/background/angles. Do not add them to training data.

Record:

| Image | True class | Prediction | Confidence | Lighting | Background | Angle/occlusion | Correct? |
|---|---|---|---:|---|---|---|---|
| TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

## 7. Final verification — 15 min

```powershell
pytest -q
git status
git log --oneline --decorate -5
```

Then review:

- `reports/experiment_report.md`
- `reports/failure_log.md`
- `reports/dataset_provenance.md`
- `reports/submission_checklist.md`
- `docs/AI_USAGE.md`

## 8. Deployment

Push the final repository with the model checkpoint and evidence artifacts. For Streamlit, use `app/app.py` as the entry point.
