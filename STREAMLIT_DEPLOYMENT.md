# Streamlit Deployment Guide

## Local

```powershell
python -m streamlit run app/app.py
```

## Streamlit Community Cloud

- Repository root: this project.
- Main file: `app/app.py`.
- Dependencies: `requirements.txt`.
- Python version: `runtime.txt`.
- Model artifact: `artifacts/checkpoints/final_model.pth`.
- Model metadata: `artifacts/metrics/final_model_meta.json`.

Run the pipeline and test the app locally before deploying.

## Large model files

If the final checkpoint is too large for normal Git hosting, use Git LFS or an external artifact store. Do not replace the real checkpoint with a fake or random file merely to make the deployment start.
