# EcoSort AI Submission Checklist

## Dataset
- [ ] Exact dataset source and license documented.
- [ ] Four target classes have adequate images.
- [ ] Duplicate leakage check completed.
- [ ] 70/15/15 stratified split generated.

## Modeling
- [ ] Baseline CNN trained.
- [ ] Transfer-learning model trained.
- [ ] Augmentation experiment completed.
- [ ] Best checkpoint saved.
- [ ] Test set evaluated only after selection.

## Evidence
- [ ] Per-class precision/recall/F1.
- [ ] Confusion matrix.
- [ ] Same-test-set model comparison.
- [ ] At least 20 misclassified examples and reasons.
- [ ] Real phone photos across different conditions.
- [ ] Practical inference speed.
- [ ] Grad-CAM examples.

## Engineering
- [ ] `pytest -q` passes.
- [ ] `python -m compileall app src tests` passes.
- [ ] README setup tested from a clean environment.
- [ ] Streamlit app tested locally.
- [ ] Git history contains meaningful commits.
- [ ] AI usage statement completed/reviewed.
- [ ] 5–8 minute demo video recorded.
