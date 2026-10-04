# EcoSort AI Submission Checklist

## Current status

This repository is structurally complete and passes code validation, but the submission is not fully evidence-complete until the remaining real-data requirements are collected and documented.

## Dataset
- [x] Exact dataset source and license documented.
- [x] Four target classes have adequate images.
- [x] Duplicate leakage check completed.
- [x] 70/15/15 stratified split generated.

## Modeling
- [x] Baseline CNN trained.
- [x] Transfer-learning model trained.
- [x] Augmentation experiment completed.
- [x] Best checkpoint saved.
- [x] Test set evaluated only after selection.

## Evidence
- [x] Per-class precision/recall/F1.
- [x] Confusion matrix.
- [x] Same-test-set model comparison.
- [ ] At least 20 misclassified examples and reasons. (Blocked: no visual inspection file or captured examples were added to the repo.)
- [ ] Real phone photos across different conditions. (Blocked: no phone-image test set was recorded in the repository.)
- [x] Practical inference speed.
- [x] Grad-CAM examples.

## Engineering
- [x] `pytest -q` passes.
- [x] `python -m compileall app src tests` passes.
- [ ] README setup tested from a clean environment. (Not yet verified from a fresh environment in this session.)
- [ ] Streamlit app tested locally. (Local smoke-test was not recorded in the repo.)
- [x] Git history contains meaningful commits.
- [x] AI usage statement completed/reviewed.
- [x] Demo video intentionally excluded from this submission package checklist.
