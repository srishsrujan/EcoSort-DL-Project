# Dataset Provenance

## Source

- Dataset name: Garbage Dataset, downloaded through Kaggle dataset handle `sumn2u/garbage-classification-v2` by `scripts/download_garbage_dataset.py`.
- URL: https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2
- Download date: Not recorded in the repository.
- License: MIT, as listed on the Kaggle dataset page when checked on 2026-10-02. Recheck the current upstream terms before redistribution.
- Citation/attribution: Dataset credited on Kaggle to Suman Kunwar and a collaborator. The page references [The Garbage Dataset (GD): A Multi-Class Image Benchmark for Automated Waste Segregation](https://journals-sol.sbc.org.br/index.php/jbcs/article/view/7774).
- Count note: The Kaggle page reports 13,348 images in the source release; this project's generated manifest records 12,259 images discovered locally. The reason for this difference was not recorded and should be checked before making claims about full-source coverage.

## Preparation

- Images discovered in the local raw folders: 12,259 (`data/processed/dataset_summary.json`).
- Images after exact duplicate removal: 12,259; 0 exact duplicates removed.
- Unreadable files removed: Not counted in the saved summary; unreadable files are logged during preparation.
- Target classes: dry / wet / recyclable / e-waste
- Source-to-target mapping: `configs/class_mapping.json`
- Stratified train/validation/test split: 70% / 15% / 15% (8,581 / 1,839 / 1,839 images)
- Target-class counts: dry 6,541; wet 699; recyclable 4,263; e-waste 756.
- Seed: 42

## Additional phone images

- Collection status and count: Not recorded in the repository.
- No separate phone-image evaluation is represented in the reported test metrics.
- Collection conditions and storage location: Not recorded.

## Notes

The local counts and split details above come from the generated dataset summary. Do not present the upstream image count as the number used for training, or claim phone-photo testing that was not recorded.
