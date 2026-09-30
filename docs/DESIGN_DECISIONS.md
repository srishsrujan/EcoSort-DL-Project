# Design Decisions and Trade-offs

## Dataset mapping

The source dataset may contain many material-level classes. `configs/class_mapping.json` maps them into the four project classes. This is intentionally configurable because “dry/wet/recyclable/e-waste” is not a single universal image taxonomy.

## Duplicate control

The pipeline calculates SHA-256 for every source image and refuses conflicting labels for the same exact image. This prevents exact duplicate leakage across train/validation/test.

## Augmentation

The baseline comparison uses standard augmentation: horizontal flip, small rotation and color jitter. A stronger transfer-learning variant additionally applies random resized crop and perspective distortion. The ablation uses validation macro-F1 rather than test data.

## Metrics

Macro precision, recall and F1 are included because class imbalance can make aggregate accuracy misleading. A confusion matrix is saved for qualitative review.

## Deployment

Streamlit was chosen because it provides a simple upload and webcam interface without requiring a separate frontend. The final model checkpoint is loaded once with Streamlit resource caching.

## CPU/GPU compatibility

The pipeline automatically uses CUDA if available and otherwise falls back to CPU. For a deadline-driven demo, reducing image size, batch size and epochs is supported through CLI flags.
