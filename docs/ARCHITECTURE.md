# EcoSort AI Architecture

```text
                         +-----------------------+
                         |   Raw labelled data   |
                         | data/raw/<class>/...  |
                         +-----------+-----------+
                                     |
                                     v
                         +-----------------------+
                         | Dataset audit         |
                         | - SHA-256 duplicates  |
                         | - class mapping       |
                         | - provenance manifest  |
                         +-----------+-----------+
                                     |
                                     v
                         +-----------------------+
                         | Stratified split      |
                         | train / val / test    |
                         +-----+-------------+---+
                               |             |
                  +------------+             +------------+
                  v                                       v
        +--------------------+                 +-----------------------+
        | Baseline CNN       |                 | MobileNetV3-Small     |
        | from scratch       |                 | ImageNet transfer     |
        +---------+----------+                 +-----------+-----------+
                  |                                        |
                  +------------------+---------------------+
                                     |
                                     v
                         +-----------------------+
                         | Validation selection |
                         | macro-F1              |
                         +-----------+-----------+
                                     |
                                     v
                         +-----------------------+
                         | Held-out test         |
                         | P/R/F1 + confusion   |
                         +-----------+-----------+
                                     |
               +---------------------+----------------------+
               |                                            |
               v                                            v
      +----------------------+                    +---------------------+
      | Grad-CAM explanation |                    | Inference benchmark |
      +----------------------+                    +---------------------+
               |                                            |
               +----------------------+---------------------+
                                      |
                                      v
                            +----------------------+
                            | Streamlit web app    |
                            | upload + webcam      |
                            +----------------------+
```

## Main design decisions

- **Four target classes:** the project brief defines dry, wet, recyclable and e-waste, while raw datasets commonly use narrower material categories. A JSON mapping layer makes that decision explicit.
- **Leakage prevention:** exact duplicate image hashes are removed before train/validation/test split.
- **Baseline:** a small CNN makes the transfer-learning comparison meaningful and easy to explain in a viva.
- **Transfer model:** MobileNetV3-Small is compact enough for a web demo and supports Grad-CAM through a final convolutional feature layer.
- **Selection rule:** the validation macro-F1 determines which trained variant becomes `final_model.pth`. The test set is not used for tuning.
- **Explainability:** Grad-CAM is presented as a debugging/evidence aid, not as a causal proof of model reasoning.
