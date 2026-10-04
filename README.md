# EcoSort AI — Deep Learning Waste Classifier

**EcoSort AI** is a deep-learning based waste classification system that uses computer vision to identify waste from images and classify it into four categories:

**Dry · Wet · Recyclable · E-Waste**

The project combines **CNN-based image classification, transfer learning, data augmentation, model evaluation, and Grad-CAM explainability** into a complete end-to-end pipeline. A Streamlit web application allows users to upload an image or capture one using a camera and receive the model's prediction along with confidence scores and alternative predictions.

### Live Demo

**Try EcoSort AI:**
    https://ecosort-dl.streamlit.app/

### 💻 Source Code

**GitHub Repository:**
    https://github.com/srishsrujan/EcoSort-DL-Project

---

## Project Overview

Waste segregation is an important step in effective waste management, but identifying the correct category manually can sometimes be difficult.

EcoSort AI explores how **deep learning and computer vision** can assist with this problem.

The system takes a waste image as input and predicts one of four project-defined categories:

| Class         | Description                                      |
| ------------- | ------------------------------------------------ |
| Dry        | General dry waste such as paper, cardboard, etc. |
| Wet        | Organic or biodegradable waste                   |
| Recyclable | Materials such as plastic, glass, metal, etc.    |
| E-Waste    | Electronic items and related waste               |

The project was designed not only to train a model, but to build a **complete machine-learning workflow** covering data preparation, experimentation, evaluation, explainability, testing, and deployment.

---

## Features

* **CNN baseline model** for image classification
* **MobileNetV3-Small transfer learning** model
* Image upload through the Streamlit interface
* Camera-based image input
* Prediction confidence scores
* Top-k alternative predictions
* **Grad-CAM** visual explanations
* Training and validation accuracy/loss curves
* Precision, recall and F1-score for each class
* Confusion matrix visualization
* Exact duplicate-image detection before dataset splitting
* Stratified train/validation/test splitting
* Data augmentation experiments
* Inference-speed benchmarking
* Automated tests
* Experiment and failure-analysis reports
* Windows setup and deployment scripts

---

## Machine Learning Approach

The project compares two approaches rather than relying on a single model.

### 1. Baseline CNN

A custom convolutional neural network is trained from scratch to establish a baseline.

This provides a useful reference point for understanding how much performance can be achieved without using a pretrained model.

### 2. MobileNetV3-Small Transfer Learning

The second approach uses **MobileNetV3-Small**, a lightweight pretrained architecture designed to provide strong image-classification performance while remaining practical for deployment.

Transfer learning allows the model to reuse visual features learned from a large image dataset and adapt them to the waste-classification task.

The two models are evaluated on the **same held-out test set**, making the comparison more meaningful.

---

## 🔄 End-to-End Pipeline

The project follows a complete workflow:

```text
              Raw Waste Images
                     │
                     ▼
            Dataset Preparation
                     │
                     ▼
          Duplicate Image Detection
                     │
                     ▼
             Class Mapping
                     │
                     ▼
        Train / Validation / Test Split
                     │
                     ▼
             Data Augmentation
                     │
             ┌───────┴────────┐
             ▼                ▼
        Baseline CNN    MobileNetV3-Small
             │                │
             └───────┬────────┘
                     ▼
                 Evaluation
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
   Classification  Confusion    Metrics
      Metrics       Matrix     Comparison
                     │
                     ▼
               Final Model
                     │
          ┌──────────┴─────────┐
          ▼                    ▼
       Streamlit             Grad-CAM
        App                   Visuals
```

---

## Dataset and Class Mapping

The project is designed to work with real labelled waste-image datasets.

Because different datasets use different class names, the repository uses:

```text
configs/class_mapping.json
```

to map the original dataset classes to the four final project classes:

```text
dry
wet
recyclable
e-waste
```

For example, a dataset might contain classes such as:

```text
cardboard
glass
metal
paper
plastic
battery
biological
...
```

These can be mapped to the final categories according to the project's classification definition.

> **Important:** The class mapping is a project decision, not a universal definition of waste categories. Ambiguous examples should be documented during failure analysis rather than silently treated as perfectly labelled data.

---

## Dataset Structure

The simplest dataset structure is one folder per source class:

```text
data/
└── raw/
    ├── cardboard/
    ├── glass/
    ├── metal/
    ├── paper/
    ├── plastic/
    ├── battery/
    ├── biological/
    └── ...
```

Supported image formats:

```text
.jpg
.jpeg
.png
.webp
.bmp
```

Before training, the pipeline:

1. Detects exact duplicate images using image hashes.
2. Applies the configured class mapping.
3. Creates stratified train/validation/test splits.
4. Applies training-time augmentation.
5. Trains and evaluates the models.

---

## Model Evaluation

The project does not rely on accuracy alone.

The evaluation pipeline generates:

* Accuracy
* Precision
* Recall
* F1-score
* Class-wise performance
* Confusion matrices
* Training/validation curves
* Model comparison
* Inference-speed benchmarks
* Predictions on incorrectly classified images

This makes it easier to understand **where the model performs well and where it fails**.

For example, if two classes are frequently confused with each other, the confusion matrix and misclassified examples can help identify whether the problem comes from visually similar objects, insufficient training data, background variation, or ambiguous labels.

---

## Explainability with Grad-CAM

A classification model can give the correct label without making it obvious *why* it made that prediction.

To make the model more interpretable, EcoSort AI uses **Grad-CAM (Gradient-weighted Class Activation Mapping)**.

Grad-CAM creates a heatmap showing the regions of an image that contributed most strongly to the model's prediction.

Example workflow:

```text
Input Image
     │
     ▼
MobileNetV3-Small
     │
     ▼
Predicted Class
     │
     ▼
Grad-CAM
     │
     ▼
Important Image Regions
```

This provides a visual way to inspect whether the model is focusing on the waste object itself rather than irrelevant background features.

---

## Streamlit Application

The trained model is integrated into a Streamlit interface.

Users can:

1. Upload a waste image.
2. Capture an image using the camera.
3. View the predicted class.
4. See the prediction confidence.
5. View alternative top predictions.
6. Inspect Grad-CAM explanations when available.

The Evidence tab summarizes model scores, per-class performance, augmentation results, and inference speed in a readable format. Confusion-matrix images remain available there.

### Live application

 **https://ecosort-dl.streamlit.app/**

---

##  Tech Stack

### Machine Learning

* Python
* PyTorch
* Torchvision
* CNN
* Transfer Learning
* MobileNetV3-Small
* Grad-CAM

### Data & Evaluation

* NumPy
* Pandas
* Matplotlib
* Image hashing
* Stratified dataset splitting
* Precision / Recall / F1
* Confusion matrices

### Deployment

* Streamlit
* GitHub
* Streamlit Community Cloud

### Testing

* Pytest
* Python compile checks

---

##  Project Structure

```text
EcoSort-DL-Project/
│
├── app/
│   └── app.py
│
├── artifacts/
│   ├── checkpoints/
│   ├── metrics/
│   └── figures/
│
├── configs/
│   └── class_mapping.json
│
├── data/
│   └── raw/
│
├── docs/
│   └── ...
│
├── reports/
│   ├── experiment_report.md
│   ├── failure_log.md
│   └── dataset_provenance.md
│
├── scripts/
│   └── ...
│
├── src/
│   └── ...
│
├── tests/
│   └── ...
│
├── requirements.txt
├── runtime.txt
├── LICENSE
└── README.md
```

---

#  Running the Project Locally

## 1. Install Python

For the most predictable environment, use **Python 3.11**.

Check your installation:

```powershell
python --version
```

---

## 2. Clone the repository

```powershell
git clone https://github.com/srishsrujan/EcoSort-DL-Project.git
cd EcoSort-DL-Project
```

---

## 3. Create the virtual environment

### PowerShell

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_windows.ps1
.\.venv\Scripts\Activate.ps1
```

### Command Prompt

```bat
scripts\setup_windows.bat
.venv\Scripts\activate.bat
```

---

## 4. Add the dataset

Place your labelled dataset inside:

```text
data/raw/
```

Then update:

```text
configs/class_mapping.json
```

so that the source classes are mapped correctly to:

```text
dry
wet
recyclable
e-waste
```

---

## 5. Run the complete pipeline

```powershell
python -m src.run_pipeline
```

For a quick CPU-friendly experiment:

```powershell
python -m src.run_pipeline --epochs 2 --image-size 160 --batch-size 16
```

For the full experiment, use the default settings and a GPU when available.

---

## 6. Run the Streamlit application

```powershell
streamlit run app/app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

---

#  Generated Artifacts

After running the pipeline, the project can generate artifacts such as:

```text
artifacts/
├── checkpoints/
│   ├── baseline_best.pth
│   ├── transfer_best.pth
│   └── final_model.pth
│
├── metrics/
│   ├── baseline_metrics.json
│   ├── transfer_metrics.json
│   ├── final_model_meta.json
│   ├── benchmark.json
│   └── predictions.csv
│
└── figures/
    ├── baseline_training_curves.png
    ├── transfer_training_curves.png
    ├── baseline_confusion_matrix.png
    ├── transfer_confusion_matrix.png
    └── gradcam_*.png
```

The reports directory contains supporting documentation such as:

```text
reports/
├── experiment_report.md
├── failure_log.md
├── demo_script.md
└── dataset_provenance.md
```

---

#  Testing

Run the automated test suite:

```powershell
pytest -q
```

You can also check the project for Python syntax errors:

```powershell
python -m compileall app src tests
```

---

#  Real-World Testing

A good image-classification model should not be evaluated only on clean dataset images.

For real-world testing, the project should include photos captured under different conditions, such as:

* Different lighting
* Different backgrounds
* Different camera angles
* Partial visibility of objects
* Different object sizes
* Real phone-camera images

Misclassified examples are especially useful because they show where the model still needs improvement.

---

#  Evidence and Experimentation

The project is structured around reproducible experiments rather than presenting a single accuracy number.

Useful evidence includes:

* Baseline vs. transfer-learning performance
* Class-wise precision, recall and F1
* Confusion matrices
* Training/validation curves
* Data-augmentation comparisons
* Misclassified test images
* Grad-CAM visualizations
* Practical inference time
* Real-world phone-camera testing

This helps answer not only **"How accurate is the model?"**, but also:

> **"How does the model behave, where does it fail, and is it practical to deploy?"**

---

#  Dataset and Evidence Note

The project is designed to work with a real labelled waste dataset.

The repository does **not** intentionally fabricate training metrics, evaluation results, or model performance. The final reported results should come from the actual dataset and experiments used for the project.

Before final submission, document the actual dataset used in:

```text
reports/dataset_provenance.md
```

including:

* Dataset name
* Dataset URL
* License
* Download date
* Original classes
* Class mapping
* Preprocessing
* Any additional images collected

This keeps the experiment reproducible and makes the reported results easier to verify.

---

#  Future Improvements

EcoSort AI can be extended in several directions:

* Collecting a larger and more diverse real-world dataset
* Improving performance on visually similar classes
* Adding stronger augmentation strategies
* Fine-tuning more lightweight architectures
* Quantizing the final model for faster inference
* Adding better uncertainty handling
* Expanding the number of waste categories
* Improving Grad-CAM visualization
* Collecting more real-world phone images for evaluation
* Deploying the model on edge/mobile devices

---

#  What I Learned

Building EcoSort AI helped me understand the complete workflow behind a practical deep-learning project, including:

* Preparing and validating image datasets
* Preventing duplicate images from leaking across dataset splits
* Training CNN models
* Using transfer learning
* Applying data augmentation
* Evaluating classification models properly
* Understanding confusion matrices and class-wise metrics
* Using Grad-CAM for model explainability
* Saving and loading model checkpoints
* Building a Streamlit interface
* Testing a machine-learning application
* Deploying a deep-learning model

The project was especially useful for understanding the difference between **training a model** and building an **end-to-end machine-learning application**.

---

#  License

The application code in this repository is released under the **MIT License**.

Dataset licenses are separate and depend on the dataset actually used for training. Always follow the original dataset's licensing and attribution requirements.

---

##  Final Note

EcoSort AI is an educational and experimental project exploring how deep learning can be applied to waste classification.

The goal is not simply to produce a prediction, but to build a workflow where the model can be **trained, evaluated, explained, tested, and deployed** in a reproducible way.

 **Live Demo:** https://ecosort-dl.streamlit.app/
 **GitHub:** https://github.com/srishsrujan/EcoSort-DL-Project
