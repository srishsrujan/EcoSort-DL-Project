# EcoSort AI

EcoSort AI is an image-classification project that identifies waste as **dry**, **wet**, **recyclable**, or **e-waste**. It combines a custom CNN and MobileNetV3-Small with a Streamlit app for image upload, camera input, confidence scores, alternative predictions, and Grad-CAM explanations.

**[Try the live app](https://ecosort-dl.streamlit.app/)** · **[View the source on GitHub](https://github.com/srishsrujan/EcoSort-DL-Project)**

Drive video link 1 for uploading image : https://drive.google.com/file/d/1tRqiZ2VnNEZYHbCIkgJ2-ykofyYjwf4J/view?usp=drive_link

Drive video link 2 for taking image from phone : https://drive.google.com/file/d/1KF4bjI2J30ETLK2R3_WBzuzCzcBujfMR/view?usp=drive_link

## What you can explore

- Upload or photograph an item to get a predicted waste class.
- Review confidence and alternative predictions.
- Inspect Grad-CAM visual explanations.
- Compare model metrics and per-class performance in the Evidence tab.
- View the confusion matrices, augmentation results, and inference benchmark.

## Classes

| Class | Examples |
| --- | --- |
| Dry | Paper, cardboard |
| Wet | Organic or biodegradable waste |
| Recyclable | Plastic, glass, metal |
| E-waste | Batteries and electronic items |

Category boundaries depend on the project's class mapping and are not universal waste-disposal guidance.

## Run locally

Use Python 3.11 for the most predictable setup.

```powershell
git clone https://github.com/srishsrujan/EcoSort-DL-Project.git
cd EcoSort-DL-Project
.\scripts\setup_windows.ps1
.\.venv\Scripts\Activate.ps1
streamlit run app/app.py
```

The app opens at `http://localhost:8501`.

## Train with your own dataset

Place labelled images in one folder per source class under `data/raw/`, then update `configs/class_mapping.json` to map each source class to `dry`, `wet`, `recyclable`, or `e-waste`.

Run the training and evaluation pipeline:

```powershell
python -m src.run_pipeline
```

For a short CPU-friendly experiment:

```powershell
python -m src.run_pipeline --epochs 2 --image-size 160 --batch-size 16
```

The pipeline prepares stratified data splits, checks exact duplicates, trains and evaluates the models, and writes metrics and figures under `artifacts/`.

## Evaluation

Model quality is evaluated with accuracy, macro precision, macro recall, F1 scores, and confusion matrices. The Evidence tab presents these results alongside augmentation comparisons and inference speed. Evaluation results depend on the dataset and training run; they should not be interpreted as a guarantee of real-world performance.

To run the tests:

```powershell
pytest -q
```

## Project structure

```text
app/          Streamlit application
configs/      Training settings and class mapping
data/         Dataset files and prepared splits
src/          Training, evaluation, and inference code
artifacts/    Generated model, metrics, and figures
reports/      Experiment and dataset notes
tests/        Automated tests
```

## AI usage

See [AI_USAGE.md](./AI_USAGE.md) for the AI usage statement and remaining human verification steps.

## License

The application code is released under the MIT License. Dataset licenses and attribution requirements are separate; check the terms of the dataset you use.
