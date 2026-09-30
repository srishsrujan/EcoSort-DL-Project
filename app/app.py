from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
from PIL import Image

from src.config import CHECKPOINT_DIR, METRIC_DIR
from src.gradcam import explain_image
from src.inference import WasteClassifier

ROOT = Path(__file__).resolve().parents[1]
FINAL_CHECKPOINT = CHECKPOINT_DIR / "final_model.pth"
FINAL_META = METRIC_DIR / "final_model_meta.json"

st.set_page_config(
    page_title="EcoSort AI",
    page_icon="♻️",
    layout="wide",
)


def load_meta():
    if not FINAL_META.exists():
        return None
    with FINAL_META.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_resource(show_spinner=False)
def load_classifier(checkpoint: str, device: str = "cpu"):
    return WasteClassifier(checkpoint, device=device)


st.title("♻️ EcoSort AI")
st.caption("Deep-learning waste classification with confidence scores and Grad-CAM explanations.")

meta = load_meta()

with st.sidebar:
    st.header("Model")
    if meta:
        st.write(f"**Architecture:** {meta.get('model_kind', 'unknown')}")
        st.write(f"**Augmentation:** {meta.get('augmentation', 'unknown')}")
        st.write(f"**Selection metric:** validation macro-F1")
        st.write(f"**Selection value:** {meta.get('selection_value', 0):.3f}")
    else:
        st.warning("No trained final checkpoint found.")
        st.code("python -m src.run_pipeline", language="powershell")


tab_classify, tab_evidence, tab_about = st.tabs(["Classify", "Evidence", "About"])

with tab_classify:
    source = st.radio("Input source", ["Upload image", "Use camera"], horizontal=True)
    image = None
    if source == "Upload image":
        uploaded = st.file_uploader("Upload a waste photo", type=["jpg", "jpeg", "png", "webp", "bmp"])
        if uploaded is not None:
            image = Image.open(uploaded).convert("RGB")
    else:
        camera = st.camera_input("Take a waste photo")
        if camera is not None:
            image = Image.open(camera).convert("RGB")

    if image is not None:
        left, right = st.columns(2)
        with left:
            st.image(image, caption="Input image", use_container_width=True)

        if not FINAL_CHECKPOINT.exists():
            with right:
                st.error("The final trained model is not present yet. Add your dataset, run the pipeline, and refresh the app.")
        else:
            classifier = load_classifier(str(FINAL_CHECKPOINT))
            with st.spinner("Running inference..."):
                prediction = classifier.predict(image, top_k=3)
            with right:
                st.subheader(f"Prediction: {prediction.label}")
                st.metric("Confidence", f"{prediction.confidence * 100:.1f}%")
                if prediction.confidence < 0.60:
                    st.warning("Low-confidence prediction — inspect alternatives and the image conditions.")
                st.write("Top alternatives")
                for label, confidence in prediction.alternatives:
                    st.progress(confidence, text=f"{label}: {confidence * 100:.1f}%")

            with st.expander("Grad-CAM explanation"):
                try:
                    heatmap = explain_image(
                        image,
                        classifier.model,
                        classifier.model_kind,
                        classifier.image_size,
                        classifier.device,
                        classifier.class_names.index(prediction.label),
                    )
                    st.image(heatmap, caption="Regions that contributed most to the predicted class", use_container_width=True)
                except Exception as exc:
                    st.warning(f"Grad-CAM could not be generated for this image: {exc}")

with tab_evidence:
    st.subheader("Model evidence")
    metric_files = [
        ("Baseline CNN", METRIC_DIR / "baseline_metrics.json"),
        ("Transfer model", METRIC_DIR / "transfer_metrics.json"),
        ("Augmentation", METRIC_DIR / "augmentation_ablation.json"),
        ("Benchmark", METRIC_DIR / "benchmark.json"),
    ]
    available = []
    for name, path in metric_files:
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                available.append((name, json.load(handle)))
    if not available:
        st.info("No trained evidence artifacts yet. Run the pipeline first.")
    else:
        for name, payload in available:
            with st.expander(name, expanded=(name in {"Baseline CNN", "Transfer model"})):
                st.json(payload)

    for image_name in ["baseline_confusion_matrix.png", "transfer_confusion_matrix.png", "transfer_strong_confusion_matrix.png"]:
        image_path = ROOT / "artifacts" / "figures" / image_name
        if image_path.exists():
            st.image(str(image_path), caption=image_name.replace("_", " ").title())

with tab_about:
    st.markdown(
        """
### Project scope

EcoSort AI is a four-class image classifier for dry, wet, recyclable and e-waste. It is intended as a demonstrator, not a substitute for formal waste-management rules.

### What the system exposes

- Upload or camera input.
- Predicted class and confidence score.
- Top alternative classes.
- Grad-CAM visual explanation.
- Model-evidence panel for precision/recall/F1, confusion matrix and speed results.

### Reproducibility

Training configuration lives in `configs/training_config.json`; dataset mapping lives in `configs/class_mapping.json`.
"""
    )
