from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CHECKPOINT_DIR, METRIC_DIR
from src.gradcam import explain_image
from src.inference import WasteClassifier

FINAL_CHECKPOINT = CHECKPOINT_DIR / "final_model.pth"
FINAL_META = METRIC_DIR / "final_model_meta.json"


@st.cache_data
def load_meta() -> dict | None:
    if not FINAL_META.exists():
        return None
    with FINAL_META.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def render_model_evidence(payload: dict) -> None:
    metrics = [
        ("Test accuracy", "test_accuracy"),
        ("Macro precision", "macro_precision"),
        ("Macro recall", "macro_recall"),
        ("Macro F1", "macro_f1"),
        ("Weighted F1", "weighted_f1"),
    ]
    available_metrics = [(label, payload[key]) for label, key in metrics if key in payload]
    if available_metrics:
        columns = st.columns(len(available_metrics))
        for column, (label, value) in zip(columns, available_metrics):
            column.metric(label, f"{float(value):.1%}")

    details = []
    if payload.get("num_test_images") is not None:
        details.append(f"{payload['num_test_images']} test images")
    if payload.get("augmentation_used_for_training"):
        details.append(f"{payload['augmentation_used_for_training']} training augmentation")
    if details:
        st.caption(" · ".join(details))

    report = payload.get("classification_report", {})
    class_rows = [
        {
            "Class": class_name.replace("-", " ").title(),
            "Precision": f"{scores['precision']:.1%}",
            "Recall": f"{scores['recall']:.1%}",
            "F1 score": f"{scores['f1-score']:.1%}",
            "Test images": int(scores["support"]),
        }
        for class_name, scores in report.items()
        if class_name not in {"accuracy", "macro avg", "weighted avg"}
        and isinstance(scores, dict)
        and all(key in scores for key in ("precision", "recall", "f1-score", "support"))
    ]
    if class_rows:
        with st.expander("Per-class performance"):
            st.dataframe(pd.DataFrame(class_rows), hide_index=True, use_container_width=True)


def render_ablation_evidence(payload: dict) -> None:
    st.caption("Validation macro-F1 by training setup")
    columns = st.columns(max(len(payload), 1))
    for column, (variant, score) in zip(columns, payload.items()):
        label = variant.replace("_", " ").title()
        column.metric(label, f"{float(score):.1%}")


def render_benchmark_evidence(payload: dict) -> None:
    latency = payload.get("milliseconds_per_image")
    throughput = payload.get("images_per_second")
    device = payload.get("device")
    metrics = []
    if latency is not None:
        metrics.append(("Per-image inference", f"{float(latency):.1f} ms"))
    if throughput is not None:
        metrics.append(("Throughput", f"{float(throughput):.1f} images/s"))
    if device:
        metrics.append(("Device", str(device).upper()))
    if metrics:
        columns = st.columns(len(metrics))
        for column, (label, value) in zip(columns, metrics):
            column.metric(label, value)
    if payload.get("note"):
        st.caption(payload["note"])


@st.cache_resource(show_spinner=False)
def load_classifier(checkpoint: str, device: str = "cpu"):
    return WasteClassifier(checkpoint, device=device)


def show_sidebar(meta: dict | None) -> None:
    with st.sidebar:
        st.header("Model")
        if meta:
            st.write(f"**Architecture:** {meta.get('model_kind', 'unknown')}")
            st.write(f"**Augmentation:** {meta.get('augmentation', 'unknown')}")
            st.write("**Selection metric:** validation macro-F1")
            st.write(f"**Selection value:** {meta.get('selection_value', 0):.3f}")
        else:
            st.warning("No trained final checkpoint found yet.")
            st.code("python -m src.run_pipeline", language="powershell")


def prepare_image() -> Image.Image | None:
    source = st.radio("Input source", ["Upload image", "Use camera"], horizontal=True)
    if source == "Upload image":
        uploaded = st.file_uploader("Upload a waste photo", type=["jpg", "jpeg", "png", "webp", "bmp"])
        if uploaded is not None:
            return Image.open(uploaded).convert("RGB")
        return None

    camera_generation = st.session_state.get("camera_generation", 0)
    if st.button("Restart camera", icon=":material/refresh:", help="Reopen the camera preview if it stopped"):
        camera_generation += 1
        st.session_state["camera_generation"] = camera_generation

    camera = st.camera_input("Take a waste photo", key=f"waste_camera_{camera_generation}")
    if camera is not None:
        return Image.open(camera).convert("RGB")
    return None


def render_prediction_panel(image: Image.Image) -> None:
    left, right = st.columns(2)
    with left:
        st.image(image, caption="Input image", use_container_width=True)

    if not FINAL_CHECKPOINT.exists():
        with right:
            st.error("The final trained model is not present yet. Add your dataset, run the pipeline, and refresh the app.")
        return

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


def render_evidence_tab() -> None:
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
                if name in {"Baseline CNN", "Transfer model"}:
                    render_model_evidence(payload)
                elif name == "Augmentation":
                    render_ablation_evidence(payload)
                elif name == "Benchmark":
                    render_benchmark_evidence(payload)

    for image_name in [
        "baseline_confusion_matrix.png",
        "transfer_confusion_matrix.png",
        "transfer_strong_confusion_matrix.png",
    ]:
        image_path = ROOT / "artifacts" / "figures" / image_name
        if image_path.exists():
            st.image(str(image_path), caption=image_name.replace("_", " ").title())


def render_about_tab() -> None:
    st.markdown(
        """
### Project scope
EcoSort AI is a four-class image classifier for dry, wet, recyclable and e-waste.

### What the app exposes
- Upload or camera input
- Predicted class and confidence score
- Top alternative classes
- Grad-CAM visual explanation
- Model evidence for metrics and speed

### Reproducibility
Training config lives in `configs/training_config.json`; dataset mapping lives in `configs/class_mapping.json`.
"""
    )


def main() -> None:
    st.set_page_config(page_title="EcoSort AI", page_icon="♻️", layout="wide")
    st.title("♻️ EcoSort AI")
    st.caption("Waste classification with confidence scores and Grad-CAM explanations.")

    meta = load_meta()
    show_sidebar(meta)

    classify_tab, evidence_tab, about_tab = st.tabs(["Classify", "Evidence", "About"])
    with classify_tab:
        image = prepare_image()
        if image is not None:
            render_prediction_panel(image)
    with evidence_tab:
        render_evidence_tab()
    with about_tab:
        render_about_tab()


if __name__ == "__main__":
    main()
