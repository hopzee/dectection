from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps


APP_TITLE = "Fake Signature Detection"
APP_SUBTITLE = "Upload a signature image and get a quick authenticity check."
MODEL_DIR = Path("models")
DEFAULT_MODEL_PATHS = [
    MODEL_DIR / "signature_model.pkl",
    MODEL_DIR / "signature_model.joblib",
    MODEL_DIR / "signature_model.h5",
]
HISTORY_STATE_KEY = "analysis_history"
PAGE_STATE_KEY = "history_page"
DEFAULT_PAGE_SIZE = 5


@st.cache_resource
def load_model():
    for path in DEFAULT_MODEL_PATHS:
        if not path.exists():
            continue

        suffix = path.suffix.lower()
        if suffix in {".pkl", ".joblib"}:
            try:
                import joblib

                return joblib.load(path), str(path)
            except Exception:
                continue

        if suffix == ".h5":
            try:
                from tensorflow.keras.models import load_model as keras_load_model

                return keras_load_model(path), str(path)
            except Exception:
                continue

    return None, None


def preprocess_image(image: Image.Image, target_size=(128, 128)) -> np.ndarray:
    gray = ImageOps.grayscale(image)
    resized = gray.resize(target_size)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    return arr


def extract_features(image_array: np.ndarray) -> dict:
    dark_pixels = image_array < 0.5
    ink_coverage = float(dark_pixels.mean())
    mean_intensity = float(image_array.mean())
    std_intensity = float(image_array.std())
    min_intensity = float(image_array.min())
    max_intensity = float(image_array.max())
    contrast = max_intensity - min_intensity

    flattened = image_array.flatten()
    hist_counts, hist_bins = np.histogram(flattened, bins=10, range=(0.0, 1.0))

    return {
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "ink_coverage": ink_coverage,
        "contrast": contrast,
        "min_intensity": min_intensity,
        "max_intensity": max_intensity,
        "hist_counts": hist_counts.tolist(),
        "hist_bins": hist_bins.tolist(),
    }


def demo_predict(image_array: np.ndarray):
    mean_intensity = float(image_array.mean())
    std_intensity = float(image_array.std())

    if mean_intensity < 0.55 and std_intensity > 0.18:
        label = "Likely Genuine"
        confidence = 0.71
    else:
        label = "Likely Forged"
        confidence = 0.69

    return label, confidence


def model_predict(model, image_array: np.ndarray):
    model_input = image_array.reshape(1, -1)

    try:
        prediction = model.predict(model_input)
    except Exception:
        prediction = model.predict(np.expand_dims(image_array, axis=(0, -1)))

    if isinstance(prediction, (list, tuple, np.ndarray)):
        pred = np.asarray(prediction)
        if pred.ndim == 2 and pred.shape[1] > 1:
            index = int(np.argmax(pred[0]))
            confidence = float(np.max(pred[0]))
            label = "Likely Genuine" if index == 1 else "Likely Forged"
            return label, confidence
        if pred.ndim == 1:
            value = float(pred[0])
        else:
            value = float(pred.squeeze())
    else:
        value = float(prediction)

    label = "Likely Genuine" if value >= 0.5 else "Likely Forged"
    confidence = value if value >= 0.5 else 1.0 - value
    return label, confidence


def classify_demo(features: dict, threshold: float):
    score = (
        (1.0 - features["mean_intensity"]) * 0.45
        + features["std_intensity"] * 0.35
        + features["ink_coverage"] * 0.20
    )
    confidence = float(np.clip(score, 0.0, 1.0))
    label = "Likely Genuine" if confidence >= threshold else "Likely Forged"
    return label, confidence


def add_history_entry(entry: dict):
    st.session_state.setdefault(HISTORY_STATE_KEY, [])
    st.session_state[HISTORY_STATE_KEY].insert(0, entry)


def paginate(items: list[dict], page: int, page_size: int):
    if page_size <= 0:
        page_size = DEFAULT_PAGE_SIZE
    total_pages = max(1, int(np.ceil(len(items) / page_size)))
    page = max(1, min(page, total_pages))
    start = (page - 1) * page_size
    end = start + page_size
    return items[start:end], total_pages


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="FS", layout="wide")
    st.session_state.setdefault(HISTORY_STATE_KEY, [])
    st.session_state.setdefault(PAGE_STATE_KEY, 1)

    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)

    model, model_path = load_model()

    with st.sidebar:
        st.header("Model Status")
        if model is None:
            st.warning("No trained model found. Running in demo mode.")
            st.write("Place a model in `models/` as:")
            st.code("signature_model.pkl\nsignature_model.joblib\nsignature_model.h5")
        else:
            st.success(f"Loaded model: {model_path}")

        st.divider()
        st.header("Analysis Controls")
        threshold = st.slider("Genuine threshold", min_value=0.10, max_value=0.95, value=0.50, step=0.01)
        page_size = st.selectbox("History page size", options=[3, 5, 10, 20], index=1)

        st.header("How it works")
        st.write("1. Upload a signature image")
        st.write("2. The app preprocesses it")
        st.write("3. The model or demo logic returns a result")

        if st.button("Clear history"):
            st.session_state[HISTORY_STATE_KEY] = []
            st.session_state[PAGE_STATE_KEY] = 1
            st.rerun()

    tab_overview, tab_analysis, tab_history = st.tabs(["Overview", "Image Analysis", "History"])

    uploaded = st.file_uploader("Upload signature image", type=["png", "jpg", "jpeg"])
    image = Image.open(uploaded).convert("RGB") if uploaded else None

    if image is None:
        with tab_overview:
            st.info("Upload an image to unlock the dashboard.")
            st.write("You’ll see the prediction, charts, and history panel here once a signature is uploaded.")
        with tab_analysis:
            st.info("No image uploaded yet.")
        with tab_history:
            st.info("No predictions recorded yet.")
        st.stop()

    processed = preprocess_image(image)
    features = extract_features(processed)

    if model is None:
        label, confidence = classify_demo(features, threshold)
        engine_name = "Demo engine"
    else:
        label, confidence = model_predict(model, processed)
        engine_name = "Loaded model"

    result_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "label": label,
        "confidence": confidence,
        "engine": engine_name,
        "mean_intensity": features["mean_intensity"],
        "std_intensity": features["std_intensity"],
        "ink_coverage": features["ink_coverage"],
        "contrast": features["contrast"],
        "file_name": getattr(uploaded, "name", "uploaded_image"),
    }
    add_history_entry(result_entry)

    score_cols = st.columns(4)
    score_cols[0].metric("Prediction", label)
    score_cols[1].metric("Confidence", f"{confidence:.2%}")
    score_cols[2].metric("Ink coverage", f"{features['ink_coverage']:.2%}")
    score_cols[3].metric("Contrast", f"{features['contrast']:.2f}")

    with tab_overview:
        left, right = st.columns([1.15, 1])

        with left:
            st.subheader("Prediction Summary")
            summary_df = pd.DataFrame(
                {
                    "Field": ["Prediction", "Confidence", "Engine", "Uploaded File"],
                    "Value": [label, f"{confidence:.2%}", engine_name, result_entry["file_name"]],
                }
            )
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            if "Genuine" in label:
                st.success("The signature looks authentic.")
            else:
                st.error("The signature looks suspicious.")

            st.download_button(
                label="Download analysis as CSV",
                data=pd.DataFrame([result_entry]).to_csv(index=False).encode("utf-8"),
                file_name="signature_analysis.csv",
                mime="text/csv",
            )

        with right:
            st.subheader("Image Preview")
            st.image(image, caption="Uploaded signature", use_column_width=True)
            st.image(processed, caption="Preprocessed signature", clamp=True, use_column_width=True)

    with tab_analysis:
        chart_left, chart_right = st.columns([1.2, 1])

        hist_df = pd.DataFrame(
            {
                "Intensity Bin": [f"{i + 1}" for i in range(len(features["hist_counts"]))],
                "Count": features["hist_counts"],
            }
        )

        feature_df = pd.DataFrame(
            {
                "Feature": ["Mean", "Std Dev", "Ink Coverage", "Contrast"],
                "Value": [
                    features["mean_intensity"],
                    features["std_intensity"],
                    features["ink_coverage"],
                    features["contrast"],
                ],
            }
        )

        with chart_left:
            st.subheader("Pixel Intensity Distribution")
            st.bar_chart(hist_df.set_index("Intensity Bin"))
            st.caption("Each bar shows how many pixels fall into a brightness range.")

            st.subheader("Feature Scores")
            st.line_chart(feature_df.set_index("Feature"))

        with chart_right:
            st.subheader("Feature Breakdown")
            st.dataframe(
                feature_df.style.format({"Value": "{:.4f}"}),
                use_container_width=True,
                hide_index=True,
            )

            st.subheader("Decision Hint")
            hint = "More dark ink and texture usually raise the demo authenticity score." if model is None else "Loaded model output is used directly."
            st.info(hint)

            st.subheader("Quick Facts")
            st.write(f"Minimum intensity: `{features['min_intensity']:.4f}`")
            st.write(f"Maximum intensity: `{features['max_intensity']:.4f}`")
            st.write(f"Contrast: `{features['contrast']:.4f}`")

    with tab_history:
        st.subheader("Prediction History")
        history = st.session_state[HISTORY_STATE_KEY]

        if not history:
            st.info("No history yet. Upload a few signature images to build the timeline.")
        else:
            if st.session_state[PAGE_STATE_KEY] < 1:
                st.session_state[PAGE_STATE_KEY] = 1

            page_items, total_pages = paginate(history, st.session_state[PAGE_STATE_KEY], page_size)
            prev_col, page_col, next_col = st.columns([1, 1, 1])

            with prev_col:
                if st.button("Previous page", disabled=st.session_state[PAGE_STATE_KEY] <= 1):
                    st.session_state[PAGE_STATE_KEY] -= 1
                    st.rerun()

            with page_col:
                st.write(f"Page **{st.session_state[PAGE_STATE_KEY]}** of **{total_pages}**")

            with next_col:
                if st.button("Next page", disabled=st.session_state[PAGE_STATE_KEY] >= total_pages):
                    st.session_state[PAGE_STATE_KEY] += 1
                    st.rerun()

            history_df = pd.DataFrame(page_items)
            st.dataframe(
                history_df[
                    ["timestamp", "file_name", "label", "confidence", "engine", "mean_intensity", "std_intensity", "ink_coverage"]
                ],
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                label="Download full history as CSV",
                data=pd.DataFrame(history).to_csv(index=False).encode("utf-8"),
                file_name="signature_history.csv",
                mime="text/csv",
            )

    st.divider()


if __name__ == "__main__":
    main()
