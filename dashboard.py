"""
dashboard.py

Streamlit dashboard for visualizing the results of train_eurosat_cnn.py:
training curves, class distribution, confusion matrix, per-class accuracy,
and a sample-image prediction viewer (true label vs predicted label,
rendered as an RGB composite from the 13-band Sentinel-2 patch).

Run from the project root (multimodal_rs_pipeline) with the rs_pipeline
conda environment active, AFTER running train_eurosat_cnn.py at least once:

    streamlit run dashboard.py

Reads:
    results/training_history.json   (written by train_eurosat_cnn.py)
    results/val_predictions.json    (written by train_eurosat_cnn.py)
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import rasterio
import streamlit as st

st.set_page_config(page_title="EuroSAT Training Dashboard", layout="wide")

RESULTS_DIR = Path("results")
HISTORY_PATH = RESULTS_DIR / "training_history.json"
PREDICTIONS_PATH = RESULTS_DIR / "val_predictions.json"

RGB_BAND_INDICES = (3, 2, 1)  # B04(red), B03(green), B02(blue)


@st.cache_data
def load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def tif_to_rgb_display(filepath: str, band_indices=RGB_BAND_INDICES) -> np.ndarray:
    with rasterio.open(filepath) as src:
        arr = src.read().astype(np.float32)
    rgb = arr[list(band_indices), :, :]
    rgb = np.transpose(rgb, (1, 2, 0))
    p2, p98 = np.percentile(rgb, (2, 98))
    if p98 > p2:
        rgb = np.clip((rgb - p2) / (p98 - p2), 0, 1)
    else:
        rgb = np.clip(rgb / max(rgb.max(), 1.0), 0, 1)
    return (rgb * 255).astype(np.uint8)


if not HISTORY_PATH.exists() or not PREDICTIONS_PATH.exists():
    st.title("EuroSAT Training Dashboard")
    st.error(
        "No results found yet. Run `python train_eurosat_cnn.py` first "
        "(from the project root, with the rs_pipeline environment active) "
        "to generate `results/training_history.json` and "
        "`results/val_predictions.json`, then reload this page."
    )
    st.stop()

history = load_json(HISTORY_PATH)
predictions = load_json(PREDICTIONS_PATH)

class_to_idx = history["class_to_idx"]
idx_to_class = {v: k for k, v in class_to_idx.items()}
class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

st.title("EuroSAT Land-Cover Classification — Training Dashboard")
st.caption("CustomCNN trained on multi-spectral (13-band) Sentinel-2 patches")

final_train_acc = history["train_acc"][-1]
final_val_acc = history["val_acc"][-1]
best_val_acc = max(history["val_acc"])

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Epochs", history["num_epochs"])
col2.metric("Train samples", history["train_samples"])
col3.metric("Val samples", history["val_samples"])
col4.metric("Final train acc", f"{final_train_acc:.1%}")
col5.metric("Best val acc", f"{best_val_acc:.1%}", delta=f"{best_val_acc - 0.10:.1%} vs random")

st.divider()

st.subheader("Training curves")
epochs = list(range(1, history["num_epochs"] + 1))
curve_col1, curve_col2 = st.columns(2)

with curve_col1:
    loss_fig = go.Figure()
    loss_fig.add_trace(go.Scatter(x=epochs, y=history["train_loss"], mode="lines+markers", name="Train loss"))
    loss_fig.add_trace(go.Scatter(x=epochs, y=history["val_loss"], mode="lines+markers", name="Val loss"))
    loss_fig.update_layout(title="Loss per epoch", xaxis_title="Epoch", yaxis_title="Loss", height=380)
    st.plotly_chart(loss_fig, use_container_width=True)

with curve_col2:
    acc_fig = go.Figure()
    acc_fig.add_trace(go.Scatter(x=epochs, y=history["train_acc"], mode="lines+markers", name="Train accuracy"))
    acc_fig.add_trace(go.Scatter(x=epochs, y=history["val_acc"], mode="lines+markers", name="Val accuracy"))
    acc_fig.add_hline(y=1.0 / len(class_names), line_dash="dot", annotation_text="Random baseline")
    acc_fig.update_layout(title="Accuracy per epoch", xaxis_title="Epoch", yaxis_title="Accuracy",
                           yaxis_tickformat=".0%", height=380)
    st.plotly_chart(acc_fig, use_container_width=True)

st.divider()

if history.get("class_counts"):
    st.subheader("Training data class distribution")
    dist_df = pd.DataFrame(
        {"class": list(history["class_counts"].keys()), "count": list(history["class_counts"].values())}
    ).sort_values("count", ascending=False)
    dist_fig = px.bar(dist_df, x="class", y="count", title="Samples used per class (after subsampling)")
    dist_fig.update_layout(height=350)
    st.plotly_chart(dist_fig, use_container_width=True)
    st.divider()

st.subheader("Confusion matrix (validation set)")
true_labels = predictions["true_labels"]
pred_labels = predictions["pred_labels"]
num_classes = len(class_names)

confusion = np.zeros((num_classes, num_classes), dtype=int)
for t, p in zip(true_labels, pred_labels):
    confusion[t, p] += 1

confusion_fig = px.imshow(
    confusion, x=class_names, y=class_names,
    labels=dict(x="Predicted", y="Actual", color="Count"),
    text_auto=True, color_continuous_scale="Blues", aspect="auto",
)
confusion_fig.update_layout(height=550)
st.plotly_chart(confusion_fig, use_container_width=True)

st.divider()

st.subheader("Per-class accuracy")
per_class_correct = Counter()
per_class_total = Counter()
for t, p in zip(true_labels, pred_labels):
    per_class_total[t] += 1
    if t == p:
        per_class_correct[t] += 1

per_class_df = pd.DataFrame({
    "class": [idx_to_class[i] for i in range(num_classes)],
    "accuracy": [per_class_correct[i] / per_class_total[i] if per_class_total[i] > 0 else 0.0
                 for i in range(num_classes)],
    "num_samples": [per_class_total[i] for i in range(num_classes)],
}).sort_values("accuracy", ascending=True)

per_class_fig = px.bar(per_class_df, x="accuracy", y="class", orientation="h",
                        title="Validation accuracy by class", text="num_samples", range_x=[0, 1])
per_class_fig.update_layout(height=450, xaxis_tickformat=".0%")
per_class_fig.update_traces(texttemplate="n=%{text}", textposition="outside")
st.plotly_chart(per_class_fig, use_container_width=True)

st.divider()

st.subheader("Sample predictions")
filter_col1, filter_col2 = st.columns(2)
with filter_col1:
    show_only = st.radio("Show", ["All", "Correct only", "Incorrect only"], horizontal=True)
with filter_col2:
    num_to_show = st.slider("Number of samples", min_value=4, max_value=24, value=8, step=4)

filepaths = predictions["filepaths"]
confidences = predictions["confidences"]

indices = list(range(len(filepaths)))
if show_only == "Correct only":
    indices = [i for i in indices if true_labels[i] == pred_labels[i]]
elif show_only == "Incorrect only":
    indices = [i for i in indices if true_labels[i] != pred_labels[i]]
indices = indices[:num_to_show]

if not indices:
    st.info("No samples match this filter.")
else:
    cols_per_row = 4
    for row_start in range(0, len(indices), cols_per_row):
        row_indices = indices[row_start:row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, idx in zip(cols, row_indices):
            true_name = idx_to_class[true_labels[idx]]
            pred_name = idx_to_class[pred_labels[idx]]
            is_correct = true_labels[idx] == pred_labels[idx]
            confidence = confidences[idx]

            with col:
                try:
                    rgb_image = tif_to_rgb_display(filepaths[idx])
                    st.image(rgb_image, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not render image: {e}")

                if is_correct:
                    st.success(f"True: {true_name}\n\nPred: {pred_name} ({confidence:.0%})")
                else:
                    st.error(f"True: {true_name}\n\nPred: {pred_name} ({confidence:.0%})")

st.divider()
st.caption(
    f"Run config — batch size: {history['batch_size']}, learning rate: {history['learning_rate']}, "
    f"epochs: {history['num_epochs']}"
)