import os
import streamlit as st
import torch
from PIL import Image

WEIGHTS_PATH = "best.pt"

st.set_page_config(page_title="Plant Disease Detection", layout="centered")

@st.cache_resource
def load_model(weights_path: str):
    # Load YOLOv5 from the LOCAL yolov5 folder in this repo
    model = torch.hub.load(
        "yolov5",        # folder name that contains hubconf.py
        "custom",
        path=weights_path,
        source="local",
        force_reload=False
    )
    return model

st.title("🌿 Plant Disease Detection (YOLOv5)")
conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Input", use_column_width=True)

    model = load_model(WEIGHTS_PATH)
    model.conf = conf

    results = model(img)
    annotated = results.render()[0]
    st.image(annotated, caption="Prediction", use_column_width=True)

    df = results.pandas().xyxy[0]
    if df.empty:
        st.warning("No detections found. Try lowering confidence.")
    else:
        st.dataframe(
            df[["name", "confidence", "xmin", "ymin", "xmax", "ymax"]]
              .sort_values("confidence", ascending=False),
            use_container_width=True
        )
else:
    st.info("Upload an image to run inference.")
