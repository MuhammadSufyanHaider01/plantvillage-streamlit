import streamlit as st
import torch
from PIL import Image
import os

st.set_page_config(page_title="Plant Disease Detection", layout="centered")

WEIGHTS_PATH = "best.pt"  # make sure this file exists in the same folder

@st.cache_resource
def load_model(weights_path: str):
    model = torch.hub.load(
        "ultralytics/yolov5",
        "custom",
        path=weights_path,
        force_reload=False
    )
    return model


st.title("🌿 Plant Disease Detection (YOLOv5)")
st.write("Upload a leaf image to get disease prediction with bounding box.")

conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Input image", use_column_width=True)

    model = load_model(WEIGHTS_PATH)
    model.conf = conf

    results = model(img)

    annotated = results.render()[0]
    st.image(annotated, caption="Prediction", use_column_width=True)

    df = results.pandas().xyxy[0]
    if df.empty:
        st.warning("No detections found. Try lowering confidence.")
    else:
        st.subheader("Detections")
        st.dataframe(
            df[["name", "confidence", "xmin", "ymin", "xmax", "ymax"]]
            .sort_values("confidence", ascending=False),
            use_container_width=True
        )
else:
    st.info("Please upload an image to start inference.")
