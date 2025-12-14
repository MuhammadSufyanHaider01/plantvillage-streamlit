import sys
from pathlib import Path


@st.cache_resource
def load_model(weights_path: str):
    model = torch.hub.load("yolov5", "custom", path=weights_path, source="local", force_reload=False)
    return model

