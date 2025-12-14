import torch

@st.cache_resource
def load_model(weights_path: str):
    return torch.hub.load(
        "yolov5",        # local folder in your repo
        "custom",
        path=weights_path,
        source="local",  # do NOT fetch from GitHub
        force_reload=False
    )
