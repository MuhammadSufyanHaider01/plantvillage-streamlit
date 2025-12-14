@st.cache_resource
def load_model(weights_path: str):
    return torch.hub.load(
        "yolov5",          # LOCAL folder name
        "custom",
        path=weights_path,
        source="local",    # IMPORTANT: do not fetch from GitHub
        force_reload=False
    )
