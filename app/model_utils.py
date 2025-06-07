import os
import requests
from ultralytics import YOLO

# === CONFIG ===
MODEL_URL = "https://drive.google.com/uc?export=download&id=15E8nST9vJ4saRbup6dZgF8DIBzWYMG9J"
MODEL_PATH = "weights/best.pt"

# === DOWNLOAD MODEL IF NEEDED ===
def download_model_if_needed(url: str, filepath: str):
    """Download model if it does not exist locally."""
    if not os.path.exists(filepath):
        response = requests.get(url)
        response.raise_for_status()
        with open(filepath, "wb") as f:
            f.write(response.content)

# === LOAD MODEL ===
def load_model(model_path: str = MODEL_PATH):
    download_model_if_needed(MODEL_URL, model_path)
    return YOLO(model_path)
