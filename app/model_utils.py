import os
import requests
import streamlit as st
from ultralytics import YOLO

# === CONFIG ===
MODEL_URL = "https://drive.google.com/uc?export=download&id=15E8nST9vJ4saRbup6dZgF8DIBzWYMG9J"
MODEL_PATH = "weights/best.pt" # Putanja do modela

# === DOWNLOAD MODEL IF NEEDED ===
def download_model_if_needed(url: str, filepath: str):
    """Download model if it does not exist locally."""

    # Izvuci putanju do direktorija iz filepath-a
    directory = os.path.dirname(filepath)
    # Stvori direktorij ako ne postoji
    if not os.path.exists(directory):
        os.makedirs(directory) # os.makedirs će stvoriti i nadređene direktorije ako je potrebno

    if not os.path.exists(filepath):
        st.info(f"Preuzimam model s {url}...") # info poruka za debug
        try:
            response = requests.get(url)
            response.raise_for_status() # Provjerava HTTP greške
            with open(filepath, "wb") as f:
                f.write(response.content)
            st.success("Model uspješno preuzet!") # Potvrdna poruka
        except requests.exceptions.RequestException as e:
            st.error(f"Greška prilikom preuzimanja modela: {e}")
            raise # Ponovno digni iznimku da se prikaže u logovima
        except Exception as e:
            st.error(f"Neočekivana greška prilikom spremanja modela: {e}")
            raise

# === LOAD MODEL ===
def load_model(model_path: str = MODEL_PATH):
    download_model_if_needed(MODEL_URL, model_path)
    return YOLO(model_path)
