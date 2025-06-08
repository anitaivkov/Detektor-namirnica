import os
import requests
from ultralytics import YOLO
import streamlit as st

# === CONFIG ===
MODEL_URL = "https://github.com/anitaivkov/Detektor-namirnica/releases/download/v1.0.0/best.pt"
MODEL_PATH = "weights/best.pt"

# === DOWNLOAD MODEL IF NEEDED ===
def download_model_if_needed(url: str, filepath: str):
    """Download model if it does not exist locally."""
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)

    if not os.path.exists(filepath):
        st.info(f"Pokušavam preuzeti model s: {url}")
        try:
            response = requests.get(url, stream=True) # stream=True za veće datoteke
            response.raise_for_status() # Provjerava HTTP greške

            total_size_in_bytes = int(response.headers.get('content-length', 0))
            st.info(f"Očekivana veličina datoteke: {total_size_in_bytes / (1024*1024):.2f} MB")

            block_size = 1024 
            downloaded_size = 0

            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        st.progress(downloaded_size / total_size_in_bytes) 

            if total_size_in_bytes > 0 and downloaded_size != total_size_in_bytes:
                st.warning(f"Upozorenje: Preuzimanje je nepotpuno. Preuzeto {downloaded_size / (1024*1024):.2f} MB od {total_size_in_bytes / (1024*1024):.2f} MB.")
                raise ValueError("Nepotpuno preuzimanje datoteke.")
            elif total_size_in_bytes == 0 and downloaded_size == 0:
                st.warning("Upozorenje: Preuzeta datoteka ima 0 bajta. Možda je to i dalje problem s URL-om ili pristupom.")
                st.error("Model nije preuzet (veličina 0 bajta). Provjerite URL ili dopuštenja.")
                raise ValueError("Model nije preuzet, veličina je 0 bajta.")
            else:
                st.success("Model uspješno preuzet!")

        except requests.exceptions.RequestException as e:
            st.error(f"Greška prilikom preuzimanja modela (mrežna greška ili HTTP status nije 2xx): {e}")
            raise # Ponovno digni iznimku da se prikaže u Streamlit logovima
        except ValueError as e:
            st.error(f"Greška u sadržaju preuzimanja ili nepotpuno preuzimanje: {e}")
            raise
        except Exception as e:
            st.error(f"Neočekivana greška prilikom spremanja ili obrade modela: {e}")
            raise

# === LOAD MODEL ===
def load_model(model_path: str = MODEL_PATH):
    download_model_if_needed(MODEL_URL, model_path)
    # Nakon što je model preuzet, provjeri veličinu datoteke radi dijagnostike
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        st.info(f"Veličina preuzetog modela '{model_path}': {size_mb:.2f} MB")
        if size_mb < 1.0 and size_mb > 0: # Prilagodi ovu vrijednost ako je model manji od 1MB
            st.warning("Upozorenje: Preuzeti model je vrlo malen. Moguće da nije ispravno preuzet ili je oštećen.")
            st.error("Pokušavam učitati maleni model, što će vjerojatno rezultirati greškom.")
        elif size_mb == 0:
            st.error("Model datoteka je preuzeta s 0 bajta. Ne mogu učitati prazan model.")
            raise ValueError(f"Model datoteka je prazna: {model_path}")
    else:
        st.error("Model datoteka ne postoji nakon pokušaja preuzimanja!")
        raise FileNotFoundError(f"Model datoteka ne postoji: {model_path}")

    return YOLO(model_path)
