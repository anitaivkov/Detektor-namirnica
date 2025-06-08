import os
import requests
import streamlit as st
from ultralytics import YOLO

# === CONFIG ===
# Ažurirani URL za direktno preuzimanje s Google Drivea (često treba 'confirm')
MODEL_URL = "https://drive.google.com/uc?export=download&id=15E8nST9vJ4saRbup6dZgF8DIBzWYMG9J"
MODEL_PATH = "weights/best.pt"

# === DOWNLOAD MODEL IF NEEDED ===
def download_model_if_needed(url: str, filepath: str):
    """Download model if it does not exist locally."""
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)

    if not os.path.exists(filepath):
        st.info(f"Pokušavam preuzeti model s {url}...")
        try:
            # Rješavanje Google Drive redirekcije i potvrde preuzimanja
            session = requests.Session()
            response = session.get(url, stream=True)
            token = None
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    token = value
                    break

            if token:
                params = {'id': url.split('id=')[1], 'confirm': token}
                response = session.get(url, params=params, stream=True)
            
            response.raise_for_status() # Provjeri HTTP greške

            # Dodaj provjeru da li je sadržaj HTML
            if 'text/html' in response.headers.get('Content-Type', ''):
                st.error("Greška: Izgleda da Google Drive vraća HTML stranicu umjesto datoteke modela.")
                st.error("Provjerite URL ili dopuštenja za dijeljenje datoteke na Google Driveu.")
                # st.code(response.text[:500]) # Ispišite prvih 500 znakova HTML-a za dijagnostiku
                raise ValueError("Preuzeta datoteka je HTML, a ne model.")

            total_size_in_bytes = int(response.headers.get('content-length', 0))
            block_size = 1024 
            downloaded_size = 0

            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk: # filtrirajte keep-alive pakete
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        # Prikaz napretka preuzimanja u Streamlitu
                        st.progress(downloaded_size / total_size_in_bytes)
            
            if total_size_in_bytes > 0 and downloaded_size < total_size_in_bytes:
                st.warning("Upozorenje: Preuzimanje nije dovršeno u potpunosti.")
                raise ValueError("Nepotpuno preuzimanje datoteke.")
            
            st.success("Model uspješno preuzet!")

        except requests.exceptions.RequestException as e:
            st.error(f"Greška prilikom preuzimanja modela (mrežna greška): {e}")
            raise # Ponovno digni iznimku da se prikaže u logovima
        except ValueError as e:
            st.error(f"Greška u sadržaju preuzimanja: {e}")
            raise
        except Exception as e:
            st.error(f"Neočekivana greška prilikom spremanja ili obrade modela: {e}")
            raise

# === LOAD MODEL ===
def load_model(model_path: str = MODEL_PATH):
    download_model_if_needed(MODEL_URL, model_path)
    # Nakon što je model preuzet, provjeri veličinu datoteke radi dijagnostike
    if os.path.exists(model_path):
        st.info(f"Veličina preuzetog modela '{model_path}': {os.path.getsize(model_path) / (1024*1024):.2f} MB")
    return YOLO(model_path)
