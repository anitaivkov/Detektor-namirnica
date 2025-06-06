import streamlit as st
import os
import cv2
import numpy as np
import sqlite3
from PIL import Image
from collections import defaultdict
from ultralytics import YOLO
from namirnice_baze import UserDB, ListDB, FoodDB
import time

# Inicijalizacija
user_db = UserDB()
list_db = ListDB()
food_db = FoodDB()

# YOLO model
model = YOLO("C:/Users/anita/Desktop/Faks_NOVO/2. GODINA/IV. semestar/Primjenjeno strojno učenje/Projekt/namirnice_dataset/runs/detect/namirnice_train5/weights/best.pt")

# Pragovi po klasama
thresholds = {
    'kikiriki': 0.55, 'jaja': 0.45, 'riza': 0.35,
    'rajcica': 0.40, 'banane': 0.40, 'kruh': 0.35,
    'krastavci': 0.30, 'pivo': 0.15
}

st.title("📷 Detekcija namirnica kamerom mobilnog uređaja")
st.markdown("Poveži IP Webcam i unesi URL za stream. Format: `http://<IP>:8080/video`")

# --- Korisnik ---
username = st.text_input("Korisničko ime:")
if username and st.button("Prijavi se"):
    user_id = user_db.create_user(username)
    st.session_state.user_id = user_id
    st.success(f"Korisnik '{username}' prijavljen!")

# --- Kamera URL ---
camera_url = st.text_input("URL kamere (npr. http://192.168.1.10:8080/video)")

# --- Pokretanje detekcije ---
start = st.button("🎬 Pokreni detekciju")

# --- Pohranjene detekcije ---
if 'detected_items' not in st.session_state:
    st.session_state.detected_items = defaultdict(lambda: {"count": 0, "confidence": 0})

if 'detecting' not in st.session_state:
    st.session_state.detecting = False

# --- Pokretanje detekcije ---
if start and camera_url and username:
    st.session_state.detecting = True
    st.success("Detekcija pokrenuta.")

# --- Detekcija ---
if st.session_state.detecting and camera_url and username:
    stframe = st.empty()
    cap = cv2.VideoCapture(camera_url)

    if not cap.isOpened():
        st.error("Ne mogu otvoriti video stream. Provjeri IP i je li IP Webcam pokrenut.")
        st.stop()

    stop_button = st.button("⏹ Zaustavi")
    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret or frame is None:
            st.warning("Nema frame-a iz kamere.")
            break

        results = model.predict(frame, imgsz=640, conf=0.25, verbose=False)[0]
        detections = {}
        for box in results.boxes:
            conf = box.conf.item()
            class_name = model.names[int(box.cls)].lower()
            if conf > thresholds.get(class_name, 0.25):
                detections[class_name] = max(detections.get(class_name, 0), conf)

        for item, conf in detections.items():
            st.session_state.detected_items[item]["count"] += 1
            st.session_state.detected_items[item]["confidence"] = max(
                st.session_state.detected_items[item]["confidence"], conf
            )

        annotated_frame = results.plot()
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        stframe.image(annotated_frame, channels="RGB", use_container_width=True)

        time.sleep(0.1)

    cap.release()
    st.session_state.detecting = False

    # --- Spremanje ---
    user_id = st.session_state.get("user_id")
    if user_id:
        items_to_save = {}
        for item, data in st.session_state.detected_items.items():
            for i in range(data["count"]):
                items_to_save[f"{item}#{i+1}"] = data["confidence"]
        list_db.save_list(user_id, items_to_save)
        st.success("Popis uspješno spremljen!")
        st.session_state.detected_items = defaultdict(lambda: {"count": 0, "confidence": 0})
    else:
        st.error("Korisnik nije pronađen.")

# --- Prikaz detekcija ---
if st.session_state.detected_items:
    st.subheader("🛒 Detektirane namirnice:")
    for item, data in st.session_state.detected_items.items():
        st.write(f"- {item} x{data['count']} (najveća točnost: {data['confidence']:.2f})")
