import streamlit as st
import cv2
import numpy as np
import time
from collections import defaultdict
from ultralytics import YOLO
from namirnice_baze import UserDB, ListDB, FoodDB

# Inicijalizacija baza i modela
user_db = UserDB()
list_db = ListDB()
food_db = FoodDB()

model = YOLO("C:/Users/anita/Desktop/Faks_NOVO/2. GODINA/IV. semestar/Primjenjeno strojno učenje/Projekt/namirnice_dataset/runs/detect/namirnice_train5/weights/best.pt")

thresholds = {
    'kikiriki': 0.55, 'jaja': 0.45, 'riza': 0.35,
    'rajcica': 0.40, 'banane': 0.40, 'kruh': 0.35,
    'krastavci': 0.30, 'pivo': 0.15
}

# Streamlit UI
st.title("📷 Detekcija namirnica kamerom mobilnog uređaja")
st.markdown("Format URL-a: `http://<IP>:8080/video`")

username = st.text_input("Korisničko ime:")
camera_url = st.text_input("URL kamere")

if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'detected_items' not in st.session_state:
    st.session_state.detected_items = defaultdict(lambda: {"count": 0, "confidence": 0})
if 'detecting' not in st.session_state:
    st.session_state.detecting = False

if st.button("Prijavi se"):
    if username:
        user_id = user_db.create_user(username)
        st.session_state.user_id = user_id
        st.success(f"Korisnik '{username}' prijavljen. ID: {user_id}")
    else:
        st.error("Unesi korisničko ime.")

if st.button("🎬 Pokreni detekciju"):
    if not username or not camera_url:
        st.error("Unesi korisničko ime i URL kamere.")
    else:
        st.session_state.detecting = True

if st.session_state.detecting:
    stframe = st.empty()
    cap = cv2.VideoCapture(camera_url)

    if not cap.isOpened():
        st.error("Ne mogu otvoriti video stream.")
        st.session_state.detecting = False
        st.stop()

    stop = st.button("⏹ Zaustavi detekciju")

    while cap.isOpened() and not stop:
        ret, frame = cap.read()
        if not ret:
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

        annotated = results.plot()
        stframe.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")

        time.sleep(0.1)

    cap.release()
    st.session_state.detecting = False

    # --- Spremanje ---
    if st.session_state.user_id:
        to_save = {}
        for item, data in st.session_state.detected_items.items():
            for i in range(data["count"]):
                to_save[f"{item}#{i+1}"] = data["confidence"]
        list_db.save_list(st.session_state.user_id, to_save)
        st.success("Popis detektiranih namirnica uspješno spremljen!")
        st.session_state.detected_items = defaultdict(lambda: {"count": 0, "confidence": 0})
    else:
        st.error("Nema korisničkog ID-a.")

# --- Prikaz detekcija ---
if st.session_state.detected_items:
    st.subheader("🛒 Detektirane namirnice:")
    for item, data in st.session_state.detected_items.items():
        st.write(f"- {item} x{data['count']} (najveća točnost: {data['confidence']:.2f})")
