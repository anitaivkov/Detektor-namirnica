import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
import sqlite3
from ultralytics import YOLO
import pandas as pd
import threading

# === Učitaj YOLO model ===
model = YOLO("C:/Users/Viktorija/Desktop/Detektor-namirnica-main/Detektor-namirnica-main/namirnice_dataset/runs/detect/namirnice_train5/weights/best.pt")

# === Class thresholds ===
CLASS_THRESHOLDS = {
    'kikiriki': 0.55,
    'jaja': 0.45,
    'riza': 0.35,
    'rajcica': 0.40,
    'banane': 0.40,
    'kruh': 0.35,
    'krastavci': 0.30,
    'pivo': 0.15
}

# === Učitaj proizvode iz CSV ===
df = pd.read_csv("C:/Users/Viktorija/Desktop/Detektor-namirnica-main/Detektor-namirnica-main/namirnice.csv")
csv_proizvodi = set(df["naziv"].str.lower())

# === Baza i lock za thread-sigurnost ===
conn = sqlite3.connect("shopping_list.db", check_same_thread=False)
cur = conn.cursor()
cur.execute('''
    CREATE TABLE IF NOT EXISTS popis (
        id INTEGER PRIMARY KEY,
        korisnik TEXT,
        proizvod TEXT,
        confidence REAL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
''')
conn.commit()
lock = threading.Lock()

# === Streamlit UI ===
st.title("YOLO Detekcija namirnica s IP kamerom")

korisnik = st.text_input("Unesi svoje korisničko ime:")
ip_link = st.text_input("Unesi URL IP kamere (npr. http://192.168.1.123:8080/video):")

detected_items = {}

def detect_from_ip_camera(url):
    cap = cap = cv2.VideoCapture("http://192.168.211.180:8080/video")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Ne mogu dohvatiti sliku s kamere.")
            break

        results = model.predict(frame, imgsz=640, conf=0.25, verbose=False)[0]

        for box in results.boxes:
            conf = box.conf.item()
            class_id = int(box.cls)
            class_name = model.names[class_id].lower()
            threshold = CLASS_THRESHOLDS.get(class_name, 0.25)

            if conf > threshold and class_name in csv_proizvodi:
                detected_items[class_name] = max(detected_items.get(class_name, 0), conf)

            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0,255,0), 2)
            cv2.putText(frame, f"{class_name} {conf:.2f}", (xyxy[0], xyxy[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        st.image(frame, channels="BGR")

        if st.button("Zaustavi"):
            break

    cap.release()

if st.button("Pokreni IP kameru i detekciju") and korisnik.strip() and ip_link.strip():
    detect_from_ip_camera(ip_link)

    if st.button("Spremi popis"):
        with lock:
            for proizvod, conf in detected_items.items():
                cur.execute("INSERT INTO popis (korisnik, proizvod, confidence) VALUES (?, ?, ?)",
                            (korisnik, proizvod, round(conf,2)))
            conn.commit()
            st.success("Popis spremljen u bazu!")

    if detected_items:
        st.subheader("Trenutno detektirani proizvodi:")
        for p, c in detected_items.items():
            st.write(f"- {p.capitalize()}: {c:.2f}")
else:
    st.info("Unesi korisničko ime i URL kamere.")
