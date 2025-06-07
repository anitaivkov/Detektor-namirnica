import cv2
import time
from collections import defaultdict

def detect_from_camera(model, camera_url, thresholds, st, user_id, list_db):
    stframe = st.empty()
    cap = cv2.VideoCapture(camera_url)

    if not cap.isOpened():
        st.error("Ne mogu otvoriti video stream.")
        return {}

    detected_items = {}
    presence_start = {}  # kad je neka namirnica prvi put primijećena
    MIN_VISIBLE_DURATION = 6  # sekundi

    start_time = time.time()
    MAX_SECONDS = 15

    while cap.isOpened() and (time.time() - start_time < MAX_SECONDS):
        ret, frame = cap.read()
        if not ret:
            st.warning("Nema frame-a iz kamere.")
            break

        results = model.predict(frame, imgsz=640, conf=0.25, verbose=False)[0]
        current_time = time.time()
        current_detections = set()

        for box in results.boxes:
            conf = box.conf.item()
            class_name = model.names[int(box.cls)].lower()
            if conf > thresholds.get(class_name, 0.25):
                current_detections.add(class_name)

                if class_name not in presence_start:
                    presence_start[class_name] = current_time

                elif current_time - presence_start[class_name] >= MIN_VISIBLE_DURATION:
                    if class_name not in detected_items:
                        detected_items[class_name] = {
                            "count": 1,
                            "confidence": conf
                        }
                    else:
                        # samo ako želimo višestruko spremanje (npr. nakon dulje pauze)
                        pass
        # resetiraj timer za one koje više ne vidi
        for name in list(presence_start.keys()):
            if name not in current_detections:
                del presence_start[name]

        annotated = results.plot()
        stframe.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")
        time.sleep(0.1)

    cap.release()
    return detected_items
