import cv2
import numpy as np
import requests
from ultralytics import YOLO

class MobileCamera:
    def __init__(self, ip_url):
        self.ip_url = ip_url.rstrip("/") + "/shot.jpg"
        self.detected_items = {}
        
    def get_frame(self):
        try:
            response = requests.get(self.ip_url, timeout=2)
            img_array = np.frombuffer(response.content, dtype=np.uint8)
            return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"Greška pri dohvatu frame-a: {e}")
            return None

class FoodDetector:
    def __init__(self, model_path):
        self.model = YOLO("C:/Users/anita/Desktop/Faks_NOVO/2. GODINA/IV. semestar/Primjenjeno strojno učenje/Projekt/namirnice_dataset/runs/detect/namirnice_train5/weights/best.pt")
        self.thresholds = {
            'kikiriki': 0.55, 'jaja': 0.45, 'riza': 0.35,
            'rajcica': 0.40, 'banane': 0.40, 'kruh': 0.35,
            'krastavci': 0.30, 'pivo': 0.15
        }
    
    def detect(self, frame):
        results = self.model.predict(frame, imgsz=640, conf=0.25, verbose=False)[0]
        detections = {}
        
        for box in results.boxes:
            conf = box.conf.item()
            class_name = self.model.names[int(box.cls)].lower()
            
            if conf > self.thresholds.get(class_name, 0.25):
                detections[class_name] = max(detections.get(class_name, 0), conf)
        
        return detections, results.plot()
