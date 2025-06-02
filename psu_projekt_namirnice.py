import time
import sqlite3
from ultralytics import YOLO
import cv2
import pandas as pd

# === 1. Učitaj trenirani model ===
model = YOLO("C:/Users/anita/Desktop/Faks_NOVO/2. GODINA/IV. semestar/Primjenjeno strojno učenje/Projekt/namirnice_dataset/runs/detect/namirnice_train4/weights/best.pt")

# === 2. Class-Specific Confidence Thresholds ===
CLASS_THRESHOLDS = {
    'kikiriki': 0.55,   # Visok threshold za overrepresented klasu
    'jaja': 0.45,
    'riza': 0.35,
    'rajcica': 0.40,
    'banane': 0.40,
    'kruh': 0.35,
    'krastavci': 0.30,
    'pivo': 0.15        # Nizak threshold za underrepresented klasu
}

# === 3. Učitaj novi CSV s relevantnim namirnicama ===
df = pd.read_csv("C:/Users/anita/Desktop/Faks_NOVO/2. GODINA/IV. semestar/Primjenjeno strojno učenje/Projekt/namirnice.csv")
csv_proizvodi = set(df["naziv"].str.lower())

# === 4. Pokreni kameru ===
cap = cv2.VideoCapture(1)  # Eksplicitno postavi željeni indeks
if not cap.isOpened():
    print("❌ Nema dostupne kamere!")
    exit()

detected_items = {}
print("📷 Pokrećem kameru... Pritisni Q za izlaz.")

# === 5. Glavna petlja detekcije ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Prošireni parametri za detekciju
    results = model.predict(
        source=frame,
        imgsz=640,
        conf=0.25,  # Globalni minimum
        verbose=False
    )[0]

    for box in results.boxes:
        conf = box.conf.item()
        class_id = int(box.cls)
        class_name = model.names[class_id].lower()
        
        # Primijeni class-specific threshold
        threshold = CLASS_THRESHOLDS.get(class_name, 0.25)
        if conf > threshold and class_name in csv_proizvodi:
            detected_items[class_name] = max(detected_items.get(class_name, 0), conf)
            
            # Debug ispis
            print(f"✅ {class_name.capitalize()}: {conf:.2f} (threshold: {threshold})")

    # Anotiraj frame s proširenim informacijama
    annotated_frame = results.plot()
    cv2.imshow("Detekcija namirnica (Q za izlaz)", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# === 6. Pohrana rezultata s vremenskom oznakom ===
conn = sqlite3.connect("shopping_list.db")
cur = conn.cursor()
cur.execute('''CREATE TABLE IF NOT EXISTS popis 
             (id INTEGER PRIMARY KEY, 
              proizvod TEXT, 
              confidence REAL,
              timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

for proizvod, conf in detected_items.items():
    cur.execute("INSERT INTO popis (proizvod, confidence) VALUES (?, ?)",
               (proizvod, round(conf, 2)))

conn.commit()
conn.close()

# === 7. Detaljni ispis rezultata ===
print("\n📋 Detektirani proizvodi (najveće povjerenje):")
for p, c in detected_items.items():
    print(f"- {p.capitalize()}: {c:.2f}")

print("\n✅ Podaci spremljeni u bazu s vremenskim oznakama.")
