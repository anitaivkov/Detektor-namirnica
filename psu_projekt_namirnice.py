import time
import sqlite3
from ultralytics import YOLO
import cv2
import pandas as pd

# === 1. Učitaj model ===
model = YOLO("yolov8m.pt")

# === 2. Učitaj CSV s opisima proizvoda ===
df = pd.read_csv("Groceries_dataset.csv")
csv_proizvodi = set(df["itemDescription"].str.lower().unique())

# === 3. Mapiranje YOLO klasa na nazive u CSV-u ===
mapa_yolo_na_csv = {
    "apple": "pip fruit",
    "banana": "tropical fruit",
    "orange": "citrus fruit",
    "carrot": "root vegetables",
    "broccoli": "other vegetables",
    "bread": "brown bread",
    "sausage": "sausage",
    "cake": "pastry",
    "beer": "canned beer",
    "sandwich": "sandwich"
}

# === 4. Pokreni kameru ===
cap = cv2.VideoCapture(1)  # probaj s 1 ako 0 ne radi   pylint: disable=no-member

detected_items = set()
print("📷 Pokrećem kameru... Pritisni Q za izlaz.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        _, _, _, _, score, class_id = result
        yolo_klasa = model.names[int(class_id)].lower()

        if yolo_klasa in mapa_yolo_na_csv:
            csv_naziv = mapa_yolo_na_csv[yolo_klasa]
            if csv_naziv in csv_proizvodi:
                detected_items.add(csv_naziv)

    # Prikaz rezultata uživo
    annotated_frame = results.plot()
    cv2.imshow("Detekcija namirnica (Q za izlaz)", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):   # pylint: disable=no-member
        break

cap.release()
cv2.destroyAllWindows() # pylint: disable=no-member

# === 5. Spremi rezultate u SQLite ===
conn = sqlite3.connect("shopping_list.db")
cur = conn.cursor()
cur.execute('''CREATE TABLE IF NOT EXISTS popis (id INTEGER PRIMARY KEY, proizvod TEXT)''')

for proizvod in detected_items:
    cur.execute("INSERT INTO popis (proizvod) VALUES (?)", (proizvod,))

conn.commit()
conn.close()

print("\n📋 Detektirani proizvodi:")
for p in detected_items:
    print(f"- {p}")
print("\n✅ Popis spremljen u bazu.")
