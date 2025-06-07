from ultralytics import YOLO

# Inicijalizacija modela
model = YOLO("yolov8m.pt")

# Konfiguracija treninga s augmentacijama i optimizacijama
model.train(
    data="C:/Users/lukai/Documents/iznimno poslovna mapa/PSU/Projekt/namirnice_dataset/data.yaml",
    epochs=30,
    imgsz=640,
    batch=8,
    name="namirnice_train",
    project="namirnice_dataset/runs/detect",
    
    # Augmentacijski parametri
    hsv_h=0.05,       # Povećana varijacija nijansi
    hsv_s=0.9,        # Veće promjene zasićenja
    hsv_v=0.6,        # Intenzivnije promjene svjetline
    degrees=30,       # Rotacija ±30°
    translate=0.2,    # Veće translacije
    scale=0.3,        # Širi raspon skaliranja
    mosaic=1.0,       # Mosaic augmentacija uvijek aktivna
    mixup=0.3,        # Pojačan mixup efekt
    copy_paste=0.2,   # Copy-paste za manjinske klase
    
    # Optimizacijski parametri
    patience=10,      # Early stopping nakon 10 epoha bez poboljšanja
    lr0=0.005,        # Smanjeni početni learning rate
    weight_decay=0.0005,  # L2 regularizacija
    warmup_epochs=3,  # Postupno zagrijavanje modela
)
