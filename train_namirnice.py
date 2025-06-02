from ultralytics import YOLO

# Učitaj osnovni model (npr. srednje velik)
model = YOLO("yolov8m.pt")

# Treniraj na vlastitom datasetu
model.train(
    data="namirnice_dataset/data.yaml",  # putanja do yaml datoteke
    epochs=50,                            # broj epoha (možeš početi s 30–50)
    imgsz=640,                            # veličina slike
    batch=8,                              # batch size – prilagodi ako imaš slab GPU
    name="namirnice_train",
    project="namirnice_dataset/runs/detect"
)
