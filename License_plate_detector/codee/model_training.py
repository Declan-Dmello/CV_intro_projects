from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="../data/archive/dataset.yaml",  
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,
workers = 0
)