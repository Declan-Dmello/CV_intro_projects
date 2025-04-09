
from ultralytics import YOLO

model = YOLO(r"C:\Users\decla\PycharmProjects\pythonProject2\runs\detect\train17\weights\best.pt")
model.predict(source=r"C:\Users\decla\PycharmProjects\pythonProject2\Computer_vision\Plate_reader\data\archive\hqtv.mp4", save=True, conf=0.25)

