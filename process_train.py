import torch_directml
from ultralytics import YOLO

device = torch_directml.device()  # Force DirectML GPU
model = YOLO("yolov8n.pt")
model.train(data="datasets/data.yaml", epochs=100, imgsz=640, device=device)
