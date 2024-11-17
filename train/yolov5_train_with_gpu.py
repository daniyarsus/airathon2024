from ultralytics import YOLO
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Используется устройство: {device}")

model = YOLO('yolov5s.pt')

model.train(data='data_large.yaml', epochs=50, imgsz=640, batch=16, device=device)
