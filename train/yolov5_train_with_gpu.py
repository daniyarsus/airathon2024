from ultralytics import YOLO
import torch

# Убедимся, что CUDA доступна и настроим устройство
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Используется устройство: {device}")

# Инициализация модели с предобученной весовой моделью
model = YOLO('yolov5s.pt')

# Тренировка модели на вашем датасете с использованием GPU
model.train(data='data.yaml', epochs=50, imgsz=640, batch=16, device='cuda')
