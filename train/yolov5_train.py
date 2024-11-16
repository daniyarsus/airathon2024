from ultralytics import YOLO

# Инициализация модели с предобученной весовой моделью (например, yolov5s)
model = YOLO('yolov5s.pt')

# Загрузка и тренировка на вашем датасете
model.train(data='data.yaml', epochs=50, imgsz=640, batch=16, device='cpu')

