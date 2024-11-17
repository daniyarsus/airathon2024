from ultralytics import YOLO
import cv2

# Загрузка обученной модели (замените путь на ваш)
model = YOLO('yolo11n.pt')

# Загрузка изображения для предсказания
img = 'test_input/traffic-light-389-_jpg.rf.02c5d25a86f657933155c9523a9b8e32.jpg'

# Выполнение предсказания
results = model(img)


for result in results:
    boxes = result.boxes  # Получение всех рамок
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]  # Координаты рамки
        conf = box.conf[0]  # Уверенность
        cls = box.cls[0]    # Класс объекта
        print(f"Coordinates: ({x1}, {y1}), ({x2}, {y2}), Class: {cls}, Accuracy: {conf}")
