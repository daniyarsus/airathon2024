from ultralytics import YOLO

# Загрузка обученной модели (укажите путь к файлу с весами)
model = YOLO('yolov5su.pt')

# Детекция на одном изображении
results = model.predict(source='images/satbayev_manasa.jpg', save=True, imgsz=640)

for result in results:
    boxes = result.boxes  # Получение всех рамок
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]  # Координаты рамки
        conf = box.conf[0]  # Уверенность
        cls = box.cls[0]    # Класс объекта
        print(f"Coordinates: ({x1}, {y1}), ({x2}, {y2}), Class: {cls}, Accuracy: {conf}")