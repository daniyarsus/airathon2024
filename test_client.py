import requests
import time

# Путь к изображению, которое будет отправляться на сервер
file_path = "test_input/satbayev_manasa.jpg"
url = "http://localhost:8000/api/v1/ai/local-photo"

# Функция для отправки POST-запроса на эндпоинт
def send_request():
    with open(file_path, 'rb') as file:
        files = {'file': (file_path, file, 'image/jpeg')}
        response = requests.post(url, files=files)
        if response.status_code == 200:
            print(f"Успешно отправлено: {response.json()}")
        else:
            print(f"Ошибка {response.status_code}: {response.text}")

# Запуск цикла с интервалом в 1 секунду
try:
    while True:
        send_request()
        time.sleep(1)
except KeyboardInterrupt:
    print("\nПрервано пользователем.")
