import os
import base64
import shutil
from typing import Any

from PIL import Image
import uvicorn
from fastapi import FastAPI, APIRouter, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from openai import OpenAI
from inference_sdk import InferenceHTTPClient

CLIENT_CV = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="3BDEDCPYaZmmWDOEAaSN"
)


# Настройки окружения
class Settings(BaseSettings):
    OPENAI_API_KEY: str

    class Config:
        env_file = ".env"


# Настройка openai клиента
client = OpenAI(api_key=Settings().OPENAI_API_KEY)


# Запрос на gpt-сервер
class SendGptTextRequest(BaseModel):
    text: str


# Роутер для ai сервиса
router = APIRouter(prefix="/api/v1/ai", tags=["AI API methods."])

IMAGES_DIR = "images"
os.makedirs(IMAGES_DIR, exist_ok=True)


def compress_image(input_path: str, output_path: str, quality: int = 50):
    """Сжимает изображение до указанного качества."""
    with Image.open(input_path) as img:
        img = img.convert("RGB")
        img.save(output_path, "JPEG", quality=quality)


def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


@router.post("/gpt-text")
async def send_gpt_text_endpoint(request: SendGptTextRequest) -> dict[str, str]:
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for people with poor eyesight."},
            {
                "role": "user",
                "content": request.text
            }
        ]
    )

    return {
        'response': completion.choices[0].message.to_dict()['content']
    }


@router.post("/gpt-photo")
async def send_gpt_photo_endpoint(file: UploadFile = File(...)) -> dict[str, str]:
    file_path = os.path.join(IMAGES_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    base64_image = encode_image(file_path)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Что находится на изображении? Отвечай по ситуации коротко если видишь пешеходную дорогу или какую-то другую опасную ситуацию для человека с плохим зрением или длинным текстом если это обычный случай по типу 'что изображено тут?' или 'скажи мне сказку'"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                ],
            }
        ],
        max_tokens=300
    )

    if os.path.exists(file_path):
        os.remove(file_path)

    return {
        'response': response.choices[0].message.to_dict()['content']
    }


@router.post("/local-photo")
async def send_local_photo_endpoint(file: UploadFile = File(...)) -> dict[str, Any]:
    # Сохраняем загруженный файл локально
    file_path = os.path.join(IMAGES_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Путь для сжатого изображения
    compressed_file_path = os.path.join(IMAGES_DIR, f"compressed_{file.filename}")

    # Сжимаем изображение до качества 50 (можно настроить)
    compress_image(file_path, compressed_file_path, quality=50)

    # Кодируем сжатое изображение в Base64
    base64_image = encode_image(compressed_file_path)

    # Отправляем сжатое изображение на сервер
    result = CLIENT_CV.infer(compressed_file_path, model_id="trafficsigndetection-vwdix/10")

    # Удаляем временные файлы после обработки
    if os.path.exists(file_path):
        os.remove(file_path)
    if os.path.exists(compressed_file_path):
        os.remove(compressed_file_path)

    # Возвращаем только predictions
    predictions = result.get('predictions', [])
    return {'predictions': predictions}


# Настройка приложения
def create_app() -> FastAPI:
    app = FastAPI(
        title="AIrathon backend application.",
        version="0.1.0"
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )
    app.include_router(router)

    return app


# Запускаем приложение :)
if __name__ == "__main__":
    uvicorn.run(
        app='src.app:create_app',
        factory=True,
        host="0.0.0.0",
        port=8000
    )
