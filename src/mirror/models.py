"""Загрузка файлов моделей (один раз, в папку models/).

В репозиторий модели не попадают (models/ в .gitignore): при первом запуске
они скачиваются автоматически, нужен интернет.
"""
import urllib.request

from .config import ROOT

MODELS_DIR = ROOT / "models"

MODEL_URLS = {
    # MediaPipe: ключевые точки лица и кисти
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    ),
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    ),
    # OpenCV Zoo: детектор лица YuNet и распознавание лиц SFace (128-мерный вектор)
    "face_detection_yunet_2023mar.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx"
    ),
    "face_recognition_sface_2021dec.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_recognition_sface/face_recognition_sface_2021dec.onnx"
    ),
}

# Файл меньше этого размера - не модель, а, например, текстовая заглушка Git LFS или страница ошибки
MIN_MODEL_BYTES = 100_000


def ensure_model(name):
    """Возвращает путь к модели; скачивает её, если файла ещё нет."""
    path = MODELS_DIR / name
    if path.exists() and path.stat().st_size >= MIN_MODEL_BYTES:
        return str(path)

    MODELS_DIR.mkdir(exist_ok=True)
    url = MODEL_URLS[name]
    print(f"Скачиваю {name} ...")
    try:
        urllib.request.urlretrieve(url, path)
    except OSError as e:
        raise SystemExit(
            f"Не удалось скачать {name}: {e}\n"
            f"Скачайте файл вручную по ссылке\n  {url}\nи положите его в папку {MODELS_DIR}"
        )
    if path.stat().st_size < MIN_MODEL_BYTES:
        path.unlink()
        raise SystemExit(
            f"Скачанный файл {name} слишком маленький, это не модель.\n"
            f"Скачайте его вручную из браузера по ссылке\n  {url}\nи положите в папку {MODELS_DIR}"
        )
    return str(path)
