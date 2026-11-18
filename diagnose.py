"""Диагностика: по шагам проверяет камеру и MediaPipe и пишет время каждого шага.

Запуск из корня проекта:  python diagnose.py
Если какой-то шаг зависнет, через 25 секунд скрипт сам напечатает, где именно
он стоит (стек всех потоков), и завершится. Этот вывод пришлите целиком.
"""
import faulthandler
import sys
import time
from pathlib import Path

# Через 25 секунд без завершения: распечатать стек всех потоков и выйти
faulthandler.dump_traceback_later(25, exit=True)

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def step(name, func):
    """Выполняет шаг, печатает его длительность и возвращает результат."""
    print(f"[..] {name}", flush=True)
    t = time.perf_counter()
    result = func()
    print(f"[OK] {name}: {time.perf_counter() - t:.2f} c", flush=True)
    return result


import cv2  # noqa: E402
import mediapipe as mp  # noqa: E402
from mediapipe.tasks.python import BaseOptions, vision  # noqa: E402

from mirror.config import load_config  # noqa: E402
from mirror.cv.landmarks_demo import ensure_model  # noqa: E402

print("Python:", sys.version.split()[0])
print("mediapipe:", mp.__version__, "| opencv:", cv2.__version__)

cfg = load_config()["camera"]
cap = step("открыть камеру", lambda: cv2.VideoCapture(cfg["index"]))
ok, frame = step("прочитать кадр", lambda: cap.read())
print("    кадр получен:", ok, "| размер:", None if frame is None else frame.shape)
cap.release()
if not ok:
    raise SystemExit("Камера не отдаёт кадры: попробуйте camera.index: 1 в config.yaml")

# Берём один реальный кадр и прогоняем его через каждую модель отдельно
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

face = step("создать FaceLandmarker (IMAGE)", lambda: vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=ensure_model("face_landmarker.task")),
        running_mode=vision.RunningMode.IMAGE)))
result = step("лицо: detect() на одном кадре", lambda: face.detect(image))
print("    лиц найдено:", len(result.face_landmarks))

hand = step("создать HandLandmarker (IMAGE)", lambda: vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=ensure_model("hand_landmarker.task")),
        running_mode=vision.RunningMode.IMAGE)))
result = step("кисти: detect() на одном кадре", lambda: hand.detect(image))
print("    рук найдено:", len(result.hand_landmarks))

# Скорость: 30 кадров подряд одной и той же картинкой
t = time.perf_counter()
for _ in range(30):
    face.detect(image)
    hand.detect(image)
print(f"[OK] 30 кадров лицо+кисти: {(time.perf_counter() - t) / 30 * 1000:.0f} мс на кадр")

faulthandler.cancel_dump_traceback_later()
print("Диагностика завершена без зависаний.")
