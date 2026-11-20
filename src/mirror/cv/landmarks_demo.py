"""Демо недели 1: камера -> отражение -> MediaPipe (лицо + кисти) -> FPS.

Использует Tasks API (mediapipe >= 0.10.x, включая 1.x): старый mp.solutions
в новых версиях удалён. Модели .task скачиваются один раз в папку models/.

Запуск (из корня проекта):  python run_demo.py
Выход:                      клавиша q
"""
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from ..config import load_config
from ..models import ensure_model  # общий загрузчик моделей: скачивает файл один раз в models/


def draw_landmarks(frame, landmarks, connections=None, color=(0, 255, 0)):
    """Рисует точки (и линии между ними) поверх кадра.

    MediaPipe возвращает координаты в долях от 0 до 1, поэтому
    умножаем на размер кадра, чтобы получить пиксели.
    """
    h, w = frame.shape[:2]
    points = [(int(p.x * w), int(p.y * h)) for p in landmarks]
    if connections:
        for c in connections:
            cv2.line(frame, points[c.start], points[c.end], color, 1)
    for x, y in points:
        cv2.circle(frame, (x, y), 2, color, -1)


def run():
    cfg = load_config()
    cam_cfg = cfg["camera"]

    # Лицо: контур и губы. num_faces=1: в кадре интересует ближайший человек
    face_options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=ensure_model("face_landmarker.task")),
        # IMAGE: каждый кадр обрабатывается независимо и синхронно.
        # Режим VIDEO у вас завис внутри нативного кода, IMAGE проще и надёжнее.
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
    )
    # Кисти: до двух рук (вторая рука пригодится при жестах)
    hand_options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=ensure_model("hand_landmarker.task")),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
    )

    cap = cv2.VideoCapture(cam_cfg["index"])
    # Просим камеру отдавать кадр нужного размера (драйвер может округлить)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["height"])
    if not cap.isOpened():
        raise SystemExit("Камера не открылась: проверьте camera.index в config.yaml")

    face_conn = vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS
    hand_conn = vision.HandLandmarksConnections.HAND_CONNECTIONS

    print("Камера открыта. Должно появиться окно 'Smart mirror - landmarks demo'.")
    print("Если окна не видно, проверьте панель задач: оно может быть позади других.")
    print("Выход: нажмите q, когда активно окно с видео (или Ctrl+C в терминале).")

    prev_time = time.perf_counter()
    last_print = prev_time
    frames = 0
    fps = 0.0

    with vision.FaceLandmarker.create_from_options(face_options) as face_lm, \
            vision.HandLandmarker.create_from_options(hand_options) as hand_lm:
        while True:
            ok, frame = cap.read()
            if not ok:
                break  # камера отключилась: выходим из цикла

            if cam_cfg["mirror"]:
                frame = cv2.flip(frame, 1)  # зеркальное отражение по горизонтали

            # MediaPipe ожидает RGB, OpenCV отдаёт BGR
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            face_result = face_lm.detect(mp_image)
            hand_result = hand_lm.detect(mp_image)

            for face in face_result.face_landmarks:
                draw_landmarks(frame, face, face_conn, color=(255, 200, 0))
            # Точки кистей позже используются для жестов и анализа чистки
            for hand in hand_result.hand_landmarks:
                draw_landmarks(frame, hand, hand_conn, color=(0, 255, 0))

            # FPS: обратная величина времени обработки кадра, со сглаживанием
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
            prev_time = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Раз в секунду пишем в терминал: так видно, что цикл идёт, даже без окна
            frames += 1
            if now - last_print >= 1.0:
                print(f"кадров: {frames} | FPS: {fps:.1f} | "
                      f"лиц: {len(face_result.face_landmarks)} | "
                      f"рук: {len(hand_result.hand_landmarks)}")
                last_print = now

            cv2.imshow("Smart mirror - landmarks demo", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


def main():
    """Запускает демо; Ctrl+C завершает его без длинного traceback."""
    try:
        run()
    except KeyboardInterrupt:
        print("Остановлено пользователем.")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
