"""Идентификация по лицу (ФТ-3): детектор YuNet + вектор лица SFace (оба из OpenCV Zoo).

Работает на CPU и не требует дополнительных библиотек: достаточно opencv.
"""
from collections import Counter

import numpy as np

from ..models import ensure_model

GUEST = "guest"


def match_embedding(emb, profiles, threshold):
    """Сопоставляет вектор лица с профилями.

    profiles: {имя: массив (n, 128)} из единичных векторов; emb: единичный вектор.
    Для каждого профиля берётся лучшее из его векторов. Возвращает (имя или None, лучшее сходство):
    None, если даже лучшее сходство ниже порога, то есть человек незнакомый.
    """
    best_name, best_score = None, -1.0
    for name, vectors in profiles.items():
        score = float(np.max(vectors @ emb))  # косинусное сходство единичных векторов
        if score > best_score:
            best_name, best_score = name, score
    if best_score >= threshold:
        return best_name, best_score
    return None, best_score


def decide(votes):
    """Итог голосования по кадрам: имя профиля, если за него проголосовала большая часть, иначе гость.

    votes: список имён профилей или None (кадр, где совпадения не нашлось).
    """
    if not votes:
        return GUEST
    name, count = Counter(votes).most_common(1)[0]
    if name is not None and count * 2 > len(votes):
        return name
    return GUEST


class IdentificationSession:
    """Собирает результаты по нескольким кадрам и выдаёт решение (устойчиво к отдельным ошибкам)."""

    def __init__(self, identifier, profiles, cfg):
        self.identifier = identifier
        self.profiles = profiles
        self.threshold = cfg["cosine_threshold"]
        self.window = cfg["vote_window"]
        self.votes = []

    def feed(self, frame_bgr):
        """Обрабатывает кадр. Возвращает решение (имя или 'guest'), когда голосов набралось достаточно."""
        emb = self.identifier.embed(frame_bgr)
        if emb is None:
            return None  # лицо в этом кадре не найдено, голос не засчитывается
        name, _score = match_embedding(emb, self.profiles, self.threshold) if self.profiles else (None, 0.0)
        self.votes.append(name)
        if len(self.votes) >= self.window:
            return decide(self.votes)
        return None

    def finish(self):
        """Принудительное решение по тем голосам, что есть (например, по истечении времени)."""
        return decide(self.votes)


class FaceIdentifier:
    """Обёртка над YuNet (поиск лица) и SFace (вектор лица)."""

    def __init__(self, cfg):
        import cv2  # импорт здесь: чистые функции выше не требуют OpenCV

        self._cv2 = cv2
        self._detector = cv2.FaceDetectorYN.create(
            ensure_model("face_detection_yunet_2023mar.onnx"), "", (320, 320),
            score_threshold=cfg["detector_score"], nms_threshold=0.3, top_k=5,
        )
        self._recognizer = cv2.FaceRecognizerSF.create(
            ensure_model("face_recognition_sface_2021dec.onnx"), ""
        )

    def embed(self, frame_bgr):
        """Возвращает единичный вектор самого крупного лица в кадре или None, если лица нет."""
        h, w = frame_bgr.shape[:2]
        self._detector.setInputSize((w, h))
        _, faces = self._detector.detect(frame_bgr)
        if faces is None or len(faces) == 0:
            return None
        face = max(faces, key=lambda f: f[2] * f[3])  # самое крупное = ближайшее к зеркалу
        aligned = self._recognizer.alignCrop(frame_bgr, face)  # выравнивание по 5 точкам лица
        feat = self._recognizer.feature(aligned).reshape(-1).astype(np.float32)
        return feat / max(float(np.linalg.norm(feat)), 1e-9)
