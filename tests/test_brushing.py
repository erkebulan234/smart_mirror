import math
import random

from mirror.cv.brushing import BrushingTracker, Features, estimate_zone, extract_features
from mirror.events import ZONES, EventType

from .helpers import Pt

FPS = 30


# ---------- Признаки ----------

def face_landmarks(mouth=(0.5, 0.55), half_width=0.15):
    """478 точек; заполнены только нужные индексы: губы 13/14 и края лица 234/454."""
    pts = [Pt(0.5, 0.5)] * 478
    pts = list(pts)
    pts[13], pts[14] = Pt(mouth[0], mouth[1] - 0.005), Pt(mouth[0], mouth[1] + 0.005)
    pts[234], pts[454] = Pt(0.5 - half_width, 0.5), Pt(0.5 + half_width, 0.5)
    return pts


def hand_at(x, y):
    pts = [Pt(0.0, 0.0)] * 21
    pts = list(pts)
    pts[9] = Pt(x, y)
    return pts


def test_extract_features_near_and_far():
    face = face_landmarks()
    near = extract_features(face, [hand_at(0.7, 0.55)], near_ratio=0.7, aspect=1.0)
    assert near.near and math.isclose(near.dx, 0.2 / 0.3, rel_tol=1e-3) and abs(near.dy) < 1e-6
    far = extract_features(face, [hand_at(0.95, 0.9)], near_ratio=0.7, aspect=1.0)
    assert not far.near


def test_extract_features_none_without_face_or_hands():
    assert extract_features(None, [hand_at(0.5, 0.5)], 0.7) is None
    assert extract_features(face_landmarks(), [], 0.7) is None


def test_features_scale_with_face_size():
    """Лицо вдвое ближе к камере и рука вдвое дальше от рта в кадре: признаки те же."""
    a = extract_features(face_landmarks(half_width=0.10), [hand_at(0.5 + 0.1, 0.55)], 0.7, aspect=1.0)
    b = extract_features(face_landmarks(half_width=0.20), [hand_at(0.5 + 0.2, 0.55)], 0.7, aspect=1.0)
    assert math.isclose(a.dx, b.dx, rel_tol=1e-3)


def test_nearest_hand_is_used():
    face = face_landmarks()
    f = extract_features(face, [hand_at(0.95, 0.9), hand_at(0.6, 0.55)], 0.7, aspect=1.0)
    assert f.near and math.isclose(f.dx, 0.1 / 0.3, rel_tol=1e-3)


def test_estimate_zone():
    assert estimate_zone(0.4, 0.3, 0.15) == "upper_right"   # рука справа и ниже рта
    assert estimate_zone(-0.4, 0.3, 0.15) == "upper_left"
    assert estimate_zone(-0.4, -0.3, 0.15) == "lower_left"
    assert estimate_zone(0.4, -0.3, 0.15) == "lower_right"
    assert estimate_zone(0.05, 0.3, 0.15) is None            # у центра рта сторона неясна
    assert estimate_zone(0.4, 0.3, 0.15, hand_below_means_upper=False) == "lower_right"


# ---------- Трекер ----------

def brushing(t, cx=0.45, cy=0.0):
    """Чистка: рука у рта колеблется 3 раза в секунду с размахом 0.12 ширины лица."""
    return Features(near=True, dx=cx + 0.06 * math.sin(2 * math.pi * 3 * t), dy=cy)


def still(cx=0.45, cy=0.0, jitter=0.0, rng=None):
    j = (rng.uniform(-jitter, jitter) if rng else 0.0)
    return Features(near=True, dx=cx + j, dy=cy + j)


def run(tracker, seconds, make, start=0.0):
    """Скармливает трекеру кадры; make(t) -> Features или None. Возвращает (тип события, время)."""
    out = []
    for i in range(int(seconds * FPS)):
        t = start + i / FPS
        for e in tracker.update(t, make(t)):
            out.append((e.type, round(t, 2)))
    return out


def test_full_session_goes_through_all_zones(cfg):
    tr = BrushingTracker(cfg["brushing"])
    first = tr.start(0.0)
    assert first[0].payload["zone"] == "upper_left"

    events = run(tr, 125, brushing, start=0.0)
    zones = [e for e in events if e[0] == EventType.BRUSHING_ZONE]
    assert len(zones) == 3                                # переходы во 2-ю, 3-ю, 4-ю зоны
    assert events[-1][0] == EventType.BRUSHING_FINISHED
    assert tr.finished and tr.remaining == 0
    assert tr.zone_time == [30.0] * 4
    # Вся сессия укладывается примерно в 2 минуты (с запасом на разгон окна движения)
    assert 119 <= events[-1][1] <= 123


def test_pause_after_still_and_resume(cfg):
    tr = BrushingTracker(cfg["brushing"])
    tr.start(0.0)
    ev = run(tr, 3, brushing)                              # 3 с чистки
    ev += run(tr, 8, lambda t: still(), start=3.0)         # рука замерла у рта
    assert [e for e, _ in ev] == [EventType.BRUSHING_PAUSED]
    assert 8.0 <= ev[0][1] <= 9.0                          # пауза примерно через 5 с после остановки
    elapsed_at_pause = tr.elapsed

    ev2 = run(tr, 2, brushing, start=11.0)                 # чистка возобновилась
    assert [e for e, _ in ev2] == [EventType.BRUSHING_RESUMED]
    assert tr.elapsed > elapsed_at_pause
    # Пока пауза, время не копится
    assert abs(elapsed_at_pause - 3.0) <= 1.0


def test_no_time_counted_while_paused(cfg):
    tr = BrushingTracker(cfg["brushing"])
    tr.start(0.0)
    run(tr, 20, lambda t: still())
    assert tr.paused and tr.elapsed == 0.0


def test_face_lost_pauses(cfg):
    tr = BrushingTracker(cfg["brushing"])
    tr.start(0.0)
    ev = run(tr, 7, lambda t: None)
    assert [e for e, _ in ev] == [EventType.BRUSHING_PAUSED]


def test_hand_far_from_mouth_is_not_brushing(cfg):
    tr = BrushingTracker(cfg["brushing"])
    tr.start(0.0)
    run(tr, 4, lambda t: Features(near=False, dx=1.5 + 0.06 * math.sin(20 * t), dy=0.0))
    assert tr.elapsed == 0.0


def test_jitter_of_static_hand_is_not_brushing(cfg):
    """Дрожание точек у неподвижной руки не должно засчитываться как чистка."""
    rng = random.Random(1)
    tr = BrushingTracker(cfg["brushing"])
    tr.start(0.0)
    run(tr, 10, lambda t: still(jitter=0.008, rng=rng))
    assert tr.elapsed == 0.0
    assert tr.paused


def test_slow_drift_is_not_brushing(cfg):
    """Рука плавно перемещается в одну сторону: размах есть, возвратных движений нет."""
    tr = BrushingTracker(cfg["brushing"])
    tr.start(0.0)
    run(tr, 6, lambda t: Features(near=True, dx=0.2 + 0.1 * t, dy=0.0))
    assert tr.elapsed == 0.0


def test_strict_zone_requires_matching_side(cfg):
    strict = dict(cfg["brushing"], strict_zone=True)
    # Задана зона upper_left (сторона left), а рука справа от рта
    tr = BrushingTracker(strict)
    tr.start(0.0)
    run(tr, 6, lambda t: brushing(t, cx=0.5, cy=0.3))
    assert tr.elapsed == 0.0

    tr = BrushingTracker(strict)
    tr.start(0.0)
    run(tr, 6, lambda t: brushing(t, cx=-0.5, cy=0.3))     # рука слева: сторона совпала
    assert tr.elapsed > 3.0

    lenient = BrushingTracker(cfg["brushing"])             # по умолчанию строгость выключена
    lenient.start(0.0)
    run(lenient, 6, lambda t: brushing(t, cx=0.5, cy=0.3))
    assert lenient.elapsed > 3.0


def test_zone_match_ratio(cfg):
    tr = BrushingTracker(cfg["brushing"])
    tr.start(0.0)
    run(tr, 6, lambda t: brushing(t, cx=-0.5, cy=0.3))     # upper_left: совпадает с заданной зоной
    assert tr.zone_match_ratio > 0.9
    assert tr.detected_zone == "upper_left"

    tr = BrushingTracker(cfg["brushing"])
    tr.start(0.0)
    run(tr, 6, lambda t: brushing(t, cx=0.5, cy=-0.3))     # lower_right: не совпадает
    assert tr.zone_match_ratio < 0.1


def test_abort_finishes_as_incomplete(cfg):
    tr = BrushingTracker(cfg["brushing"])
    tr.start(0.0)
    run(tr, 5, brushing)
    ev = tr.abort()
    assert ev[0].type == EventType.BRUSHING_FINISHED and ev[0].payload["completed"] is False
    assert tr.update(10.0, brushing(10.0)) == []           # после завершения события не выдаются


def test_zones_progress_in_declared_order(cfg):
    tr = BrushingTracker(cfg["brushing"])
    first = tr.start(0.0)
    names = [first[0].payload["zone"]]
    ev = []
    for i in range(int(125 * FPS)):
        t = i / FPS
        ev += tr.update(t, brushing(t))
    names += [e.payload["zone"] for e in ev if e.type == EventType.BRUSHING_ZONE]
    assert names == list(ZONES)
