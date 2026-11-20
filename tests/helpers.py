"""Общие вспомогательные функции для тестов."""
from collections import namedtuple

Pt = namedtuple("Pt", "x y")  # точка в нормированных координатах кадра (как у MediaPipe)

# Смещения оснований пальцев по горизонтали: указательный, средний, безымянный, мизинец
_FINGER_DX = (-0.05, -0.017, 0.017, 0.05)
_FINGER_BASE = ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))


def make_hand(fingers, thumb="folded", cx=0.5, cy=0.7):
    """Строит 21 точку кисти: запястье внизу, пальцы направлены вверх.

    fingers: четыре флага (указательный, средний, безымянный, мизинец), True = выпрямлен.
    thumb: 'folded' (прижат к ладони), 'out' (отведён в сторону) или 'up' (вверх).
    """
    pts = [Pt(cx, cy)] * 21
    pts = list(pts)
    for (mcp, pip, dip, tip), dx, ext in zip(_FINGER_BASE, _FINGER_DX, fingers):
        pts[mcp] = Pt(cx + dx, cy - 0.15)
        pts[pip] = Pt(cx + dx, cy - 0.21)
        pts[dip] = Pt(cx + dx, cy - 0.25 if ext else cy - 0.17)
        pts[tip] = Pt(cx + dx, cy - 0.28 if ext else cy - 0.13)
    if thumb == "out":
        pts[1], pts[2] = Pt(cx - 0.04, cy - 0.03), Pt(cx - 0.07, cy - 0.07)
        pts[3], pts[4] = Pt(cx - 0.11, cy - 0.11), Pt(cx - 0.15, cy - 0.14)
    elif thumb == "up":
        pts[1], pts[2] = Pt(cx - 0.04, cy - 0.03), Pt(cx - 0.06, cy - 0.08)
        pts[3], pts[4] = Pt(cx - 0.06, cy - 0.14), Pt(cx - 0.06, cy - 0.20)
    else:  # folded
        pts[1], pts[2] = Pt(cx - 0.04, cy - 0.03), Pt(cx - 0.07, cy - 0.07)
        pts[3], pts[4] = Pt(cx - 0.03, cy - 0.10), Pt(cx, cy - 0.09)
    return pts


def palm(cx=0.5, cy=0.7):
    return make_hand((True, True, True, True), "out", cx, cy)


def thumb_up(cx=0.5, cy=0.7):
    return make_hand((False, False, False, False), "up", cx, cy)


def fist(cx=0.5, cy=0.7):
    return make_hand((False, False, False, False), "folded", cx, cy)
