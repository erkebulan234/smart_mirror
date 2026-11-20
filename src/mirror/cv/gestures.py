"""Распознавание жестов по 21 точке кисти MediaPipe (ФТ-10).

Жесты: palm (ладонь), thumb_up (большой палец вверх), swipe_left, swipe_right.
Кадр уже отражён по горизонтали, поэтому "влево" и "вправо" соответствуют тому,
что человек видит на экране.
"""
import math
from collections import deque

# Индексы точек MediaPipe Hands
WRIST = 0
THUMB_MCP, THUMB_TIP = 2, 4
MIDDLE_MCP = 9
# Пары (средний сустав PIP, кончик) для указательного, среднего, безымянного и мизинца
FINGERS = ((6, 8), (10, 12), (14, 16), (18, 20))

THUMB_EXTENDED_RATIO = 1.5   # большой палец отведён: кончик дальше от запястья, чем его основание, в 1.5 раза
FINGER_FOLDED_RATIO = 1.0    # палец согнут: кончик не дальше от запястья, чем средний сустав
THUMB_UP_MIN_VERTICAL = 0.7  # доля вертикальной составляющей направления большого пальца


def _dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def finger_ratios(lm):
    """Для каждого из четырёх пальцев: расстояние запястье-кончик / запястье-сустав.

    Отношение не зависит ни от размера кисти в кадре, ни от её поворота:
    у выпрямленного пальца оно заметно больше 1, у согнутого меньше 1.
    """
    wrist = lm[WRIST]
    return [_dist(lm[tip], wrist) / max(_dist(lm[pip], wrist), 1e-6) for pip, tip in FINGERS]


def thumb_ratio(lm):
    """Расстояние запястье-кончик большого пальца / запястье-его основание."""
    wrist = lm[WRIST]
    return _dist(lm[THUMB_TIP], wrist) / max(_dist(lm[THUMB_MCP], wrist), 1e-6)


def classify_static(lm, extended_ratio=1.15):
    """Определяет статичный жест по одной кисти: 'palm', 'thumb_up' или None."""
    ratios = finger_ratios(lm)
    extended = [r > extended_ratio for r in ratios]
    folded = [r < FINGER_FOLDED_RATIO for r in ratios]

    # Ладонь: выпрямлены все четыре пальца (положение большого не учитываем:
    # у многих людей он прижат к ладони)
    if all(extended):
        return "palm"

    # Большой палец вверх: четыре пальца согнуты, большой отведён и направлен вверх
    if all(folded) and thumb_ratio(lm) > THUMB_EXTENDED_RATIO:
        vx = lm[THUMB_TIP].x - lm[THUMB_MCP].x
        vy = lm[THUMB_TIP].y - lm[THUMB_MCP].y
        length = math.hypot(vx, vy)
        # В координатах изображения ось y направлена вниз, поэтому "вверх" это отрицательный vy
        if length > 0 and (-vy / length) > THUMB_UP_MIN_VERTICAL:
            return "thumb_up"
    return None


class GestureRecognizer:
    """Превращает поток кадров с кистью в редкие события-жесты.

    Защита от ложных срабатываний:
    - статичный жест должен удерживаться hold_ms почти без движения руки;
    - после срабатывания действует пауза cooldown_ms;
    - сработавший жест не повторяется, пока его не отпустят (рука ушла или жест сменился).
    """

    def __init__(self, cfg):
        self.hold = cfg["hold_ms"] / 1000
        self.cooldown = cfg["cooldown_ms"] / 1000
        self.max_move = cfg["max_move_while_holding"]
        self.swipe_window = cfg["swipe_window_ms"] / 1000
        self.swipe_min_dx = cfg["swipe_min_dx"]
        self.swipe_max_dy_ratio = cfg["swipe_max_dy_ratio"]
        self.extended_ratio = cfg["finger_extended_ratio"]

        self._candidate = None      # жест, который сейчас удерживается
        self._cand_start = 0.0      # когда начали удерживать
        self._cand_pos = (0.0, 0.0)  # где находилась рука в начале удержания
        self._latched = None        # жест, который уже сработал и ещё не отпущен
        self._last_fire = -1e9
        self._trail = deque()       # (время, x, y) центра ладони для поиска свайпа

    def _fire(self, name, t, latch):
        self._last_fire = t
        self._latched = latch
        self._candidate = None
        self._trail.clear()
        return name

    def update(self, hand, t):
        """hand: список из 21 точки или None; t: время в секундах. Возвращает имя жеста или None."""
        if hand is None:
            # Рука ушла из кадра: всё сбрасываем, жест считается отпущенным
            self._candidate = None
            self._latched = None
            self._trail.clear()
            return None

        centre = hand[MIDDLE_MCP]
        name = classify_static(hand, self.extended_ratio)
        in_cooldown = (t - self._last_fire) < self.cooldown

        # --- Свайп: открытая ладонь быстро прошла по горизонтали ---
        self._trail.append((t, centre.x, centre.y))
        while self._trail and t - self._trail[0][0] > self.swipe_window:
            self._trail.popleft()
        if name == "palm" and not in_cooldown and len(self._trail) >= 3:
            t0, x0, y0 = self._trail[0]
            dx, dy = centre.x - x0, centre.y - y0
            if abs(dx) >= self.swipe_min_dx and abs(dy) <= abs(dx) * self.swipe_max_dy_ratio:
                # Ладонь остаётся "защёлкнутой": иначе после свайпа сразу сработал бы жест palm
                return self._fire("swipe_left" if dx < 0 else "swipe_right", t, latch="palm")

        # --- Статичные жесты ---
        if name is None:
            self._candidate = None
            self._latched = None
            return None

        if name != self._candidate:
            self._candidate, self._cand_start, self._cand_pos = name, t, (centre.x, centre.y)
            return None

        # Тот же жест: если рука заметно сместилась, отсчёт удержания начинается заново
        moved = math.hypot(centre.x - self._cand_pos[0], centre.y - self._cand_pos[1])
        if moved > self.max_move:
            self._cand_start, self._cand_pos = t, (centre.x, centre.y)
            return None

        if name == self._latched:
            return None
        if (t - self._cand_start) >= self.hold and not in_cooldown:
            return self._fire(name, t, latch=name)
        return None
