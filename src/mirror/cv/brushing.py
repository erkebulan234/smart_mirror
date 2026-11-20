"""Трекер чистки зубов (ФТ-6, ФТ-7).

Идея: щётка находится в руке, рука у рта и совершает возвратно-поступательные движения.
Модуль не ищет саму щётку (на веб-камере она мелкая и часто закрыта рукой), а использует
положение кисти относительно рта:
  - "чистка идёт" = рука у рта и колеблется (размах и число разворотов выше порога);
  - сторона и челюсть оцениваются по смещению руки от центра рта (эвристика, см. estimate_zone);
  - время засчитывается заданному квадранту, пока чистка идёт; по 30 с на каждый из четырёх.

Если движения нет дольше pause_after_still_sec, тренер ставится на паузу.
"""
import math
from collections import deque
from dataclasses import dataclass

from ..events import ZONES, Event, EventType

# Индексы точек MediaPipe Face Mesh
UPPER_LIP_INNER, LOWER_LIP_INNER = 13, 14   # внутренний край верхней и нижней губы
FACE_LEFT, FACE_RIGHT = 234, 454            # крайние точки лица на уровне скул
HAND_CENTER = 9                             # основание среднего пальца: середина ладони


@dataclass
class Features:
    """Положение руки относительно рта. Все расстояния в ширинах лица, поэтому не зависят от дальности."""
    near: bool    # рука достаточно близко ко рту
    dx: float     # смещение руки от центра рта по горизонтали (вправо на экране = положительное)
    dy: float     # смещение по вертикали (вниз на экране = положительное)


def extract_features(face_lm, hands, near_ratio, aspect=4 / 3):
    """Считает Features по точкам лица и кистей; None, если лица или рук нет.

    aspect = ширина/высота кадра: координаты MediaPipe нормированы по каждой оси отдельно,
    поэтому горизонталь домножается на aspect, чтобы расстояния не искажались.
    """
    if not face_lm or not hands:
        return None

    def xy(p):
        return p.x * aspect, p.y

    mouth = [(a + b) / 2 for a, b in zip(xy(face_lm[UPPER_LIP_INNER]), xy(face_lm[LOWER_LIP_INNER]))]
    left, right = xy(face_lm[FACE_LEFT]), xy(face_lm[FACE_RIGHT])
    face_width = math.hypot(right[0] - left[0], right[1] - left[1])
    if face_width < 1e-6:
        return None

    # Рука, ближайшая ко рту, считается держащей щётку
    best = None
    for hand in hands:
        hx, hy = xy(hand[HAND_CENTER])
        dx, dy = (hx - mouth[0]) / face_width, (hy - mouth[1]) / face_width
        if best is None or math.hypot(dx, dy) < math.hypot(best[0], best[1]):
            best = (dx, dy)
    return Features(near=math.hypot(*best) <= near_ratio, dx=best[0], dy=best[1])


def estimate_zone(dx, dy, dead_zone, hand_below_means_upper=True):
    """Оценивает квадрант по смещению руки от центра рта; None, если сторона неясна.

    Это стартовая гипотеза, её нужно проверить и откалибровать на записанных видео:
    при чистке верхних зубов рука с щёткой обычно ниже линии рта, при чистке нижних выше.
    Стороны даны как в зеркале (слева на экране = слева у человека).
    """
    if dx < -dead_zone:
        side = "left"
    elif dx > dead_zone:
        side = "right"
    else:
        return None
    below = dy > 0  # ось y направлена вниз
    upper = below if hand_below_means_upper else not below
    return f"{'upper' if upper else 'lower'}_{side}"


class BrushingTracker:
    def __init__(self, cfg):
        self.zone_sec = cfg["quadrant_sec"]
        self.pause_after = cfg["pause_after_still_sec"]
        self.window = cfg["motion_window_sec"]
        self.min_extent = cfg["min_motion_extent"]
        self.min_ratio = cfg["min_oscillation_ratio"]
        self.dead_zone = cfg["side_dead_zone"]
        self.hand_below_means_upper = cfg["hand_below_means_upper"]
        self.strict_zone = cfg["strict_zone"]
        self.reset()

    # ---------- состояние ----------
    def reset(self):
        self.running = False
        self.finished = False
        self.paused = False
        self.zone_index = 0
        self.zone_time = [0.0] * len(ZONES)
        self.detected_zone = None
        self.samples = 0        # сколько кадров чистки оценивали зону
        self.matches = 0        # в скольких из них оценка совпала с заданной зоной
        self._trail = deque()
        self._last_t = 0.0
        self._last_active = 0.0

    @property
    def current_zone(self):
        return None if self.finished else ZONES[self.zone_index]

    @property
    def total_sec(self):
        return self.zone_sec * len(ZONES)

    @property
    def elapsed(self):
        return sum(self.zone_time)

    @property
    def remaining(self):
        return max(0.0, self.total_sec - self.elapsed)

    @property
    def zone_match_ratio(self):
        """Доля кадров, где оценка зоны совпала с заданной (для проверки метрики по квадрантам)."""
        return self.matches / self.samples if self.samples else 0.0

    # ---------- управление ----------
    def start(self, t):
        """Начинает сессию; возвращает событие о первой зоне."""
        self.reset()
        self.running = True
        self._last_t = self._last_active = t
        return [self._zone_event()]

    def abort(self):
        """Досрочное завершение (например, жестом): сессия закрывается как незавершённая."""
        if not self.running or self.finished:
            return []
        self.running, self.finished = False, True
        return [Event(EventType.BRUSHING_FINISHED, {"completed": False, "zone_time": list(self.zone_time)})]

    def _zone_event(self):
        return Event(EventType.BRUSHING_ZONE, {"zone": ZONES[self.zone_index], "index": self.zone_index})

    # ---------- движение ----------
    def _is_oscillating(self):
        """Рука колеблется: достаточный размах и путь заметно больше размаха (движение туда-сюда)."""
        if len(self._trail) < 5 or self._trail[-1][0] - self._trail[0][0] < 0.5 * self.window:
            return False  # окно ещё не набралось
        xs = [p[1] for p in self._trail]
        ys = [p[2] for p in self._trail]
        extent = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
        if extent < self.min_extent:
            return False  # дрожание точек, а не движение
        path = sum(math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1]) for i in range(1, len(xs)))
        return path >= self.min_ratio * extent

    # ---------- основной шаг ----------
    def update(self, t, features):
        """Принимает кадр (время и Features или None); возвращает список событий."""
        if not self.running or self.finished:
            return []
        dt = min(max(t - self._last_t, 0.0), 0.5)  # ограничение: после долгой паузы кадров не копим время скачком
        self._last_t = t

        if features is not None and features.near:
            self._trail.append((t, features.dx, features.dy))
            while self._trail and t - self._trail[0][0] > self.window:
                self._trail.popleft()
            active = self._is_oscillating()
            self.detected_zone = estimate_zone(features.dx, features.dy, self.dead_zone,
                                               self.hand_below_means_upper)
        else:
            self._trail.clear()
            active = False
            self.detected_zone = None

        events = []
        if active:
            self._last_active = t
            if self.paused:
                self.paused = False
                events.append(Event(EventType.BRUSHING_RESUMED))

            target = ZONES[self.zone_index]
            self.samples += 1
            self.matches += self.detected_zone == target
            # В строгом режиме время идёт, только если определённая сторона совпала с заданной
            side_ok = self.detected_zone is not None and self.detected_zone.endswith(target.split("_")[1])
            if not self.strict_zone or side_ok:
                self.zone_time[self.zone_index] += dt

            if self.zone_time[self.zone_index] >= self.zone_sec:
                self.zone_time[self.zone_index] = self.zone_sec
                self.zone_index += 1
                if self.zone_index >= len(ZONES):
                    self.zone_index = len(ZONES) - 1
                    self.finished, self.running = True, False
                    events.append(Event(EventType.BRUSHING_FINISHED,
                                        {"completed": True, "zone_time": list(self.zone_time)}))
                else:
                    events.append(self._zone_event())
        elif not self.paused and t - self._last_active >= self.pause_after:
            self.paused = True
            events.append(Event(EventType.BRUSHING_PAUSED))
        return events
