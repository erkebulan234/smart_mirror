"""События, через которые модули общаются друг с другом (раздел 6.2 ТЗ).

CV-модуль, логика приложения, хранилище и интерфейс не вызывают друг друга
напрямую: они публикуют и принимают события через EventBus.
"""
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class EventType(str, Enum):
    FACE_APPEARED = "face_appeared"        # в кадре появилось лицо
    FACE_LOST = "face_lost"                # лицо пропало из кадра
    USER_IDENTIFIED = "user_identified"    # payload: {"profile": имя}, либо "guest"
    GESTURE = "gesture"                    # payload: {"name": "swipe_left" | ...}
    BRUSHING_STARTED = "brushing_started"  # начало чистки зубов
    BRUSHING_ZONE = "brushing_zone"        # payload: {"zone": "upper_left" | ...}
    BRUSHING_PAUSED = "brushing_paused"    # движение отсутствует > 5 с
    BRUSHING_RESUMED = "brushing_resumed"  # движение возобновилось после паузы
    BRUSHING_FINISHED = "brushing_finished"
    TIMEOUT = "timeout"                    # истёк тайм-аут отсутствия лица


# Допустимые названия зон чистки (4 квадранта рта)
ZONES = ("upper_left", "upper_right", "lower_left", "lower_right")


@dataclass
class Event:
    type: EventType
    payload: dict[str, Any] = field(default_factory=dict)


class EventBus:
    """Минимальная шина событий: подписка по типу, синхронная рассылка."""

    def __init__(self):
        self._handlers = defaultdict(list)

    def subscribe(self, event_type, handler: Callable[[Event], None]):
        self._handlers[event_type].append(handler)

    def publish(self, event: Event):
        # Копия списка: обработчик может подписываться во время рассылки
        for handler in list(self._handlers[event.type]):
            handler(event)
