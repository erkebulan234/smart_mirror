"""Конечный автомат интерфейса (раздел 6.2 ТЗ):
ожидание -> идентификация -> персональный экран -> тренер -> итог.
"""
from enum import Enum

from .events import Event, EventType


class State(str, Enum):
    IDLE = "idle"                  # часы, дата, погода
    IDENTIFYING = "identifying"    # лицо найдено, определяем, кто это
    PERSONAL = "personal"          # персональный или гостевой экран
    TRAINER = "trainer"            # тренер чистки зубов
    SUMMARY = "summary"            # итог сессии чистки


# Таблица переходов: (текущее состояние, тип события) -> новое состояние.
# Любая пара, которой нет в таблице, игнорируется: состояние не меняется.
TRANSITIONS = {
    (State.IDLE, EventType.FACE_APPEARED): State.IDENTIFYING,
    (State.IDENTIFYING, EventType.USER_IDENTIFIED): State.PERSONAL,
    (State.IDENTIFYING, EventType.FACE_LOST): State.IDLE,
    (State.PERSONAL, EventType.BRUSHING_STARTED): State.TRAINER,
    (State.PERSONAL, EventType.TIMEOUT): State.IDLE,
    (State.TRAINER, EventType.BRUSHING_FINISHED): State.SUMMARY,
    (State.SUMMARY, EventType.TIMEOUT): State.IDLE,
}


class StateMachine:
    def __init__(self, bus=None):
        self.state = State.IDLE
        self.bus = bus
        if bus is not None:
            # Автомат слушает все события, которые встречаются в таблице переходов
            for event_type in {e for (_, e) in TRANSITIONS}:
                bus.subscribe(event_type, self.handle)

    def handle(self, event: Event):
        """Применяет событие; возвращает новое состояние."""
        new_state = TRANSITIONS.get((self.state, event.type))
        if new_state is not None:
            self.state = new_state
        return self.state
