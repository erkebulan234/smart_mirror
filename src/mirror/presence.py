"""Отслеживание присутствия лица перед зеркалом (ФТ-2).

Превращает поток "лицо есть / лица нет" в события FACE_APPEARED, FACE_LOST и TIMEOUT.
Какое состояние интерфейса из этого получится, решает конечный автомат.
"""
from .events import Event, EventType


class PresenceTracker:
    def __init__(self, cfg):
        self.debounce = cfg["activation_debounce_sec"]
        self.lost_grace = cfg["lost_grace_sec"]
        self.idle_timeout = cfg["idle_timeout_sec"]

        self._visible = False         # лицо подтверждено (событие FACE_APPEARED уже было)
        self._seen_since = None       # с какого момента лицо непрерывно в кадре
        self._absent_since = None     # с какого момента лица нет
        self._timeout_sent = False

    def update(self, face_present, t):
        """Принимает признак наличия лица и время; возвращает список событий."""
        events = []
        if face_present:
            self._absent_since = None
            self._timeout_sent = False
            if self._seen_since is None:
                self._seen_since = t
            # Короткое мелькание лица (меньше debounce) не считается появлением
            if not self._visible and (t - self._seen_since) >= self.debounce:
                self._visible = True
                events.append(Event(EventType.FACE_APPEARED))
        else:
            self._seen_since = None
            if self._absent_since is None:
                self._absent_since = t
            absent = t - self._absent_since
            # Короткая потеря лица (моргание детектора, поворот головы) не считается уходом
            if self._visible and absent >= self.lost_grace:
                self._visible = False
                events.append(Event(EventType.FACE_LOST))
            if not self._timeout_sent and absent >= self.idle_timeout:
                self._timeout_sent = True
                events.append(Event(EventType.TIMEOUT))
        return events
