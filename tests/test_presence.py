from mirror.events import EventType
from mirror.presence import PresenceTracker


def run(tracker, samples):
    """samples: (есть ли лицо, время). Возвращает список (тип события, время)."""
    out = []
    for present, t in samples:
        for e in tracker.update(present, t):
            out.append((e.type, round(t, 2)))
    return out


def steps(present, start, end, dt=0.1):
    n = int(round((end - start) / dt))
    return [(present, round(start + i * dt, 2)) for i in range(n + 1)]


def test_face_appears_after_debounce(cfg):
    tr = PresenceTracker(cfg["presence"])
    ev = run(tr, steps(True, 0.0, 1.0))
    assert ev == [(EventType.FACE_APPEARED, 0.3)]


def test_activation_within_one_second(cfg):
    """ФТ-2: переход при появлении лица не дольше 1 с."""
    tr = PresenceTracker(cfg["presence"])
    ev = run(tr, steps(True, 5.0, 7.0))
    assert ev[0][0] == EventType.FACE_APPEARED
    assert ev[0][1] - 5.0 <= 1.0


def test_flicker_is_ignored(cfg):
    tr = PresenceTracker(cfg["presence"])
    # Лицо мелькнуло на 0.2 с, потом исчезло
    ev = run(tr, steps(True, 0.0, 0.2) + steps(False, 0.3, 1.0))
    assert ev == []


def test_short_loss_does_not_emit_face_lost(cfg):
    tr = PresenceTracker(cfg["presence"])
    ev = run(tr, steps(True, 0.0, 1.0) + steps(False, 1.1, 1.6) + steps(True, 1.7, 3.0))
    assert [e for e, _ in ev] == [EventType.FACE_APPEARED]


def test_face_lost_then_timeout(cfg):
    tr = PresenceTracker(cfg["presence"])
    timeout = cfg["presence"]["idle_timeout_sec"]
    ev = run(tr, steps(True, 0.0, 1.0) + steps(False, 1.1, 1.0 + timeout + 2))
    kinds = [e for e, _ in ev]
    assert kinds == [EventType.FACE_APPEARED, EventType.FACE_LOST, EventType.TIMEOUT]
    # Тайм-аут считается от момента, когда лицо пропало (по умолчанию 15 с)
    t_timeout = ev[-1][1]
    assert abs(t_timeout - (1.1 + timeout)) <= 0.2


def test_timeout_sent_once_per_absence(cfg):
    tr = PresenceTracker(cfg["presence"])
    ev = run(tr, steps(False, 0.0, 60.0, dt=0.5))
    assert [e for e, _ in ev].count(EventType.TIMEOUT) == 1


def test_returning_user_triggers_new_appearance(cfg):
    tr = PresenceTracker(cfg["presence"])
    samples = steps(True, 0.0, 1.0) + steps(False, 1.1, 20.0) + steps(True, 20.1, 21.5)
    kinds = [e for e, _ in run(tr, samples)]
    assert kinds == [EventType.FACE_APPEARED, EventType.FACE_LOST, EventType.TIMEOUT,
                     EventType.FACE_APPEARED]
