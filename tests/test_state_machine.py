from mirror.events import Event, EventBus, EventType
from mirror.state_machine import State, StateMachine


def test_full_morning_scenario():
    """Сценарий 1 + 2 из ТЗ: от ожидания до итога чистки и обратно."""
    bus = EventBus()
    sm = StateMachine(bus)
    assert sm.state == State.IDLE

    bus.publish(Event(EventType.FACE_APPEARED))
    assert sm.state == State.IDENTIFYING

    bus.publish(Event(EventType.USER_IDENTIFIED, {"profile": "Aidar"}))
    assert sm.state == State.PERSONAL

    bus.publish(Event(EventType.BRUSHING_STARTED))
    assert sm.state == State.TRAINER

    bus.publish(Event(EventType.BRUSHING_FINISHED))
    assert sm.state == State.SUMMARY

    bus.publish(Event(EventType.TIMEOUT))
    assert sm.state == State.IDLE


def test_face_lost_during_identification_returns_to_idle():
    bus = EventBus()
    sm = StateMachine(bus)
    bus.publish(Event(EventType.FACE_APPEARED))
    bus.publish(Event(EventType.FACE_LOST))
    assert sm.state == State.IDLE


def test_unexpected_event_is_ignored():
    """Жест в режиме ожидания не должен менять состояние."""
    bus = EventBus()
    sm = StateMachine(bus)
    bus.publish(Event(EventType.BRUSHING_STARTED))
    assert sm.state == State.IDLE


def test_trainer_ignores_timeout():
    """Во время чистки пропадание лица не должно прерывать тренер."""
    bus = EventBus()
    sm = StateMachine(bus)
    bus.publish(Event(EventType.FACE_APPEARED))
    bus.publish(Event(EventType.USER_IDENTIFIED, {"profile": "guest"}))
    bus.publish(Event(EventType.BRUSHING_STARTED))
    bus.publish(Event(EventType.TIMEOUT))
    assert sm.state == State.TRAINER
