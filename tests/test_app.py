"""Сквозные тесты логики приложения: камеры нет, вместо неё синтетические точки лица и рук."""
import math

import numpy as np
import pytest

from mirror.app import MirrorApp
from mirror.state_machine import State
from mirror.storage.profiles import EMBEDDING_DIM, ProfileStore

from .helpers import palm, thumb_up
from .test_brushing import face_landmarks, hand_at
from .test_profiles_identity import new_shot, person

FPS = 30
FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
FACE = face_landmarks()            # рот в (0.5, 0.55), ширина лица 0.3 по оси x

SCHEDULE = {
    "Айдар": {"events": [{"time": f"{9 + i:02d}:00", "title": f"Событие {i}"} for i in range(5)],
              "reminders": ["Выпить воды"]},
    "guest": {"events": [], "reminders": ["Добро пожаловать!"]},
}


class FakeIdentifier:
    """Вместо YuNet+SFace: возвращает заданный вектор лица (None = лицо не найдено)."""

    def __init__(self, vec):
        self.vec = vec

    def embed(self, frame):
        return self.vec


def brush_hands(t, cx=-0.45, cy=0.3):
    """Рука у рта колеблется 3 раза в секунду. Смещения заданы в ширинах лица, затем переведены в кадр."""
    dx = cx + 0.06 * math.sin(2 * math.pi * 3 * t)
    # Ширина лица в единицах расчёта = 0.3 * (640/480) = 0.4; по оси x нормировка ещё делится на aspect
    return [hand_at(0.5 + dx * 0.3, 0.55 + cy * 0.4)]


class Sim:
    def __init__(self, app):
        self.app, self.t, self.states = app, 0.0, [app.sm.state]

    def step(self, face=True, hands=None):
        hands = hands(self.t) if callable(hands) else (hands or [])
        self.app.process(FRAME, FACE if face else None, hands, self.t)
        self.t += 1 / FPS
        if self.app.sm.state != self.states[-1]:
            self.states.append(self.app.sm.state)

    def run(self, seconds, face=True, hands=None):
        for _ in range(int(seconds * FPS)):
            self.step(face, hands)


@pytest.fixture
def known(cfg):
    """Приложение с одним зарегистрированным профилем и идентификатором, который видит этого человека."""
    base, shots = person(1)
    store = ProfileStore(":memory:")
    store.add_profile("Айдар", shots)
    app = MirrorApp(cfg, FakeIdentifier(new_shot(base, 77)), store, weather=None, schedule=SCHEDULE)
    return app, base


def test_full_scenario_from_idle_to_summary_and_back(known):
    app, _ = known
    sim = Sim(app)

    sim.run(1.0, face=False)
    assert app.sm.state == State.IDLE

    sim.run(1.0)                                  # человек подошёл: появление, идентификация
    assert app.sm.state == State.PERSONAL and app.user == "Айдар"

    sim.run(0.7, hands=[thumb_up()])              # большой палец вверх: начать чистку
    assert app.sm.state == State.TRAINER
    sim.run(0.3)                                  # руку убрали

    sim.run(125, hands=brush_hands)               # чистка с запасом по времени
    assert app.sm.state == State.SUMMARY
    assert app.summary["completed"] is True and app.summary["zone_time"] == [30.0] * 4

    sim.run(16, hands=None)                       # итог закрывается сам через 15 с
    assert app.sm.state == State.IDLE
    assert app.user is None

    assert sim.states == [State.IDLE, State.IDENTIFYING, State.PERSONAL, State.TRAINER,
                          State.SUMMARY, State.IDLE]


def test_activation_time_is_within_one_second(known):
    """ФТ-2: от появления лица до персонального экрана не больше 1 с."""
    app, _ = known
    sim = Sim(app)
    t_start = sim.t
    while app.sm.state != State.PERSONAL and sim.t - t_start < 3:
        sim.step()
    assert app.sm.state == State.PERSONAL
    assert sim.t - t_start <= 1.0


def test_unknown_person_becomes_guest(known):
    app, _ = known
    stranger, _ = person(999)
    app.identifier.vec = new_shot(stranger, 5)
    sim = Sim(app)
    sim.run(1.5)
    assert app.sm.state == State.PERSONAL and app.user == "guest"
    assert app.build_view().reminders == ["Добро пожаловать!"]


def test_no_profiles_means_guest(cfg):
    store = ProfileStore(":memory:")
    base, _ = person(1)
    app = MirrorApp(cfg, FakeIdentifier(new_shot(base, 1)), store, weather=None, schedule=SCHEDULE)
    Sim(app).run(1.5)
    assert app.user == "guest"


def test_identification_falls_back_to_guest_when_detector_finds_no_face(known):
    """MediaPipe видит лицо, а детектор идентификации нет: через 3 с решаем как гость."""
    app, _ = known
    app.identifier.vec = None
    sim = Sim(app)
    sim.run(1.5)
    assert app.sm.state == State.IDENTIFYING
    sim.run(3.0)
    assert app.sm.state == State.PERSONAL and app.user == "guest"


def test_palm_closes_personal_screen_and_forgets_user(known):
    app, _ = known
    sim = Sim(app)
    sim.run(1.5)
    sim.run(0.7, hands=[palm()])
    assert app.sm.state == State.IDLE and app.user is None


def test_face_leaving_returns_to_idle_after_timeout(known):
    app, _ = known
    sim = Sim(app)
    sim.run(1.5)
    assert app.sm.state == State.PERSONAL
    sim.run(14.0, face=False)
    assert app.sm.state == State.PERSONAL         # 15 с ещё не прошло
    sim.run(2.5, face=False)
    assert app.sm.state == State.IDLE


def test_gestures_do_nothing_on_idle_screen(known):
    app, _ = known
    sim = Sim(app)
    sim.run(1.0, face=False, hands=[thumb_up()])
    assert app.sm.state == State.IDLE


def test_swipe_changes_schedule_page_and_wraps(known):
    app, _ = known
    Sim(app).run(1.5)
    assert app.build_view().page == 0 and len(app.build_view().events) == 5
    app.handle_gesture("swipe_left")
    assert app.build_view().page == 1
    app.handle_gesture("swipe_left")
    assert app.build_view().page == 0             # страниц две: переход по кругу
    app.handle_gesture("swipe_right")
    assert app.build_view().page == 1


def test_pause_is_visible_in_view_and_resumes(known):
    app, _ = known
    sim = Sim(app)
    sim.run(1.5)
    sim.run(0.7, hands=[thumb_up()])
    sim.run(5, hands=brush_hands)
    assert not app.build_view().paused
    sim.run(7, hands=None)                        # рука ушла: через 5 с пауза
    v = app.build_view()
    assert v.state == "trainer" and v.paused
    sim.run(3, hands=brush_hands)
    assert not app.build_view().paused


def test_palm_aborts_training_and_summary_shows_incomplete(known):
    app, _ = known
    sim = Sim(app)
    sim.run(1.5)
    sim.run(0.7, hands=[thumb_up()])
    sim.run(0.3)
    sim.run(10, hands=brush_hands)
    sim.run(0.7, hands=[palm()])
    assert app.sm.state == State.SUMMARY
    v = app.build_view()
    assert v.completed is False and 5 < sum(v.zone_time) < 12


def test_view_for_every_state_is_consistent(known):
    app, _ = known
    sim = Sim(app)
    assert app.build_view().state == "idle"
    sim.run(0.4)
    assert app.build_view().state == "identifying"
    sim.run(1.0)
    v = app.build_view()
    assert v.state == "personal" and v.user == "Айдар" and v.reminders == ["Выпить воды"]
    sim.run(0.7, hands=[thumb_up()])
    v = app.build_view()
    assert v.state == "trainer" and v.remaining == 120 and v.zone_index == 0
