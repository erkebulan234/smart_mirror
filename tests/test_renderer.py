from datetime import datetime

import numpy as np
import pytest

from mirror.services.weather import Weather
from mirror.ui.renderer import (Renderer, View, format_clock, format_date, greeting)

NOW = datetime(2026, 11, 18, 7, 42)          # среда
EVENTS = [{"time": f"{9 + i}:00", "title": f"Событие {i}", "place": "Онлайн"} for i in range(5)]


@pytest.fixture(scope="module")
def renderer(cfg):
    return Renderer(cfg["ui"])


def camera_frame():
    """Синтетический кадр 640x480 с градиентом: достаточно, чтобы проверить отрисовку."""
    x = np.tile(np.linspace(0, 255, 640, dtype=np.uint8), (480, 1))
    return np.dstack([x, x[::-1], np.full_like(x, 90)])


VIEWS = {
    "idle": View(state="idle", now=NOW, weather=Weather(-3, "Пасмурно", 4.2)),
    "idle_no_weather": View(state="idle", now=NOW),
    "identifying": View(state="identifying", now=NOW),
    "personal": View(state="personal", now=NOW, user="Айдар", events=EVENTS,
                     reminders=["Выпить воды", "Загрузить отчёт недели на Google Drive и сделать коммит"]),
    "personal_guest": View(state="personal", now=NOW, user="guest"),
    "trainer": View(state="trainer", now=NOW, user="Айдар", zone_index=1,
                    zone_time=[30.0, 18.0, 0.0, 0.0], remaining=72.0, hand_near=True),
    "trainer_paused": View(state="trainer", now=NOW, user="guest", paused=True, remaining=100.0),
    "summary": View(state="summary", now=NOW, user="Айдар", zone_time=[30.0] * 4, summary_left=12),
    "summary_aborted": View(state="summary", now=NOW, user="Айдар", completed=False,
                            zone_time=[30.0, 12.0, 0.0, 0.0]),
}


@pytest.mark.parametrize("name", list(VIEWS))
def test_every_screen_renders_with_correct_shape(renderer, cfg, name):
    out = renderer.render(camera_frame(), VIEWS[name])
    assert out.shape == (cfg["ui"]["height"], cfg["ui"]["width"], 3)
    assert out.dtype == np.uint8


def test_screens_differ_from_each_other(renderer):
    a = renderer.render(camera_frame(), VIEWS["idle"])
    b = renderer.render(camera_frame(), VIEWS["personal"])
    assert np.abs(a.astype(int) - b.astype(int)).mean() > 1.0


def test_debug_overlay_renders(renderer):
    v = View(state="idle", now=NOW, debug=True, debug_lines=["FPS 25", "state idle"], points=[(0.5, 0.5), (0.1, 0.1)])
    assert renderer.render(camera_frame(), v).shape[2] == 3


def test_helpers():
    assert greeting(7) == "Доброе утро" and greeting(14) == "Добрый день"
    assert greeting(20) == "Добрый вечер" and greeting(2) == "Доброй ночи"
    assert format_date(NOW) == "Среда, 18 ноября"
    assert format_clock(72) == "1:12" and format_clock(120) == "2:00" and format_clock(-5) == "0:00"
    assert Renderer.EVENTS_PER_PAGE == 3
