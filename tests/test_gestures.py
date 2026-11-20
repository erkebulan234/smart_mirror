from .helpers import fist, make_hand, palm, thumb_up
from mirror.cv.gestures import GestureRecognizer, classify_static

FPS = 30


def feed(rec, frames):
    """Подаёт последовательность (рука, время); возвращает список сработавших жестов."""
    fired = []
    for hand, t in frames:
        g = rec.update(hand, t)
        if g:
            fired.append(g)
    return fired


def hold(hand, start, seconds):
    n = int(seconds * FPS) + 1
    return [(hand, start + i / FPS) for i in range(n)]


def test_classify_palm_thumb_up_and_others():
    assert classify_static(palm()) == "palm"
    assert classify_static(thumb_up()) == "thumb_up"
    assert classify_static(fist()) is None
    # Только указательный палец выпрямлен (указание): не жест
    assert classify_static(make_hand((True, False, False, False), "folded")) is None
    # Ладонь с прижатым большим пальцем тоже считается ладонью
    assert classify_static(make_hand((True, True, True, True), "folded")) == "palm"


def test_classification_does_not_depend_on_hand_position():
    assert classify_static(palm(cx=0.2, cy=0.9)) == "palm"
    assert classify_static(thumb_up(cx=0.8, cy=0.5)) == "thumb_up"


def test_static_gesture_fires_after_hold_and_only_once(cfg):
    rec = GestureRecognizer(cfg["gestures"])
    fired = feed(rec, hold(thumb_up(), 0.0, 2.0))
    assert fired == ["thumb_up"]  # удерживаем 2 с, а событие одно


def test_too_short_hold_does_not_fire(cfg):
    rec = GestureRecognizer(cfg["gestures"])
    assert feed(rec, hold(palm(), 0.0, 0.25)) == []


def test_gesture_fires_again_after_release_and_cooldown(cfg):
    rec = GestureRecognizer(cfg["gestures"])
    frames = hold(thumb_up(), 0.0, 0.6)
    frames += [(None, 0.7)]                       # рука убрана
    frames += hold(thumb_up(), 2.0, 0.6)          # снова показана после паузы
    assert feed(rec, frames) == ["thumb_up", "thumb_up"]


def test_cooldown_blocks_immediate_repeat(cfg):
    rec = GestureRecognizer(cfg["gestures"])
    frames = hold(thumb_up(), 0.0, 0.6)
    frames += [(None, 0.65)]
    frames += hold(thumb_up(), 0.7, 0.6)          # меньше cooldown_ms после первого срабатывания
    assert feed(rec, frames) == ["thumb_up"]


def test_moving_hand_does_not_fire_static_gesture(cfg):
    rec = GestureRecognizer(cfg["gestures"])
    # Ладонь медленно ползёт, каждые 0.4 с смещаясь сильнее допустимого
    frames = [(palm(cx=0.2 + 0.25 * i / FPS), i / FPS) for i in range(int(2 * FPS))]
    assert "palm" not in feed(rec, frames)


def test_swipe_left_and_right(cfg):
    rec = GestureRecognizer(cfg["gestures"])
    left = [(palm(cx=0.75 - 0.5 * i / 12), i / FPS) for i in range(13)]   # 0.4 с, путь 0.5
    assert feed(rec, left) == ["swipe_left"]

    rec = GestureRecognizer(cfg["gestures"])
    right = [(palm(cx=0.25 + 0.5 * i / 12), i / FPS) for i in range(13)]
    assert feed(rec, right) == ["swipe_right"]


def test_slow_movement_is_not_a_swipe(cfg):
    rec = GestureRecognizer(cfg["gestures"])
    frames = [(palm(cx=0.75 - 0.4 * i / (2 * FPS)), i / FPS) for i in range(2 * FPS + 1)]
    assert not any(g.startswith("swipe") for g in feed(rec, frames))


def test_vertical_movement_is_not_a_swipe(cfg):
    rec = GestureRecognizer(cfg["gestures"])
    frames = [(palm(cx=0.5, cy=0.9 - 0.5 * i / 12), i / FPS) for i in range(13)]
    assert not any(g.startswith("swipe") for g in feed(rec, frames))


def test_no_palm_event_right_after_swipe_while_hand_stays_open(cfg):
    rec = GestureRecognizer(cfg["gestures"])
    frames = [(palm(cx=0.75 - 0.5 * i / 12), i / FPS) for i in range(13)]
    frames += hold(palm(cx=0.25), 13 / FPS, 3.0)  # рука остаётся открытой на новом месте
    assert feed(rec, frames) == ["swipe_left"]


def test_hand_lost_resets_state(cfg):
    rec = GestureRecognizer(cfg["gestures"])
    feed(rec, hold(palm(), 0.0, 0.3))
    assert rec.update(None, 0.4) is None
    # После потери руки отсчёт удержания идёт заново
    assert feed(rec, hold(palm(), 0.5, 0.25)) == []
