"""Главное приложение «Умное зеркало».

Класс MirrorApp содержит всю логику (присутствие, идентификация, жесты, тренер, переходы между
экранами) и не зависит от камеры: ему передаются готовые точки лица и рук. Благодаря этому
сценарий целиком проверяется тестами. Функция main() подключает камеру, MediaPipe и окно.

Запуск:  python run_app.py
"""
import math
import time
from datetime import datetime

from .config import load_config, resolve_path
from .cv.brushing import BrushingTracker, extract_features
from .cv.gestures import GestureRecognizer
from .cv.identity import IdentificationSession
from .events import Event, EventBus, EventType
from .presence import PresenceTracker
from .services.schedule import get_for_profile
from .state_machine import State, StateMachine
from .ui.renderer import View

IDENTIFY_MAX_SEC = 3.0   # если за это время голосов не набралось, решаем по тем, что есть


class MirrorApp:
    def __init__(self, cfg, identifier, store, weather=None, schedule=None):
        self.cfg = cfg
        self.identifier, self.store, self.weather = identifier, store, weather
        self.schedule = schedule or {}

        self.bus = EventBus()
        self.sm = StateMachine(self.bus)          # подписывается на события первым: он меняет состояние
        self.presence = PresenceTracker(cfg["presence"])
        self.gestures = GestureRecognizer(cfg["gestures"])
        self.tracker = BrushingTracker(cfg["brushing"])
        self.near_ratio = cfg["brushing"]["mouth_distance_ratio"]
        self.summary_timeout = cfg["ui"]["summary_timeout_sec"]

        # Данные текущей сессии
        self.user = None            # имя профиля или "guest"
        self.page = 0               # страница расписания
        self.summary = None         # итог чистки: {"completed": bool, "zone_time": [...]}
        self.last_gesture = None
        self._ident = None          # идёт ли голосование по лицу
        self._ident_start = 0.0
        self._summary_start = 0.0
        self._features = None
        self._t = 0.0

        # Приложение подписывается на все события после автомата: к этому моменту состояние уже обновлено
        for event_type in EventType:
            self.bus.subscribe(event_type, self._on_event)

    # ---------- реакция на события ----------
    def _on_event(self, event):
        et, state = event.type, self.sm.state
        if et == EventType.FACE_APPEARED and state == State.IDENTIFYING:
            # Профили читаются при каждом появлении лица: только что добавленный профиль работает сразу
            self._ident = IdentificationSession(self.identifier, self.store.load_all(), self.cfg["identification"])
            self._ident_start = self._t
        elif et == EventType.USER_IDENTIFIED:
            self.user, self.page, self._ident = event.payload["profile"], 0, None
        elif et == EventType.BRUSHING_STARTED and state == State.TRAINER:
            for e in self.tracker.start(self._t):
                self.bus.publish(e)
        elif et == EventType.BRUSHING_FINISHED and state == State.SUMMARY:
            self.summary, self._summary_start = event.payload, self._t
        elif state == State.IDLE and et in (EventType.TIMEOUT, EventType.FACE_LOST):
            self.user, self._ident = None, None

    # ---------- жесты ----------
    def handle_gesture(self, name):
        """Применяет жест к текущему экрану. Вызывается и распознавателем, и клавишами для отладки."""
        self.last_gesture = name
        state = self.sm.state
        if state == State.PERSONAL:
            if name == "thumb_up":
                self.bus.publish(Event(EventType.BRUSHING_STARTED))
            elif name == "palm":
                self.bus.publish(Event(EventType.TIMEOUT))   # закрыть экран: автомат вернёт в ожидание
            elif name in ("swipe_left", "swipe_right"):
                pages = max(1, math.ceil(len(self._events()) / 3))
                self.page = (self.page + (1 if name == "swipe_left" else -1)) % pages
        elif state == State.TRAINER and name == "palm":
            for e in self.tracker.abort():
                self.bus.publish(e)
        elif state == State.SUMMARY and name == "palm":
            self.bus.publish(Event(EventType.TIMEOUT))

    # ---------- один кадр ----------
    def process(self, frame, face_lm, hands, now):
        """Логика одного кадра. face_lm: точки лица или None; hands: список кистей (может быть пустым)."""
        self._t = now

        # 1. Лицо в кадре -> события FACE_APPEARED / FACE_LOST / TIMEOUT
        for e in self.presence.update(bool(face_lm), now):
            self.bus.publish(e)

        # 2. Идентификация: голосование по нескольким кадрам
        if self.sm.state == State.IDENTIFYING and self._ident is not None:
            decision = self._ident.feed(frame)
            if decision is None and now - self._ident_start >= IDENTIFY_MAX_SEC:
                decision = self._ident.finish()
            if decision is not None:
                self.bus.publish(Event(EventType.USER_IDENTIFIED, {"profile": decision}))

        # 3. Жесты
        gesture = self.gestures.update(hands[0] if hands else None, now)
        if gesture:
            self.handle_gesture(gesture)

        # 4. Тренер: положение руки относительно рта -> время по зонам
        if self.sm.state == State.TRAINER:
            aspect = frame.shape[1] / frame.shape[0]
            self._features = extract_features(face_lm, hands, self.near_ratio, aspect)
            for e in self.tracker.update(now, self._features):
                self.bus.publish(e)

        # 5. Итог закрывается сам через summary_timeout_sec
        if self.sm.state == State.SUMMARY and now - self._summary_start >= self.summary_timeout:
            self.bus.publish(Event(EventType.TIMEOUT))

    # ---------- данные для экрана ----------
    def _events(self):
        return get_for_profile(self.schedule, self.user or "guest")[0]

    def build_view(self, now_dt=None, debug=False, debug_lines=None, points=None):
        state = self.sm.state
        v = View(state=state.value, now=now_dt or datetime.now(), user=self.user,
                 weather=self.weather.current if self.weather else None,
                 debug=debug, debug_lines=debug_lines or [], points=points or [])
        if state == State.PERSONAL:
            v.events, v.reminders = get_for_profile(self.schedule, self.user or "guest", v.now.date())
            v.page = self.page
        elif state == State.TRAINER:
            t = self.tracker
            v.zone_index, v.zone_time, v.zone_sec = t.zone_index, list(t.zone_time), t.zone_sec
            v.remaining, v.paused, v.detected_zone = t.remaining, t.paused, t.detected_zone
            v.hand_near = bool(self._features and self._features.near)
        elif state == State.SUMMARY and self.summary:
            v.completed, v.zone_time = self.summary["completed"], list(self.summary["zone_time"])
            v.summary_left = max(0.0, self.summary_timeout - (self._t - self._summary_start))
        return v


def main():
    # Тяжёлые импорты здесь: тесты логики не требуют камеры и MediaPipe
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    from .cv.identity import FaceIdentifier
    from .models import ensure_model
    from .services.schedule import load_schedule
    from .services.weather import WeatherService
    from .storage.profiles import ProfileStore
    from .ui.renderer import Renderer

    cfg = load_config()
    cam = cfg["camera"]

    store = ProfileStore(resolve_path(cfg["storage"]["db_path"]))
    profiles = store.list_profiles()
    print("Профили:", ", ".join(profiles) if profiles else "нет (все будут гостями). "
          "Добавить: python enroll.py add Имя")

    identifier = FaceIdentifier(cfg["identification"])
    weather = WeatherService(cfg["weather"])
    weather.start()
    schedule = load_schedule(resolve_path(cfg["schedule"]["path"]))
    renderer = Renderer(cfg["ui"])
    app = MirrorApp(cfg, identifier, store, weather, schedule)

    face_lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=ensure_model("face_landmarker.task")),
        running_mode=vision.RunningMode.IMAGE, num_faces=1))
    hand_lm = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=ensure_model("hand_landmarker.task")),
        running_mode=vision.RunningMode.IMAGE, num_hands=2))

    cap = cv2.VideoCapture(cam["index"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam["height"])
    if not cap.isOpened():
        raise SystemExit("Камера не открылась: проверьте camera.index в config.yaml")

    win = "Smart mirror"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, cfg["ui"]["width"], cfg["ui"]["height"])
    print("Клавиши (в окне): q выход | d отладка | t большой палец | o ладонь | , и . свайп влево/вправо")

    debug, fps, prev = False, 0.0, time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if cam["mirror"]:
                frame = cv2.flip(frame, 1)
            now = time.perf_counter()

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            face_res = face_lm.detect(mp_image)
            hand_res = hand_lm.detect(mp_image)
            face = face_res.face_landmarks[0] if face_res.face_landmarks else None
            hands = list(hand_res.hand_landmarks)

            app.process(frame, face, hands, now)

            fps = 0.9 * fps + 0.1 / max(now - prev, 1e-6)
            prev = now
            lines, points = [], []
            if debug:
                lines = [f"FPS {fps:.1f}  состояние: {app.sm.state.value}",
                         f"лицо: {bool(face)}  рук: {len(hands)}  жест: {app.last_gesture}",
                         f"зона (оценка): {app.tracker.detected_zone}  совпадение: {app.tracker.zone_match_ratio:.0%}"]
                points = [(p.x, p.y) for h in hands for p in h]
                if face:
                    points += [(face[i].x, face[i].y) for i in (13, 14, 234, 454)]

            cv2.imshow(win, renderer.render(frame, app.build_view(debug=debug, debug_lines=lines, points=points)))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("d"):
                debug = not debug
            elif key in (ord("t"), ord("o"), ord(","), ord(".")):
                app.handle_gesture({ord("t"): "thumb_up", ord("o"): "palm",
                                    ord(","): "swipe_left", ord("."): "swipe_right"}[key])
    except KeyboardInterrupt:
        pass
    finally:
        weather.stop()
        cap.release()
        cv2.destroyAllWindows()
        store.close()


if __name__ == "__main__":
    main()
