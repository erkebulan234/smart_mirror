"""Отрисовка интерфейса зеркала поверх кадра камеры (Pillow + OpenCV).

Все координаты в коде заданы в системе макета 1280x720 и масштабируются под размер окна
(ui.width x ui.height в config.yaml). Кириллицу cv2.putText не рисует, поэтому текст выводит Pillow.
"""
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DESIGN_W, DESIGN_H = 1280, 720

COL = {
    "surface": (20, 26, 34), "surface2": (28, 36, 48), "border": (42, 52, 65),
    "text": (242, 245, 248), "muted": (139, 152, 168),
    "accent": (79, 209, 197), "accent_dim": (15, 59, 56),
    "ok": (91, 213, 138), "warn": (246, 185, 76),
}

WEEKDAYS = ["Понедельник", "Вторник", "Среда", "Четверг", "Пятница", "Суббота", "Воскресенье"]
MONTHS = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа",
          "сентября", "октября", "ноября", "декабря"]
ZONE_RU = {"upper_left": "Верх · слева", "upper_right": "Верх · справа",
           "lower_left": "Низ · слева", "lower_right": "Низ · справа"}
ZONE_ORDER = ["upper_left", "upper_right", "lower_left", "lower_right"]

# Кандидаты шрифтов с кириллицей: для каждого начертания перебираются по порядку
FONT_CANDIDATES = {
    "light": ["C:/Windows/Fonts/segoeuil.ttf", "C:/Windows/Fonts/segoeui.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/System/Library/Fonts/Supplemental/Arial.ttf"],
    "regular": ["C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/System/Library/Fonts/Supplemental/Arial.ttf"],
    "bold": ["C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
             "/System/Library/Fonts/Supplemental/Arial Bold.ttf"],
}


def greeting(hour):
    if 5 <= hour < 12:
        return "Доброе утро"
    if 12 <= hour < 18:
        return "Добрый день"
    if 18 <= hour < 23:
        return "Добрый вечер"
    return "Доброй ночи"


def format_date(now):
    return f"{WEEKDAYS[now.weekday()]}, {now.day} {MONTHS[now.month - 1]}"


def format_clock(seconds):
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60}:{seconds % 60:02d}"


@dataclass
class View:
    """Всё, что нужно знать отрисовщику о текущем моменте. Приложение заполняет его на каждом кадре."""
    state: str = "idle"                      # значение State: idle, identifying, personal, trainer, summary
    now: datetime = field(default_factory=datetime.now)
    weather: object = None                   # Weather или None, если данных ещё нет
    user: str = None                         # имя профиля или "guest"
    events: list = field(default_factory=list)
    reminders: list = field(default_factory=list)
    page: int = 0                            # страница расписания (по 3 события)
    # тренер чистки
    zone_index: int = 0
    zone_time: list = field(default_factory=lambda: [0.0] * 4)
    zone_sec: float = 30.0
    remaining: float = 120.0
    paused: bool = False
    hand_near: bool = False                  # рука у рта
    detected_zone: str = None
    # итог
    completed: bool = True
    summary_left: float = 0.0
    # отладка
    debug: bool = False
    debug_lines: list = field(default_factory=list)
    points: list = field(default_factory=list)   # [(x, y)] в долях кадра камеры


class Renderer:
    EVENTS_PER_PAGE = 3

    def __init__(self, ui_cfg):
        self.W, self.H = ui_cfg["width"], ui_cfg["height"]
        self.s = self.W / DESIGN_W
        self.dim = ui_cfg["background_dim"]
        self.custom_font = ui_cfg.get("font_path") or ""
        self._fonts = {}
        self.pages = lambda n: max(1, -(-n // self.EVENTS_PER_PAGE))

    # ---------- шрифты и примитивы ----------
    def _font(self, size, weight="regular"):
        key = (round(size * self.s), weight)
        if key not in self._fonts:
            paths = ([self.custom_font] if self.custom_font else []) + FONT_CANDIDATES[weight]
            font = None
            for p in paths:
                if p and os.path.exists(p):
                    font = ImageFont.truetype(p, key[0])
                    break
            if font is None:
                print("Предупреждение: не найден шрифт с кириллицей, укажите ui.font_path в config.yaml",
                      file=sys.stderr)
                font = ImageFont.load_default(key[0])
            self._fonts[key] = font
        return self._fonts[key]

    def _text(self, d, x, y, txt, size, weight="regular", color="text", anchor="la"):
        d.text((x * self.s, y * self.s), txt, font=self._font(size, weight), fill=COL[color], anchor=anchor)

    def _width(self, d, txt, size, weight="regular"):
        """Ширина текста в координатах макета."""
        return d.textlength(txt, font=self._font(size, weight)) / self.s

    def _card(self, d, x, y, w, h, fill="surface", r=28, outline=None, ow=1, alpha=225):
        d.rounded_rectangle(
            [x * self.s, y * self.s, (x + w) * self.s, (y + h) * self.s], radius=r * self.s,
            fill=(*COL[fill], alpha), outline=COL[outline] if outline else None,
            width=max(1, round(ow * self.s)))

    def _dot(self, d, cx, cy, r, color, ring=False):
        box = [(cx - r) * self.s, (cy - r) * self.s, (cx + r) * self.s, (cy + r) * self.s]
        if ring:
            d.ellipse(box, outline=COL[color], width=max(1, round(2 * self.s)))
        else:
            d.ellipse(box, fill=COL[color])

    def _wrap(self, d, txt, size, max_w, weight="regular"):
        lines, cur = [], ""
        for word in txt.split():
            trial = f"{cur} {word}".strip()
            if self._width(d, trial, size, weight) <= max_w or not cur:
                cur = trial
            else:
                lines.append(cur)
                cur = word
        return lines + ([cur] if cur else [])

    def _ellipsize(self, d, txt, size, max_w, weight="regular"):
        if self._width(d, txt, size, weight) <= max_w:
            return txt
        while txt and self._width(d, txt + "…", size, weight) > max_w:
            txt = txt[:-1]
        return txt + "…"

    def _chip(self, d, x, y, parts, fill="surface", outline=None, dot=None, h=48, anchor="left"):
        """Плашка с цепочкой фрагментов текста [(текст, размер, начертание, цвет)]; возвращает её ширину."""
        pad = 20
        w = 2 * pad + sum(self._width(d, t, sz, wt) for t, sz, wt, _ in parts) + 10 * (len(parts) - 1)
        w += 22 if dot else 0
        if anchor == "right":
            x -= w
        self._card(d, x, y, w, h, fill=fill, r=h / 2, outline=outline)
        cx = x + pad
        if dot:
            self._dot(d, cx + 5, y + h / 2, 5, dot)
            cx += 22
        for t, sz, wt, col in parts:
            self._text(d, cx, y + h / 2, t, sz, wt, col, anchor="lm")
            cx += self._width(d, t, sz, wt) + 10
        return w

    # ---------- основной вызов ----------
    def render(self, frame_bgr, view):
        """Возвращает BGR-кадр окна: затемнённая камера + интерфейс текущего состояния."""
        base = self._background(frame_bgr)
        layer = Image.new("RGBA", (self.W, self.H), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)

        draw = {
            "idle": self._idle, "identifying": self._identifying, "personal": self._personal,
            "trainer": self._trainer, "summary": self._summary,
        }[view.state]
        draw(d, view)
        if view.debug:
            self._debug(d, view, frame_bgr.shape)

        out = Image.alpha_composite(base, layer).convert("RGB")
        return cv2.cvtColor(np.asarray(out), cv2.COLOR_RGB2BGR)

    def _crop_box(self, w, h):
        """Какая часть кадра камеры попадает в окно (обрезка по центру до пропорций окна)."""
        target = self.W / self.H
        if w / h > target:
            cw = int(h * target)
            return (w - cw) // 2, 0, cw, h
        ch = int(w / target)
        return 0, (h - ch) // 2, w, ch

    def _background(self, frame):
        x, y, cw, ch = self._crop_box(frame.shape[1], frame.shape[0])
        img = cv2.resize(frame[y:y + ch, x:x + cw], (self.W, self.H))
        img = (img.astype(np.float32) * self.dim).astype(np.uint8)
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("RGBA")

    # ---------- экран 1: ожидание ----------
    def _idle(self, d, v):
        self._chip(d, 56, 56, [("Режим ожидания", 20, "regular", "muted")], dot="muted")
        self._weather_card(d, v)
        self._text(d, 640, 330, v.now.strftime("%H:%M"), 220, "light", anchor="mm")
        self._text(d, 640, 500, format_date(v.now), 36, "regular", "muted", anchor="mm")
        txt = "Подойдите к зеркалу: лицо определится меньше чем за 1 с"
        total = 34 + self._width(d, txt, 22)
        x0 = 640 - total / 2
        self._dot(d, x0 + 10, 650, 10, "accent", ring=True)
        self._text(d, x0 + 34, 650, txt, 22, "regular", "muted", anchor="lm")

    def _weather_card(self, d, v):
        if v.weather:
            top, bottom = v.weather.temp_text(), f"{v.weather.description} · ветер {v.weather.wind:.0f} м/с"
        else:
            top, bottom = "— °C", "Нет данных о погоде"
        w = max(self._width(d, top, 40, "bold"), self._width(d, bottom, 20)) + 56
        x = DESIGN_W - 56 - w
        self._card(d, x, 56, w, 92, r=24)
        self._text(d, x + 28, 66, top, 40, "bold")
        self._text(d, x + 28, 116, bottom, 20, "regular", "muted")

    # ---------- идентификация ----------
    def _identifying(self, d, v):
        t = v.now.timestamp()
        cx, cy, r = 640, 330, 54
        box = [(cx - r) * self.s, (cy - r) * self.s, (cx + r) * self.s, (cy + r) * self.s]
        d.arc(box, 0, 360, fill=(*COL["border"], 255), width=max(2, round(8 * self.s)))
        start = (t * 360) % 360  # вращающаяся дуга: видно, что система работает
        d.arc(box, start, start + 100, fill=(*COL["accent"], 255), width=max(2, round(8 * self.s)))
        self._text(d, 640, 450, "Определяю вас…", 40, "bold", anchor="mm")
        self._text(d, 640, 505, "Смотрите в камеру", 22, "regular", "muted", anchor="mm")

    # ---------- экран 2: персональный ----------
    def _personal(self, d, v):
        guest = v.user in (None, "guest")
        name = "гость" if guest else v.user
        self._text(d, 56, 50, f"{greeting(v.now.hour)}, {name}", 56, "bold")
        self._text(d, 56, 126, f"{format_date(v.now)} · {v.now.strftime('%H:%M')}", 24, "regular", "muted")

        # Карточка профиля справа вверху
        sub = "профиль не найден" if guest else "профиль распознан"
        w = 2 * 20 + 44 + 14 + max(self._width(d, "Гость" if guest else name, 20, "bold"),
                                   self._width(d, sub, 16))
        x = DESIGN_W - 56 - w
        self._card(d, x, 54, w, 68, r=34)
        self._dot(d, x + 20 + 22, 88, 22, "accent_dim")
        self._text(d, x + 42, 88, ("Г" if guest else name[0]).upper(), 22, "bold", "accent", anchor="mm")
        self._text(d, x + 20 + 58, 74, "Гость" if guest else name, 20, "bold")
        self._text(d, x + 20 + 58, 100, sub, 16, "regular", "muted")

        self._schedule_card(d, v)
        self._reminders_card(d, v)

        # Подсказки жестов
        self._text(d, 56, 641, "Жесты", 18, "regular", "muted", anchor="lm")
        x = 128
        for g, desc in (("Ладонь", "закрыть экран"), ("Большой палец", "начать чистку зубов"),
                        ("Свайп", "листать расписание")):
            x += self._chip(d, x, 617, [(g, 18, "bold", "accent"), (desc, 18, "regular", "text")],
                            fill="surface", outline="border", h=48) + 14

    def _schedule_card(self, d, v):
        self._card(d, 56, 176, 744, 412)
        self._text(d, 88, 204, "Расписание на сегодня", 26, "bold")
        per = self.EVENTS_PER_PAGE
        pages = self.pages(len(v.events))
        page = v.page % pages
        shown = v.events[page * per:(page + 1) * per]
        if not shown:
            self._text(d, 428, 400, "На сегодня событий нет", 24, "regular", "muted", anchor="mm")
        for i, e in enumerate(shown):
            y = 258 + i * 104
            self._card(d, 88, y, 680, 88, fill="surface2", r=18)
            self._card(d, 108, y + 20, 104, 48, fill="accent_dim", r=12)
            self._text(d, 160, y + 44, e.get("time", ""), 22, "bold", "accent", anchor="mm")
            title = self._ellipsize(d, e.get("title", ""), 24, 520)
            self._text(d, 232, y + 14, title, 24)
            if e.get("place"):
                self._text(d, 232, y + 52, e["place"], 18, "regular", "muted")
        if pages > 1:
            self._text(d, 768, 566, f"Стр. {page + 1}/{pages} · свайп", 16, "regular", "muted", anchor="rm")

    def _reminders_card(self, d, v):
        self._card(d, 824, 176, 400, 412)
        self._text(d, 856, 204, "Напоминания", 26, "bold")
        if not v.reminders:
            self._text(d, 1024, 400, "Напоминаний нет", 22, "regular", "muted", anchor="mm")
        y = 262
        for r in v.reminders[:6]:
            lines = self._wrap(d, r, 22, 310)
            self._dot(d, 862, y + 15, 5, "accent")
            for ln in lines:
                self._text(d, 884, y, ln, 22)
                y += 30
            y += 16

    # ---------- экран 3: тренер ----------
    def _trainer(self, d, v):
        self._chip(d, 56, 56, [("Тренер чистки зубов", 20, "regular", "accent")], fill="accent_dim", dot="accent")
        name = "гость" if v.user in (None, "guest") else v.user
        self._text(d, 1224, 80, name, 22, "regular", "muted", anchor="rm")

        self._card(d, 56, 130, 620, 468)
        self._text(d, 84, 156, "Зоны чистки · по 30 с", 22, "bold")
        for i, zone in enumerate(ZONE_ORDER):
            x = 84 + (i % 2) * 290
            y = 206 + (i // 2) * 166
            self._zone_cell(d, v, i, zone, x, y)
        self._text(d, 84, 552, "Стороны показаны как в зеркале", 16, "regular", "muted")

        # Таймер и прогресс справа
        self._text(d, 724, 160, "Осталось", 22, "regular", "muted")
        self._text(d, 716, 186, format_clock(v.remaining), 180, "light")
        done = 1 - v.remaining / (v.zone_sec * 4)
        self._bar(d, 724, 430, 480, 14, done, "accent", "surface2")
        idx = min(v.zone_index, 3)
        self._text(d, 724, 470, f"Квадрант {idx + 1} из 4 · {ZONE_RU[ZONE_ORDER[idx]]}", 28, "bold")
        for i, line in enumerate(self._wrap(d, "Мелкие круговые движения, без сильного нажима", 22, 480)):
            self._text(d, 724, 518 + 30 * i, line, 22, "regular", "muted")

        # Строка состояния
        if v.paused:
            col, txt = "warn", "Пауза: возьмите щётку и продолжайте"
        elif v.hand_near:
            col, txt = "ok", "Движение определено"
        else:
            col, txt = "muted", "Поднесите щётку ко рту"
        self._dot(d, 62, 652, 6, col)
        self._text(d, 82, 652, txt, 20, "bold", col, anchor="lm")
        self._text(d, 1224, 652, "Пауза, если движения нет больше 5 с", 18, "regular", "muted", anchor="rm")

    def _zone_cell(self, d, v, i, zone, x, y):
        w, h = 274, 150
        done = v.zone_time[i] >= v.zone_sec - 1e-6
        active = i == v.zone_index and not done
        if active:
            self._card(d, x, y, w, h, fill="accent_dim", r=22, outline="accent", ow=2)
        else:
            self._card(d, x, y, w, h, fill="surface2" if done else "surface", r=22, outline="border")
        color = "accent" if active else "text" if done else "muted"
        self._text(d, x + 20, y + 24, ZONE_RU[zone], 24, "bold", color)
        if done:
            self._text(d, x + 20, y + 70, f"Готово · {v.zone_sec:.0f} с", 20, "bold", "ok")
        elif active:
            self._text(d, x + 20, y + 70, f"{v.zone_time[i]:.0f} с из {v.zone_sec:.0f} с", 20, "regular", "text")
            self._bar(d, x + 20, y + 112, w - 40, 10, v.zone_time[i] / v.zone_sec, "accent", "border")
        else:
            self._text(d, x + 20, y + 70, "Ожидает", 20, "regular", "muted")

    def _bar(self, d, x, y, w, h, fraction, fill, track):
        self._card(d, x, y, w, h, fill=track, r=h / 2, alpha=255)
        fw = max(0.0, min(1.0, fraction)) * w
        if fw >= h:  # слишком короткую заливку не рисуем: скруглённые края не поместятся
            self._card(d, x, y, fw, h, fill=fill, r=h / 2, alpha=255)

    # ---------- экран 4: итог ----------
    def _summary(self, d, v):
        self._chip(d, 56, 56, [("Итог чистки", 20, "regular", "muted")], dot="ok" if v.completed else "warn")
        name = "гость" if v.user in (None, "guest") else v.user
        self._text(d, 1224, 80, name, 22, "regular", "muted", anchor="rm")

        cx, cy = 640, 190
        self._dot(d, cx, cy, 60, "accent_dim" if v.completed else "surface2")
        pts = ([(-22, 2), (-6, 18), (24, -14)] if v.completed else [(-20, 0), (20, 0)])
        color = (*COL["accent"], 255) if v.completed else (*COL["warn"], 255)
        d.line([((cx + px) * self.s, (cy + py) * self.s) for px, py in pts], fill=color,
               width=max(3, round(8 * self.s)), joint="curve")

        total = sum(v.zone_time)
        title = "Чистка завершена" if v.completed else "Чистка прервана"
        who = "" if v.user in (None, "guest") else f", {v.user}"
        self._text(d, 640, 320, title, 64, "bold", anchor="mm")
        self._text(d, 640, 385, f"Время чистки: {format_clock(total)}{who}", 28, "regular", "muted", anchor="mm")

        for i, zone in enumerate(ZONE_ORDER):
            x = 56 + i * (277 + 20)
            full = v.zone_time[i] >= v.zone_sec - 1e-6
            self._card(d, x, 440, 277, 140, r=24)
            self._text(d, x + 138, 470, ZONE_RU[zone], 20, "regular", "muted", anchor="mm")
            self._text(d, x + 138, 515, f"{v.zone_time[i]:.0f} с", 44, "bold", anchor="mm")
            self._text(d, x + 138, 556, "Готово" if full else "Не завершено", 18, "bold",
                       "ok" if full else "muted", anchor="mm")

        txt = f"Иначе экран закроется через {max(0, round(v.summary_left))} с"
        chip_w = 2 * 20 + self._width(d, "Ладонь", 18, "bold") + self._width(d, "закрыть экран", 18) + 10
        total_w = chip_w + 16 + self._width(d, txt, 18)
        x0 = 640 - total_w / 2
        self._chip(d, x0, 626, [("Ладонь", 18, "bold", "accent"), ("закрыть экран", 18, "regular", "text")],
                   outline="border")
        self._text(d, x0 + chip_w + 16, 650, txt, 18, "regular", "muted", anchor="lm")

    # ---------- отладка ----------
    def _debug(self, d, v, frame_shape):
        fh, fw = frame_shape[:2]
        bx, by, bw, bh = self._crop_box(fw, fh)
        for px, py in v.points:
            # Точка в долях кадра камеры -> пиксель окна с учётом обрезки фона
            x = ((px * fw - bx) / bw) * self.W
            y = ((py * fh - by) / bh) * self.H
            if 0 <= x < self.W and 0 <= y < self.H:
                d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(*COL["accent"], 255))
        if v.debug_lines:
            h = 14 + 18 * len(v.debug_lines)
            d.rectangle([0, self.H - h * 1.0 - 4, 320 * self.s + 60, self.H], fill=(0, 0, 0, 170))
            for i, line in enumerate(v.debug_lines):
                self._text(d, 8 / self.s, (self.H - h + 6 + 18 * i) / self.s, line, 14 / self.s * self.s, "regular", "warn")
