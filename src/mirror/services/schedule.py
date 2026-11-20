"""Расписание и напоминания из JSON-файла (ФТ-5).

Формат data/schedule.json: ключ верхнего уровня = имя профиля (как при регистрации) или "guest".
    {
      "Имя": {
        "events": [{"time": "09:00", "title": "Занятие", "place": "Аудитория 305", "weekdays": [0, 1, 2]}],
        "reminders": ["Выпить воды"]
      },
      "guest": {"events": [], "reminders": []}
    }
"weekdays" необязателен: 0 = понедельник ... 6 = воскресенье; без него событие показывается каждый день.
"""
import json
from datetime import date


def load_schedule(path):
    """Читает файл расписания; если файла нет или он повреждён, возвращает пустое расписание."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def get_for_profile(data, name, today=None):
    """Возвращает (события на сегодня по времени, напоминания) для профиля."""
    entry = data.get(name, {})
    weekday = (today or date.today()).weekday()
    events = [e for e in entry.get("events", []) if "weekdays" not in e or weekday in e["weekdays"]]
    events.sort(key=lambda e: e.get("time", ""))
    return events, list(entry.get("reminders", []))
