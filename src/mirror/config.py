"""Загрузка настроек из config.yaml.

Все пороги и тайм-ауты хранятся в одном файле (требование 6.6 ТЗ),
поэтому остальные модули читают значения только через этот загрузчик.
"""
from pathlib import Path

import yaml

# Корень проекта: src/mirror/config.py -> на три уровня выше
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "config.yaml"


def load_config(path=None):
    """Читает YAML-файл и возвращает вложенный словарь настроек."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
