"""Запуск демо из корня проекта на любой ОС и в любой оболочке.

Пакет mirror лежит в src/, поэтому добавляем эту папку в путь поиска модулей:
так не нужно задавать PYTHONPATH вручную (в PowerShell и cmd это делается по-разному).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from mirror.cv.landmarks_demo import main  # noqa: E402

if __name__ == "__main__":
    main()
