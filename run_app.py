"""Запуск приложения «Умное зеркало» из корня проекта (папка src добавляется в путь поиска модулей)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from mirror.app import main  # noqa: E402

if __name__ == "__main__":
    main()
