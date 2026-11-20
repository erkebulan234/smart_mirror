"""Регистрация, список и удаление профилей (ФТ-4).

    python enroll.py add Айдар      # снять 8 векторов лица и сохранить профиль
    python enroll.py list           # показать профили
    python enroll.py delete Айдар   # удалить профиль вместе с векторами

Сохраняются только векторы лица (не фотографии). Кадры с камеры нигде не записываются.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from mirror.config import load_config, resolve_path  # noqa: E402
from mirror.cv.identity import FaceIdentifier  # noqa: E402
from mirror.storage.profiles import ProfileStore  # noqa: E402


def cmd_add(name, cfg, store):
    if name in store.list_profiles():
        raise SystemExit(f"Профиль '{name}' уже есть. Сначала удалите его: python enroll.py delete {name}")

    ident_cfg = cfg["identification"]
    needed = ident_cfg["vectors_per_profile"]
    identifier = FaceIdentifier(ident_cfg)

    cam = cfg["camera"]
    cap = cv2.VideoCapture(cam["index"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam["height"])
    if not cap.isOpened():
        raise SystemExit("Камера не открылась: проверьте camera.index в config.yaml")

    print(f"Регистрация '{name}': смотрите в камеру и медленно поворачивайте голову "
          f"влево-вправо и слегка вверх-вниз. Отмена: клавиша q.")
    vectors, last_capture = [], 0.0
    try:
        while len(vectors) < needed:
            ok, frame = cap.read()
            if not ok:
                raise SystemExit("Камера перестала отдавать кадры")
            if cam["mirror"]:
                frame = cv2.flip(frame, 1)

            now = time.perf_counter()
            if now - last_capture >= ident_cfg["enroll_interval_sec"]:
                emb = identifier.embed(frame)
                if emb is not None:
                    vectors.append(emb)
                    last_capture = now
                    print(f"  снято {len(vectors)}/{needed}")

            # Подсказка на экране латиницей: cv2.putText не умеет кириллицу
            cv2.putText(frame, f"Captured {len(vectors)}/{needed}  (q = cancel)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Enroll", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise SystemExit("Отменено, профиль не сохранён")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    store.add_profile(name, np.vstack(vectors))
    print(f"Профиль '{name}' сохранён ({len(vectors)} векторов).")


def main():
    args = sys.argv[1:]
    if not args or args[0] not in {"add", "list", "delete"}:
        raise SystemExit(__doc__)

    cfg = load_config()
    store = ProfileStore(resolve_path(cfg["storage"]["db_path"]))
    try:
        if args[0] == "list":
            names = store.list_profiles()
            print("Профили:", ", ".join(names) if names else "(пусто)")
        elif len(args) < 2:
            raise SystemExit("Укажите имя: python enroll.py " + args[0] + " <имя>")
        elif args[0] == "add":
            cmd_add(" ".join(args[1:]), cfg, store)
        else:
            name = " ".join(args[1:])
            print("Удалён." if store.delete_profile(name) else f"Профиль '{name}' не найден.")
    finally:
        store.close()


if __name__ == "__main__":
    main()
