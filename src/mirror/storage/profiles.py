"""Хранилище профилей (ФТ-4).

В SQLite лежат только векторы лиц (по 5-10 штук на человека). Фотографии и кадры
не сохраняются нигде: по вектору восстановить изображение лица нельзя.
"""
import sqlite3
import time
from pathlib import Path

import numpy as np

EMBEDDING_DIM = 128  # длина вектора модели SFace


def _normalize(vectors):
    """Приводит каждый вектор к единичной длине: тогда сходство лиц равно скалярному произведению."""
    v = np.asarray(vectors, dtype=np.float32).reshape(-1, EMBEDDING_DIM)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(norms, 1e-9)


class ProfileStore:
    def __init__(self, db_path, min_vectors=5, max_vectors=10):
        self.min_vectors = min_vectors
        self.max_vectors = max_vectors
        if str(db_path) != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS profiles (
                id         INTEGER PRIMARY KEY,
                name       TEXT NOT NULL UNIQUE,
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS embeddings (
                id         INTEGER PRIMARY KEY,
                profile_id INTEGER NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
                vector     BLOB NOT NULL
            );
            """
        )
        self._conn.commit()

    def add_profile(self, name, vectors):
        """Создаёт профиль. Бросает ValueError при пустом/занятом имени или неверном числе векторов."""
        name = name.strip()
        if not name:
            raise ValueError("Имя профиля не может быть пустым")
        vecs = _normalize(vectors)
        if not (self.min_vectors <= len(vecs) <= self.max_vectors):
            raise ValueError(
                f"Нужно от {self.min_vectors} до {self.max_vectors} векторов, получено {len(vecs)}"
            )
        try:
            with self._conn:  # транзакция: профиль и его векторы сохраняются вместе или не сохраняются
                cur = self._conn.execute(
                    "INSERT INTO profiles (name, created_at) VALUES (?, ?)", (name, time.time())
                )
                self._conn.executemany(
                    "INSERT INTO embeddings (profile_id, vector) VALUES (?, ?)",
                    [(cur.lastrowid, v.tobytes()) for v in vecs],
                )
        except sqlite3.IntegrityError:
            raise ValueError(f"Профиль '{name}' уже существует")

    def delete_profile(self, name):
        """Удаляет профиль вместе с векторами; возвращает True, если он существовал."""
        with self._conn:
            cur = self._conn.execute("DELETE FROM profiles WHERE name = ?", (name.strip(),))
        return cur.rowcount > 0

    def list_profiles(self):
        rows = self._conn.execute("SELECT name FROM profiles ORDER BY name").fetchall()
        return [r[0] for r in rows]

    def load_all(self):
        """Возвращает {имя: массив векторов формы (n, 128)} для сопоставления лиц."""
        rows = self._conn.execute(
            "SELECT p.name, e.vector FROM embeddings e JOIN profiles p ON p.id = e.profile_id"
        ).fetchall()
        result = {}
        for name, blob in rows:
            result.setdefault(name, []).append(np.frombuffer(blob, dtype=np.float32))
        return {name: np.vstack(vs) for name, vs in result.items()}

    def close(self):
        self._conn.close()
