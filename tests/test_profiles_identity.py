import numpy as np
import pytest

from mirror.cv.identity import GUEST, IdentificationSession, decide, match_embedding
from mirror.storage.profiles import EMBEDDING_DIM, ProfileStore


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def person(seed, n=8, noise=0.15):
    """Синтетический человек: базовый вектор + небольшой шум на каждый снимок."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=EMBEDDING_DIM)
    shots = np.vstack([unit(base + noise * rng.normal(size=EMBEDDING_DIM)) for _ in range(n)])
    return base, shots


def new_shot(base, seed, noise=0.15):
    rng = np.random.default_rng(seed)
    return unit(base + noise * rng.normal(size=EMBEDDING_DIM))


# ---------- Хранилище ----------

def test_add_list_load_delete(tmp_path):
    store = ProfileStore(tmp_path / "m.db")
    _, a = person(1)
    _, b = person(2)
    store.add_profile("Айдар", a)
    store.add_profile("Дана", b)
    assert store.list_profiles() == ["Айдар", "Дана"]

    loaded = store.load_all()
    assert set(loaded) == {"Айдар", "Дана"}
    assert loaded["Айдар"].shape == (8, EMBEDDING_DIM)

    assert store.delete_profile("Айдар") is True
    assert store.delete_profile("Айдар") is False
    assert store.list_profiles() == ["Дана"]
    assert set(store.load_all()) == {"Дана"}  # векторы удалены вместе с профилем


def test_profiles_survive_reopen(tmp_path):
    path = tmp_path / "m.db"
    store = ProfileStore(path)
    _, a = person(1)
    store.add_profile("Айдар", a)
    store.close()
    assert ProfileStore(path).list_profiles() == ["Айдар"]


def test_vectors_are_stored_normalized(tmp_path):
    store = ProfileStore(tmp_path / "m.db")
    rng = np.random.default_rng(0)
    store.add_profile("X", rng.normal(size=(6, EMBEDDING_DIM)) * 7)  # вектор с произвольной длиной
    norms = np.linalg.norm(store.load_all()["X"], axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_validation(tmp_path):
    store = ProfileStore(tmp_path / "m.db")
    _, a = person(1)
    with pytest.raises(ValueError):
        store.add_profile("", a)                      # пустое имя
    with pytest.raises(ValueError):
        store.add_profile("X", a[:3])                 # меньше 5 векторов
    with pytest.raises(ValueError):
        store.add_profile("X", np.vstack([a, a]))     # больше 10 векторов
    store.add_profile("X", a)
    with pytest.raises(ValueError):
        store.add_profile("X", a)                     # имя занято
    assert store.list_profiles() == ["X"]             # неудачные попытки ничего не оставили


# ---------- Сопоставление ----------

def test_known_person_is_recognized_and_stranger_is_guest():
    base_a, a = person(1)
    base_b, b = person(2)
    stranger_base, _ = person(99)
    profiles = {"Айдар": a, "Дана": b}
    thr = 0.40

    name, score = match_embedding(new_shot(base_a, 10), profiles, thr)
    assert name == "Айдар" and score >= thr
    name, _ = match_embedding(new_shot(base_b, 11), profiles, thr)
    assert name == "Дана"
    name, score = match_embedding(new_shot(stranger_base, 12), profiles, thr)
    assert name is None and score < thr


def test_error_rates_on_synthetic_people():
    """Грубая проверка метрик: 4 профиля, 200 проверок своих и 200 чужих."""
    people = [person(s) for s in (1, 2, 3, 4)]
    profiles = {f"p{i}": shots for i, (_, shots) in enumerate(people)}
    ok = sum(match_embedding(new_shot(people[i % 4][0], 1000 + i), profiles, 0.40)[0] == f"p{i % 4}"
             for i in range(200))
    strangers = [person(500 + s)[0] for s in range(200)]
    false_accepts = sum(match_embedding(new_shot(b, 2000 + i), profiles, 0.40)[0] is not None
                        for i, b in enumerate(strangers))
    assert ok / 200 >= 0.95
    assert false_accepts / 200 <= 0.05


# ---------- Голосование ----------

def test_decide_majority_and_guest():
    assert decide(["Айдар"] * 7 + [None] * 3) == "Айдар"
    assert decide(["Айдар"] * 4 + [None] * 6) == GUEST       # меньше половины
    assert decide(["Айдар"] * 5 + ["Дана"] * 5) == GUEST     # ничья
    assert decide([None] * 10) == GUEST
    assert decide([]) == GUEST


class FakeIdentifier:
    """Подставка вместо FaceIdentifier: отдаёт заранее заданные векторы (None = лица в кадре нет)."""

    def __init__(self, embeddings):
        self.queue = list(embeddings)

    def embed(self, frame):
        return self.queue.pop(0)


def test_session_decides_after_vote_window():
    base_a, a = person(1)
    shots = [new_shot(base_a, 50 + i) for i in range(10)]
    session = IdentificationSession(FakeIdentifier(shots), {"Айдар": a},
                                    {"cosine_threshold": 0.40, "vote_window": 10})
    results = [session.feed(None) for _ in range(10)]
    assert results[:-1] == [None] * 9
    assert results[-1] == "Айдар"


def test_session_skips_frames_without_face_and_guest_without_profiles():
    base_a, _ = person(1)
    embs = [None, new_shot(base_a, 1), None, new_shot(base_a, 2), new_shot(base_a, 3)]
    session = IdentificationSession(FakeIdentifier(embs), {}, {"cosine_threshold": 0.40, "vote_window": 3})
    assert [session.feed(None) for _ in range(5)][-1] == GUEST
    assert len(session.votes) == 3  # кадры без лица не считаются голосами
