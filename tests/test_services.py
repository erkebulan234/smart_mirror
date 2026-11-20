import pytest
import json
from datetime import date

from mirror.services.schedule import get_for_profile, load_schedule
from mirror.services.weather import Weather, WeatherService, build_url, parse_response

SAMPLE = {"current": {"time": "2026-11-18T08:00", "temperature_2m": -3.4, "weather_code": 3,
                      "wind_speed_10m": 4.2}}


def test_parse_response():
    w = parse_response(SAMPLE)
    assert w == Weather(temperature=-3, description="Пасмурно", wind=4.2)
    assert w.temp_text() == "−3 °C"
    assert Weather(5, "Ясно", 1.0).temp_text() == "5 °C"


def test_unknown_weather_code_has_fallback_text():
    data = {"current": {"temperature_2m": 1, "weather_code": 12345, "wind_speed_10m": 0}}
    assert parse_response(data).description == "Погода"


def test_url_contains_coordinates():
    url = build_url(51.17, 71.43)
    assert "latitude=51.17" in url and "longitude=71.43" in url and "open-meteo.com" in url


def test_service_keeps_last_value_when_network_fails():
    calls = {"n": 0}

    def fetch(url):
        calls["n"] += 1
        if calls["n"] == 1:
            return SAMPLE
        raise OSError("нет сети")

    svc = WeatherService({"latitude": 1, "longitude": 2, "refresh_min": 15}, fetch=fetch)
    assert svc.refresh_once() is True and svc.current.temperature == -3
    assert svc.error is None
    assert svc.refresh_once() is False
    assert svc.current.temperature == -3   # старое значение не потеряно
    assert "нет сети" in svc.error         # причина сохранена и выводится в консоль


def test_schedule_filters_by_weekday_and_sorts(tmp_path):
    path = tmp_path / "s.json"
    path.write_text(json.dumps({
        "Аня": {"events": [
            {"time": "14:00", "title": "B"},
            {"time": "09:00", "title": "A", "weekdays": [2]},       # только по средам
            {"time": "10:00", "title": "C", "weekdays": [0]},       # только по понедельникам
        ], "reminders": ["r1"]},
        "guest": {"reminders": ["hello"]},
    }, ensure_ascii=False), encoding="utf-8")
    data = load_schedule(path)

    wednesday = date(2026, 11, 18)
    events, reminders = get_for_profile(data, "Аня", wednesday)
    assert [e["title"] for e in events] == ["A", "B"]
    assert reminders == ["r1"]

    assert get_for_profile(data, "guest", wednesday) == ([], ["hello"])
    assert get_for_profile(data, "Неизвестный", wednesday) == ([], [])


def test_missing_or_broken_file_gives_empty_schedule(tmp_path):
    assert load_schedule(tmp_path / "нет.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{ не json", encoding="utf-8")
    assert load_schedule(bad) == {}


def test_sample_schedule_file_is_valid():
    from mirror.config import ROOT
    data = load_schedule(ROOT / "data" / "schedule.json")
    assert "guest" in data and len(data) >= 2


# ---------- Проверка сертификатов: повтор с certifi ----------

class _Resp:
    """Минимальная подставка под ответ urlopen."""
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, *args):
        return json.dumps(SAMPLE).encode()


def test_cert_error_retries_with_certifi_context(monkeypatch):
    import ssl
    import urllib.error

    from mirror.services import weather

    contexts = []

    def fake_urlopen(url, timeout, context=None):
        contexts.append(context)
        if context is None:  # системное хранилище: проверка не прошла
            raise urllib.error.URLError(ssl.SSLCertVerificationError("certificate has expired"))
        return _Resp()

    monkeypatch.setattr(weather.urllib.request, "urlopen", fake_urlopen)
    assert weather._http_get_json("https://example.org") == SAMPLE
    assert contexts[0] is None and isinstance(contexts[1], ssl.SSLContext)
    assert contexts[1].verify_mode == ssl.CERT_REQUIRED   # проверка сертификата не отключена


def test_other_network_errors_are_not_retried(monkeypatch):
    import urllib.error

    from mirror.services import weather

    calls = []

    def fake_urlopen(url, timeout, context=None):
        calls.append(context)
        raise urllib.error.URLError("нет сети")

    monkeypatch.setattr(weather.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(urllib.error.URLError):
        weather._http_get_json("https://example.org")
    assert len(calls) == 1