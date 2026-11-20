"""Погода через Open-Meteo (ключ API не нужен). Запрос идёт в фоновом потоке, чтобы не тормозить видео."""
import json
import ssl
import sys
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass

# Коды погоды WMO, которые возвращает Open-Meteo
WEATHER_CODES = {
    0: "Ясно", 1: "Преимущественно ясно", 2: "Переменная облачность", 3: "Пасмурно",
    45: "Туман", 48: "Изморозь",
    51: "Лёгкая морось", 53: "Морось", 55: "Сильная морось",
    56: "Ледяная морось", 57: "Ледяная морось",
    61: "Небольшой дождь", 63: "Дождь", 65: "Сильный дождь",
    66: "Ледяной дождь", 67: "Ледяной дождь",
    71: "Небольшой снег", 73: "Снег", 75: "Сильный снег", 77: "Снежная крупа",
    80: "Ливень", 81: "Ливень", 82: "Сильный ливень",
    85: "Снегопад", 86: "Сильный снегопад",
    95: "Гроза", 96: "Гроза с градом", 99: "Гроза с градом",
}


@dataclass
class Weather:
    temperature: int
    description: str
    wind: float  # м/с

    def temp_text(self):
        """Температура с настоящим знаком минус, например '−3 °C'."""
        sign = "−" if self.temperature < 0 else ""
        return f"{sign}{abs(self.temperature)} °C"


def build_url(latitude, longitude):
    return (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}&longitude={longitude}"
        "&current=temperature_2m,weather_code,wind_speed_10m"
        "&wind_speed_unit=ms&timezone=auto"
    )


def parse_response(data):
    """Достаёт текущую погоду из ответа Open-Meteo."""
    cur = data["current"]
    return Weather(
        temperature=round(cur["temperature_2m"]),
        description=WEATHER_CODES.get(cur["weather_code"], "Погода"),
        wind=float(cur["wind_speed_10m"]),
    )


def _open_json(url, context):
    with urllib.request.urlopen(url, timeout=5, context=context) as resp:
        return json.load(resp)


def _certifi_context():
    """Контекст TLS с набором корневых сертификатов certifi (None, если пакета нет)."""
    try:
        import certifi
    except ImportError:
        return None
    return ssl.create_default_context(cafile=certifi.where())


def _http_get_json(url):
    """GET-запрос с JSON-ответом.

    Проверка сертификата всегда включена. Если системное хранилище сертификатов не смогло
    проверить цепочку (на некоторых ПК с Windows так бывает из-за устаревших корневых
    сертификатов), запрос повторяется с набором certifi.
    """
    try:
        return _open_json(url, None)
    except urllib.error.URLError as e:
        if isinstance(e.reason, ssl.SSLCertVerificationError):
            context = _certifi_context()
            if context is not None:
                return _open_json(url, context)
        raise


class WeatherService:
    """Хранит последнюю успешно полученную погоду; при ошибке сети показывает прежнее значение."""

    def __init__(self, cfg, fetch=None):
        self.url = build_url(cfg["latitude"], cfg["longitude"])
        self.refresh_sec = cfg["refresh_min"] * 60
        self._fetch = fetch or _http_get_json  # fetch подменяется в тестах
        self.current = None
        self.error = None            # текст последней ошибки или None, если последний запрос удался
        self._stop = threading.Event()

    def refresh_once(self):
        try:
            self.current = parse_response(self._fetch(self.url))
            self.error = None
            return True
        except Exception as e:
            # Нет сети или неожиданный ответ: прежнее значение остаётся, а причина выводится в консоль
            message = f"{type(e).__name__}: {e}"
            if message != self.error:  # одинаковую ошибку при каждом обновлении не повторяем
                print(f"Погода: не удалось получить данные ({message})", file=sys.stderr)
            self.error = message
            return False

    def start(self):
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while not self._stop.is_set():
            self.refresh_once()
            self._stop.wait(self.refresh_sec)

    def stop(self):
        self._stop.set()