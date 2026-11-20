"""Общие фикстуры для тестов."""
import pytest

from mirror.config import load_config


@pytest.fixture(scope="session")
def cfg():
    """Настройки из реального config.yaml: тесты проверяют те же пороги, что и приложение."""
    return load_config()
