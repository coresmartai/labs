import pytest


@pytest.fixture(autouse=True)
def _settings(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-a-real-credential")
    from app.config import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
