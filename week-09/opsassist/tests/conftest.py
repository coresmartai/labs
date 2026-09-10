"""Test fixtures - sets required env vars so no real .env file is needed."""
import pytest
from app.config import get_settings


@pytest.fixture(autouse=True)
def patch_settings(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-a-real-credential")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
