import os
import pytest
from jarvis.core.config import Settings, get_settings, reload_settings


def test_defaults(monkeypatch):
    # Clear env vars to test actual defaults (not .env file values)
    monkeypatch.delenv("JARVIS_LLM_MODEL", raising=False)
    monkeypatch.delenv("JARVIS_MEM0_ENABLED", raising=False)
    monkeypatch.delenv("JARVIS_CALENDAR_POLL_SECONDS", raising=False)
    monkeypatch.delenv("JARVIS_EMAIL_POLL_SECONDS", raising=False)
    monkeypatch.delenv("JARVIS_ALLOW_SEND_EMAIL", raising=False)
    # Create a Settings instance that reads from defaults, not .env
    s = Settings(_env_file=None)
    assert s.llm_model == "claude-sonnet-4-6"
    assert s.mem0_enabled is True
    assert s.calendar_poll_seconds == 300
    assert s.email_poll_seconds == 600
    assert s.allow_send_email is False


def test_env_override(monkeypatch):
    monkeypatch.setenv("JARVIS_LLM_MODEL", "claude-opus-4-7")
    monkeypatch.setenv("JARVIS_CALENDAR_POLL_SECONDS", "60")
    s = Settings(_env_file=None)
    assert s.llm_model == "claude-opus-4-7"
    assert s.calendar_poll_seconds == 60


def test_reload_settings(monkeypatch):
    monkeypatch.setenv("JARVIS_LLM_MODEL", "claude-haiku-4-5-20251001")
    s = reload_settings()
    assert s.llm_model == "claude-haiku-4-5-20251001"
    reload_settings()


def test_allowed_chat_ids():
    s = Settings(telegram_allowed_chats="123,456,789")
    assert s.allowed_chat_ids == {123, 456, 789}


def test_google_paths_expand(tmp_path):
    s = Settings(
        google_credentials=str(tmp_path / "creds.json"),
        google_token=str(tmp_path / "token.json"),
    )
    assert "creds.json" in str(s.google_credentials)
