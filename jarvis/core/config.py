from __future__ import annotations

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="JARVIS_",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    llm_model: str = "claude-sonnet-4-6"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1024
    llm_timeout: float = 120.0

    # STT
    stt_model: str = "base.en"
    stt_device: str = "auto"
    stt_compute_type: str = "int8"
    stt_language: str = "en"

    # TTS (Kokoro)
    tts_voice: str = "af_heart"
    tts_speed: float = 1.0

    # Audio
    input_device: str | None = None
    output_device: str | None = None
    sample_rate: int = 16000
    vad_aggressiveness: int = 2
    silence_ms: int = 700
    min_utterance_ms: int = 300
    max_utterance_ms: int = 30_000

    # Wake word
    wake_word: str = "hey jarvis"
    require_wake_word: bool = False
    wake_enabled: bool = True
    wake_model: str = "hey_jarvis_v0.1"
    wake_threshold: float = 0.5

    # Workspace + safety
    workspace: Path = Path(".")
    allow_writes: bool = False
    allow_shell: bool = False
    allow_send_email: bool = False
    stream_voice: bool = True

    # Google
    google_credentials: Path = Path("~/.jarvis/credentials.json")
    google_token: Path = Path("~/.jarvis/google_token.json")

    # Web search
    brave_api_key: str = ""

    # Memory
    mem0_path: Path = Path("~/.jarvis/mem0")
    mem0_enabled: bool = True

    # Telegram
    telegram_token: str | None = None
    telegram_allowed_chats: str = ""
    telegram_notify_chat: str | None = None

    # Proactive monitors
    calendar_poll_seconds: int = 300
    email_poll_seconds: int = 600
    watch_paths: str = "."
    watch_command: str | None = None
    watch_debounce_ms: int = 400

    @field_validator("google_credentials", "google_token", "mem0_path", "workspace", mode="before")
    @classmethod
    def expand_path(cls, v: object) -> object:
        if v:
            return Path(str(v)).expanduser()
        return v

    @property
    def allowed_chat_ids(self) -> set[int]:
        raw = [c.strip() for c in self.telegram_allowed_chats.split(",") if c.strip()]
        return {int(c) for c in raw}

    @property
    def notify_chat_id(self) -> int | None:
        return int(self.telegram_notify_chat) if self.telegram_notify_chat else None

    @property
    def watch_path_list(self) -> list[str]:
        return [p.strip() for p in self.watch_paths.split(",") if p.strip()]


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    global _settings
    _settings = Settings()
    return _settings
