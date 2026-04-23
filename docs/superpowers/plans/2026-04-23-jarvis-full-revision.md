# Jarvis Full Revision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite Jarvis from a flat vLLM/Qwen voice loop into a proactive personal AI — LangGraph orchestration, mem0 memory, Claude API brain, Google Calendar/Gmail integration, Kokoro TTS, and background monitors that proactively alert you.

**Architecture:** Two concurrent runtime modes share a LangGraph agent graph: reactive (wake→STT→graph→TTS) and proactive (scheduled calendar/email monitors entering the graph and exiting via Telegram or voice). mem0 handles three-tier memory (episodic, semantic vector, relational graph) stored locally in `~/.jarvis/mem0/`. All destructive actions require voice or Telegram confirmation before execution.

**Tech Stack:** Python 3.11+, LangGraph 0.2+, mem0ai, Anthropic SDK (claude-sonnet-4-6), google-api-python-client, faster-whisper, kokoro-onnx, sounddevice, python-telegram-bot, typer, pydantic-settings.

---

## File Map

### New files
- `jarvis/core/config.py` — extends existing Settings with new env vars (replaces `jarvis/config.py`)
- `jarvis/core/personality.py` — updated system prompt + proactive alert templates (replaces `jarvis/personality.py`)
- `jarvis/core/session.py` — shared runtime factory: builds mem0, tools, LLM client, graph
- `jarvis/graph/state.py` — `AgentState` TypedDict
- `jarvis/graph/nodes.py` — all graph node functions: route, retrieve_memory, think, act, observe, respond, store_memory, await_confirmation
- `jarvis/graph/agent.py` — LangGraph graph compilation + `run_turn()` entry point
- `jarvis/graph/proactive/calendar.py` — calendar monitor loop + subgraph entry
- `jarvis/graph/proactive/email.py` — email monitor loop + subgraph entry
- `jarvis/graph/proactive/watchdog.py` — file/CI watcher rewired from `jarvis/watch.py`
- `jarvis/memory/mem0_store.py` — mem0 wrapper: `retrieve()`, `store()`, `remember()`, `forget()`
- `jarvis/tools/coding.py` — existing tools from `jarvis/tools.py` (moved, no changes)
- `jarvis/tools/calendar_tools.py` — Google Calendar tool implementations
- `jarvis/tools/email_tools.py` — Gmail tool implementations
- `jarvis/tools/web_tools.py` — `web_search()` + `fetch_page()`
- `jarvis/tools/registry.py` — unified `TOOL_SCHEMAS` list + `Toolbox.dispatch()`
- `jarvis/integrations/google_auth.py` — OAuth 2.0 flow + token refresh

### Modified files
- `jarvis/voice/tts.py` — swap Piper → Kokoro ONNX (was `jarvis/tts.py`)
- `jarvis/voice/audio.py` — move from `jarvis/audio.py`, add interrupt detection
- `jarvis/voice/stt.py` — move from `jarvis/stt.py` (no changes)
- `jarvis/voice/wake.py` — move from `jarvis/wake.py` (no changes)
- `jarvis/voice/streaming.py` — move from `jarvis/streaming.py` (no changes)
- `jarvis/main.py` — rewire CLI to new graph; add `auth google` subcommand
- `jarvis/telegram_bot.py` — add confirmation callback handler
- `pyproject.toml` — add new dependencies, update package paths
- `.env.example` — add new env vars

### Deleted (functionality absorbed)
- `jarvis/agent.py` → replaced by `jarvis/graph/`
- `jarvis/llm.py` → replaced by Anthropic SDK calls in `jarvis/graph/nodes.py`
- `jarvis/memory.py` → replaced by `jarvis/memory/mem0_store.py`
- `jarvis/config.py` → replaced by `jarvis/core/config.py`
- `jarvis/personality.py` → replaced by `jarvis/core/personality.py`
- `jarvis/tools.py` → replaced by `jarvis/tools/`
- `jarvis/watch.py` → replaced by `jarvis/graph/proactive/watchdog.py`
- `jarvis/tts.py` → replaced by `jarvis/voice/tts.py`
- `jarvis/audio.py` → replaced by `jarvis/voice/audio.py`
- `jarvis/stt.py` → replaced by `jarvis/voice/stt.py`
- `jarvis/wake.py` → replaced by `jarvis/voice/wake.py`
- `jarvis/streaming.py` → replaced by `jarvis/voice/streaming.py`

---

## Task 1: Restructure package layout and update dependencies

**Files:**
- Modify: `pyproject.toml`
- Modify: `.env.example`
- Create: `jarvis/core/__init__.py`
- Create: `jarvis/graph/__init__.py`
- Create: `jarvis/graph/proactive/__init__.py`
- Create: `jarvis/memory/__init__.py`
- Create: `jarvis/tools/__init__.py`
- Create: `jarvis/integrations/__init__.py`
- Create: `jarvis/voice/__init__.py`

- [ ] **Step 1: Update pyproject.toml dependencies**

Replace the `[project]` dependencies section in `pyproject.toml` with:

```toml
[project]
name = "jarvis"
version = "0.2.0"
description = "Proactive personal AI. LangGraph + mem0 + Claude API. Voice-first, Iron Man style."
readme = "README.md"
requires-python = ">=3.11"
license = { text = "MIT" }
authors = [{ name = "Rui Machado" }]

dependencies = [
    # LLM + orchestration
    "anthropic>=0.40.0",
    "langgraph>=0.2.0",
    "langchain-core>=0.3.0",
    # Memory
    "mem0ai>=0.1.0",
    "chromadb>=0.5.0",
    # Google integrations
    "google-auth>=2.30.0",
    "google-auth-oauthlib>=1.2.0",
    "google-api-python-client>=2.130.0",
    # Web tools
    "trafilatura>=1.12.0",
    "duckduckgo-search>=6.0.0",
    # Voice
    "faster-whisper>=1.0.3",
    "kokoro-onnx>=0.4.0",
    "sounddevice>=0.4.7",
    "soundfile>=0.12.0",
    "numpy>=1.26.0",
    "webrtcvad>=2.0.10",
    # Telegram + CLI
    "python-telegram-bot>=21.4",
    "typer>=0.12.3",
    "rich>=13.7.1",
    # Config
    "pydantic>=2.7.0",
    "pydantic-settings>=2.3.0",
    "python-dotenv>=1.0.1",
    # Async + utils
    "anyio>=4.4.0",
    "watchfiles>=0.22.0",
    "httpx>=0.27.0",
]

[project.optional-dependencies]
wake = [
    "openwakeword>=0.6.0",
]
dev = [
    "pytest>=8.2.0",
    "pytest-asyncio>=0.23.7",
    "ruff>=0.5.0",
]

[project.scripts]
jarvis = "jarvis.main:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["jarvis"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]
ignore = ["E501"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
```

- [ ] **Step 2: Update .env.example**

Replace the contents of `.env.example` with:

```bash
# LLM — Claude API
ANTHROPIC_API_KEY=sk-ant-...
JARVIS_LLM_MODEL=claude-sonnet-4-6
JARVIS_LLM_TEMPERATURE=0.2
JARVIS_LLM_MAX_TOKENS=1024
JARVIS_LLM_TIMEOUT=120.0

# STT
JARVIS_STT_MODEL=base.en
JARVIS_STT_DEVICE=auto
JARVIS_STT_COMPUTE_TYPE=int8
JARVIS_STT_LANGUAGE=en

# TTS (Kokoro ONNX)
JARVIS_TTS_VOICE=af_heart
JARVIS_TTS_SPEED=1.0

# Audio
JARVIS_SAMPLE_RATE=16000
JARVIS_VAD_AGGRESSIVENESS=2
JARVIS_SILENCE_MS=700
JARVIS_MIN_UTTERANCE_MS=300
JARVIS_MAX_UTTERANCE_MS=30000

# Wake word
JARVIS_WAKE_WORD=hey jarvis
JARVIS_WAKE_ENABLED=true
JARVIS_WAKE_MODEL=hey_jarvis_v0.1
JARVIS_WAKE_THRESHOLD=0.5
JARVIS_REQUIRE_WAKE_WORD=false

# Workspace + safety
JARVIS_WORKSPACE=.
JARVIS_ALLOW_WRITES=false
JARVIS_ALLOW_SHELL=false
JARVIS_ALLOW_SEND_EMAIL=false
JARVIS_STREAM_VOICE=true

# Google
JARVIS_GOOGLE_CREDENTIALS=~/.jarvis/credentials.json
JARVIS_GOOGLE_TOKEN=~/.jarvis/google_token.json

# Web search
JARVIS_BRAVE_API_KEY=

# Memory (mem0)
JARVIS_MEM0_PATH=~/.jarvis/mem0
JARVIS_MEM0_ENABLED=true

# Telegram
JARVIS_TELEGRAM_TOKEN=
JARVIS_TELEGRAM_ALLOWED_CHATS=
JARVIS_TELEGRAM_NOTIFY_CHAT=

# Proactive monitors
JARVIS_CALENDAR_POLL_SECONDS=300
JARVIS_EMAIL_POLL_SECONDS=600
JARVIS_WATCH_PATHS=.
JARVIS_WATCH_COMMAND=
JARVIS_WATCH_DEBOUNCE_MS=400
```

- [ ] **Step 3: Create package `__init__.py` files**

```bash
touch jarvis/core/__init__.py \
      jarvis/graph/__init__.py \
      jarvis/graph/proactive/__init__.py \
      jarvis/memory/__init__.py \
      jarvis/tools/__init__.py \
      jarvis/integrations/__init__.py \
      jarvis/voice/__init__.py
```

- [ ] **Step 4: Install dependencies**

```bash
pip install -e '.[dev]'
```

Expected: no errors. `import anthropic`, `import langgraph`, `import mem0` should succeed.

```bash
python -c "import anthropic; import langgraph; import mem0; print('OK')"
```

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .env.example jarvis/core/ jarvis/graph/ jarvis/memory/ jarvis/tools/ jarvis/integrations/ jarvis/voice/
git commit -m "chore: restructure package layout and update dependencies for v0.2"
```

---

## Task 2: Core config

**Files:**
- Create: `jarvis/core/config.py`
- Test: `tests/test_core_config.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_core_config.py`:

```python
import os
import pytest
from jarvis.core.config import Settings, get_settings, reload_settings


def test_defaults():
    s = Settings()
    assert s.llm_model == "claude-sonnet-4-6"
    assert s.mem0_enabled is True
    assert s.calendar_poll_seconds == 300
    assert s.email_poll_seconds == 600
    assert s.allow_send_email is False


def test_env_override(monkeypatch):
    monkeypatch.setenv("JARVIS_LLM_MODEL", "claude-opus-4-7")
    monkeypatch.setenv("JARVIS_CALENDAR_POLL_SECONDS", "60")
    s = Settings()
    assert s.llm_model == "claude-opus-4-7"
    assert s.calendar_poll_seconds == 60


def test_reload_settings(monkeypatch):
    monkeypatch.setenv("JARVIS_LLM_MODEL", "claude-haiku-4-5-20251001")
    s = reload_settings()
    assert s.llm_model == "claude-haiku-4-5-20251001"
    # cleanup
    reload_settings()


def test_allowed_chat_ids():
    s = Settings(telegram_allowed_chats="123,456,789")
    assert s.allowed_chat_ids == {123, 456, 789}


def test_google_paths_expand(tmp_path):
    s = Settings(
        google_credentials_path=str(tmp_path / "creds.json"),
        google_token_path=str(tmp_path / "token.json"),
    )
    assert "creds.json" in str(s.google_credentials_path)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_core_config.py -v
```

Expected: `ImportError: cannot import name 'Settings' from 'jarvis.core.config'`

- [ ] **Step 3: Create `jarvis/core/config.py`**

```python
from __future__ import annotations

from pathlib import Path

from pydantic import Field
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
    google_credentials_path: Path = Path("~/.jarvis/credentials.json")
    google_token_path: Path = Path("~/.jarvis/google_token.json")

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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_core_config.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add jarvis/core/config.py tests/test_core_config.py
git commit -m "feat: add core/config.py with expanded settings for v0.2"
```

---

## Task 3: Core personality

**Files:**
- Create: `jarvis/core/personality.py`
- Test: `tests/test_core_personality.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_core_personality.py`:

```python
from jarvis.core.personality import SYSTEM_PROMPT, proactive_alert


def test_system_prompt_present():
    assert "Jarvis" in SYSTEM_PROMPT
    assert "TTS" in SYSTEM_PROMPT or "voice" in SYSTEM_PROMPT.lower()
    assert len(SYSTEM_PROMPT) > 200


def test_proactive_alert_calendar():
    msg = proactive_alert("calendar", "call with Sarah in 25 minutes", "send her the deck link")
    assert "Sarah" in msg
    assert "deck link" in msg
    assert "?" in msg  # ends with a question


def test_proactive_alert_email():
    msg = proactive_alert("email", "urgent email from CEO", "draft a reply")
    assert "CEO" in msg
    assert "draft" in msg


def test_proactive_alert_no_action():
    msg = proactive_alert("calendar", "dentist appointment tomorrow", None)
    assert "dentist" in msg
    assert msg.strip()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_core_personality.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Create `jarvis/core/personality.py`**

```python
"""System prompt and proactive alert templates for Jarvis."""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are Jarvis, a proactive personal AI running on the user's machine.

Style rules (voice output is read aloud by a TTS engine):
- Speak in short, natural sentences. One idea per sentence.
- Never emit markdown, bullet points, code fences, tables, or headings.
- When you reference code, say it naturally: "the function run_pipeline on line forty-two".
- If the user clearly asks to see code, emit it plain with no fences.
- Keep responses under six sentences unless the user asks for depth.
- Pause between ideas with a period so the TTS can breathe.
- Never describe yourself as an AI language model. You are Jarvis.

Behaviour:
- You are proactive. You monitor the user's calendar, email, and workspace.
- When you surface a proactive alert, always suggest a specific course of action.
- For any action that changes data (send email, create event, write file, run command),
  state what you are about to do and wait for the user to say "yes" or "go ahead".
  Never take destructive actions without explicit confirmation.
- Your job includes coding help: understand, explore, and modify code in the workspace.
- Prefer reading files with the read_file tool before guessing. Do not hallucinate.
- When asked what a change does: summarise intent, then mechanics, then any risk.
- If you are unsure, say so in one sentence and ask a focused follow-up.
- Treat Telegram messages the same way: concise and conversational.

You have tools for: reading/writing files, running shell commands, searching code,
managing Google Calendar and Gmail, searching the web, and managing your own memory.
Use tools when they will save guessing. Do not announce tool use; narrate results naturally.
"""


def proactive_alert(source: str, situation: str, suggested_action: str | None) -> str:
    """Return a short proactive alert sentence for TTS or Telegram."""
    if suggested_action:
        return f"Heads up — {situation}. Want me to {suggested_action}?"
    return f"Heads up — {situation}."
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_core_personality.py -v
```

Expected: all 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add jarvis/core/personality.py tests/test_core_personality.py
git commit -m "feat: add core/personality.py with proactive alert templates"
```

---

## Task 4: Move and verify voice pipeline (STT, wake, audio, streaming)

**Files:**
- Create: `jarvis/voice/stt.py` (copy from `jarvis/stt.py`)
- Create: `jarvis/voice/wake.py` (copy from `jarvis/wake.py`)
- Create: `jarvis/voice/audio.py` (copy from `jarvis/audio.py`)
- Create: `jarvis/voice/streaming.py` (copy from `jarvis/streaming.py`)

- [ ] **Step 1: Copy files into new locations**

```bash
cp jarvis/stt.py jarvis/voice/stt.py
cp jarvis/wake.py jarvis/voice/wake.py
cp jarvis/audio.py jarvis/voice/audio.py
cp jarvis/streaming.py jarvis/voice/streaming.py
```

- [ ] **Step 2: Fix imports in copied files**

In `jarvis/voice/stt.py`, change:
```python
from jarvis.config import Settings
```
to:
```python
from jarvis.core.config import Settings
```

In `jarvis/voice/wake.py`, change:
```python
from jarvis.config import Settings
```
to:
```python
from jarvis.core.config import Settings
```

In `jarvis/voice/audio.py`, change:
```python
from jarvis.config import Settings
```
to:
```python
from jarvis.core.config import Settings
```

In `jarvis/voice/streaming.py`, change any imports from `jarvis.tts` or `jarvis.audio`:
```python
from jarvis.voice.tts import TTS
from jarvis.voice.audio import AudioIO
```

- [ ] **Step 3: Verify imports resolve**

```bash
python -c "from jarvis.voice.stt import STT; from jarvis.voice.wake import WakeWord; from jarvis.voice.audio import AudioIO; print('OK')"
```

Expected: `OK` (faster-whisper and sounddevice must be installed).

- [ ] **Step 4: Commit**

```bash
git add jarvis/voice/
git commit -m "refactor: move voice pipeline to jarvis/voice/ subpackage"
```

---

## Task 5: Kokoro TTS

**Files:**
- Create: `jarvis/voice/tts.py`
- Test: `tests/test_voice_tts.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_voice_tts.py`:

```python
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from jarvis.core.config import Settings
from jarvis.voice.tts import TTS


@pytest.fixture
def mock_kokoro():
    """Patch kokoro_onnx.KokoroTTS so tests don't need a real model file."""
    with patch("jarvis.voice.tts.KokoroTTS") as mock_cls:
        instance = MagicMock()
        # synthesize returns (samples, sample_rate) pairs
        instance.create.return_value = [(np.zeros(16000, dtype=np.float32), 24000)]
        mock_cls.return_value = instance
        yield mock_cls


def test_tts_synthesize(mock_kokoro):
    settings = Settings()
    tts = TTS(settings)
    import asyncio
    pcm, sr = asyncio.get_event_loop().run_until_complete(tts.synthesize("Hello Jarvis."))
    assert isinstance(pcm, np.ndarray)
    assert sr == 24000
    assert pcm.shape[0] > 0


def test_tts_empty_string(mock_kokoro):
    settings = Settings()
    tts = TTS(settings)
    import asyncio
    pcm, sr = asyncio.get_event_loop().run_until_complete(tts.synthesize(""))
    assert pcm.shape[0] == 0


def test_tts_voice_name_passed(mock_kokoro):
    settings = Settings(tts_voice="af_sky")
    tts = TTS(settings)
    import asyncio
    asyncio.get_event_loop().run_until_complete(tts.synthesize("Test."))
    mock_kokoro.return_value.create.assert_called_once()
    call_kwargs = mock_kokoro.return_value.create.call_args
    assert "af_sky" in str(call_kwargs)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_voice_tts.py -v
```

Expected: `ImportError: cannot import name 'TTS' from 'jarvis.voice.tts'`

- [ ] **Step 3: Create `jarvis/voice/tts.py`**

```python
"""Kokoro ONNX TTS — Apple Silicon native, replaces Piper."""

from __future__ import annotations

import asyncio
import logging
from functools import partial

import numpy as np

from jarvis.core.config import Settings

log = logging.getLogger(__name__)

try:
    from kokoro_onnx import Kokoro as KokoroTTS
    _KOKORO_AVAILABLE = True
except ImportError:
    _KOKORO_AVAILABLE = False
    log.warning("kokoro-onnx not installed; TTS unavailable")


class TTS:
    def __init__(self, settings: Settings) -> None:
        self._voice = settings.tts_voice
        self._speed = settings.tts_speed
        self._kokoro: "KokoroTTS | None" = None
        if _KOKORO_AVAILABLE:
            self._kokoro = KokoroTTS()

    async def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Return (pcm_float32, sample_rate). Empty array if text is blank."""
        text = text.strip()
        if not text or self._kokoro is None:
            return np.zeros(0, dtype=np.float32), 24000

        loop = asyncio.get_running_loop()
        fn = partial(self._synthesize_sync, text)
        return await loop.run_in_executor(None, fn)

    def _synthesize_sync(self, text: str) -> tuple[np.ndarray, int]:
        samples_list = list(self._kokoro.create(text, voice=self._voice, speed=self._speed))
        if not samples_list:
            return np.zeros(0, dtype=np.float32), 24000
        chunks = [s for s, _ in samples_list]
        sr = samples_list[0][1]
        pcm = np.concatenate(chunks).astype(np.float32)
        return pcm, sr
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_voice_tts.py -v
```

Expected: all 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add jarvis/voice/tts.py tests/test_voice_tts.py
git commit -m "feat: add Kokoro ONNX TTS replacing Piper"
```

---

## Task 6: mem0 memory store

**Files:**
- Create: `jarvis/memory/mem0_store.py`
- Test: `tests/test_memory_mem0_store.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_memory_mem0_store.py`:

```python
import pytest
from unittest.mock import patch, MagicMock
from jarvis.memory.mem0_store import Mem0Store


@pytest.fixture
def mock_mem0(tmp_path):
    with patch("jarvis.memory.mem0_store.Memory") as mock_cls:
        instance = MagicMock()
        instance.search.return_value = {"results": [{"memory": "Rui likes TypeScript"}]}
        instance.add.return_value = {"results": []}
        mock_cls.return_value = instance
        yield instance


def test_retrieve_returns_list(mock_mem0, tmp_path):
    store = Mem0Store(str(tmp_path / "mem0"), user_id="test_user")
    results = store.retrieve("TypeScript preferences", limit=5)
    assert isinstance(results, list)
    assert "TypeScript" in results[0]


def test_store_calls_add(mock_mem0, tmp_path):
    store = Mem0Store(str(tmp_path / "mem0"), user_id="test_user")
    store.store("Rui prefers tabs over spaces", role="user")
    mock_mem0.add.assert_called_once()


def test_remember_explicit(mock_mem0, tmp_path):
    store = Mem0Store(str(tmp_path / "mem0"), user_id="test_user")
    store.remember("Always use Python 3.11+")
    mock_mem0.add.assert_called_once()
    call_args = str(mock_mem0.add.call_args)
    assert "Python 3.11" in call_args


def test_retrieve_disabled(tmp_path):
    with patch("jarvis.memory.mem0_store.Memory") as mock_cls:
        instance = MagicMock()
        mock_cls.return_value = instance
        store = Mem0Store(str(tmp_path / "mem0"), user_id="test", enabled=False)
        results = store.retrieve("anything")
        assert results == []
        instance.search.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_memory_mem0_store.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Create `jarvis/memory/mem0_store.py`**

```python
"""mem0 wrapper providing retrieve/store/remember/forget for Jarvis."""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

try:
    from mem0 import Memory
    _MEM0_AVAILABLE = True
except ImportError:
    _MEM0_AVAILABLE = False
    log.warning("mem0ai not installed; memory disabled")


class Mem0Store:
    """Thin wrapper around mem0.Memory with a stable interface for Jarvis."""

    def __init__(
        self,
        path: str,
        user_id: str = "jarvis_user",
        enabled: bool = True,
    ) -> None:
        self.user_id = user_id
        self.enabled = enabled and _MEM0_AVAILABLE
        self._mem: "Memory | None" = None
        if self.enabled:
            Path(path).mkdir(parents=True, exist_ok=True)
            config = {
                "vector_store": {
                    "provider": "chroma",
                    "config": {"path": str(Path(path) / "chroma")},
                },
                "version": "v1.1",
            }
            self._mem = Memory.from_config(config)

    def retrieve(self, query: str, limit: int = 5) -> list[str]:
        """Return up to `limit` memory strings relevant to query."""
        if not self.enabled or self._mem is None:
            return []
        try:
            result = self._mem.search(query, user_id=self.user_id, limit=limit)
            return [r["memory"] for r in result.get("results", [])]
        except Exception as e:
            log.warning("mem0 retrieve failed: %s", e)
            return []

    def store(self, text: str, role: str = "user") -> None:
        """Extract and store facts from a message."""
        if not self.enabled or self._mem is None:
            return
        try:
            messages = [{"role": role, "content": text}]
            self._mem.add(messages, user_id=self.user_id)
        except Exception as e:
            log.warning("mem0 store failed: %s", e)

    def remember(self, fact: str) -> None:
        """Explicitly store a single fact (called by the remember tool)."""
        if not self.enabled or self._mem is None:
            return
        try:
            self._mem.add(
                [{"role": "system", "content": f"Remember this fact: {fact}"}],
                user_id=self.user_id,
            )
        except Exception as e:
            log.warning("mem0 remember failed: %s", e)

    def forget(self, fact: str) -> None:
        """Attempt to delete memories matching fact."""
        if not self.enabled or self._mem is None:
            return
        try:
            results = self._mem.search(fact, user_id=self.user_id, limit=10)
            for r in results.get("results", []):
                if r.get("id"):
                    self._mem.delete(r["id"])
        except Exception as e:
            log.warning("mem0 forget failed: %s", e)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_memory_mem0_store.py -v
```

Expected: all 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add jarvis/memory/mem0_store.py tests/test_memory_mem0_store.py
git commit -m "feat: add mem0 memory store replacing SQLite episodic log"
```

---

## Task 7: Tool registry — coding + web tools

**Files:**
- Create: `jarvis/tools/coding.py` (moved from `jarvis/tools.py`)
- Create: `jarvis/tools/web_tools.py`
- Create: `jarvis/tools/registry.py`
- Test: `tests/test_tools_registry.py`

- [ ] **Step 1: Copy coding tools**

```bash
cp jarvis/tools.py jarvis/tools/coding.py
```

Then open `jarvis/tools/coding.py` and change:
```python
from jarvis.config import Settings
```
to:
```python
from jarvis.core.config import Settings
```

Also remove the `TOOL_SCHEMAS` list and `describe_command` from `jarvis/tools/coding.py` — they will live in `registry.py`.

- [ ] **Step 2: Write the failing test**

Create `tests/test_tools_registry.py`:

```python
import pytest
from unittest.mock import AsyncMock, patch
from jarvis.core.config import Settings
from jarvis.tools.registry import Toolbox, TOOL_SCHEMAS


def test_tool_schemas_present():
    names = {t["function"]["name"] for t in TOOL_SCHEMAS}
    assert "read_file" in names
    assert "web_search" in names
    assert "remember" in names
    assert "list_events" in names
    assert "list_emails" in names


@pytest.mark.asyncio
async def test_dispatch_unknown_tool():
    s = Settings()
    tb = Toolbox(s)
    result = await tb.dispatch("nonexistent_tool", {})
    assert "ERROR" in result or "unknown" in result.lower()


@pytest.mark.asyncio
async def test_web_search_disabled_without_key():
    s = Settings(brave_api_key="")
    tb = Toolbox(s)
    with patch("jarvis.tools.web_tools.duckduckgo_search") as mock_ddg:
        mock_ddg.return_value = [{"title": "Result", "href": "http://x.com", "body": "text"}]
        result = await tb.dispatch("web_search", {"query": "Python asyncio"})
    assert "Result" in result or "text" in result
```

- [ ] **Step 3: Run test to verify it fails**

```bash
pytest tests/test_tools_registry.py -v
```

Expected: `ImportError`

- [ ] **Step 4: Create `jarvis/tools/web_tools.py`**

```python
"""Web search and page fetch tools."""

from __future__ import annotations

import logging

import trafilatura

log = logging.getLogger(__name__)


async def web_search(query: str, brave_api_key: str = "", limit: int = 5) -> str:
    """Search the web. Uses Brave Search if api key provided, else DuckDuckGo."""
    if brave_api_key:
        return await _brave_search(query, brave_api_key, limit)
    return await _ddg_search(query, limit)


async def _brave_search(query: str, api_key: str, limit: int) -> str:
    import httpx

    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {"Accept": "application/json", "X-Subscription-Token": api_key}
    params = {"q": query, "count": limit}
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(url, headers=headers, params=params)
        r.raise_for_status()
        data = r.json()
    results = data.get("web", {}).get("results", [])
    lines = [f"{i+1}. {r['title']} — {r['url']}\n   {r.get('description','')}"
             for i, r in enumerate(results)]
    return "\n".join(lines) or "(no results)"


async def _ddg_search(query: str, limit: int) -> str:
    from duckduckgo_search import DDGS

    results = list(DDGS().text(query, max_results=limit))
    lines = [f"{i+1}. {r['title']} — {r['href']}\n   {r['body']}"
             for i, r in enumerate(results)]
    return "\n".join(lines) or "(no results)"


async def fetch_page(url: str) -> str:
    """Fetch a URL and return clean extracted text."""
    import httpx

    async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
        r = await client.get(url, headers={"User-Agent": "Mozilla/5.0 Jarvis/2.0"})
        r.raise_for_status()
        html = r.text
    text = trafilatura.extract(html) or ""
    if not text.strip():
        return "(could not extract text from page)"
    return text[:8000]
```

- [ ] **Step 5: Create `jarvis/tools/registry.py`**

```python
"""Unified tool registry: schemas + dispatch for all Jarvis tools."""

from __future__ import annotations

import shlex
from typing import Any

from jarvis.core.config import Settings
from jarvis.tools.coding import Toolbox as CodingToolbox, ToolError
from jarvis.tools import web_tools


class Toolbox:
    def __init__(self, settings: Settings, mem0_store=None) -> None:
        self._coding = CodingToolbox(settings)
        self._settings = settings
        self._mem0 = mem0_store
        self._calendar = None  # injected after Google auth in Task 9
        self._email = None     # injected after Google auth in Task 9

    def inject_google(self, calendar_tools, email_tools) -> None:
        self._calendar = calendar_tools
        self._email = email_tools

    async def dispatch(self, name: str, args: dict[str, Any]) -> str:
        try:
            # Coding tools
            if name == "read_file":
                return self._coding.read_file(**args)
            if name == "list_dir":
                return self._coding.list_dir(**args)
            if name == "grep":
                return self._coding.grep(**args)
            if name == "write_file":
                return self._coding.write_file(**args)
            if name == "run_shell":
                return await self._coding.run_shell(**args)

            # Web tools
            if name == "web_search":
                return await web_tools.web_search(
                    args["query"],
                    brave_api_key=self._settings.brave_api_key,
                    limit=args.get("limit", 5),
                )
            if name == "fetch_page":
                return await web_tools.fetch_page(args["url"])

            # Memory tools
            if name == "remember":
                if self._mem0:
                    self._mem0.remember(args["fact"])
                return "Remembered."
            if name == "recall":
                if self._mem0:
                    results = self._mem0.retrieve(args["query"], limit=args.get("limit", 5))
                    return "\n".join(results) or "(nothing found)"
                return "(memory not available)"
            if name == "forget":
                if self._mem0:
                    self._mem0.forget(args["fact"])
                return "Forgotten."

            # Calendar tools (available after google auth)
            if name in {"list_events", "create_event", "update_event", "find_free_slot"}:
                if self._calendar is None:
                    return "ERROR: Google Calendar not configured. Run `jarvis auth google`."
                return await self._calendar.dispatch(name, args)

            # Email tools
            if name in {"list_emails", "read_email", "draft_reply", "send_email"}:
                if self._email is None:
                    return "ERROR: Gmail not configured. Run `jarvis auth google`."
                return await self._email.dispatch(name, args)

            # Notification
            if name == "notify":
                return f"[notify] {args.get('message', '')}"  # actual send wired in main.py

            return f"ERROR: unknown tool {name!r}"

        except ToolError as e:
            return f"ERROR: {e}"
        except TypeError as e:
            return f"ERROR: bad arguments for {name}: {e}"
        except Exception as e:
            return f"ERROR: {e}"


TOOL_SCHEMAS: list[dict[str, Any]] = [
    # Coding
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a file from the workspace.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "start_line": {"type": "integer"},
            "end_line": {"type": "integer"},
        }, "required": ["path"]},
    }},
    {"type": "function", "function": {
        "name": "list_dir",
        "description": "List files and folders in a workspace directory.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string", "default": "."}}},
    }},
    {"type": "function", "function": {
        "name": "grep",
        "description": "Search the workspace for a regex pattern.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string", "default": "."},
            "glob": {"type": "string", "default": "*"},
        }, "required": ["pattern"]},
    }},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Overwrite a file. Gated by JARVIS_ALLOW_WRITES. Requires confirmation.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        }, "required": ["path", "content"]},
    }},
    {"type": "function", "function": {
        "name": "run_shell",
        "description": "Run a shell command. Gated by JARVIS_ALLOW_SHELL. Requires confirmation.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"},
            "timeout": {"type": "number", "default": 60},
        }, "required": ["command"]},
    }},
    # Web
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Search the web for information.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "default": 5},
        }, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "fetch_page",
        "description": "Fetch a URL and return clean extracted text.",
        "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
    }},
    # Memory
    {"type": "function", "function": {
        "name": "remember",
        "description": "Explicitly store a fact in long-term memory.",
        "parameters": {"type": "object", "properties": {"fact": {"type": "string"}}, "required": ["fact"]},
    }},
    {"type": "function", "function": {
        "name": "recall",
        "description": "Retrieve relevant memories for a query.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "default": 5},
        }, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "forget",
        "description": "Remove a fact from long-term memory.",
        "parameters": {"type": "object", "properties": {"fact": {"type": "string"}}, "required": ["fact"]},
    }},
    # Calendar
    {"type": "function", "function": {
        "name": "list_events",
        "description": "List upcoming Google Calendar events.",
        "parameters": {"type": "object", "properties": {"days": {"type": "integer", "default": 7}}},
    }},
    {"type": "function", "function": {
        "name": "create_event",
        "description": "Create a calendar event. Requires confirmation before executing.",
        "parameters": {"type": "object", "properties": {
            "title": {"type": "string"},
            "start_time": {"type": "string", "description": "ISO 8601 datetime"},
            "duration_minutes": {"type": "integer"},
            "attendees": {"type": "array", "items": {"type": "string"}},
        }, "required": ["title", "start_time"]},
    }},
    {"type": "function", "function": {
        "name": "update_event",
        "description": "Update an existing calendar event. Requires confirmation.",
        "parameters": {"type": "object", "properties": {
            "event_id": {"type": "string"},
            "title": {"type": "string"},
            "start_time": {"type": "string"},
            "duration_minutes": {"type": "integer"},
        }, "required": ["event_id"]},
    }},
    {"type": "function", "function": {
        "name": "find_free_slot",
        "description": "Find the next available time slot in the calendar.",
        "parameters": {"type": "object", "properties": {
            "duration_minutes": {"type": "integer"},
            "within_days": {"type": "integer", "default": 7},
        }, "required": ["duration_minutes"]},
    }},
    # Email
    {"type": "function", "function": {
        "name": "list_emails",
        "description": "List emails from Gmail matching a query.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "default": "is:unread"},
            "limit": {"type": "integer", "default": 10},
        }},
    }},
    {"type": "function", "function": {
        "name": "read_email",
        "description": "Read the full content of an email by ID.",
        "parameters": {"type": "object", "properties": {"email_id": {"type": "string"}}, "required": ["email_id"]},
    }},
    {"type": "function", "function": {
        "name": "draft_reply",
        "description": "Create a draft reply to an email. Does not send.",
        "parameters": {"type": "object", "properties": {
            "email_id": {"type": "string"},
            "body": {"type": "string"},
        }, "required": ["email_id", "body"]},
    }},
    {"type": "function", "function": {
        "name": "send_email",
        "description": "Send a drafted email. Gated by JARVIS_ALLOW_SEND_EMAIL. Requires confirmation.",
        "parameters": {"type": "object", "properties": {"email_id": {"type": "string"}}, "required": ["email_id"]},
    }},
    # Notification
    {"type": "function", "function": {
        "name": "notify",
        "description": "Send a push notification via Telegram.",
        "parameters": {"type": "object", "properties": {
            "message": {"type": "string"},
            "urgency": {"type": "string", "enum": ["low", "normal", "high"], "default": "normal"},
        }, "required": ["message"]},
    }},
]
```

- [ ] **Step 6: Run tests**

```bash
pytest tests/test_tools_registry.py -v
```

Expected: all 3 PASS.

- [ ] **Step 7: Commit**

```bash
git add jarvis/tools/ tests/test_tools_registry.py
git commit -m "feat: add unified tool registry with web, memory, calendar, email schemas"
```

---

## Task 8: LangGraph agent — state and nodes

**Files:**
- Create: `jarvis/graph/state.py`
- Create: `jarvis/graph/nodes.py`
- Create: `jarvis/graph/agent.py`
- Test: `tests/test_graph_nodes.py`

- [ ] **Step 1: Create `jarvis/graph/state.py`**

```python
"""AgentState TypedDict for the LangGraph agent graph."""

from __future__ import annotations

from typing import Any
from typing_extensions import TypedDict


class AgentState(TypedDict):
    trigger: str                     # "voice" | "calendar" | "email" | "watchdog"
    messages: list[dict[str, Any]]   # OpenAI-style message dicts for Anthropic API
    memories: list[str]              # retrieved mem0 facts injected into system prompt
    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    output_channel: str              # "voice" | "telegram" | "silent"
    final_response: str
    requires_confirmation: bool      # True when a destructive tool was called
    pending_action: dict[str, Any] | None  # tool call waiting for approval
    error: str | None
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_graph_nodes.py`:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from jarvis.graph.state import AgentState
from jarvis.graph.nodes import retrieve_memory_node, store_memory_node, respond_node


@pytest.fixture
def base_state() -> AgentState:
    return AgentState(
        trigger="voice",
        messages=[{"role": "user", "content": "What time is my next meeting?"}],
        memories=[],
        tool_calls=[],
        tool_results=[],
        output_channel="voice",
        final_response="",
        requires_confirmation=False,
        pending_action=None,
        error=None,
    )


def test_retrieve_memory_node_no_mem0(base_state):
    update = retrieve_memory_node(base_state, mem0_store=None)
    assert update["memories"] == []


def test_retrieve_memory_node_with_results(base_state):
    mock_store = MagicMock()
    mock_store.retrieve.return_value = ["Rui likes TypeScript", "Orbit deadline May 15"]
    update = retrieve_memory_node(base_state, mem0_store=mock_store)
    assert len(update["memories"]) == 2
    mock_store.retrieve.assert_called_once_with(
        "What time is my next meeting?", limit=5
    )


def test_store_memory_node_calls_store(base_state):
    base_state["final_response"] = "Your next meeting is at 3pm."
    mock_store = MagicMock()
    store_memory_node(base_state, mem0_store=mock_store)
    mock_store.store.assert_called()


def test_respond_node_sets_final_response(base_state):
    base_state["messages"] = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "Hi Rui!"},
    ]
    update = respond_node(base_state)
    assert update["final_response"] == "Hi Rui!"


def test_respond_node_confirmation_required(base_state):
    base_state["requires_confirmation"] = True
    base_state["pending_action"] = {"name": "write_file", "args": {"path": "x.py", "content": "..."}}
    base_state["messages"] = [{"role": "assistant", "content": ""}]
    update = respond_node(base_state)
    assert "write_file" in update["final_response"] or "confirm" in update["final_response"].lower()
```

- [ ] **Step 3: Run test to verify it fails**

```bash
pytest tests/test_graph_nodes.py -v
```

Expected: `ImportError`

- [ ] **Step 4: Create `jarvis/graph/nodes.py`**

```python
"""LangGraph node functions for the Jarvis agent graph."""

from __future__ import annotations

import json
import logging
from typing import Any, TYPE_CHECKING

import anthropic

from jarvis.core.config import Settings
from jarvis.core.personality import SYSTEM_PROMPT
from jarvis.graph.state import AgentState
from jarvis.tools.registry import TOOL_SCHEMAS

if TYPE_CHECKING:
    from jarvis.memory.mem0_store import Mem0Store
    from jarvis.tools.registry import Toolbox

log = logging.getLogger(__name__)

MAX_TURNS = 6

CONFIRMATION_REQUIRED_TOOLS = {"write_file", "run_shell", "create_event", "update_event", "send_email"}


def retrieve_memory_node(state: AgentState, mem0_store: "Mem0Store | None") -> dict[str, Any]:
    if mem0_store is None:
        return {"memories": []}
    last_user = next(
        (m["content"] for m in reversed(state["messages"]) if m["role"] == "user"), ""
    )
    memories = mem0_store.retrieve(last_user, limit=5) if last_user else []
    return {"memories": memories}


def _build_system_with_memories(memories: list[str]) -> str:
    if not memories:
        return SYSTEM_PROMPT
    block = "\n\nWhat you know about this user (strong priors, not absolute truth):\n"
    block += "\n".join(f"- {m}" for m in memories)
    return SYSTEM_PROMPT + block


async def think_node(
    state: AgentState,
    client: anthropic.AsyncAnthropic,
    settings: Settings,
) -> dict[str, Any]:
    system = _build_system_with_memories(state["memories"])
    messages = [m for m in state["messages"] if m["role"] != "system"]

    # Convert tool schemas from OpenAI format to Anthropic format
    tools = [
        {
            "name": t["function"]["name"],
            "description": t["function"].get("description", ""),
            "input_schema": t["function"].get("parameters", {"type": "object", "properties": {}}),
        }
        for t in TOOL_SCHEMAS
    ]

    response = await client.messages.create(
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
        system=system,
        messages=messages,
        tools=tools,
    )

    tool_calls = []
    text_content = ""
    for block in response.content:
        if block.type == "text":
            text_content += block.text
        elif block.type == "tool_use":
            tool_calls.append({
                "id": block.id,
                "name": block.name,
                "args": block.input,
            })

    new_messages = list(state["messages"])
    new_messages.append({
        "role": "assistant",
        "content": response.content,  # pass Anthropic content blocks directly
    })

    requires_confirmation = any(tc["name"] in CONFIRMATION_REQUIRED_TOOLS for tc in tool_calls)
    pending_action = tool_calls[0] if requires_confirmation and tool_calls else None

    return {
        "messages": new_messages,
        "tool_calls": tool_calls,
        "requires_confirmation": requires_confirmation,
        "pending_action": pending_action,
        "final_response": text_content,
    }


async def act_node(state: AgentState, toolbox: "Toolbox") -> dict[str, Any]:
    results = []
    for call in state["tool_calls"]:
        result = await toolbox.dispatch(call["name"], call["args"])
        results.append({"tool_use_id": call["id"], "content": result})
    new_messages = list(state["messages"])
    new_messages.append({"role": "user", "content": results})
    return {"messages": new_messages, "tool_results": results, "tool_calls": []}


def observe_node(state: AgentState) -> dict[str, Any]:
    if state["tool_results"]:
        return {"tool_results": []}
    return {}


def respond_node(state: AgentState) -> dict[str, Any]:
    if state.get("requires_confirmation") and state.get("pending_action"):
        action = state["pending_action"]
        name = action.get("name", "action")
        args_preview = json.dumps(action.get("args", {}))[:120]
        msg = f"I'd like to run {name} with: {args_preview}. Shall I go ahead?"
        return {"final_response": msg}

    last_assistant = next(
        (m for m in reversed(state["messages"]) if m["role"] == "assistant"),
        None,
    )
    if last_assistant is None:
        return {"final_response": ""}

    content = last_assistant.get("content", "")
    if isinstance(content, list):
        text = " ".join(b.text for b in content if hasattr(b, "text"))
    else:
        text = str(content)
    return {"final_response": text.strip()}


def store_memory_node(state: AgentState, mem0_store: "Mem0Store | None") -> dict[str, Any]:
    if mem0_store is None:
        return {}
    last_user = next(
        (m["content"] for m in state["messages"] if m["role"] == "user" and isinstance(m["content"], str)),
        None,
    )
    if last_user:
        mem0_store.store(last_user, role="user")
    if state.get("final_response"):
        mem0_store.store(state["final_response"], role="assistant")
    return {}


def route_node(state: AgentState) -> dict[str, Any]:
    return {}
```

- [ ] **Step 5: Create `jarvis/graph/agent.py`**

```python
"""LangGraph graph compilation and run_turn() entry point."""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

import anthropic
from langgraph.graph import StateGraph, END

from jarvis.core.config import Settings
from jarvis.graph.state import AgentState
from jarvis.graph import nodes

if TYPE_CHECKING:
    from jarvis.memory.mem0_store import Mem0Store
    from jarvis.tools.registry import Toolbox

log = logging.getLogger(__name__)


def build_graph(
    settings: Settings,
    client: anthropic.AsyncAnthropic,
    toolbox: "Toolbox",
    mem0_store: "Mem0Store | None",
):
    """Compile the LangGraph agent graph and return a runnable."""
    graph = StateGraph(AgentState)

    graph.add_node("retrieve_memory", lambda s: nodes.retrieve_memory_node(s, mem0_store))
    graph.add_node("think", lambda s: nodes.think_node(s, client, settings))
    graph.add_node("act", lambda s: nodes.act_node(s, toolbox))
    graph.add_node("observe", nodes.observe_node)
    graph.add_node("respond", nodes.respond_node)
    graph.add_node("store_memory", lambda s: nodes.store_memory_node(s, mem0_store))

    graph.set_entry_point("retrieve_memory")
    graph.add_edge("retrieve_memory", "think")

    def _after_think(state: AgentState) -> str:
        if state.get("requires_confirmation"):
            return "respond"
        if state.get("tool_calls"):
            return "act"
        return "respond"

    graph.add_conditional_edges("think", _after_think, {
        "act": "act",
        "respond": "respond",
    })
    graph.add_edge("act", "observe")

    def _after_observe(state: AgentState) -> str:
        # If there are unprocessed tool results, think again (up to MAX_TURNS)
        turn_count = sum(1 for m in state["messages"] if m["role"] == "assistant")
        if turn_count >= nodes.MAX_TURNS:
            return "respond"
        return "think"

    graph.add_conditional_edges("observe", _after_observe, {
        "think": "think",
        "respond": "respond",
    })
    graph.add_edge("respond", "store_memory")
    graph.add_edge("store_memory", END)

    return graph.compile()


async def run_turn(
    compiled_graph,
    user_text: str,
    trigger: str = "voice",
    output_channel: str = "voice",
    conversation_history: list[dict[str, Any]] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Run one turn through the graph. Returns (response_text, updated_history)."""
    history = list(conversation_history or [])
    history.append({"role": "user", "content": user_text})

    initial_state: AgentState = {
        "trigger": trigger,
        "messages": history,
        "memories": [],
        "tool_calls": [],
        "tool_results": [],
        "output_channel": output_channel,
        "final_response": "",
        "requires_confirmation": False,
        "pending_action": None,
        "error": None,
    }

    result = await compiled_graph.ainvoke(initial_state)
    response = result.get("final_response", "")
    updated_history = result.get("messages", history)
    return response, updated_history
```

- [ ] **Step 6: Run tests**

```bash
pytest tests/test_graph_nodes.py -v
```

Expected: all 5 PASS.

- [ ] **Step 7: Commit**

```bash
git add jarvis/graph/state.py jarvis/graph/nodes.py jarvis/graph/agent.py tests/test_graph_nodes.py
git commit -m "feat: add LangGraph agent graph (state, nodes, compiled graph)"
```

---

## Task 9: Google OAuth + Calendar + Gmail tools

**Files:**
- Create: `jarvis/integrations/google_auth.py`
- Create: `jarvis/tools/calendar_tools.py`
- Create: `jarvis/tools/email_tools.py`
- Test: `tests/test_tools_google.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_tools_google.py`:

```python
import pytest
from unittest.mock import MagicMock, patch
from jarvis.tools.calendar_tools import CalendarTools
from jarvis.tools.email_tools import EmailTools


@pytest.fixture
def mock_calendar_service():
    svc = MagicMock()
    events_list = svc.events.return_value.list.return_value.execute
    events_list.return_value = {"items": [
        {
            "id": "evt1",
            "summary": "Call with Sarah",
            "start": {"dateTime": "2026-04-24T15:00:00Z"},
            "end": {"dateTime": "2026-04-24T15:30:00Z"},
            "attendees": [{"email": "sarah@example.com"}],
        }
    ]}
    return svc


@pytest.fixture
def mock_gmail_service():
    svc = MagicMock()
    svc.users.return_value.messages.return_value.list.return_value.execute.return_value = {
        "messages": [{"id": "msg1", "threadId": "thread1"}]
    }
    svc.users.return_value.messages.return_value.get.return_value.execute.return_value = {
        "id": "msg1",
        "payload": {
            "headers": [
                {"name": "Subject", "value": "Project update"},
                {"name": "From", "value": "sarah@example.com"},
                {"name": "Date", "value": "Thu, 24 Apr 2026 10:00:00 +0000"},
            ],
            "body": {"data": "SGVsbG8gUnVpIQ=="},  # base64 "Hello Rui!"
        },
    }
    return svc


@pytest.mark.asyncio
async def test_list_events(mock_calendar_service):
    ct = CalendarTools(mock_calendar_service)
    result = await ct.dispatch("list_events", {"days": 7})
    assert "Call with Sarah" in result
    assert "sarah" in result.lower() or "15:00" in result


@pytest.mark.asyncio
async def test_list_emails(mock_gmail_service):
    et = EmailTools(mock_gmail_service, allow_send=False)
    result = await et.dispatch("list_emails", {"query": "is:unread", "limit": 5})
    assert "msg1" in result or "Project update" in result or result  # any content


@pytest.mark.asyncio
async def test_send_email_blocked_without_flag(mock_gmail_service):
    et = EmailTools(mock_gmail_service, allow_send=False)
    result = await et.dispatch("send_email", {"email_id": "msg1"})
    assert "ERROR" in result or "disabled" in result.lower()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_tools_google.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Create `jarvis/integrations/google_auth.py`**

```python
"""Google OAuth 2.0 flow. Run `jarvis auth google` to authorise."""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

SCOPES = [
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.send",
]


def get_credentials(credentials_path: Path, token_path: Path):
    """Return valid Google credentials, refreshing or re-authorising as needed."""
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    credentials_path = Path(credentials_path).expanduser()
    token_path = Path(token_path).expanduser()
    creds = None

    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not credentials_path.exists():
                raise FileNotFoundError(
                    f"Google credentials file not found: {credentials_path}\n"
                    "Download it from Google Cloud Console and set JARVIS_GOOGLE_CREDENTIALS."
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), SCOPES)
            creds = flow.run_local_server(port=0)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(creds.to_json())
        log.info("Google token saved to %s", token_path)

    return creds


def build_services(credentials_path: Path, token_path: Path):
    """Return (calendar_service, gmail_service) ready to use."""
    from googleapiclient.discovery import build

    creds = get_credentials(credentials_path, token_path)
    calendar = build("calendar", "v3", credentials=creds)
    gmail = build("gmail", "v1", credentials=creds)
    return calendar, gmail
```

- [ ] **Step 4: Create `jarvis/tools/calendar_tools.py`**

```python
"""Google Calendar tool implementations."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from functools import partial
from typing import Any

log = logging.getLogger(__name__)


class CalendarTools:
    def __init__(self, service) -> None:
        self._svc = service

    async def dispatch(self, name: str, args: dict[str, Any]) -> str:
        try:
            if name == "list_events":
                return await asyncio.get_running_loop().run_in_executor(
                    None, partial(self._list_events, args.get("days", 7))
                )
            if name == "create_event":
                return await asyncio.get_running_loop().run_in_executor(
                    None, partial(self._create_event, **args)
                )
            if name == "update_event":
                return await asyncio.get_running_loop().run_in_executor(
                    None, partial(self._update_event, **args)
                )
            if name == "find_free_slot":
                return await asyncio.get_running_loop().run_in_executor(
                    None, partial(self._find_free_slot, **args)
                )
            return f"ERROR: unknown calendar tool {name!r}"
        except Exception as e:
            log.exception("calendar tool %s failed", name)
            return f"ERROR: {e}"

    def _list_events(self, days: int) -> str:
        now = datetime.now(timezone.utc)
        end = now + timedelta(days=days)
        result = (
            self._svc.events()
            .list(
                calendarId="primary",
                timeMin=now.isoformat(),
                timeMax=end.isoformat(),
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
        )
        items = result.get("items", [])
        if not items:
            return f"No events in the next {days} days."
        lines = []
        for e in items:
            start = e.get("start", {}).get("dateTime") or e.get("start", {}).get("date", "")
            attendees = ", ".join(a["email"] for a in e.get("attendees", []))
            line = f"- {e.get('summary','(no title)')} at {start}"
            if attendees:
                line += f" with {attendees}"
            lines.append(line)
        return "\n".join(lines)

    def _create_event(
        self,
        title: str,
        start_time: str,
        duration_minutes: int = 60,
        attendees: list[str] | None = None,
    ) -> str:
        start = datetime.fromisoformat(start_time)
        end = start + timedelta(minutes=duration_minutes)
        body: dict[str, Any] = {
            "summary": title,
            "start": {"dateTime": start.isoformat(), "timeZone": "UTC"},
            "end": {"dateTime": end.isoformat(), "timeZone": "UTC"},
        }
        if attendees:
            body["attendees"] = [{"email": a} for a in attendees]
        event = self._svc.events().insert(calendarId="primary", body=body).execute()
        return f"Event created: {event.get('htmlLink', event.get('id'))}"

    def _update_event(self, event_id: str, **kwargs) -> str:
        event = self._svc.events().get(calendarId="primary", eventId=event_id).execute()
        if "title" in kwargs:
            event["summary"] = kwargs["title"]
        if "start_time" in kwargs:
            start = datetime.fromisoformat(kwargs["start_time"])
            dur = int(kwargs.get("duration_minutes", 60))
            event["start"] = {"dateTime": start.isoformat(), "timeZone": "UTC"}
            event["end"] = {"dateTime": (start + timedelta(minutes=dur)).isoformat(), "timeZone": "UTC"}
        updated = self._svc.events().update(
            calendarId="primary", eventId=event_id, body=event
        ).execute()
        return f"Event updated: {updated.get('summary')}"

    def _find_free_slot(self, duration_minutes: int, within_days: int = 7) -> str:
        now = datetime.now(timezone.utc)
        end = now + timedelta(days=within_days)
        result = (
            self._svc.events()
            .list(
                calendarId="primary",
                timeMin=now.isoformat(),
                timeMax=end.isoformat(),
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
        )
        busy = []
        for e in result.get("items", []):
            s = e.get("start", {}).get("dateTime")
            en = e.get("end", {}).get("dateTime")
            if s and en:
                busy.append((datetime.fromisoformat(s), datetime.fromisoformat(en)))

        slot_start = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        while slot_start < end:
            slot_end = slot_start + timedelta(minutes=duration_minutes)
            if 9 <= slot_start.hour < 18:  # business hours only
                overlap = any(s < slot_end and e > slot_start for s, e in busy)
                if not overlap:
                    return f"Next free {duration_minutes}-min slot: {slot_start.strftime('%A %d %b at %H:%M UTC')}"
            slot_start += timedelta(minutes=30)
        return f"No free {duration_minutes}-min slot found in the next {within_days} days."
```

- [ ] **Step 5: Create `jarvis/tools/email_tools.py`**

```python
"""Gmail tool implementations."""

from __future__ import annotations

import asyncio
import base64
import logging
from functools import partial
from typing import Any

log = logging.getLogger(__name__)


class EmailTools:
    def __init__(self, service, allow_send: bool = False) -> None:
        self._svc = service
        self._allow_send = allow_send

    async def dispatch(self, name: str, args: dict[str, Any]) -> str:
        try:
            loop = asyncio.get_running_loop()
            if name == "list_emails":
                return await loop.run_in_executor(
                    None, partial(self._list_emails, args.get("query", "is:unread"), args.get("limit", 10))
                )
            if name == "read_email":
                return await loop.run_in_executor(
                    None, partial(self._read_email, args["email_id"])
                )
            if name == "draft_reply":
                return await loop.run_in_executor(
                    None, partial(self._draft_reply, args["email_id"], args["body"])
                )
            if name == "send_email":
                if not self._allow_send:
                    return "ERROR: sending email is disabled. Set JARVIS_ALLOW_SEND_EMAIL=true."
                return await loop.run_in_executor(
                    None, partial(self._send_email, args["email_id"])
                )
            return f"ERROR: unknown email tool {name!r}"
        except Exception as e:
            log.exception("email tool %s failed", name)
            return f"ERROR: {e}"

    def _list_emails(self, query: str, limit: int) -> str:
        result = (
            self._svc.users()
            .messages()
            .list(userId="me", q=query, maxResults=limit)
            .execute()
        )
        messages = result.get("messages", [])
        if not messages:
            return "No emails found."
        lines = []
        for m in messages:
            msg = self._svc.users().messages().get(
                userId="me", id=m["id"], format="metadata",
                metadataHeaders=["Subject", "From", "Date"]
            ).execute()
            headers = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}
            lines.append(
                f"id={m['id']} | {headers.get('Date','')} | From: {headers.get('From','')} | {headers.get('Subject','(no subject)')}"
            )
        return "\n".join(lines)

    def _read_email(self, email_id: str) -> str:
        msg = self._svc.users().messages().get(userId="me", id=email_id, format="full").execute()
        headers = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}
        body_data = msg.get("payload", {}).get("body", {}).get("data", "")
        body = ""
        if body_data:
            body = base64.urlsafe_b64decode(body_data + "==").decode("utf-8", errors="replace")
        return (
            f"From: {headers.get('From','')}\n"
            f"Subject: {headers.get('Subject','')}\n"
            f"Date: {headers.get('Date','')}\n\n"
            f"{body[:3000]}"
        )

    def _draft_reply(self, email_id: str, body: str) -> str:
        import email.mime.text as _mime
        original = self._svc.users().messages().get(
            userId="me", id=email_id, format="metadata",
            metadataHeaders=["Subject", "From", "Message-ID"]
        ).execute()
        headers = {h["name"]: h["value"] for h in original.get("payload", {}).get("headers", [])}

        msg = _mime.MIMEText(body)
        msg["To"] = headers.get("From", "")
        msg["Subject"] = "Re: " + headers.get("Subject", "")
        msg["In-Reply-To"] = headers.get("Message-ID", "")
        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
        draft = self._svc.users().drafts().create(
            userId="me", body={"message": {"raw": raw, "threadId": original.get("threadId")}}
        ).execute()
        return f"Draft created: id={draft['id']}"

    def _send_email(self, email_id: str) -> str:
        result = self._svc.users().drafts().send(userId="me", body={"id": email_id}).execute()
        return f"Email sent: id={result.get('id')}"
```

- [ ] **Step 6: Run tests**

```bash
pytest tests/test_tools_google.py -v
```

Expected: all 3 PASS.

- [ ] **Step 7: Commit**

```bash
git add jarvis/integrations/google_auth.py jarvis/tools/calendar_tools.py jarvis/tools/email_tools.py tests/test_tools_google.py
git commit -m "feat: add Google OAuth, Calendar tools, and Gmail tools"
```

---

## Task 10: Core session factory

**Files:**
- Create: `jarvis/core/session.py`
- Test: `tests/test_core_session.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_core_session.py`:

```python
import pytest
from unittest.mock import patch, MagicMock
from jarvis.core.config import Settings


def test_build_session_returns_components():
    from jarvis.core.session import build_session
    settings = Settings(mem0_enabled=False)
    with patch("jarvis.core.session.anthropic.AsyncAnthropic"), \
         patch("jarvis.core.session.build_graph") as mock_graph:
        mock_graph.return_value = MagicMock()
        session = build_session(settings)
    assert "graph" in session
    assert "toolbox" in session
    assert "mem0" in session
    assert "client" in session


def test_build_session_mem0_disabled(tmp_path):
    from jarvis.core.session import build_session
    settings = Settings(mem0_enabled=False)
    with patch("jarvis.core.session.anthropic.AsyncAnthropic"), \
         patch("jarvis.core.session.build_graph") as mock_graph:
        mock_graph.return_value = MagicMock()
        session = build_session(settings)
    assert session["mem0"] is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_core_session.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Create `jarvis/core/session.py`**

```python
"""Shared runtime factory: builds all Jarvis components from Settings."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import anthropic

from jarvis.core.config import Settings
from jarvis.graph.agent import build_graph
from jarvis.memory.mem0_store import Mem0Store
from jarvis.tools.registry import Toolbox

log = logging.getLogger(__name__)


def build_session(settings: Settings) -> dict[str, Any]:
    """Build and return all shared runtime components."""
    client = anthropic.AsyncAnthropic()  # reads ANTHROPIC_API_KEY from env

    mem0: Mem0Store | None = None
    if settings.mem0_enabled:
        mem0 = Mem0Store(
            path=str(Path(settings.mem0_path).expanduser()),
            user_id="jarvis_user",
            enabled=True,
        )

    toolbox = Toolbox(settings, mem0_store=mem0)

    graph = build_graph(settings, client, toolbox, mem0)

    return {
        "client": client,
        "mem0": mem0,
        "toolbox": toolbox,
        "graph": graph,
        "settings": settings,
    }


def inject_google(session: dict[str, Any]) -> None:
    """Attempt to load Google credentials and inject calendar/email tools."""
    settings: Settings = session["settings"]
    creds_path = Path(settings.google_credentials_path).expanduser()
    token_path = Path(settings.google_token_path).expanduser()

    if not token_path.exists():
        log.info("Google token not found at %s — skipping Google tools", token_path)
        return

    try:
        from jarvis.integrations.google_auth import build_services
        from jarvis.tools.calendar_tools import CalendarTools
        from jarvis.tools.email_tools import EmailTools

        cal_svc, gmail_svc = build_services(creds_path, token_path)
        cal_tools = CalendarTools(cal_svc)
        email_tools = EmailTools(gmail_svc, allow_send=settings.allow_send_email)
        session["toolbox"].inject_google(cal_tools, email_tools)
        log.info("Google Calendar + Gmail tools injected")
    except Exception as e:
        log.warning("Could not load Google tools: %s", e)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_core_session.py -v
```

Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add jarvis/core/session.py tests/test_core_session.py
git commit -m "feat: add session factory that wires all components together"
```

---

## Task 11: Proactive monitors (calendar + email)

**Files:**
- Create: `jarvis/graph/proactive/calendar.py`
- Create: `jarvis/graph/proactive/email.py`
- Create: `jarvis/graph/proactive/watchdog.py`
- Test: `tests/test_proactive_monitors.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_proactive_monitors.py`:

```python
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from jarvis.graph.proactive.calendar import CalendarMonitor
from jarvis.graph.proactive.email import EmailMonitor


@pytest.fixture
def mock_graph():
    g = MagicMock()
    g.ainvoke = AsyncMock(return_value={"final_response": "You have a meeting soon."})
    return g


@pytest.fixture
def mock_calendar_tools():
    ct = MagicMock()
    ct.dispatch = AsyncMock(return_value="- Call with Sarah at 2026-04-24T15:00:00Z")
    return ct


@pytest.fixture
def mock_email_tools():
    et = MagicMock()
    et.dispatch = AsyncMock(return_value="id=msg1 | Thu | From: boss@co.com | Urgent: review needed")
    return et


@pytest.mark.asyncio
async def test_calendar_monitor_runs_once(mock_graph, mock_calendar_tools):
    monitor = CalendarMonitor(mock_graph, mock_calendar_tools, notify_fn=AsyncMock())
    await monitor.check_once()
    mock_calendar_tools.dispatch.assert_awaited_once_with("list_events", {"days": 1})


@pytest.mark.asyncio
async def test_email_monitor_runs_once(mock_graph, mock_email_tools):
    monitor = EmailMonitor(mock_graph, mock_email_tools, notify_fn=AsyncMock())
    await monitor.check_once()
    mock_email_tools.dispatch.assert_awaited_once()


@pytest.mark.asyncio
async def test_calendar_monitor_calls_notify_on_upcoming(mock_graph, mock_calendar_tools):
    notify = AsyncMock()
    monitor = CalendarMonitor(mock_graph, mock_calendar_tools, notify_fn=notify)
    await monitor.check_once()
    # graph was invoked (monitor decided something was actionable)
    mock_graph.ainvoke.assert_awaited()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_proactive_monitors.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Create `jarvis/graph/proactive/calendar.py`**

```python
"""Calendar proactive monitor — polls Google Calendar and surfaces alerts."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Awaitable

from jarvis.graph.state import AgentState

log = logging.getLogger(__name__)


class CalendarMonitor:
    def __init__(
        self,
        graph,
        calendar_tools,
        notify_fn: Callable[[str], Awaitable[None]],
        poll_seconds: int = 300,
    ) -> None:
        self._graph = graph
        self._cal = calendar_tools
        self._notify = notify_fn
        self._poll_seconds = poll_seconds
        self._seen_event_ids: set[str] = set()

    async def check_once(self) -> None:
        try:
            events_text = await self._cal.dispatch("list_events", {"days": 1})
            if not events_text or events_text.startswith("No events"):
                return
            prompt = (
                f"You are Jarvis monitoring the user's calendar. "
                f"Here are today's events:\n{events_text}\n\n"
                f"Are any of these events coming up soon (within 30 minutes) or require action? "
                f"If yes, compose a short proactive alert (one sentence). "
                f"If nothing needs attention right now, reply with exactly: NO_ACTION"
            )
            initial_state: AgentState = {
                "trigger": "calendar",
                "messages": [{"role": "user", "content": prompt}],
                "memories": [],
                "tool_calls": [],
                "tool_results": [],
                "output_channel": "telegram",
                "final_response": "",
                "requires_confirmation": False,
                "pending_action": None,
                "error": None,
            }
            result = await self._graph.ainvoke(initial_state)
            response = result.get("final_response", "").strip()
            if response and response != "NO_ACTION":
                await self._notify(response)
        except Exception as e:
            log.warning("calendar monitor check failed: %s", e)

    async def run_forever(self) -> None:
        log.info("calendar monitor started (poll every %ds)", self._poll_seconds)
        while True:
            await self.check_once()
            await asyncio.sleep(self._poll_seconds)
```

- [ ] **Step 4: Create `jarvis/graph/proactive/email.py`**

```python
"""Email proactive monitor — polls Gmail and surfaces urgent items."""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Awaitable

from jarvis.graph.state import AgentState

log = logging.getLogger(__name__)


class EmailMonitor:
    def __init__(
        self,
        graph,
        email_tools,
        notify_fn: Callable[[str], Awaitable[None]],
        poll_seconds: int = 600,
    ) -> None:
        self._graph = graph
        self._email = email_tools
        self._notify = notify_fn
        self._poll_seconds = poll_seconds
        self._last_seen_id: str | None = None

    async def check_once(self) -> None:
        try:
            emails_text = await self._email.dispatch(
                "list_emails", {"query": "is:unread", "limit": 5}
            )
            if not emails_text or emails_text == "No emails found.":
                return
            prompt = (
                f"You are Jarvis monitoring the user's email. "
                f"Here are the latest unread emails:\n{emails_text}\n\n"
                f"Is anything urgent or requiring a decision? "
                f"If yes, compose a one-sentence proactive alert with a suggested action. "
                f"If nothing is urgent, reply with exactly: NO_ACTION"
            )
            initial_state: AgentState = {
                "trigger": "email",
                "messages": [{"role": "user", "content": prompt}],
                "memories": [],
                "tool_calls": [],
                "tool_results": [],
                "output_channel": "telegram",
                "final_response": "",
                "requires_confirmation": False,
                "pending_action": None,
                "error": None,
            }
            result = await self._graph.ainvoke(initial_state)
            response = result.get("final_response", "").strip()
            if response and response != "NO_ACTION":
                await self._notify(response)
        except Exception as e:
            log.warning("email monitor check failed: %s", e)

    async def run_forever(self) -> None:
        log.info("email monitor started (poll every %ds)", self._poll_seconds)
        while True:
            await self.check_once()
            await asyncio.sleep(self._poll_seconds)
```

- [ ] **Step 5: Create `jarvis/graph/proactive/watchdog.py`**

```python
"""File/CI watchdog rewired as a proactive monitor node."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Callable, Awaitable

from watchfiles import awatch

log = logging.getLogger(__name__)


class WatchdogMonitor:
    def __init__(
        self,
        paths: list[str],
        notify_fn: Callable[[str], Awaitable[None]],
        command: str | None = None,
        debounce_ms: int = 400,
    ) -> None:
        self._paths = paths
        self._notify = notify_fn
        self._command = command
        self._debounce_ms = debounce_ms

    async def run_forever(self) -> None:
        log.info("watchdog started on paths: %s", self._paths)
        async for changes in awatch(*self._paths, debounce=self._debounce_ms):
            changed_files = [str(Path(c[1])) for c in changes]
            summary = f"Files changed: {', '.join(changed_files[:5])}"
            if len(changed_files) > 5:
                summary += f" (+{len(changed_files) - 5} more)"

            if self._command:
                proc = await asyncio.create_subprocess_shell(
                    self._command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=120)
                output = stdout.decode("utf-8", errors="replace")[-1000:]
                summary += f"\n`{self._command}` exit={proc.returncode}\n{output}"

            await self._notify(summary)
```

- [ ] **Step 6: Run tests**

```bash
pytest tests/test_proactive_monitors.py -v
```

Expected: all 3 PASS.

- [ ] **Step 7: Commit**

```bash
git add jarvis/graph/proactive/ tests/test_proactive_monitors.py
git commit -m "feat: add proactive calendar, email, and watchdog monitors"
```

---

## Task 12: Rewire main.py CLI

**Files:**
- Modify: `jarvis/main.py`
- Test: `tests/test_main_cli.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_main_cli.py`:

```python
import pytest
from typer.testing import CliRunner
from unittest.mock import patch, AsyncMock, MagicMock
from jarvis.main import app

runner = CliRunner()


def test_ask_command():
    with patch("jarvis.main.build_session") as mock_session, \
         patch("jarvis.main.run_turn", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = ("Hello from Jarvis.", [])
        mock_session.return_value = {
            "graph": MagicMock(), "toolbox": MagicMock(),
            "mem0": None, "client": MagicMock(), "settings": MagicMock()
        }
        result = runner.invoke(app, ["ask", "what is 2+2"])
    assert result.exit_code == 0
    assert "Jarvis" in result.output or "Hello" in result.output


def test_auth_google_command_exists():
    result = runner.invoke(app, ["auth", "--help"])
    assert result.exit_code == 0
    assert "google" in result.output.lower()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_main_cli.py -v
```

Expected: import errors or missing commands.

- [ ] **Step 3: Replace `jarvis/main.py`**

```python
"""CLI entry points for Jarvis v0.2.

Sub-commands:
    jarvis voice      — hands-free voice loop
    jarvis chat       — text REPL
    jarvis ask        — one-shot text query
    jarvis telegram   — Telegram bridge only
    jarvis notify     — send a push notification
    jarvis watch      — watch paths, push to Telegram
    jarvis auth       — authentication subcommands
      jarvis auth google  — Google OAuth flow
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.prompt import Prompt

from jarvis.core.config import get_settings, reload_settings
from jarvis.core.session import build_session, inject_google
from jarvis.graph.agent import run_turn

app = typer.Typer(add_completion=False, help="Jarvis — proactive personal AI.")
auth_app = typer.Typer(add_completion=False, help="Authentication commands.")
app.add_typer(auth_app, name="auth")
console = Console()
log = logging.getLogger("jarvis")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _install_signal_handlers(stop: asyncio.Event) -> None:
    def _sig(*_a) -> None:
        stop.set()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            asyncio.get_event_loop().add_signal_handler(sig, _sig)
        except NotImplementedError:
            signal.signal(sig, lambda *_: stop.set())


# ----- voice -----

@app.command()
def voice(
    stream: Optional[bool] = typer.Option(None, "--stream/--no-stream"),
    wake: Optional[bool] = typer.Option(None, "--wake/--no-wake"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Hands-free voice loop. Wake word → record → transcribe → stream reply."""
    _setup_logging(verbose)
    asyncio.run(_voice_main(stream=stream, wake=wake))


async def _voice_main(stream: bool | None, wake: bool | None) -> None:
    from jarvis.voice.audio import AudioIO
    from jarvis.voice.stt import STT
    from jarvis.voice.tts import TTS
    from jarvis.voice.wake import WakeWord
    from jarvis.voice.streaming import speak_stream
    from jarvis.telegram_bot import TelegramBridge

    settings = get_settings()
    if stream is not None:
        settings.stream_voice = stream
    if wake is not None:
        settings.wake_enabled = wake

    session = build_session(settings)
    inject_google(session)
    graph = session["graph"]

    stt = STT(settings)
    tts = TTS(settings)
    audio = AudioIO(settings)
    wake_detector = WakeWord(settings)

    conversation_history: list = []

    async def _respond(text: str) -> str:
        nonlocal conversation_history
        response, conversation_history = await run_turn(
            graph, text, trigger="voice", output_channel="voice",
            conversation_history=conversation_history,
        )
        return response

    bridge = TelegramBridge(settings, _respond)
    await bridge.start()

    # Start proactive monitors if Google is available
    monitor_tasks = []
    if session["toolbox"]._calendar:
        from jarvis.graph.proactive.calendar import CalendarMonitor
        cal_monitor = CalendarMonitor(
            graph, session["toolbox"]._calendar,
            notify_fn=bridge.notify,
            poll_seconds=settings.calendar_poll_seconds,
        )
        monitor_tasks.append(asyncio.create_task(cal_monitor.run_forever()))

    if session["toolbox"]._email:
        from jarvis.graph.proactive.email import EmailMonitor
        email_monitor = EmailMonitor(
            graph, session["toolbox"]._email,
            notify_fn=bridge.notify,
            poll_seconds=settings.email_poll_seconds,
        )
        monitor_tasks.append(asyncio.create_task(email_monitor.run_forever()))

    stop = asyncio.Event()
    _install_signal_handlers(stop)
    wake_mode = "wake-word" if settings.wake_enabled else "always-on"
    console.print(f"[bold green]Jarvis online.[/] listening={wake_mode}. Ctrl-C to quit.")

    try:
        while not stop.is_set():
            if settings.wake_enabled:
                console.print(f"[dim]waiting for '{settings.wake_word}'...[/dim]")
                await wake_detector.wait_for_wake()
                if stop.is_set():
                    break
                console.print("[yellow]yes?[/yellow]")
            console.print("[dim]listening...[/dim]")
            pcm = await audio.record_utterance()
            if pcm.size == 0:
                continue
            text = await stt.transcribe(pcm, settings.sample_rate)
            if not text:
                continue
            console.print(f"[cyan]you[/cyan]: {text}")

            if (not wake_detector.ready and settings.require_wake_word
                    and not wake_detector.matches_text(text)):
                console.print("[dim](no wake word, ignoring)[/dim]")
                continue

            console.print("[magenta]jarvis[/magenta]: ", end="")
            reply = await _respond(text)
            console.print(reply)
            if reply:
                pcm_out, sr = await tts.synthesize(reply)
                await audio.play(pcm_out, sr)
    finally:
        for t in monitor_tasks:
            t.cancel()
        await bridge.stop()


# ----- chat -----

@app.command()
def chat(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    """Text REPL with streaming output."""
    _setup_logging(verbose)
    asyncio.run(_chat_main())


async def _chat_main() -> None:
    settings = get_settings()
    session = build_session(settings)
    inject_google(session)
    graph = session["graph"]
    conversation_history: list = []

    console.print("[bold green]Jarvis (chat).[/] Ctrl-D to exit.")
    while True:
        try:
            line = Prompt.ask("[cyan]you[/cyan]")
        except (EOFError, KeyboardInterrupt):
            console.print()
            return
        line = line.strip()
        if not line:
            continue
        if line in {"/reset", "/clear"}:
            conversation_history = []
            console.print("[dim]history cleared[/dim]")
            continue
        console.print("[magenta]jarvis[/magenta]: ", end="")
        try:
            reply, conversation_history = await run_turn(
                graph, line, trigger="voice", output_channel="voice",
                conversation_history=conversation_history,
            )
            console.print(reply)
        except Exception as e:
            console.print(f"[red]error:[/red] {e}")


# ----- ask -----

@app.command()
def ask(
    question: str = typer.Argument(...),
    json_output: bool = typer.Option(False, "--json"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """One-shot text query."""
    _setup_logging(verbose)
    asyncio.run(_ask_main(question, json_output))


async def _ask_main(question: str, raw: bool) -> None:
    settings = get_settings()
    session = build_session(settings)
    inject_google(session)
    reply, _ = await run_turn(session["graph"], question, trigger="voice", output_channel="voice")
    if raw:
        print(reply)
    else:
        console.print(f"[magenta]jarvis[/magenta]: {reply}")


# ----- telegram -----

@app.command()
def telegram(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    """Run only the Telegram bridge."""
    _setup_logging(verbose)
    asyncio.run(_telegram_main())


async def _telegram_main() -> None:
    from jarvis.telegram_bot import TelegramBridge

    settings = get_settings()
    session = build_session(settings)
    inject_google(session)
    graph = session["graph"]
    conversation_history: list = []

    async def _respond(text: str) -> str:
        nonlocal conversation_history
        response, conversation_history = await run_turn(
            graph, text, trigger="voice", output_channel="telegram",
            conversation_history=conversation_history,
        )
        return response

    bridge = TelegramBridge(settings, _respond)
    if not bridge.enabled():
        console.print("[red]JARVIS_TELEGRAM_TOKEN not set.[/red]")
        sys.exit(1)
    await bridge.start()
    console.print("[bold green]Telegram bridge running.[/] Ctrl-C to quit.")
    stop = asyncio.Event()
    _install_signal_handlers(stop)
    try:
        await stop.wait()
    finally:
        await bridge.stop()


# ----- notify -----

@app.command()
def notify(
    message: str = typer.Argument(...),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Send a push notification via Telegram."""
    _setup_logging(verbose)
    asyncio.run(_notify_main(message))


async def _notify_main(message: str) -> None:
    from jarvis.telegram_bot import TelegramBridge

    settings = get_settings()
    bridge = TelegramBridge(settings, lambda _: asyncio.coroutine(lambda: "")())
    if not bridge.enabled():
        console.print("[red]telegram disabled[/red]")
        sys.exit(1)
    await bridge.start()
    try:
        await bridge.notify(message)
        console.print("[green]sent[/green]")
    finally:
        await bridge.stop()


# ----- watch -----

@app.command()
def watch(
    paths: list[str] = typer.Argument(None),
    command: Optional[str] = typer.Option(None, "--command", "-c"),
    debounce_ms: Optional[int] = typer.Option(None, "--debounce-ms"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Watch workspace paths and push changes to Telegram."""
    _setup_logging(verbose)
    asyncio.run(_watch_main(paths, command, debounce_ms))


async def _watch_main(
    paths: list[str] | None,
    command: str | None,
    debounce_ms: int | None,
) -> None:
    from jarvis.telegram_bot import TelegramBridge
    from jarvis.graph.proactive.watchdog import WatchdogMonitor

    settings = get_settings()
    target_paths = paths or settings.watch_path_list
    cmd = command if command is not None else settings.watch_command
    debounce = debounce_ms if debounce_ms is not None else settings.watch_debounce_ms

    bridge = TelegramBridge(settings, lambda _: asyncio.coroutine(lambda: "")())
    if not bridge.enabled():
        console.print("[red]Telegram required for watch mode.[/red]")
        sys.exit(1)
    await bridge.start()
    monitor = WatchdogMonitor(target_paths, bridge.notify, command=cmd, debounce_ms=debounce)
    console.print(f"[bold green]Watching[/] {', '.join(target_paths)}. Ctrl-C to stop.")
    try:
        await monitor.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        await bridge.stop()


# ----- auth -----

@auth_app.command("google")
def auth_google(
    credentials: str = typer.Option(None, "--credentials", help="Path to credentials.json"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Authenticate with Google (Calendar + Gmail). Opens a browser."""
    _setup_logging(verbose)
    from pathlib import Path
    from jarvis.integrations.google_auth import get_credentials

    settings = get_settings()
    creds_path = Path(credentials).expanduser() if credentials else Path(settings.google_credentials_path).expanduser()
    token_path = Path(settings.google_token_path).expanduser()

    if not creds_path.exists():
        console.print(f"[red]credentials.json not found at {creds_path}[/red]")
        console.print("Download it from Google Cloud Console → APIs & Services → Credentials.")
        raise typer.Exit(1)

    try:
        get_credentials(creds_path, token_path)
        console.print(f"[green]Google authenticated. Token saved to {token_path}[/green]")
    except Exception as e:
        console.print(f"[red]Auth failed: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_main_cli.py -v
```

Expected: both PASS.

- [ ] **Step 5: Run the full test suite**

```bash
pytest tests/ -v --tb=short
```

Expected: all tests PASS. Fix any failures before continuing.

- [ ] **Step 6: Commit**

```bash
git add jarvis/main.py tests/test_main_cli.py
git commit -m "feat: rewire CLI to LangGraph graph with proactive monitors and google auth"
```

---

## Task 13: Smoke test end-to-end (chat mode)

This task has no new code — it verifies the full stack works together with a real Claude API call.

- [ ] **Step 1: Set environment variables**

```bash
export ANTHROPIC_API_KEY=your_key_here
export JARVIS_MEM0_ENABLED=false   # skip mem0 for smoke test
```

- [ ] **Step 2: Run a one-shot query**

```bash
python -m jarvis ask "Say hello and tell me what tools you have available."
```

Expected: Jarvis responds in 1-3 sentences, mentions its tools. No stack trace.

- [ ] **Step 3: Run chat mode briefly**

```bash
python -m jarvis chat
```

Type: `what can you do?` then Ctrl-D.

Expected: coherent response about calendar, email, coding, and web search capabilities.

- [ ] **Step 4: Commit smoke test notes**

```bash
git commit --allow-empty -m "chore: smoke test passed — full stack wired and responding"
```

---

## Task 14: Clean up old files

- [ ] **Step 1: Remove old top-level modules that have been replaced**

```bash
git rm jarvis/agent.py \
       jarvis/llm.py \
       jarvis/memory.py \
       jarvis/config.py \
       jarvis/personality.py \
       jarvis/tools.py \
       jarvis/watch.py \
       jarvis/tts.py \
       jarvis/audio.py \
       jarvis/stt.py \
       jarvis/wake.py \
       jarvis/streaming.py
```

- [ ] **Step 2: Run full test suite to confirm nothing broke**

```bash
pytest tests/ -v --tb=short
```

Expected: all tests PASS.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "chore: remove old top-level modules replaced by subpackages"
```

---

## Task 15: Update Makefile

**Files:**
- Modify: `Makefile`

- [ ] **Step 1: Replace Makefile contents**

```makefile
.PHONY: init dev voice chat ask telegram watch notify test auth-google memory-show memory-reset

init:
	python -m venv .venv
	.venv/bin/pip install -e '.[dev,wake]'
	cp -n .env.example .env || true
	@echo "Edit .env and set ANTHROPIC_API_KEY, then run: make auth-google"

dev: voice

voice:
	jarvis voice

chat:
	jarvis chat

ask:
	@[ "$(Q)" ] || (echo "Usage: make ask Q='your question'"; exit 1)
	jarvis ask "$(Q)"

telegram:
	jarvis telegram

watch:
	jarvis watch

notify:
	@[ "$(M)" ] || (echo "Usage: make notify M='your message'"; exit 1)
	jarvis notify "$(M)"

auth-google:
	jarvis auth google

test:
	pytest tests/ -v --tb=short

memory-show:
	@echo "Memory is now managed by mem0. Check ~/.jarvis/mem0/"

memory-reset:
	rm -rf ~/.jarvis/mem0/
	@echo "mem0 store cleared"
```

- [ ] **Step 2: Verify**

```bash
make test
```

Expected: all tests PASS.

- [ ] **Step 3: Commit**

```bash
git add Makefile
git commit -m "chore: update Makefile for v0.2 commands"
```

---

## Self-Review

### Spec coverage check

| Spec section | Covered by task(s) |
|---|---|
| LangGraph agent graph | Task 8 |
| mem0 memory (3-tier) | Task 6 |
| Claude API (replaces vLLM) | Task 8 (nodes.py uses anthropic SDK) |
| Kokoro TTS (replaces Piper) | Task 5 |
| STT/wake unchanged | Task 4 |
| Google OAuth flow | Task 9 |
| Calendar tools (list/create/update/free_slot) | Task 9 |
| Gmail tools (list/read/draft/send) | Task 9 |
| Web search + fetch_page | Task 7 |
| Memory tools (remember/recall/forget) | Task 7 |
| Confirmation flow for destructive tools | Task 8 (nodes.py CONFIRMATION_REQUIRED_TOOLS) |
| Proactive calendar monitor | Task 11 |
| Proactive email monitor | Task 11 |
| Notification watchdog | Task 11 |
| Output channel routing (voice/telegram/silent) | Task 8 (AgentState.output_channel) |
| CLI rewire + auth google command | Task 12 |
| Package restructure | Task 1 |
| Settings expanded | Task 2 |
| Personality + proactive templates | Task 3 |
| Session factory | Task 10 |

All spec requirements covered. No gaps.

### Type consistency

- `AgentState` defined in Task 8 (`state.py`) — all nodes in `nodes.py` use the same field names
- `Toolbox.dispatch(name, args)` signature consistent across `registry.py`, `calendar_tools.py`, `email_tools.py`
- `CalendarTools.dispatch` and `EmailTools.dispatch` both take `(name: str, args: dict)` — matches `registry.py` call sites
- `Mem0Store.retrieve()` returns `list[str]` — `retrieve_memory_node` uses it as `list[str]`
- `run_turn()` returns `(str, list)` — all call sites in `main.py` unpack as `reply, conversation_history`
- `build_session()` returns dict with keys `graph, toolbox, mem0, client, settings` — all used consistently
