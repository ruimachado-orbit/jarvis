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
        self._kokoro: KokoroTTS | None = None
        if _KOKORO_AVAILABLE:
            import os
            from pathlib import Path
            # Look for model files relative to project root or in kokoro/ subdir
            base = Path(__file__).parent.parent.parent
            model_path = base / "kokoro" / "kokoro-v1.0.onnx"
            voices_path = base / "kokoro" / "voices-v1.0.bin"
            self._kokoro = KokoroTTS(str(model_path), str(voices_path))

    async def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Return (pcm_float32, sample_rate). Empty array if text is blank."""
        text = text.strip()
        if not text or self._kokoro is None:
            return np.zeros(0, dtype=np.float32), 24000

        loop = asyncio.get_running_loop()
        fn = partial(self._synthesize_sync, text)
        return await loop.run_in_executor(None, fn)

    def _synthesize_sync(self, text: str) -> tuple[np.ndarray, int]:
        samples, sr = self._kokoro.create(text, voice=self._voice, speed=self._speed, lang="en-gb")
        return samples.astype(np.float32), sr
