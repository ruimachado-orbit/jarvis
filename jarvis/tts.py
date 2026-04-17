"""Text-to-speech using Piper. Sentence-level synthesis keeps latency low."""

from __future__ import annotations

import asyncio
import logging
import wave
from functools import lru_cache
from io import BytesIO

import numpy as np

from jarvis.config import Settings

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_voice(voice_path: str):
    from piper import PiperVoice  # type: ignore[import-not-found]

    log.info("loading piper voice=%s", voice_path)
    return PiperVoice.load(voice_path)


class TTS:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _voice(self):
        return _load_voice(str(self.settings.tts_voice))

    async def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Return (int16 PCM mono, sample_rate) for the sentence."""
        text = text.strip()
        if not text:
            return np.zeros(0, dtype=np.int16), 22050

        def _run() -> tuple[np.ndarray, int]:
            voice = self._voice()
            buf = BytesIO()
            with wave.open(buf, "wb") as wav:
                voice.synthesize(text, wav, length_scale=1.0 / max(self.settings.tts_speed, 0.1))
            buf.seek(0)
            with wave.open(buf, "rb") as wav:
                rate = wav.getframerate()
                n = wav.getnframes()
                raw = wav.readframes(n)
            pcm = np.frombuffer(raw, dtype=np.int16)
            return pcm, rate

        return await asyncio.to_thread(_run)
