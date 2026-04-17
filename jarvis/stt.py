"""Speech-to-text powered by faster-whisper."""

from __future__ import annotations

import asyncio
import logging
from functools import lru_cache

import numpy as np

from jarvis.config import Settings

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_model(settings_key: tuple[str, str, str]):
    # Imported lazily so the package loads without the model installed.
    from faster_whisper import WhisperModel

    model_name, device, compute_type = settings_key
    log.info("loading faster-whisper model=%s device=%s compute=%s", model_name, device, compute_type)
    return WhisperModel(model_name, device=device, compute_type=compute_type)


class STT:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _model(self):
        return _load_model(
            (self.settings.stt_model, self.settings.stt_device, self.settings.stt_compute_type)
        )

    async def transcribe(self, pcm16: np.ndarray, sample_rate: int) -> str:
        """Transcribe mono int16 PCM audio to text."""
        audio = pcm16.astype(np.float32) / 32768.0
        if sample_rate != 16000:
            # faster-whisper resamples internally when given a numpy array via audio=...
            # but it expects 16kHz; resample with simple linear interpolation for robustness.
            ratio = 16000 / sample_rate
            new_len = int(len(audio) * ratio)
            audio = np.interp(
                np.linspace(0, len(audio), new_len, endpoint=False),
                np.arange(len(audio)),
                audio,
            ).astype(np.float32)

        def _run() -> str:
            segments, _info = self._model().transcribe(
                audio,
                language=self.settings.stt_language,
                vad_filter=False,
                beam_size=1,
            )
            return " ".join(seg.text.strip() for seg in segments).strip()

        return await asyncio.to_thread(_run)
