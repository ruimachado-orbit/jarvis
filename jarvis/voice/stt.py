"""Speech-to-text powered by faster-whisper."""

from __future__ import annotations

import asyncio
import logging
import re
from functools import lru_cache

import numpy as np

from jarvis.core.config import Settings

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

        # reject if audio energy is too low (background noise)
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < 0.001:
            return ""

        def _run() -> str:
            # vad_filter=True → Silero VAD strips non-speech frames before
            # decode, which eliminates "You" / "Okay. Okay." / "Thanks."
            # hallucinations on ambient-noise captures.
            segments, info = self._model().transcribe(
                audio,
                language=self.settings.stt_language,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 300},
                beam_size=5,
                no_speech_threshold=0.6,
                condition_on_previous_text=False,
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
            if _is_whisper_hallucination(text):
                log.debug("stt: dropped likely hallucination %r", text)
                return ""
            return text

        return await asyncio.to_thread(_run)


# Known faster-whisper garbage on silent / near-silent input. If the whole
# transcription is one of these, treat as empty.
_HALLUCINATIONS = {
    "you", "okay.", "thanks.", "thank you.", "thank you for watching.",
    "bye.", ".", "...", "uh", "um",
}


def _is_whisper_hallucination(text: str) -> bool:
    if not text:
        return True
    t = text.lower().strip().rstrip(".").strip()
    # collapse whitespace and trailing filler so "okay. okay. okay." → "okay"
    t = re.sub(r"[\s.]+", " ", t).strip()
    if not t:
        return True
    words = t.split()
    if all(w == words[0] for w in words) and words[0] in {"you", "okay", "yeah", "uh", "um"}:
        return True
    return text.lower().strip() in _HALLUCINATIONS
