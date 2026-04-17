"""Wake-word detector.

Strategy
--------
* If ``openwakeword`` is installed, run its ``hey_jarvis`` model against a
  continuous 16 kHz mic stream. Latency is low and CPU is cheap.
* If it is not installed, fall back to the STT-based keyword check that the
  voice loop already uses (``JARVIS_REQUIRE_WAKE_WORD``).

The detector is disabled entirely when ``JARVIS_WAKE_ENABLED=false``, turning
Jarvis into an always-on listener.
"""

from __future__ import annotations

import asyncio
import logging
import queue

import numpy as np

from jarvis.config import Settings

log = logging.getLogger(__name__)

FRAME_SAMPLES = 1280  # 80 ms at 16 kHz — required by openwakeword


class WakeWord:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model = None
        self._tried_load = False

    def _load(self) -> bool:
        if self._tried_load:
            return self._model is not None
        self._tried_load = True
        try:
            from openwakeword.model import Model  # type: ignore[import-not-found]
        except ImportError:
            log.info("openwakeword not installed; wake detection falls back to STT match")
            return False
        try:
            self._model = Model(
                wakeword_models=[self.settings.wake_model],
                inference_framework="onnx",
            )
            log.info("wake-word model loaded: %s", self.settings.wake_model)
            return True
        except Exception as e:  # pragma: no cover - model download side effects
            log.warning("failed to load wake model %s: %s", self.settings.wake_model, e)
            self._model = None
            return False

    @property
    def ready(self) -> bool:
        return self._load()

    async def wait_for_wake(self) -> None:
        """Block until the wake phrase is heard."""
        if not self.settings.wake_enabled:
            return
        if not self._load():
            # No audio wake detector available; STT layer will enforce the phrase.
            return

        import sounddevice as sd

        q: queue.Queue[np.ndarray] = queue.Queue()

        def _cb(indata, _frames, _time, status):  # noqa: D401 - sounddevice callback
            if status:
                log.debug("wake stream status: %s", status)
            q.put(indata[:, 0].copy())

        threshold = self.settings.wake_threshold
        log.info("listening for wake phrase '%s' ...", self.settings.wake_word)
        with sd.InputStream(
            samplerate=16000,
            channels=1,
            dtype="int16",
            blocksize=FRAME_SAMPLES,
            device=self.settings.input_device or None,
            callback=_cb,
        ):
            loop = asyncio.get_event_loop()
            while True:
                frame = await loop.run_in_executor(None, q.get)
                if len(frame) < FRAME_SAMPLES:
                    continue
                scores = self._model.predict(frame[:FRAME_SAMPLES])
                hit = max(scores.values()) if scores else 0.0
                if hit >= threshold:
                    log.info("wake detected (score=%.3f)", hit)
                    # Drain the queue so the next VAD read starts fresh.
                    while not q.empty():
                        try:
                            q.get_nowait()
                        except queue.Empty:
                            break
                    return

    def matches_text(self, text: str) -> bool:
        """Fallback STT-based match used when the audio model is unavailable."""
        return self.settings.wake_word.lower() in text.lower()
