"""Audio I/O: VAD-gated recording plus queued playback."""

from __future__ import annotations

import asyncio
import logging
import queue
from collections.abc import AsyncIterator

import numpy as np

from jarvis.core.config import Settings

log = logging.getLogger(__name__)

FRAME_MS = 30  # webrtcvad only accepts 10/20/30 ms frames


class AudioIO:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._playback_lock = asyncio.Lock()

    # ----- recording -----

    async def record_utterance(self) -> np.ndarray:
        """Block until the user speaks, then return int16 mono PCM at sample_rate."""
        import sounddevice as sd
        import webrtcvad

        vad = webrtcvad.Vad(self.settings.vad_aggressiveness)
        sr = self.settings.sample_rate
        frame_samples = int(sr * FRAME_MS / 1000)
        silence_frames = max(1, self.settings.silence_ms // FRAME_MS)
        min_frames = max(1, self.settings.min_utterance_ms // FRAME_MS)
        max_frames = max(min_frames, self.settings.max_utterance_ms // FRAME_MS)

        q: queue.Queue[np.ndarray] = queue.Queue()

        def _cb(indata, _frames, _time, status):
            if status:
                log.debug("sounddevice status: %s", status)
            q.put(indata[:, 0].copy())

        collected: list[np.ndarray] = []
        triggered = False
        trailing_silence = 0
        total_frames = 0

        with sd.InputStream(
            samplerate=sr,
            channels=1,
            dtype="int16",
            blocksize=frame_samples,
            device=self.settings.input_device or None,
            callback=_cb,
        ):
            loop = asyncio.get_running_loop()
            while True:
                frame = await loop.run_in_executor(None, q.get)
                if len(frame) < frame_samples:
                    continue
                pcm_bytes = frame[:frame_samples].tobytes()
                is_speech = vad.is_speech(pcm_bytes, sr)

                if triggered:
                    collected.append(frame[:frame_samples])
                    total_frames += 1
                    trailing_silence = 0 if is_speech else trailing_silence + 1
                    if trailing_silence >= silence_frames and total_frames >= min_frames:
                        break
                    if total_frames >= max_frames:
                        break
                else:
                    if is_speech:
                        triggered = True
                        collected.append(frame[:frame_samples])
                        total_frames += 1

        return np.concatenate(collected) if collected else np.zeros(0, dtype=np.int16)

    # ----- playback -----

    async def play(self, pcm: np.ndarray, sample_rate: int) -> None:
        if pcm.size == 0:
            return
        import sounddevice as sd

        async with self._playback_lock:
            def _run() -> None:
                sd.play(
                    pcm,
                    samplerate=sample_rate,
                    device=self.settings.output_device or None,
                    blocking=True,
                )
                sd.wait()

            await asyncio.to_thread(_run)

    async def play_queue(self, clips: AsyncIterator[tuple[np.ndarray, int]]) -> None:
        """Play clips serially so sentences come out in order."""
        async for pcm, sr in clips:
            await self.play(pcm, sr)
