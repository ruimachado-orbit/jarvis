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


def _pick_input_device() -> int | None:
    """Return the best available input device index, or None for system default."""
    import sounddevice as sd
    devices = sd.query_devices()
    default_in = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else sd.default.device

    # Priority: default first, then any device with input channels
    # Test each by sampling 0.1s and checking RMS > noise floor
    candidates = [default_in] + [
        i for i, d in enumerate(devices)
        if d["max_input_channels"] > 0 and i != default_in
    ]

    import queue as _queue
    for dev_id in candidates:
        try:
            q: _queue.Queue = _queue.Queue()
            sr = 16000
            blocksize = 480

            def _cb(indata, *_):
                q.put(indata[:, 0].copy())

            with sd.InputStream(samplerate=sr, channels=1, dtype="int16",
                                blocksize=blocksize, device=dev_id, callback=_cb):
                frames = [q.get() for _ in range(6)]  # ~0.18s

            import numpy as np
            pcm = np.concatenate(frames).astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(pcm ** 2)))
            log.debug("device [%d] RMS=%.4f", dev_id, rms)
            if rms > 0.0001:  # not completely silent
                log.info("auto-selected input device [%d]: %s", dev_id, devices[dev_id]["name"])
                return dev_id
        except Exception:
            continue

    log.warning("no working input device found, using system default")
    return None


class AudioIO:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._playback_lock = asyncio.Lock()
        self._input_device: int | None = None  # resolved at first use

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

        # Use system default device — no hardcoding
        dev = None

        # Calibrate noise floor: sample 0.5s of ambient audio to set adaptive thresholds
        noise_rms = 0.002
        try:
            _cal_q: queue.Queue[np.ndarray] = queue.Queue()
            def _cal_cb(indata, *_): _cal_q.put(indata[:, 0].copy())
            cal_frames: list[np.ndarray] = []
            with sd.InputStream(samplerate=sr, channels=1, dtype="int16",
                                blocksize=frame_samples, device=dev, callback=_cal_cb):
                for _ in range(int(0.5 * sr / frame_samples)):
                    try:
                        cal_frames.append(_cal_q.get(timeout=1.0))
                    except Exception:
                        break
            if cal_frames:
                cal = np.concatenate(cal_frames).astype(np.float32) / 32768.0
                noise_rms = float(np.sqrt(np.mean(cal ** 2)))
                log.debug("noise floor rms=%.4f", noise_rms)
        except Exception as e:
            log.debug("noise calibration failed: %s", e)

        # Speech threshold: 4× noise floor, minimum 0.003
        SPEECH_RMS = max(0.003, noise_rms * 4.0)
        SILENCE_RMS = max(0.002, noise_rms * 2.0)
        log.debug("thresholds: speech=%.4f silence=%.4f", SPEECH_RMS, SILENCE_RMS)

        with sd.InputStream(
            samplerate=sr,
            channels=1,
            dtype="int16",
            blocksize=frame_samples,
            device=dev,
            callback=_cb,
        ):
            loop = asyncio.get_running_loop()
            while True:
                frame = await loop.run_in_executor(None, q.get)
                if len(frame) < frame_samples:
                    continue
                frame = frame[:frame_samples]
                frame_rms = float(np.sqrt(np.mean((frame.astype(np.float32) / 32768.0) ** 2)))

                try:
                    is_speech = vad.is_speech(frame.tobytes(), sr) or frame_rms > SPEECH_RMS
                except Exception:
                    is_speech = frame_rms > SPEECH_RMS

                is_silence = frame_rms < SILENCE_RMS

                if triggered:
                    collected.append(frame)
                    total_frames += 1
                    trailing_silence = 0 if is_speech else trailing_silence + 1
                    if trailing_silence >= silence_frames and total_frames >= min_frames:
                        break
                    if total_frames >= max_frames:
                        break
                else:
                    if is_speech:
                        triggered = True
                        collected.append(frame)
                        total_frames += 1

        return np.concatenate(collected) if collected else np.zeros(0, dtype=np.int16)

    # ----- playback -----

    async def play_boot_sound(self) -> None:
        """Play Iron Man HUD power-on chime."""
        import sounddevice as sd

        sr = 44100

        def _tone(freq, dur, vol=0.4, fade=0.02):
            t = np.linspace(0, dur, int(sr * dur), False)
            wave = (np.sin(2 * np.pi * freq * t) + 0.15 * np.sin(4 * np.pi * freq * t)) * vol
            fade_s = int(sr * fade)
            if fade_s > 0:
                wave[:fade_s] *= np.linspace(0, 1, fade_s)
                wave[-fade_s:] *= np.linspace(1, 0, fade_s)
            return wave

        def _silence(dur):
            return np.zeros(int(sr * dur))

        crackle = np.random.randn(int(sr * 0.08)) * 0.1
        hum = np.concatenate([_tone(80 + i * 8, 0.03, 0.15) for i in range(20)])
        blips = np.concatenate([
            _tone(1200, 0.05), _silence(0.03),
            _tone(1500, 0.05), _silence(0.03),
            _tone(1800, 0.07), _silence(0.02),
        ])
        chord = (
            _tone(330, 0.6, 0.3) +
            _tone(415, 0.6, 0.25) +
            _tone(494, 0.6, 0.2) +
            _tone(659, 0.6, 0.15)
        )
        ping = _tone(2093, 0.4, 0.25, fade=0.05)
        boot = np.concatenate([crackle, hum, _silence(0.05), blips, chord * 0.7, _silence(0.02), ping])
        boot = np.clip(boot, -1, 1).astype(np.float32)

        await asyncio.to_thread(lambda: (sd.play(boot, sr, device=self._output_device()), sd.wait()))

    async def play_sleep_sound(self) -> None:
        """Play power-down chime."""
        import sounddevice as sd

        sr = 44100

        def _tone(freq, dur, vol=0.3, fade=0.02):
            t = np.linspace(0, dur, int(sr * dur), False)
            wave = (np.sin(2 * np.pi * freq * t) + 0.1 * np.sin(4 * np.pi * freq * t)) * vol
            fade_s = int(sr * fade)
            if fade_s > 0:
                wave[:fade_s] *= np.linspace(0, 1, fade_s)
                wave[-fade_s:] *= np.linspace(1, 0, fade_s)
            return wave

        def _silence(dur):
            return np.zeros(int(sr * dur))

        # descending tones + fade out hum
        down = np.concatenate([
            _tone(659, 0.08), _silence(0.02),
            _tone(494, 0.08), _silence(0.02),
            _tone(330, 0.08), _silence(0.02),
            _tone(220, 0.12), _silence(0.02),
        ])
        hum_down = np.concatenate([_tone(80 - i * 3, 0.03, 0.12) for i in range(15)])
        sleep_snd = np.concatenate([down, hum_down])
        sleep_snd = np.clip(sleep_snd, -1, 1).astype(np.float32)

        await asyncio.to_thread(lambda: (sd.play(sleep_snd, sr, device=self._output_device()), sd.wait()))

    def stop(self) -> None:
        """Stop current playback immediately."""
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass

    def _output_device(self):
        dev = self.settings.output_device or None
        if dev and str(dev).isdigit():
            return int(dev)
        return dev

    async def play(self, pcm: np.ndarray, sample_rate: int) -> None:
        if pcm.size == 0:
            return
        import sounddevice as sd

        async with self._playback_lock:
            def _run() -> None:
                sd.play(
                    pcm,
                    samplerate=sample_rate,
                    device=self._output_device(),
                    blocking=True,
                )
                sd.wait()

            await asyncio.to_thread(_run)

    async def play_queue(self, clips: AsyncIterator[tuple[np.ndarray, int]]) -> None:
        """Play clips serially so sentences come out in order."""
        async for pcm, sr in clips:
            await self.play(pcm, sr)
