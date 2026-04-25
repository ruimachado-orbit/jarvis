"""TTS engines: Sesame CSM 1B (preferred) and Kokoro ONNX (fallback).

Engine selected by ``JARVIS_TTS_ENGINE``:
- ``csm``    → ``transformers.CsmForConditionalGeneration`` (needs HF gated
                 access to ``sesame/csm-1b`` + ``meta-llama/Llama-3.2-1B``).
- ``kokoro`` → ``kokoro-onnx`` with local model files under ``kokoro/``.
                 No network, no gated repos.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np

from jarvis.core.config import Settings

log = logging.getLogger(__name__)

_SAMPLE_RATE = 24000  # Both CSM and Kokoro emit 24 kHz
_PHRASE_CACHE_DIR = Path.home() / ".cache" / "jarvis" / "tts"


def _check_csm() -> bool:
    try:
        import torch  # noqa: F401
        import torchaudio  # noqa: F401
        from transformers import CsmForConditionalGeneration  # noqa: F401
        return True
    except ImportError as e:
        log.warning("CSM dependencies not available (%s). Run: make csm-install", e)
        return False


@lru_cache(maxsize=1)
def _get_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=1)
def _load_csm_generator():
    """Load CSM model + processor once and cache."""
    if not _check_csm():
        raise RuntimeError("CSM dependencies not available")

    import torch
    from transformers import AutoProcessor, CsmForConditionalGeneration

    device = _get_device()
    model_id = "sesame/csm-1b"
    log.info("loading CSM 1B from %s on device=%s (first run downloads ~5GB)", model_id, device)

    processor = AutoProcessor.from_pretrained(model_id)
    # bf16 on MPS/CUDA, fp32 on CPU — MPS doesn't love fp32 perf but works.
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    model = CsmForConditionalGeneration.from_pretrained(model_id, dtype=dtype).to(device)
    model.eval()
    log.info("CSM 1B ready on %s", device)
    return processor, model, device


@lru_cache(maxsize=1)
def _load_kokoro():
    """Load Kokoro ONNX model + voices file from ./kokoro/."""
    from kokoro_onnx import Kokoro
    base = Path(__file__).resolve().parent.parent.parent
    model_path = base / "kokoro" / "kokoro-v1.0.onnx"
    voices_path = base / "kokoro" / "voices-v1.0.bin"
    if not model_path.exists() or not voices_path.exists():
        raise RuntimeError(
            f"Kokoro model files missing. Expected:\n  {model_path}\n  {voices_path}"
        )
    log.info("loading Kokoro ONNX from %s", model_path)
    return Kokoro(str(model_path), str(voices_path))


class TTS:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._engine = (settings.tts_engine or "csm").lower()
        self._voice = settings.tts_voice
        self._speed = settings.tts_speed
        # CSM-only: map ms → max_new_tokens (1 token ≈ 80 samples at 24 kHz).
        self._max_new_tokens = max(32, settings.tts_max_audio_ms // 80)
        self._csm_loaded = False

    def _ensure_csm(self):
        if not self._csm_loaded:
            self._processor, self._model, self._device = _load_csm_generator()
            self._csm_loaded = True
        return self._processor, self._model, self._device

    async def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        text = text.strip()
        if not text:
            return np.zeros(0, dtype=np.float32), _SAMPLE_RATE

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._synthesize_sync, text)

    def _phrase_cache_path(self, phrase: str) -> Path:
        # Key over engine/voice/speed so a config change invalidates the cache.
        key = f"{self._engine}|{self._voice}|{self._speed}|{phrase}"
        h = hashlib.sha1(key.encode()).hexdigest()[:16]
        return _PHRASE_CACHE_DIR / f"{h}.npz"

    async def prewarm(self, phrases: list[str]) -> dict[str, tuple[np.ndarray, int]]:
        """Synthesise fixed phrases up-front, caching PCM on disk.

        Returns a dict mapping each phrase → (pcm, sr) that callers can use
        to skip TTS synthesis at runtime. First run pays the one-time synth
        cost per phrase; subsequent runs load the cached .npz files instantly.
        """
        _PHRASE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache: dict[str, tuple[np.ndarray, int]] = {}
        for phrase in phrases:
            if not phrase.strip():
                continue
            f = self._phrase_cache_path(phrase)
            if f.exists():
                try:
                    data = np.load(f)
                    cache[phrase] = (data["pcm"].astype(np.float32), int(data["sr"]))
                    continue
                except Exception as e:
                    log.warning("failed to load %s, resynthesising: %s", f.name, e)
            pcm, sr = await self.synthesize(phrase)
            cache[phrase] = (pcm, sr)
            try:
                np.savez_compressed(f, pcm=pcm, sr=np.int32(sr))
            except Exception as e:
                log.warning("failed to cache %s: %s", f.name, e)
        return cache

    def _synthesize_sync(self, text: str) -> tuple[np.ndarray, int]:
        if self._engine == "kokoro":
            return self._synthesize_kokoro(text)
        return self._synthesize_csm(text)

    def _synthesize_kokoro(self, text: str) -> tuple[np.ndarray, int]:
        kokoro = _load_kokoro()
        voice = self._voice if self._voice and not self._voice.isdigit() else "af_heart"
        samples, sr = kokoro.create(text, voice=voice, speed=self._speed, lang="en-gb")
        return samples.astype(np.float32), sr

    def _synthesize_csm(self, text: str) -> tuple[np.ndarray, int]:
        import torch

        processor, model, device = self._ensure_csm()
        speaker = str(int(self._voice)) if self._voice.isdigit() else "0"

        conversation = [
            {"role": speaker, "content": [{"type": "text", "text": text}]},
        ]
        inputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
        ).to(device)

        with torch.inference_mode():
            audio = model.generate(
                **inputs,
                output_audio=True,
                max_new_tokens=self._max_new_tokens,
            )

        wav = audio[0] if isinstance(audio, (list, tuple)) else audio
        if wav.ndim > 1:
            wav = wav.squeeze(0)
        pcm = wav.detach().to("cpu", dtype=torch.float32).numpy()

        if self._speed != 1.0:
            import torchaudio
            t = torch.from_numpy(pcm).unsqueeze(0)
            t = torchaudio.functional.speed(t, _SAMPLE_RATE, 1.0 / self._speed)[0]
            pcm = t.squeeze(0).cpu().numpy().astype(np.float32)

        return pcm.astype(np.float32), _SAMPLE_RATE
