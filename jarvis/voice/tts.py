"""Sesame CSM 1B TTS — state-of-the-art conversational speech.

Replaces Kokoro ONNX. CSM (Conversational Speech Model) from Sesame Labs
produces natural, prosodic, conversational TTS with human-like turn-taking cues.

Requirements:
- transformers >= 4.52.1
- torch + torchaudio
- Access to HuggingFace models: sesame/csm-1b, meta-llama/Llama-3.2-1B
  (run: huggingface-cli login)

Usage:
    export HF_TOKEN=hf_...   # get from https://huggingface.co/settings/tokens
"""

from __future__ import annotations

import asyncio
import logging
from functools import lru_cache

import numpy as np

from jarvis.core.config import Settings

log = logging.getLogger(__name__)

_CSM_AVAILABLE = False
_GENERATOR = None
_SAMPLE_RATE = 24000


def _check_csm() -> bool:
    global _CSM_AVAILABLE
    if _CSM_AVAILABLE:
        return True
    try:
        import torch  # noqa: F401
        import torchaudio  # noqa: F401
        import transformers  # noqa: F401
        _CSM_AVAILABLE = True
        return True
    except ImportError:
        log.warning("CSM dependencies not installed: pip install 'transformers[torch]' torch torchaudio")
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
    global _GENERATOR, _SAMPLE_RATE
    if not _check_csm():
        raise RuntimeError("CSM dependencies not available")

    from jarvis.voice.csm_generator import load_csm_1b
    device = _get_device()
    log.info("loading CSM 1B on device=%s (first run downloads ~5GB from HuggingFace)", device)
    gen = load_csm_1b(device=device)
    _SAMPLE_RATE = gen.sample_rate
    return gen


class TTS:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._generator = None
        self._speaker = int(settings.tts_voice) if settings.tts_voice.isdigit() else 0
        self._max_audio_ms = settings.tts_max_audio_ms
        self._speed = settings.tts_speed

    def _ensure_generator(self):
        if self._generator is None:
            self._generator = _load_csm_generator()
        return self._generator

    async def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        text = text.strip()
        if not text:
            return np.zeros(0, dtype=np.float32), _SAMPLE_RATE

        loop = asyncio.get_running_loop()

        def fn():
            return self._synthesize_sync(text)

        return await loop.run_in_executor(None, fn)

    def _synthesize_sync(self, text: str) -> tuple[np.ndarray, int]:
        gen = self._ensure_generator()
        audio = gen.generate(
            text=text,
            speaker=self._speaker,
            context=[],
            max_audio_length_ms=self._max_audio_ms,
        )
        arr = audio.squeeze(0) if audio.ndim > 1 else audio
        if hasattr(arr, "cpu"):
            pcm = arr.cpu().numpy().astype(np.float32)
        else:
            pcm = np.asarray(arr, dtype=np.float32)
        if self._speed != 1.0:
            import torch
            import torchaudio
            wav = torch.from_numpy(pcm).unsqueeze(0)
            wav = torchaudio.functional.speed(wav, _SAMPLE_RATE, 1.0 / self._speed)
            pcm = wav.squeeze(0).cpu().numpy().astype(np.float32)
        return pcm, _SAMPLE_RATE
