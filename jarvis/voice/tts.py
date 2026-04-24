"""Sesame CSM 1B TTS — state-of-the-art conversational speech.

Uses the native ``transformers`` implementation (``CsmForConditionalGeneration``)
so we don't have to vendor the CSM repo or depend on ``torchtune`` / ``moshi`` /
``silentcipher``.

Requirements:
- transformers >= 4.52.1 (ships the Csm* classes)
- torch + torchaudio
- Access to HuggingFace model sesame/csm-1b (``huggingface-cli login`` or ``HF_TOKEN``).
"""

from __future__ import annotations

import asyncio
import logging
from functools import lru_cache

import numpy as np

from jarvis.core.config import Settings

log = logging.getLogger(__name__)

_SAMPLE_RATE = 24000  # CSM generates at 24 kHz


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


class TTS:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._loaded = False
        self._speaker = str(int(settings.tts_voice)) if settings.tts_voice.isdigit() else "0"
        # max_new_tokens ≈ frames of 1920 samples each at 24kHz; 780 tokens ≈ 60s.
        # Using tts_max_audio_ms: 24000 * ms/1000 / 1920 ≈ ms/80.
        self._max_new_tokens = max(32, settings.tts_max_audio_ms // 80)
        self._speed = settings.tts_speed

    def _ensure_loaded(self):
        if not self._loaded:
            self._processor, self._model, self._device = _load_csm_generator()
            self._loaded = True
        return self._processor, self._model, self._device

    async def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        text = text.strip()
        if not text:
            return np.zeros(0, dtype=np.float32), _SAMPLE_RATE

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._synthesize_sync, text)

    def _synthesize_sync(self, text: str) -> tuple[np.ndarray, int]:
        import torch

        processor, model, device = self._ensure_loaded()

        conversation = [
            {"role": self._speaker, "content": [{"type": "text", "text": text}]},
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

        # ``output_audio=True`` returns a list (batch) of 1-D float tensors at 24 kHz.
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
