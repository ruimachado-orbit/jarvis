"""Real-time speech-to-text using Mistral Voxtral-Mini-4B-Realtime-2602.

Voxtral provides:
- <500ms transcription latency (vs 1-3s for faster-whisper)
- Streaming architecture for real-time ASR
- 13 languages support
- 4B parameters optimized for on-device deployment
- Configurable delay (240ms to 2.4s) for latency/accuracy tradeoff

Optimized for Apple Silicon (MPS) with bf16 precision.
"""

from __future__ import annotations

import asyncio
import logging
import re
from functools import lru_cache

import numpy as np
import torch

from jarvis.core.config import Settings

log = logging.getLogger(__name__)

_VOXTRAL_SAMPLE_RATE = 16000  # Voxtral native sample rate


@lru_cache(maxsize=1)
def _get_device() -> str:
    """Detect best available device: MPS > CUDA > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=1)
def _load_voxtral():
    """Load Voxtral model + processor once and cache globally."""
    from transformers import AutoProcessor, AutoModel

    device = _get_device()
    model_id = "mistralai/Voxtral-Mini-4B-Realtime-2602"

    log.info("Loading Voxtral-Mini-4B from %s on device=%s (first run downloads ~8GB)", model_id, device)

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    # Use bf16 on MPS/CUDA for optimal performance, fp32 on CPU
    dtype = torch.bfloat16 if device in ("mps", "cuda") else torch.float32

    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)

    model.eval()

    log.info("Voxtral-Mini-4B ready on %s (dtype=%s)", device, dtype)
    return processor, model, device


class STT:
    """Real-time speech-to-text powered by Voxtral-Mini-4B-Realtime."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._processor = None
        self._model = None
        self._device = None
        self._loaded = False

    def _ensure_loaded(self):
        """Lazy-load model on first use."""
        if not self._loaded:
            self._processor, self._model, self._device = _load_voxtral()
            self._loaded = True
        return self._processor, self._model, self._device

    async def transcribe(self, pcm16: np.ndarray, sample_rate: int) -> str:
        """Transcribe mono int16 PCM audio to text.

        Args:
            pcm16: Audio as int16 numpy array
            sample_rate: Input sample rate (will be resampled to 16kHz if needed)

        Returns:
            Transcribed text string
        """
        # Convert to float32 [-1, 1]
        audio = pcm16.astype(np.float32) / 32768.0

        # Resample to 16kHz if needed (Voxtral's native rate)
        if sample_rate != _VOXTRAL_SAMPLE_RATE:
            ratio = _VOXTRAL_SAMPLE_RATE / sample_rate
            new_len = int(len(audio) * ratio)
            audio = np.interp(
                np.linspace(0, len(audio), new_len, endpoint=False),
                np.arange(len(audio)),
                audio,
            ).astype(np.float32)

        # Reject if audio energy is too low (background noise)
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < 0.001:
            log.debug("STT: rejected low-energy audio (rms=%.4f)", rms)
            return ""

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._transcribe_sync, audio)

    def _transcribe_sync(self, audio: np.ndarray) -> str:
        """Synchronous transcription (runs in thread pool)."""
        processor, model, device = self._ensure_loaded()

        # Prepare input features using the chat template format
        # Voxtral uses a conversational format for transcription
        inputs = processor(
            audio,
            sampling_rate=_VOXTRAL_SAMPLE_RATE,
            return_tensors="pt"
        )

        # Move inputs to device and convert to the same dtype as the model
        model_dtype = next(model.parameters()).dtype

        if hasattr(inputs, 'to'):
            inputs = inputs.to(device, dtype=model_dtype)
        elif isinstance(inputs, dict):
            inputs = {k: v.to(device, dtype=model_dtype) if isinstance(v, torch.Tensor) else v
                     for k, v in inputs.items()}

        # Generate transcription with Voxtral's recommended settings
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.0,  # Recommended by Voxtral docs
                do_sample=False,
            )

        # Decode text
        if hasattr(processor, 'batch_decode'):
            transcription = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )[0]
        elif hasattr(processor, 'decode'):
            transcription = processor.decode(
                generated_ids[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
        else:
            # Fallback: use tokenizer directly
            transcription = processor.tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

        text = transcription.strip()

        # Filter common hallucinations
        if _is_hallucination(text):
            log.debug("STT: dropped likely hallucination %r", text)
            return ""

        return text


def _is_hallucination(text: str) -> bool:
    """Detect common STT hallucinations on silent/noisy input."""
    if not text:
        return True

    t = text.lower().strip().rstrip(".!?").strip()

    # Empty after cleanup
    if not t:
        return True

    # Known garbage outputs
    HALLUCINATIONS = {
        "you", "okay", "thanks", "thank you", "thank you for watching",
        "bye", "um", "uh", "hmm", "mhm", "yeah",
    }

    if t in HALLUCINATIONS:
        return True

    # Repeated single word (e.g., "okay okay okay")
    words = t.split()
    if len(words) > 1 and all(w == words[0] for w in words) and words[0] in HALLUCINATIONS:
        return True

    return False
