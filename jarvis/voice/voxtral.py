"""Unified speech-to-speech engine using Mistral Voxtral-Mini-4B-Realtime.

Voxtral provides:
- Speech-to-text (STT): audio → text transcription
- Text-to-speech (TTS): text → audio synthesis
- Speech-to-speech (S2S): direct audio → audio with low latency

Optimized for Apple Silicon (MPS) with bf16 precision.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from jarvis.core.config import Settings

log = logging.getLogger(__name__)

_SAMPLE_RATE = 24000  # Voxtral native sample rate
_PHRASE_CACHE_DIR = Path.home() / ".cache" / "jarvis" / "voxtral"


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
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

    device = _get_device()
    model_id = "mistralai/Voxtral-Mini-4B-Realtime-2602"

    log.info("Loading Voxtral-Mini-4B from %s on device=%s (first run downloads ~8GB)", model_id, device)

    processor = AutoProcessor.from_pretrained(model_id)

    # Use bf16 on MPS/CUDA for optimal performance, fp32 on CPU
    dtype = torch.bfloat16 if device in ("mps", "cuda") else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)

    model.eval()

    log.info("Voxtral-Mini-4B ready on %s (dtype=%s)", device, dtype)
    return processor, model, device


class VoxtralEngine:
    """Unified STT + TTS engine powered by Voxtral-Mini-4B-Realtime."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._processor = None
        self._model = None
        self._device = None
        self._loaded = False

        # TTS-specific settings
        self._speed = settings.tts_speed
        self._max_new_tokens = max(512, settings.tts_max_audio_ms // 50)  # ~50ms per token

        # STT-specific settings
        self._language = settings.stt_language

    def _ensure_loaded(self):
        """Lazy-load model on first use."""
        if not self._loaded:
            self._processor, self._model, self._device = _load_voxtral()
            self._loaded = True
        return self._processor, self._model, self._device

    # ========== STT (Speech-to-Text) ==========

    async def transcribe(self, pcm16: np.ndarray, sample_rate: int) -> str:
        """Transcribe mono int16 PCM audio to text.

        Args:
            pcm16: Audio as int16 numpy array
            sample_rate: Input sample rate (will be resampled to 24kHz if needed)

        Returns:
            Transcribed text string
        """
        # Convert to float32 [-1, 1]
        audio = pcm16.astype(np.float32) / 32768.0

        # Resample to 24kHz if needed
        if sample_rate != _SAMPLE_RATE:
            import torchaudio
            audio_tensor = torch.from_numpy(audio).unsqueeze(0)
            resampler = torchaudio.transforms.Resample(sample_rate, _SAMPLE_RATE)
            audio_tensor = resampler(audio_tensor)
            audio = audio_tensor.squeeze(0).numpy()

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

        # Prepare input features
        inputs = processor(
            audio,
            sampling_rate=_SAMPLE_RATE,
            return_tensors="pt"
        ).to(device)

        # Generate transcription
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                num_beams=1,  # Greedy for speed
                do_sample=False,
            )

        # Decode text
        transcription = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]

        text = transcription.strip()

        # Filter common Whisper-style hallucinations
        if _is_hallucination(text):
            log.debug("STT: dropped likely hallucination %r", text)
            return ""

        return text

    # ========== TTS (Text-to-Speech) ==========

    async def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Synthesize speech from text.

        Args:
            text: Input text to speak

        Returns:
            (pcm_audio, sample_rate) tuple where pcm_audio is float32 [-1, 1]
        """
        text = text.strip()
        if not text:
            return np.zeros(0, dtype=np.float32), _SAMPLE_RATE

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._synthesize_sync, text)

    def _synthesize_sync(self, text: str) -> tuple[np.ndarray, int]:
        """Synchronous synthesis (runs in thread pool)."""
        processor, model, device = self._ensure_loaded()

        # Tokenize text input
        inputs = processor(
            text=text,
            return_tensors="pt",
            padding=True,
        ).to(device)

        # Generate audio
        with torch.inference_mode():
            audio_values = model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
            )

        # Extract waveform
        if isinstance(audio_values, dict):
            wav = audio_values["audio_values"]
        else:
            wav = audio_values

        # Ensure correct shape and dtype
        if wav.ndim > 1:
            wav = wav.squeeze(0)

        pcm = wav.detach().to("cpu", dtype=torch.float32).numpy()

        # Apply speed adjustment if needed
        if self._speed != 1.0:
            import torchaudio
            t = torch.from_numpy(pcm).unsqueeze(0)
            t = torchaudio.functional.speed(t, _SAMPLE_RATE, 1.0 / self._speed)[0]
            pcm = t.squeeze(0).cpu().numpy().astype(np.float32)

        return pcm.astype(np.float32), _SAMPLE_RATE

    async def prewarm(self, phrases: list[str]) -> dict[str, tuple[np.ndarray, int]]:
        """Pre-synthesize fixed phrases and cache on disk for instant playback.

        Args:
            phrases: List of phrases to pre-generate (e.g., acknowledgements)

        Returns:
            Dict mapping phrase → (pcm, sample_rate) for instant retrieval
        """
        _PHRASE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache: dict[str, tuple[np.ndarray, int]] = {}

        for phrase in phrases:
            if not phrase.strip():
                continue

            cache_path = self._phrase_cache_path(phrase)

            # Try loading from disk cache
            if cache_path.exists():
                try:
                    data = np.load(cache_path)
                    cache[phrase] = (data["pcm"].astype(np.float32), int(data["sr"]))
                    log.debug("TTS cache hit: %s", phrase[:40])
                    continue
                except Exception as e:
                    log.warning("Failed to load cached phrase %s, regenerating: %s", cache_path.name, e)

            # Generate and cache
            log.debug("TTS pre-warming: %s", phrase[:40])
            pcm, sr = await self.synthesize(phrase)
            cache[phrase] = (pcm, sr)

            try:
                np.savez_compressed(cache_path, pcm=pcm, sr=np.int32(sr))
            except Exception as e:
                log.warning("Failed to cache phrase %s: %s", cache_path.name, e)

        return cache

    def _phrase_cache_path(self, phrase: str) -> Path:
        """Generate consistent cache path for a phrase."""
        key = f"voxtral|{self._speed}|{phrase}"
        h = hashlib.sha1(key.encode()).hexdigest()[:16]
        return _PHRASE_CACHE_DIR / f"{h}.npz"


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
