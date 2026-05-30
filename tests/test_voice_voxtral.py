"""Tests for Voxtral unified STT+TTS engine."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from jarvis.core.config import Settings
from jarvis.voice.voxtral import VoxtralEngine


@pytest.fixture
def mock_voxtral(monkeypatch):
    """Stub _load_voxtral → (processor, model, device)."""
    # Clear any env overrides so Settings gets defaults
    for k in ("JARVIS_TTS_SPEED", "JARVIS_STT_LANGUAGE"):
        monkeypatch.delenv(k, raising=False)

    with patch("jarvis.voice.voxtral._load_voxtral") as mock_load:
        import torch

        processor = MagicMock()

        # STT: processor(audio, ...) returns dict-like with .to(device)
        stt_inputs = MagicMock()
        stt_inputs.to.return_value = {"input_features": torch.zeros(1, 80, 100)}
        processor.return_value = stt_inputs

        # TTS: processor(text=...) returns dict-like with .to(device)
        tts_inputs = MagicMock()
        tts_inputs.to.return_value = {"input_ids": torch.zeros(1, 10, dtype=torch.long)}

        # Configure side_effect to return different results based on arguments
        def processor_side_effect(*args, **kwargs):
            if "text" in kwargs:
                return tts_inputs
            else:
                return stt_inputs
        processor.side_effect = processor_side_effect

        # Batch decode for STT
        processor.batch_decode.return_value = ["Hello Jarvis"]

        model = MagicMock()
        # TTS: model.generate returns audio waveform
        model.generate.return_value = torch.zeros(24000, dtype=torch.float32)

        mock_load.return_value = (processor, model, "cpu")
        yield mock_load, processor, model


def test_voxtral_tts_synthesize(mock_voxtral):
    """Test TTS synthesis."""
    settings = Settings(_env_file=None)
    voxtral = VoxtralEngine(settings)
    import asyncio
    pcm, sr = asyncio.run(voxtral.synthesize("Hello Jarvis."))
    assert isinstance(pcm, np.ndarray)
    assert sr == 24000
    assert pcm.shape[0] > 0


def test_voxtral_tts_empty_string(mock_voxtral):
    """Test TTS with empty input."""
    settings = Settings(_env_file=None)
    voxtral = VoxtralEngine(settings)
    import asyncio
    pcm, sr = asyncio.run(voxtral.synthesize(""))
    assert pcm.shape[0] == 0
    assert sr == 24000


def test_voxtral_stt_transcribe(mock_voxtral):
    """Test STT transcription."""
    _, processor, model = mock_voxtral

    # Configure model.generate to return token IDs for STT
    import torch
    model.generate.return_value = torch.zeros(1, 10, dtype=torch.long)

    settings = Settings(_env_file=None)
    voxtral = VoxtralEngine(settings)

    # Create fake audio (1 second at 16kHz)
    audio = np.random.randint(-1000, 1000, 16000, dtype=np.int16)

    import asyncio
    text = asyncio.run(voxtral.transcribe(audio, 16000))

    assert isinstance(text, str)
    assert text == "Hello Jarvis"


def test_voxtral_stt_low_energy_rejection(mock_voxtral):
    """Test that very quiet audio is rejected."""
    settings = Settings(_env_file=None)
    voxtral = VoxtralEngine(settings)

    # Create near-silent audio
    audio = np.random.randint(-10, 10, 16000, dtype=np.int16)

    import asyncio
    text = asyncio.run(voxtral.transcribe(audio, 16000))

    assert text == ""


def test_voxtral_prewarm_caching(mock_voxtral, tmp_path, monkeypatch):
    """Test phrase pre-warming and disk caching."""
    # Override cache dir to temp location
    import jarvis.voice.voxtral as voxtral_module
    monkeypatch.setattr(voxtral_module, "_PHRASE_CACHE_DIR", tmp_path)

    settings = Settings(_env_file=None)
    voxtral = VoxtralEngine(settings)

    phrases = ["Right away, Sir.", "One moment, Sir."]

    import asyncio
    cache = asyncio.run(voxtral.prewarm(phrases))

    # Check cache returned
    assert len(cache) == 2
    for phrase in phrases:
        assert phrase in cache
        pcm, sr = cache[phrase]
        assert isinstance(pcm, np.ndarray)
        assert sr == 24000

    # Check files were written
    cache_files = list(tmp_path.glob("*.npz"))
    assert len(cache_files) == 2
