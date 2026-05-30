"""Tests for Voxtral real-time STT."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from jarvis.core.config import Settings
from jarvis.voice.stt import STT


@pytest.fixture
def mock_voxtral(monkeypatch):
    """Stub _load_voxtral → (processor, model, device)."""
    for k in ("JARVIS_STT_LANGUAGE",):
        monkeypatch.delenv(k, raising=False)

    with patch("jarvis.voice.stt._load_voxtral") as mock_load:
        import torch

        processor = MagicMock()

        # processor(audio, ...) returns dict-like with .to(device)
        inputs = MagicMock()
        inputs.to.return_value = {"input_features": torch.zeros(1, 80, 100)}
        processor.return_value = inputs

        # Batch decode for STT
        processor.batch_decode.return_value = ["Hello Jarvis"]

        model = MagicMock()
        # model.generate returns token IDs for STT
        model.generate.return_value = torch.zeros(1, 10, dtype=torch.long)

        mock_load.return_value = (processor, model, "cpu")
        yield mock_load, processor, model


def test_stt_transcribe(mock_voxtral):
    """Test STT transcription."""
    _, processor, model = mock_voxtral

    settings = Settings(_env_file=None)
    stt = STT(settings)

    # Create fake audio (1 second at 16kHz)
    audio = np.random.randint(-1000, 1000, 16000, dtype=np.int16)

    import asyncio
    text = asyncio.run(stt.transcribe(audio, 16000))

    assert isinstance(text, str)
    assert text == "Hello Jarvis"


def test_stt_low_energy_rejection(mock_voxtral):
    """Test that very quiet audio is rejected."""
    settings = Settings(_env_file=None)
    stt = STT(settings)

    # Create near-silent audio
    audio = np.random.randint(-10, 10, 16000, dtype=np.int16)

    import asyncio
    text = asyncio.run(stt.transcribe(audio, 16000))

    assert text == ""


def test_stt_resampling(mock_voxtral):
    """Test audio resampling to 16kHz."""
    _, processor, model = mock_voxtral

    settings = Settings(_env_file=None)
    stt = STT(settings)

    # Create audio at different sample rate (48kHz)
    audio_48k = np.random.randint(-1000, 1000, 48000, dtype=np.int16)

    import asyncio
    text = asyncio.run(stt.transcribe(audio_48k, 48000))

    # Should resample internally and still work
    assert isinstance(text, str)
