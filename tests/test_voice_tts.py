import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from jarvis.core.config import Settings
from jarvis.voice.tts import TTS


@pytest.fixture
def mock_kokoro():
    with patch("jarvis.voice.tts.KokoroTTS") as mock_cls:
        instance = MagicMock()
        instance.create.return_value = (np.zeros(16000, dtype=np.float32), 24000)
        mock_cls.return_value = instance
        yield mock_cls


def test_tts_synthesize(mock_kokoro):
    settings = Settings(_env_file=None)
    tts = TTS(settings)
    import asyncio
    pcm, sr = asyncio.run(tts.synthesize("Hello Jarvis."))
    assert isinstance(pcm, np.ndarray)
    assert sr == 24000
    assert pcm.shape[0] > 0


def test_tts_empty_string(mock_kokoro):
    settings = Settings(_env_file=None)
    tts = TTS(settings)
    import asyncio
    pcm, sr = asyncio.run(tts.synthesize(""))
    assert pcm.shape[0] == 0


def test_tts_voice_name_passed(mock_kokoro):
    settings = Settings(_env_file=None, tts_voice="af_sky")
    tts = TTS(settings)
    import asyncio
    asyncio.run(tts.synthesize("Test."))
    mock_kokoro.return_value.create.assert_called_once()
    call_args = mock_kokoro.return_value.create.call_args
    assert "af_sky" in str(call_args)
