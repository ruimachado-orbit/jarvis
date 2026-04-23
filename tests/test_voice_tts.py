from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from jarvis.core.config import Settings
from jarvis.voice.tts import TTS


@pytest.fixture
def mock_csm_generator():
    with patch("jarvis.voice.tts._load_csm_generator") as mock_load:
        instance = MagicMock()
        instance.sample_rate = 24000
        instance.generate.return_value = np.zeros(16000, dtype=np.float32)
        mock_load.return_value = instance
        yield mock_load


def test_tts_synthesize(mock_csm_generator):
    settings = Settings(_env_file=None)
    tts = TTS(settings)
    import asyncio
    pcm, sr = asyncio.run(tts.synthesize("Hello Jarvis."))
    assert isinstance(pcm, np.ndarray)
    assert sr == 24000
    assert pcm.shape[0] > 0


def test_tts_empty_string(mock_csm_generator):
    settings = Settings(_env_file=None)
    tts = TTS(settings)
    import asyncio
    pcm, sr = asyncio.run(tts.synthesize(""))
    assert pcm.shape[0] == 0


def test_tts_speaker_passed(mock_csm_generator):
    settings = Settings(_env_file=None, tts_voice="2")
    tts = TTS(settings)
    import asyncio
    asyncio.run(tts.synthesize("Test."))
    mock_csm_generator.return_value.generate.assert_called_once()
    call_kwargs = mock_csm_generator.return_value.generate.call_args
    assert call_kwargs.kwargs.get("speaker") == 2
