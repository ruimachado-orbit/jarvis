from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from jarvis.core.config import Settings
from jarvis.voice.tts import TTS


@pytest.fixture
def mock_csm():
    """Stub _load_csm_generator → (processor, model, device).

    - processor.apply_chat_template returns a dict-like with .to(device) echoing itself.
    - model.generate returns a list with a single 1-D float tensor shaped [N].
    """
    with patch("jarvis.voice.tts._load_csm_generator") as mock_load:
        import torch

        processor = MagicMock()
        template_out = MagicMock()
        template_out.to.return_value = {"input_ids": torch.zeros(1, 4, dtype=torch.long)}
        processor.apply_chat_template.return_value = template_out

        model = MagicMock()
        model.generate.return_value = [torch.zeros(16000, dtype=torch.float32)]

        mock_load.return_value = (processor, model, "cpu")
        yield mock_load, processor, model


def test_tts_synthesize(mock_csm):
    settings = Settings(_env_file=None)
    tts = TTS(settings)
    import asyncio
    pcm, sr = asyncio.run(tts.synthesize("Hello Jarvis."))
    assert isinstance(pcm, np.ndarray)
    assert sr == 24000
    assert pcm.shape[0] > 0


def test_tts_empty_string(mock_csm):
    settings = Settings(_env_file=None)
    tts = TTS(settings)
    import asyncio
    pcm, sr = asyncio.run(tts.synthesize(""))
    assert pcm.shape[0] == 0


def test_tts_speaker_passed(mock_csm):
    _, processor, _ = mock_csm
    settings = Settings(_env_file=None, tts_voice="2")
    tts = TTS(settings)
    import asyncio
    asyncio.run(tts.synthesize("Test."))
    processor.apply_chat_template.assert_called_once()
    conversation = processor.apply_chat_template.call_args.args[0]
    # conversation = [{"role": "2", "content": [{"type": "text", "text": "Test."}]}]
    assert conversation[0]["role"] == "2"
    assert conversation[0]["content"][0]["text"] == "Test."
