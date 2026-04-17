import pytest

from jarvis.streaming import SentenceSplitter, speak_stream


def test_sentence_splitter_emits_incrementally():
    sp = SentenceSplitter()
    assert sp.feed("Hello") == []
    # A boundary needs a following word, so the period alone doesn't flush yet.
    assert sp.feed(" there. ") == []
    assert sp.feed("How are ") == ["Hello there."]
    assert sp.feed("you? I am") == ["How are you?"]
    assert sp.flush() == "I am"


def test_sentence_splitter_handles_paragraph_breaks():
    sp = SentenceSplitter()
    assert sp.feed("First thought\n\nSecond thought") == ["First thought"]
    assert sp.flush() == "Second thought"


class _FakeTTS:
    async def synthesize(self, text):
        return (f"pcm:{text}", 22050)


class _FakeAudio:
    def __init__(self):
        self.played = []

    async def play(self, pcm, sr):
        self.played.append((pcm, sr))


async def _sentences():
    for s in ["One.", "Two.", "Three."]:
        yield s


@pytest.mark.asyncio
async def test_speak_stream_plays_in_order():
    audio = _FakeAudio()
    await speak_stream(_sentences(), _FakeTTS(), audio)
    assert audio.played == [
        ("pcm:One.", 22050),
        ("pcm:Two.", 22050),
        ("pcm:Three.", 22050),
    ]
