import pytest

from jarvis.streaming import stream_sentences


async def _feed(chunks):
    for c in chunks:
        yield c


@pytest.mark.asyncio
async def test_sentence_splitter_emits_on_boundary():
    tokens = ["Hello", " there. ", "How are you?", " I am", " fine."]
    got = [s async for s in stream_sentences(_feed(tokens))]
    assert got == ["Hello there.", "How are you?", "I am fine."]


@pytest.mark.asyncio
async def test_sentence_splitter_flushes_tail():
    tokens = ["A lonely fragment"]
    got = [s async for s in stream_sentences(_feed(tokens))]
    assert got == ["A lonely fragment"]
