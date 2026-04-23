"""Stream utilities shared by the agent and tests.

Contains:
- ``stream_sentences``: buffer a token stream into TTS-friendly sentences.
- ``SentenceSplitter``: stateful incremental variant used during streaming.
- ``speak_stream``: pipeline sentences through TTS and serial playback.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from jarvis.voice.audio import AudioIO
    from jarvis.voice.tts import TTS

_SENT_BOUNDARY = re.compile(r"(?<=[\.\!\?])\s+(?=[A-Z0-9\"'\(\[])|\n\n+")


async def stream_sentences(tokens: AsyncIterator[str]) -> AsyncIterator[str]:
    """Buffer streamed tokens and yield complete sentences for TTS.

    The tail (non-terminated remainder) is flushed on stream end.
    """
    buffer = ""
    async for token in tokens:
        buffer += token
        while True:
            match = _SENT_BOUNDARY.search(buffer)
            if not match:
                break
            cut = match.end()
            sentence = buffer[:cut].strip()
            buffer = buffer[cut:]
            if sentence:
                yield sentence
    tail = buffer.strip()
    if tail:
        yield tail


class SentenceSplitter:
    """Incremental sentence extractor used inside the agent's stream loop."""

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, chunk: str) -> list[str]:
        self._buffer += chunk
        out: list[str] = []
        while True:
            match = _SENT_BOUNDARY.search(self._buffer)
            if not match:
                break
            cut = match.end()
            sentence = self._buffer[:cut].strip()
            self._buffer = self._buffer[cut:]
            if sentence:
                out.append(sentence)
        return out

    def flush(self) -> str | None:
        tail = self._buffer.strip()
        self._buffer = ""
        return tail or None


async def speak_stream(
    sentences: AsyncIterator[str],
    tts: TTS,
    audio: AudioIO,
    max_pending: int = 3,
) -> None:
    """Synthesize sentences and play them in order, pipelining the two stages.

    As soon as the first sentence finishes synthesising, playback starts.
    Later sentences synthesise in parallel (bounded by ``max_pending``) so the
    speaker is never starved while the LLM keeps streaming.
    """
    queue: asyncio.Queue[tuple] = asyncio.Queue(maxsize=max_pending)
    STOP = object()

    async def producer() -> None:
        try:
            async for sentence in sentences:
                clip = await tts.synthesize(sentence)
                await queue.put(clip)
        finally:
            await queue.put(STOP)

    async def consumer() -> None:
        while True:
            item = await queue.get()
            if item is STOP:
                return
            pcm, sr = item
            await audio.play(pcm, sr)

    await asyncio.gather(producer(), consumer())
