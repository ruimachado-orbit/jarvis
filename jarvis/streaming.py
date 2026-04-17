"""Stream utilities shared by the agent and tests.

Split out from agent.py so consumers (tests, tools) don't have to import the
OpenAI client just to use the sentence splitter.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator

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
