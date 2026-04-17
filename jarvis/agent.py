"""The agent loop: orchestrates the LLM + tool calls + voice-friendly output."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from jarvis.config import Settings
from jarvis.llm import LLMClient, parse_tool_args
from jarvis.personality import SYSTEM_PROMPT
from jarvis.streaming import SentenceSplitter, stream_sentences
from jarvis.tools import TOOL_SCHEMAS, Toolbox

__all__ = ["Agent", "stream_sentences"]

log = logging.getLogger(__name__)

MAX_TURNS = 6


class Agent:
    """A thin multi-turn orchestrator with tool calls.

    It keeps a rolling message history per session. Tool calls run in a silent inner
    loop; only the final assistant text is yielded for speech.
    """

    def __init__(self, settings: Settings, llm: LLMClient, tools: Toolbox) -> None:
        self.settings = settings
        self.llm = llm
        self.tools = tools
        self._messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    def reset(self) -> None:
        self._messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self._messages)

    async def respond(self, user_text: str) -> str:
        """Append user input, run tool-calling loop, return final assistant text."""
        self._messages.append({"role": "user", "content": user_text})

        for _ in range(MAX_TURNS):
            message = await self.llm.complete(self._messages, tools=TOOL_SCHEMAS)
            self._messages.append(message)

            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                return (message.get("content") or "").strip()

            for call in tool_calls:
                name = call["function"]["name"]
                args = parse_tool_args(call["function"]["arguments"])
                log.info("tool call %s %s", name, args)
                result = await self.tools.dispatch(name, args)
                self._messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "name": name,
                        "content": result,
                    }
                )

        # Loop cap hit; ask the model for a plain summary with no more tools.
        final = await self.llm.complete(
            self._messages
            + [
                {
                    "role": "user",
                    "content": "Give me your best short answer now. Do not call tools.",
                }
            ]
        )
        self._messages.append(final)
        return (final.get("content") or "").strip()

    async def respond_stream(self, user_text: str) -> AsyncIterator[str]:
        """Same loop as ``respond`` but yields sentences as the final turn streams."""
        self._messages.append({"role": "user", "content": user_text})

        for _ in range(MAX_TURNS):
            content_acc = ""
            tool_calls: list[dict[str, Any]] = []
            splitter = SentenceSplitter()

            async for event in self.llm.stream_with_tools(
                self._messages, tools=TOOL_SCHEMAS
            ):
                if event["type"] == "text":
                    content_acc += event["delta"]
                    for sentence in splitter.feed(event["delta"]):
                        yield sentence
                elif event["type"] == "done":
                    tool_calls = event["tool_calls"]

            message: dict[str, Any] = {"role": "assistant", "content": content_acc}
            if tool_calls:
                message["tool_calls"] = tool_calls
            self._messages.append(message)

            if not tool_calls:
                tail = splitter.flush()
                if tail:
                    yield tail
                return

            for call in tool_calls:
                name = call["function"]["name"]
                args = parse_tool_args(call["function"]["arguments"])
                log.info("tool call %s %s", name, args)
                result = await self.tools.dispatch(name, args)
                self._messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "name": name,
                        "content": result,
                    }
                )

        final = await self.llm.complete(
            self._messages
            + [
                {
                    "role": "user",
                    "content": "Give me your best short answer now. Do not call tools.",
                }
            ]
        )
        self._messages.append(final)
        text = (final.get("content") or "").strip()
        if text:
            splitter = SentenceSplitter()
            for sentence in splitter.feed(text):
                yield sentence
            tail = splitter.flush()
            if tail:
                yield tail


