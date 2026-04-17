"""The agent loop: orchestrates the LLM + tool calls + voice-friendly output."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

from jarvis.config import Settings
from jarvis.llm import LLMClient, parse_tool_args
from jarvis.memory import Memory, distill_profile, format_profile_block
from jarvis.personality import SYSTEM_PROMPT
from jarvis.streaming import SentenceSplitter, stream_sentences
from jarvis.tools import TOOL_SCHEMAS, Toolbox

__all__ = ["Agent", "stream_sentences"]

log = logging.getLogger(__name__)

MAX_TURNS = 6


class Agent:
    """A thin multi-turn orchestrator with tool calls and persistent memory.

    State:
    - ``_messages``: in-memory conversation for the current session.
    - ``memory``: SQLite-backed episodic log plus a distilled user profile that
      is folded into the system prompt.

    After every assistant turn, each message (user, assistant, tool result) is
    appended to the episodic log. The profile is refreshed in a background
    task once the user has spoken ``settings.memory_refresh_every`` times
    since the last refresh.
    """

    def __init__(
        self,
        settings: Settings,
        llm: LLMClient,
        tools: Toolbox,
        memory: Memory | None = None,
    ) -> None:
        self.settings = settings
        self.llm = llm
        self.tools = tools
        self.memory = memory or Memory(settings.memory_db, enabled=False)
        self._refresh_task: asyncio.Task | None = None
        self._messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._build_system_prompt()}
        ]

    # ----- prompt / memory plumbing -----

    def _build_system_prompt(self) -> str:
        profile, _ = self.memory.current_profile()
        return SYSTEM_PROMPT + format_profile_block(profile)

    def reload_profile(self) -> None:
        """Rebuild the system prompt from the latest stored profile."""
        self._messages[0] = {"role": "system", "content": self._build_system_prompt()}

    def reset(self) -> None:
        """Clear in-memory chat. Does not touch the persistent memory."""
        self._messages = [{"role": "system", "content": self._build_system_prompt()}]

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self._messages)

    # ----- public API -----

    async def respond(self, user_text: str) -> str:
        self._messages.append({"role": "user", "content": user_text})
        self.memory.log("user", user_text)

        for _ in range(MAX_TURNS):
            message = await self.llm.complete(self._messages, tools=TOOL_SCHEMAS)
            self._messages.append(message)

            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                text = (message.get("content") or "").strip()
                self.memory.log("assistant", text)
                self._schedule_refresh()
                return text

            self.memory.log(
                "assistant",
                message.get("content") or "",
                metadata={"tool_calls": [tc["function"]["name"] for tc in tool_calls]},
            )
            for call in tool_calls:
                await self._dispatch_and_log(call)

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
        self.memory.log("assistant", text, metadata={"capped": True})
        self._schedule_refresh()
        return text

    async def respond_stream(self, user_text: str) -> AsyncIterator[str]:
        """Same loop as ``respond`` but yields sentences as the final turn streams."""
        self._messages.append({"role": "user", "content": user_text})
        self.memory.log("user", user_text)

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
                self.memory.log("assistant", content_acc.strip())
                self._schedule_refresh()
                return

            self.memory.log(
                "assistant",
                content_acc,
                metadata={"tool_calls": [tc["function"]["name"] for tc in tool_calls]},
            )
            for call in tool_calls:
                await self._dispatch_and_log(call)

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
        self.memory.log("assistant", text, metadata={"capped": True})
        self._schedule_refresh()
        if text:
            splitter = SentenceSplitter()
            for sentence in splitter.feed(text):
                yield sentence
            tail = splitter.flush()
            if tail:
                yield tail

    # ----- internals -----

    async def _dispatch_and_log(self, call: dict[str, Any]) -> None:
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
        self.memory.log(
            "tool",
            result,
            tool_name=name,
            tool_call_id=call["id"],
            metadata={"args": args},
        )

    def _schedule_refresh(self) -> None:
        if not self.memory.enabled:
            return
        _, covered = self.memory.current_profile()
        if self.memory.user_turns_since(covered) < self.settings.memory_refresh_every:
            return
        if self._refresh_task and not self._refresh_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        log.info("kicking off background profile refresh")
        self._refresh_task = loop.create_task(self._refresh_profile())

    async def _refresh_profile(self) -> None:
        try:
            await distill_profile(
                self.memory, self.llm, context_turns=self.settings.memory_context_turns
            )
            self.reload_profile()
        except Exception as e:
            log.warning("background profile refresh failed: %s", e)
