"""Calendar proactive monitor — polls Google Calendar and surfaces alerts."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Awaitable

from jarvis.graph.state import AgentState

log = logging.getLogger(__name__)


class CalendarMonitor:
    def __init__(
        self,
        graph,
        calendar_tools,
        notify_fn: Callable[[str], Awaitable[None]],
        poll_seconds: int = 300,
    ) -> None:
        self._graph = graph
        self._cal = calendar_tools
        self._notify = notify_fn
        self._poll_seconds = poll_seconds

    async def check_once(self) -> None:
        try:
            events_text = await self._cal.dispatch("list_events", {"days": 1})
            if not events_text or events_text.startswith("No events"):
                return
            prompt = (
                f"You are Jarvis monitoring the user's calendar. "
                f"Here are today's events:\n{events_text}\n\n"
                f"Are any of these events coming up soon (within 30 minutes) or require action? "
                f"If yes, compose a short proactive alert (one sentence). "
                f"If nothing needs attention right now, reply with exactly: NO_ACTION"
            )
            initial_state: AgentState = {
                "trigger": "calendar",
                "messages": [{"role": "user", "content": prompt}],
                "memories": [],
                "tool_calls": [],
                "tool_results": [],
                "output_channel": "telegram",
                "final_response": "",
                "requires_confirmation": False,
                "pending_action": None,
                "error": None,
            }
            result = await self._graph.ainvoke(initial_state)
            response = result.get("final_response", "").strip()
            if response and response != "NO_ACTION":
                await self._notify(response)
        except Exception as e:
            log.warning("calendar monitor check failed: %s", e)

    async def run_forever(self) -> None:
        log.info("calendar monitor started (poll every %ds)", self._poll_seconds)
        while True:
            await self.check_once()
            await asyncio.sleep(self._poll_seconds)
