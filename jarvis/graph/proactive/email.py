"""Email proactive monitor — polls Gmail and surfaces urgent items."""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Awaitable

from jarvis.graph.state import AgentState

log = logging.getLogger(__name__)


class EmailMonitor:
    def __init__(
        self,
        graph,
        email_tools,
        notify_fn: Callable[[str], Awaitable[None]],
        poll_seconds: int = 600,
    ) -> None:
        self._graph = graph
        self._email = email_tools
        self._notify = notify_fn
        self._poll_seconds = poll_seconds

    async def check_once(self) -> None:
        try:
            emails_text = await self._email.dispatch(
                "list_emails", {"query": "is:unread", "limit": 5}
            )
            if not emails_text or emails_text == "No emails found.":
                return
            prompt = (
                f"You are Jarvis monitoring the user's email. "
                f"Here are the latest unread emails:\n{emails_text}\n\n"
                f"Is anything urgent or requiring a decision? "
                f"If yes, compose a one-sentence proactive alert with a suggested action. "
                f"If nothing is urgent, reply with exactly: NO_ACTION"
            )
            initial_state: AgentState = {
                "trigger": "email",
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
            log.warning("email monitor check failed: %s", e)

    async def run_forever(self) -> None:
        log.info("email monitor started (poll every %ds)", self._poll_seconds)
        while True:
            await self.check_once()
            await asyncio.sleep(self._poll_seconds)
