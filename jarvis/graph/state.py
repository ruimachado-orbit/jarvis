"""AgentState TypedDict for the LangGraph agent graph."""

from __future__ import annotations

from typing import Any

from typing_extensions import TypedDict


class AgentState(TypedDict):
    trigger: str                     # "voice" | "calendar" | "email" | "watchdog"
    messages: list[dict[str, Any]]   # OpenAI-style message dicts for Anthropic API
    memories: list[str]              # retrieved mem0 facts injected into system prompt
    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    output_channel: str              # "voice" | "telegram" | "silent"
    final_response: str
    requires_confirmation: bool      # True when a destructive tool was called
    pending_action: dict[str, Any] | None  # tool call waiting for approval
    error: str | None
