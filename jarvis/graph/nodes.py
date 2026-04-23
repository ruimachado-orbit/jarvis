"""LangGraph node functions for the Jarvis agent graph."""

from __future__ import annotations

import json
import logging
from typing import Any, TYPE_CHECKING

import anthropic

from jarvis.core.config import Settings
from jarvis.core.personality import SYSTEM_PROMPT
from jarvis.graph.state import AgentState
from jarvis.tools.registry import TOOL_SCHEMAS

if TYPE_CHECKING:
    from jarvis.memory.mem0_store import Mem0Store
    from jarvis.tools.registry import Toolbox

log = logging.getLogger(__name__)

MAX_TURNS = 6

CONFIRMATION_REQUIRED_TOOLS = {"write_file", "run_shell", "create_event", "update_event", "send_email"}


def retrieve_memory_node(state: AgentState, mem0_store: Mem0Store | None) -> dict[str, Any]:
    if mem0_store is None:
        return {"memories": []}
    last_user = next(
        (m["content"] for m in reversed(state["messages"]) if m["role"] == "user" and isinstance(m.get("content"), str)),
        ""
    )
    memories = mem0_store.retrieve(last_user, limit=5) if last_user else []
    return {"memories": memories}


def _build_system_with_memories(memories: list[str]) -> str:
    if not memories:
        return SYSTEM_PROMPT
    block = "\n\nWhat you know about this user (strong priors, not absolute truth):\n"
    block += "\n".join(f"- {m}" for m in memories)
    return SYSTEM_PROMPT + block


def _openai_schemas_to_anthropic(schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI-style tool schemas to Anthropic format."""
    tools = []
    for t in schemas:
        fn = t["function"]
        tools.append({
            "name": fn["name"],
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
        })
    return tools


async def think_node(
    state: AgentState,
    client: anthropic.AsyncAnthropic,
    settings: Settings,
) -> dict[str, Any]:
    system = _build_system_with_memories(state["memories"])

    # Build messages in Anthropic format (skip system messages)
    messages = [m for m in state["messages"] if m.get("role") != "system"]

    tools = _openai_schemas_to_anthropic(TOOL_SCHEMAS)

    response = await client.messages.create(
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
        system=system,
        messages=messages,
        tools=tools,
    )

    tool_calls = []
    text_content = ""
    for block in response.content:
        if block.type == "text":
            text_content += block.text
        elif block.type == "tool_use":
            tool_calls.append({
                "id": block.id,
                "name": block.name,
                "args": block.input,
            })

    # Build assistant message in Anthropic format (content is list of blocks)
    new_messages = list(state["messages"])
    new_messages.append({
        "role": "assistant",
        "content": response.content,
    })

    requires_confirmation = any(tc["name"] in CONFIRMATION_REQUIRED_TOOLS for tc in tool_calls)
    pending_action = tool_calls[0] if requires_confirmation and tool_calls else None

    return {
        "messages": new_messages,
        "tool_calls": tool_calls,
        "requires_confirmation": requires_confirmation,
        "pending_action": pending_action,
        "final_response": text_content,
    }


async def act_node(state: AgentState, toolbox: Toolbox) -> dict[str, Any]:
    results = []
    for call in state["tool_calls"]:
        result = await toolbox.dispatch(call["name"], call["args"])
        results.append({"tool_use_id": call["id"], "content": result})

    # Tool results sent as user message with tool_result content blocks
    new_messages = list(state["messages"])
    new_messages.append({
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": r["tool_use_id"], "content": r["content"]} for r in results],
    })
    return {"messages": new_messages, "tool_results": results, "tool_calls": []}


def observe_node(state: AgentState) -> dict[str, Any]:
    if state["tool_results"]:
        return {"tool_results": []}
    return {}


def respond_node(state: AgentState) -> dict[str, Any]:
    if state.get("requires_confirmation") and state.get("pending_action"):
        action = state["pending_action"]
        name = action.get("name", "action")
        args_preview = json.dumps(action.get("args", {}))[:120]
        msg = f"I'd like to run {name} with: {args_preview}. Shall I go ahead?"
        return {"final_response": msg}

    last_assistant = next(
        (m for m in reversed(state["messages"]) if m["role"] == "assistant"),
        None,
    )
    if last_assistant is None:
        return {"final_response": ""}

    content = last_assistant.get("content", "")
    if isinstance(content, list):
        # Anthropic content blocks — extract text
        text = " ".join(b.text for b in content if hasattr(b, "text"))
    else:
        text = str(content)
    return {"final_response": text.strip()}


def store_memory_node(state: AgentState, mem0_store: Mem0Store | None) -> dict[str, Any]:
    if mem0_store is None:
        return {}
    last_user = next(
        (m["content"] for m in state["messages"] if m["role"] == "user" and isinstance(m.get("content"), str)),
        None,
    )
    if last_user:
        mem0_store.store(last_user, role="user")
    if state.get("final_response"):
        mem0_store.store(state["final_response"], role="assistant")
    return {}


def route_node(state: AgentState) -> dict[str, Any]:
    return {}
