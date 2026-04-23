"""LangGraph node functions for the Jarvis agent graph."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import subprocess
import time
from typing import TYPE_CHECKING, Any

from jarvis.core.personality import SYSTEM_PROMPT
from jarvis.graph.state import AgentState

if TYPE_CHECKING:
    from jarvis.memory.mem0_store import Mem0Store
    from jarvis.tools.registry import Toolbox

log = logging.getLogger(__name__)

MAX_TURNS = 6

# --- OAuth token cache (read from macOS keychain once, refresh when expired) ---
_token_cache: dict = {}

def _get_oauth_token() -> str:
    """Return a valid Claude OAuth access token from the macOS keychain."""
    now_ms = time.time() * 1000
    if _token_cache.get("token") and _token_cache.get("expires_at", 0) > now_ms + 60_000:
        return _token_cache["token"]

    result = subprocess.run(
        ["security", "find-generic-password", "-s", "Claude Code-credentials",
         "-a", "ruimachado", "-w"],
        capture_output=True, text=True, timeout=5,
    )
    if result.returncode != 0:
        raise RuntimeError("Cannot read Claude credentials from keychain")

    creds = json.loads(result.stdout.strip())
    oauth = creds["claudeAiOauth"]
    _token_cache["token"] = oauth["accessToken"]
    _token_cache["expires_at"] = oauth["expiresAt"]
    return _token_cache["token"]

CONFIRMATION_REQUIRED_TOOLS = {"write_file", "run_shell", "create_event", "update_event", "send_email"}

TOOL_CALL_RE = re.compile(r"```tool_call\s*\n(.*?)\n```", re.DOTALL)


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


def _build_claude_prompt(system: str, messages: list[dict]) -> str:
    """Convert message history to a single prompt string for claude -p."""
    parts = [f"<system>\n{system}\n</system>\n"]
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block["text"])
                    elif block.get("type") == "tool_result":
                        text_parts.append(f"[Tool result for {block.get('tool_use_id','?')}]: {block.get('content','')}")
                    elif block.get("type") == "tool_use":
                        text_parts.append(f"[Tool call {block.get('name','?')}]: {json.dumps(block.get('input',{}))}")
            content = "\n".join(text_parts)
        parts.append(f"<{role}>\n{content}\n</{role}>")
    return "\n".join(parts)


_SENT_RE = re.compile(r"([.!?—])\s+")


async def _run_claude(
    system: str,
    messages: list[dict],
    model: str,
    sentence_callback: "asyncio.coroutines | None" = None,
) -> str:
    """Stream claude --bare (no hooks/plugins), firing sentence_callback per sentence."""
    token = _get_oauth_token()

    # Build just the conversation part (no system block) for -p
    conv_parts = []
    for msg in messages:
        if msg.get("role") == "system":
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        conv_parts.append(f"<{msg['role']}>\n{content}\n</{msg['role']}>")
    conversation = "\n".join(conv_parts)

    proc = await asyncio.create_subprocess_exec(
        "claude", "--bare", "-p", conversation,
        "--model", model,
        "--system-prompt", system,
        "--output-format", "stream-json",
        "--verbose",
        "--include-partial-messages",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**__import__("os").environ, "ANTHROPIC_API_KEY": token},
    )

    full_text: list[str] = []
    sentence_buf = ""

    async for raw_line in proc.stdout:
        line = raw_line.decode().strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        token_text = None
        etype = event.get("type", "")
        if etype == "stream_event":
            inner = event.get("event", {})
            if inner.get("type") == "content_block_delta":
                delta = inner.get("delta", {})
                if delta.get("type") == "text_delta":
                    token_text = delta.get("text", "")
        elif etype == "result" and not full_text and not event.get("is_error"):
            token_text = event.get("result", "")

        if token_text:
            full_text.append(token_text)
            sentence_buf += token_text
            if sentence_callback:
                while True:
                    m = _SENT_RE.search(sentence_buf)
                    if not m:
                        break
                    sentence = sentence_buf[:m.end()].strip()
                    sentence_buf = sentence_buf[m.end():]
                    if sentence:
                        await sentence_callback(sentence)

    if sentence_callback and sentence_buf.strip():
        await sentence_callback(sentence_buf.strip())

    await proc.wait()
    if proc.returncode not in (0, None):
        err = (await proc.stderr.read()).decode().strip()
        raise RuntimeError(f"claude --bare failed (rc={proc.returncode}): {err}")

    return "".join(full_text).strip()


async def think_node(state: AgentState) -> dict[str, Any]:
    settings = state.get("settings")
    toolbox: Toolbox | None = state.get("toolbox")

    if settings is not None and hasattr(settings, "llm_model"):
        model = settings.llm_model
    else:
        model = "claude-sonnet-4-6"

    memories = state.get("memories", [])
    system = _build_system_with_memories(memories)

    tool_descriptions = ""
    if toolbox:
        lines = []
        for schema in toolbox.schemas:
            name = schema["name"]
            desc = schema.get("description", "")
            props = schema.get("input_schema", {}).get("properties", {})
            params = ", ".join(f"{k}: {v.get('type','any')}" for k, v in props.items())
            lines.append(f"- {name}({params}): {desc}")
        tool_descriptions = "\n\nAvailable tools:\n" + "\n".join(lines)
        tool_descriptions += (
            "\n\nTo call a tool output EXACTLY:\n```tool_call\n{\"name\": \"tool_name\", \"input\": {...}}\n```"
        )

    system = system + tool_descriptions

    messages = [m for m in state.get("messages", []) if m.get("role") != "system"]
    turns = 0
    response_text = ""
    clean_response = ""

    while turns < MAX_TURNS:
        turns += 1
        tts_callback = state.get("tts_callback")
        try:
            response_text = await _run_claude(system, messages, model, sentence_callback=tts_callback)
        except Exception as e:
            log.error("LLM error: %s", e)
            return {**state, "final_response": f"Sorry, I encountered an error: {e}", "error": str(e)}

        tool_calls_raw = TOOL_CALL_RE.findall(response_text)
        clean_response = TOOL_CALL_RE.sub("", response_text).strip()

        if not tool_calls_raw:
            messages.append({"role": "assistant", "content": response_text})
            return {
                **state,
                "messages": messages,
                "tool_calls": [],
                "requires_confirmation": False,
                "pending_action": None,
                "final_response": clean_response or response_text,
            }

        tool_calls = []
        for raw in tool_calls_raw:
            try:
                tc = json.loads(raw.strip())
                tool_calls.append(tc)
            except json.JSONDecodeError as e:
                log.warning("Failed to parse tool call JSON: %s | %s", raw, e)

        if not tool_calls:
            messages.append({"role": "assistant", "content": response_text})
            return {
                **state,
                "messages": messages,
                "tool_calls": [],
                "requires_confirmation": False,
                "pending_action": None,
                "final_response": clean_response or response_text,
            }

        needs_confirm = any(tc.get("name") in CONFIRMATION_REQUIRED_TOOLS for tc in tool_calls)
        if needs_confirm and not state.get("_confirmed"):
            action_desc = "; ".join(f"{tc['name']}({tc.get('input',{})})" for tc in tool_calls)
            return {
                **state,
                "messages": messages,
                "tool_calls": tool_calls,
                "requires_confirmation": True,
                "pending_action": {"tool_calls": tool_calls, "messages": messages},
                "final_response": f"I'd like to: {action_desc}. Shall I proceed?",
            }

        messages.append({"role": "assistant", "content": response_text})
        tool_results_text = []
        for tc in tool_calls:
            name = tc.get("name", "")
            inp = tc.get("input", {})
            try:
                if toolbox:
                    result = await toolbox.dispatch(name, inp)
                else:
                    result = f"No toolbox available for {name}"
            except Exception as e:
                result = f"Error: {e}"
            tool_results_text.append(f"[Result of {name}]: {result}")
            log.info("Tool %s -> %s", name, str(result)[:120])

        messages.append({"role": "user", "content": "\n".join(tool_results_text)})

    final = clean_response or response_text or "I ran out of steps."
    return {
        **state,
        "messages": messages,
        "tool_calls": [],
        "requires_confirmation": False,
        "pending_action": None,
        "final_response": final,
    }


async def act_node(state: AgentState, toolbox: Toolbox) -> dict[str, Any]:
    results = []
    for call in state["tool_calls"]:
        name = call.get("name", "")
        args = call.get("args", call.get("input", {}))
        result = await toolbox.dispatch(name, args)
        results.append({"tool_use_id": call.get("id", name), "content": result})

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
        # pending_action may be the new format {tool_calls, messages} or old {name, args}
        if "name" in action:
            name = action.get("name", "action")
            args_preview = json.dumps(action.get("args", {}))[:120]
            msg = f"I'd like to run {name} with: {args_preview}. Shall I go ahead?"
        else:
            tcs = action.get("tool_calls", [])
            action_desc = "; ".join(f"{tc.get('name','?')}({tc.get('input',{})})" for tc in tcs)
            msg = f"I'd like to: {action_desc}. Shall I proceed?"
        return {"final_response": msg}

    last_assistant = next(
        (m for m in reversed(state["messages"]) if m["role"] == "assistant"),
        None,
    )
    if last_assistant is None:
        return {"final_response": ""}

    content = last_assistant.get("content", "")
    if isinstance(content, list):
        text = " ".join(
            b.get("text", "") if isinstance(b, dict) else (b.text if hasattr(b, "text") else "")
            for b in content
            if (b.get("type") == "text" if isinstance(b, dict) else getattr(b, "type", "") == "text")
        )
    else:
        text = str(content)
    return {"final_response": text.strip()}


def store_memory_node(state: AgentState, mem0_store: Mem0Store | None) -> dict[str, Any]:
    if mem0_store is None:
        return {}
    last_user = next(
        (m["content"] for m in reversed(state["messages"]) if m["role"] == "user" and isinstance(m.get("content"), str)),
        None,
    )
    if last_user:
        mem0_store.store(last_user, role="user")
    if state.get("final_response"):
        mem0_store.store(state["final_response"], role="assistant")
    return {}
