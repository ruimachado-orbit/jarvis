"""LangGraph graph compilation and run_turn() entry point."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from langgraph.graph import END, StateGraph

from jarvis.core.config import Settings
from jarvis.graph import nodes
from jarvis.graph.state import AgentState

if TYPE_CHECKING:
    from jarvis.memory.mem0_store import Mem0Store
    from jarvis.tools.registry import Toolbox

log = logging.getLogger(__name__)


def build_graph(
    settings: Settings,
    toolbox: Toolbox,
    mem0_store: Mem0Store | None,
):
    """Compile and return the LangGraph agent graph."""
    graph = StateGraph(AgentState)

    graph.add_node("retrieve_memory", lambda s: nodes.retrieve_memory_node(s, mem0_store))
    graph.add_node("think", nodes.think_node)
    graph.add_node("act", lambda s: nodes.act_node(s, toolbox))
    graph.add_node("observe", nodes.observe_node)
    graph.add_node("respond", nodes.respond_node)
    graph.add_node("store_memory", lambda s: nodes.store_memory_node(s, mem0_store))

    graph.set_entry_point("retrieve_memory")
    graph.add_edge("retrieve_memory", "think")

    def _after_think(state: AgentState) -> str:
        if state.get("requires_confirmation"):
            return "respond"
        if state.get("tool_calls"):
            return "act"
        return "respond"

    graph.add_conditional_edges("think", _after_think, {
        "act": "act",
        "respond": "respond",
    })
    graph.add_edge("act", "observe")

    def _after_observe(state: AgentState) -> str:
        turn_count = sum(1 for m in state["messages"] if m["role"] == "assistant")
        if turn_count >= nodes.MAX_TURNS:
            return "respond"
        return "think"

    graph.add_conditional_edges("observe", _after_observe, {
        "think": "think",
        "respond": "respond",
    })
    graph.add_edge("respond", "store_memory")
    graph.add_edge("store_memory", END)

    return graph.compile()


async def run_turn(
    compiled_graph,
    user_text: str,
    trigger: str = "voice",
    output_channel: str = "voice",
    conversation_history: list[dict[str, Any]] | None = None,
    settings: Settings | None = None,
    toolbox: Toolbox | None = None,
    tts_callback=None,
) -> tuple[str, list[dict[str, Any]]]:
    """Run one turn. Returns (response_text, updated_history)."""
    history = list(conversation_history or [])
    history.append({"role": "user", "content": user_text})

    initial_state: AgentState = {
        "trigger": trigger,
        "messages": history,
        "memories": [],
        "tool_calls": [],
        "tool_results": [],
        "output_channel": output_channel,
        "final_response": "",
        "requires_confirmation": False,
        "pending_action": None,
        "error": None,
        "settings": settings,
        "toolbox": toolbox,
        "tts_callback": tts_callback,
    }

    result = await compiled_graph.ainvoke(initial_state)
    response = result.get("final_response", "")
    updated_history = result.get("messages", history)
    return response, updated_history
