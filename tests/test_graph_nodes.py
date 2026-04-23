import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from jarvis.graph.state import AgentState
from jarvis.graph.nodes import retrieve_memory_node, store_memory_node, respond_node


@pytest.fixture
def base_state() -> AgentState:
    return AgentState(
        trigger="voice",
        messages=[{"role": "user", "content": "What time is my next meeting?"}],
        memories=[],
        tool_calls=[],
        tool_results=[],
        output_channel="voice",
        final_response="",
        requires_confirmation=False,
        pending_action=None,
        error=None,
    )


def test_retrieve_memory_node_no_mem0(base_state):
    update = retrieve_memory_node(base_state, mem0_store=None)
    assert update["memories"] == []


def test_retrieve_memory_node_with_results(base_state):
    mock_store = MagicMock()
    mock_store.retrieve.return_value = ["Rui likes TypeScript", "Orbit deadline May 15"]
    update = retrieve_memory_node(base_state, mem0_store=mock_store)
    assert len(update["memories"]) == 2
    mock_store.retrieve.assert_called_once_with(
        "What time is my next meeting?", limit=5
    )


def test_store_memory_node_calls_store(base_state):
    base_state["final_response"] = "Your next meeting is at 3pm."
    mock_store = MagicMock()
    store_memory_node(base_state, mem0_store=mock_store)
    mock_store.store.assert_called()


def test_respond_node_sets_final_response(base_state):
    base_state["messages"] = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "Hi Rui!"},
    ]
    update = respond_node(base_state)
    assert update["final_response"] == "Hi Rui!"


def test_respond_node_confirmation_required(base_state):
    base_state["requires_confirmation"] = True
    base_state["pending_action"] = {"name": "write_file", "args": {"path": "x.py", "content": "..."}}
    base_state["messages"] = [{"role": "assistant", "content": ""}]
    update = respond_node(base_state)
    assert "write_file" in update["final_response"] or "confirm" in update["final_response"].lower()
