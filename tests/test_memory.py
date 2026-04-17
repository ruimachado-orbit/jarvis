from pathlib import Path

import pytest

from jarvis.memory import (
    EMPTY_PROFILE_MARKER,
    Memory,
    distill_profile,
    format_profile_block,
)


def test_memory_disabled_is_a_noop(tmp_path: Path) -> None:
    mem = Memory(tmp_path / "m.db", enabled=False)
    assert mem.log("user", "hi") is None
    assert mem.recent_episodes() == []
    assert mem.current_profile() == ("", 0)
    mem.save_profile("whatever", 0)
    assert mem.current_profile() == ("", 0)


def test_memory_round_trip(tmp_path: Path) -> None:
    mem = Memory(tmp_path / "m.db")
    uid = mem.log("user", "hello")
    aid = mem.log("assistant", "hi there")
    tid = mem.log("tool", "file contents", tool_name="read_file", tool_call_id="c1")
    assert uid and aid and tid and uid < aid < tid

    episodes = mem.recent_episodes(limit=10)
    assert [e.role for e in episodes] == ["user", "assistant", "tool"]
    assert episodes[2].tool_name == "read_file"

    assert mem.max_episode_id() == tid
    assert mem.user_turns_since(0) == 1


def test_profile_save_and_retrieval(tmp_path: Path) -> None:
    mem = Memory(tmp_path / "m.db")
    mem.log("user", "I love Python")
    mem.save_profile("likes python", covered_up_to=1)
    text, covered = mem.current_profile()
    assert text == "likes python"
    assert covered == 1

    mem.save_profile("prefers fastapi", covered_up_to=2)
    text2, covered2 = mem.current_profile()
    assert text2 == "prefers fastapi"
    assert covered2 == 2


def test_format_profile_block_suppresses_empty() -> None:
    assert format_profile_block("") == ""
    assert format_profile_block(EMPTY_PROFILE_MARKER) == ""
    block = format_profile_block("likes python")
    assert "likes python" in block
    assert block.startswith("\n\n")


class _FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[list[dict]] = []

    async def complete(self, messages, tools=None, tool_choice="auto"):
        self.calls.append(messages)
        return {"role": "assistant", "content": self.reply}


@pytest.mark.asyncio
async def test_distill_profile_writes_new_profile(tmp_path: Path) -> None:
    mem = Memory(tmp_path / "m.db")
    mem.log("user", "I only use Python and FastAPI")
    mem.log("assistant", "got it")
    llm = _FakeLLM("likes python\nuses fastapi")

    profile = await distill_profile(mem, llm)
    assert profile == "likes python\nuses fastapi"
    stored, covered = mem.current_profile()
    assert stored == "likes python\nuses fastapi"
    assert covered == mem.max_episode_id()


@pytest.mark.asyncio
async def test_distill_profile_skips_when_no_new_episodes(tmp_path: Path) -> None:
    mem = Memory(tmp_path / "m.db")
    mem.log("user", "hello")
    mem.save_profile("initial", covered_up_to=mem.max_episode_id())
    llm = _FakeLLM("SHOULD NOT BE USED")

    profile = await distill_profile(mem, llm)
    assert profile == "initial"
    assert llm.calls == []


@pytest.mark.asyncio
async def test_distill_profile_handles_empty_marker(tmp_path: Path) -> None:
    mem = Memory(tmp_path / "m.db")
    mem.log("user", "what time is it")
    llm = _FakeLLM(EMPTY_PROFILE_MARKER)

    profile = await distill_profile(mem, llm)
    assert profile == ""  # existing profile (none) preserved
    # coverage still advanced so we don't redistill the same episodes:
    _, covered = mem.current_profile()
    assert covered == mem.max_episode_id()
