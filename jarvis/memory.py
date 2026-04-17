"""Long-term memory for Jarvis.

Two tiers, both backed by a single SQLite file:

1. **Episodic log** — every user message, assistant message, and tool call is
   appended to ``episodes``. Nothing is summarised at write time; this is the
   raw record.
2. **Distilled profile** — every N user turns, the LLM is asked to read the
   recent episodes and produce a short bullet list of durable facts about the
   user (preferred languages, style, recurring projects, likes/dislikes). The
   latest profile is injected into the system prompt so every future turn
   benefits from it.

The distillation prompt explicitly forbids fabrication and caps the profile at
~800 characters so the system prompt stays small.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from jarvis.llm import LLMClient

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    ts REAL NOT NULL,
    role TEXT NOT NULL,
    content TEXT,
    tool_name TEXT,
    tool_call_id TEXT,
    metadata TEXT
);
CREATE INDEX IF NOT EXISTS idx_episodes_ts ON episodes(ts);
CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id);
CREATE INDEX IF NOT EXISTS idx_episodes_role ON episodes(role);

CREATE TABLE IF NOT EXISTS profile (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    covered_up_to_episode_id INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);
"""

DISTILL_PROMPT = """You are a note-taker for a coding assistant named Jarvis.
Read the interactions below and produce a short profile of durable facts about
the user. Examples of useful facts: preferred languages and frameworks, coding
style, project names they work on, libraries they reach for, terminology they
use, and things they have clearly said they like or dislike.

Rules:
- Output only the profile. No greetings, no meta commentary.
- Use short lines, one fact per line, no markdown bullets.
- Prefer facts the user stated directly. Do not invent anything.
- When a newer fact contradicts an older one, keep the newer one.
- Keep the whole profile under 800 characters.
- If there is nothing durable worth saving, output exactly: (no durable facts yet)

Existing profile (may be empty):
{existing}

Recent interactions (oldest first):
{transcript}

New distilled profile:
"""

EMPTY_PROFILE_MARKER = "(no durable facts yet)"
MAX_EPISODE_CONTENT = 600
PROFILE_CHAR_LIMIT = 1000  # hard cap even if the model ignores the soft limit


@dataclass
class Episode:
    id: int
    ts: float
    role: str
    content: str | None
    tool_name: str | None


class Memory:
    """Persistent conversation log + distilled profile."""

    def __init__(
        self,
        db_path: Path,
        session_id: str | None = None,
        enabled: bool = True,
    ) -> None:
        self.db_path = Path(db_path)
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.enabled = enabled
        if not self.enabled:
            return
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ----- writes -----

    def log(
        self,
        role: str,
        content: str | None = None,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int | None:
        if not self.enabled:
            return None
        with self._conn() as c:
            cur = c.execute(
                """
                INSERT INTO episodes (session_id, ts, role, content, tool_name,
                                      tool_call_id, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.session_id,
                    time.time(),
                    role,
                    content,
                    tool_name,
                    tool_call_id,
                    json.dumps(metadata) if metadata else None,
                ),
            )
            return cur.lastrowid

    def save_profile(self, content: str, covered_up_to: int) -> None:
        if not self.enabled:
            return
        content = content.strip()[:PROFILE_CHAR_LIMIT]
        with self._conn() as c:
            c.execute(
                "INSERT INTO profile (content, covered_up_to_episode_id, created_at) "
                "VALUES (?, ?, ?)",
                (content, covered_up_to, time.time()),
            )

    # ----- reads -----

    def recent_episodes(self, limit: int = 80) -> list[Episode]:
        if not self.enabled:
            return []
        with self._conn() as c:
            rows = c.execute(
                """
                SELECT id, ts, role, content, tool_name
                FROM episodes ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return list(reversed([Episode(*r) for r in rows]))

    def current_profile(self) -> tuple[str, int]:
        """Return (profile_text, covered_up_to_episode_id). Empty string if none."""
        if not self.enabled:
            return "", 0
        with self._conn() as c:
            row = c.execute(
                "SELECT content, covered_up_to_episode_id FROM profile "
                "ORDER BY id DESC LIMIT 1"
            ).fetchone()
        if not row:
            return "", 0
        return row[0] or "", row[1] or 0

    def max_episode_id(self) -> int:
        if not self.enabled:
            return 0
        with self._conn() as c:
            row = c.execute("SELECT COALESCE(MAX(id), 0) FROM episodes").fetchone()
        return int(row[0])

    def user_turns_since(self, episode_id: int) -> int:
        """How many user messages have been logged since ``episode_id``."""
        if not self.enabled:
            return 0
        with self._conn() as c:
            row = c.execute(
                "SELECT COUNT(*) FROM episodes WHERE id > ? AND role = 'user'",
                (episode_id,),
            ).fetchone()
        return int(row[0])

    def stats(self) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        with self._conn() as c:
            totals = c.execute(
                """
                SELECT role, COUNT(*) FROM episodes GROUP BY role
                """
            ).fetchall()
            sessions = c.execute(
                "SELECT COUNT(DISTINCT session_id) FROM episodes"
            ).fetchone()[0]
            profile_rows = c.execute("SELECT COUNT(*) FROM profile").fetchone()[0]
        return {
            "enabled": True,
            "db_path": str(self.db_path),
            "sessions": int(sessions),
            "profile_versions": int(profile_rows),
            "turns_by_role": {role: int(n) for role, n in totals},
        }

    def reset(self) -> None:
        if not self.enabled:
            return
        with self._conn() as c:
            c.execute("DELETE FROM episodes")
            c.execute("DELETE FROM profile")


# ----- distillation -----

def _render_episode(e: Episode) -> str:
    content = (e.content or "").strip().replace("\n", " ")
    if len(content) > MAX_EPISODE_CONTENT:
        content = content[:MAX_EPISODE_CONTENT] + "…"
    if e.role == "tool":
        return f"tool[{e.tool_name}]: {content}"
    return f"{e.role}: {content}"


async def distill_profile(
    memory: Memory,
    llm: LLMClient,
    context_turns: int = 60,
) -> str:
    """Ask the LLM for an updated profile. Returns the new profile text.

    Does nothing and returns the existing profile when there are no new
    episodes since the last distillation.
    """
    if not memory.enabled:
        return ""
    existing, covered = memory.current_profile()
    episodes = memory.recent_episodes(limit=context_turns)
    new_episodes = [e for e in episodes if e.id > covered]
    if not new_episodes:
        return existing

    transcript = "\n".join(_render_episode(e) for e in new_episodes)
    prompt = DISTILL_PROMPT.format(
        existing=existing or "(none)",
        transcript=transcript,
    )
    try:
        message = await llm.complete([{"role": "user", "content": prompt}])
    except Exception as e:  # pragma: no cover - network
        log.warning("profile distillation failed: %s", e)
        return existing

    new_profile = (message.get("content") or "").strip()
    if not new_profile:
        return existing
    if new_profile == EMPTY_PROFILE_MARKER:
        # Still record coverage so we don't re-ask about the same episodes.
        memory.save_profile(existing or EMPTY_PROFILE_MARKER, new_episodes[-1].id)
        return existing

    memory.save_profile(new_profile, new_episodes[-1].id)
    log.info("profile refreshed (%d chars, covered through episode %d)",
             len(new_profile), new_episodes[-1].id)
    return new_profile


def format_profile_block(profile: str) -> str:
    """Produce the chunk that gets appended to the system prompt."""
    profile = profile.strip()
    if not profile or profile == EMPTY_PROFILE_MARKER:
        return ""
    return (
        "\n\nWhat you know about this user (from past conversations, "
        "treat as strong priors, not absolute truth):\n"
        + profile
    )
