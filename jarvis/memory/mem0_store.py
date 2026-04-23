"""mem0 wrapper providing retrieve/store/remember/forget for Jarvis."""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

try:
    from mem0 import Memory
    _MEM0_AVAILABLE = True
except ImportError:
    _MEM0_AVAILABLE = False
    log.warning("mem0ai not installed; memory disabled")


class Mem0Store:
    """Thin wrapper around mem0.Memory with a stable interface for Jarvis."""

    def __init__(
        self,
        path: str,
        user_id: str = "jarvis_user",
        enabled: bool = True,
    ) -> None:
        self.user_id = user_id
        self.enabled = enabled and _MEM0_AVAILABLE
        self._mem: Memory | None = None
        if self.enabled:
            Path(path).mkdir(parents=True, exist_ok=True)
            config = {
                "vector_store": {
                    "provider": "chroma",
                    "config": {"path": str(Path(path) / "chroma")},
                },
                "version": "v1.1",
            }
            self._mem = Memory(config=config)

    def retrieve(self, query: str, limit: int = 5) -> list[str]:
        """Return up to `limit` memory strings relevant to query."""
        if not self.enabled or self._mem is None:
            return []
        try:
            result = self._mem.search(query, user_id=self.user_id, limit=limit)
            return [r["memory"] for r in result.get("results", [])]
        except Exception as e:
            log.warning("mem0 retrieve failed: %s", e)
            return []

    def store(self, text: str, role: str = "user") -> None:
        """Extract and store facts from a message."""
        if not self.enabled or self._mem is None:
            return
        try:
            messages = [{"role": role, "content": text}]
            self._mem.add(messages, user_id=self.user_id)
        except Exception as e:
            log.warning("mem0 store failed: %s", e)

    def remember(self, fact: str) -> None:
        """Explicitly store a single fact."""
        if not self.enabled or self._mem is None:
            return
        try:
            self._mem.add(
                [{"role": "system", "content": f"Remember this fact: {fact}"}],
                user_id=self.user_id,
            )
        except Exception as e:
            log.warning("mem0 remember failed: %s", e)

    def forget(self, fact: str) -> None:
        """Attempt to delete memories matching fact."""
        if not self.enabled or self._mem is None:
            return
        try:
            results = self._mem.search(fact, user_id=self.user_id, limit=10)
            for r in results.get("results", []):
                if r.get("id"):
                    self._mem.delete(r["id"])
        except Exception as e:
            log.warning("mem0 forget failed: %s", e)
