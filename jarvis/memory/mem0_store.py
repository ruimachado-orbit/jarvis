"""mem0-free memory store: fastembed embeddings + Chroma vector DB + claude -p for extraction."""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import subprocess
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

try:
    import chromadb
    from fastembed import TextEmbedding
    _DEPS_OK = True
except ImportError as _e:
    _DEPS_OK = False
    log.warning("chromadb/fastembed not installed; memory disabled: %s", _e)


_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
_EXTRACT_PROMPT = """\
Extract concise, standalone facts from the following conversation message.
Return a JSON array of short fact strings (max 10 words each).
Only include facts worth remembering long-term (preferences, names, projects, decisions).
If there are no memorable facts, return an empty array.

Message:
{text}

Return ONLY valid JSON, no other text."""


class Mem0Store:
    """Lightweight memory store: fastembed + Chroma + claude -p for fact extraction."""

    def __init__(self, path: str, user_id: str = "jarvis_user", enabled: bool = True) -> None:
        self.user_id = user_id
        self.enabled = enabled and _DEPS_OK
        self._collection: Any = None
        self._embed_model: Any = None

        if self.enabled:
            Path(path).mkdir(parents=True, exist_ok=True)
            self._embed_model = TextEmbedding(model_name=_EMBED_MODEL)
            client = chromadb.PersistentClient(path=str(Path(path) / "chroma"))
            self._collection = client.get_or_create_collection(
                name=f"jarvis_{user_id}",
                metadata={"hnsw:space": "cosine"},
            )

    def _embed(self, text: str) -> list[float]:
        emb = list(self._embed_model.embed([text]))[0]
        return emb.tolist() if hasattr(emb, "tolist") else list(emb)

    def retrieve(self, query: str, limit: int = 5) -> list[str]:
        if not self.enabled or self._collection is None:
            return []
        try:
            emb = self._embed(query)
            results = self._collection.query(
                query_embeddings=[emb],
                n_results=min(limit, self._collection.count() or 1),
                include=["documents"],
            )
            docs = results.get("documents", [[]])[0]
            return docs
        except Exception as e:
            log.warning("memory retrieve failed: %s", e)
            return []

    def store(self, text: str, role: str = "user") -> None:
        if not self.enabled or self._collection is None:
            return
        try:
            facts = self._extract_facts(text)
            for fact in facts:
                self._upsert(fact)
        except Exception as e:
            log.warning("memory store failed: %s", e)

    def remember(self, fact: str) -> None:
        if not self.enabled or self._collection is None:
            return
        try:
            self._upsert(fact)
        except Exception as e:
            log.warning("memory remember failed: %s", e)

    def forget(self, fact: str) -> None:
        if not self.enabled or self._collection is None:
            return
        try:
            results = self._collection.query(
                query_embeddings=[self._embed(fact)],
                n_results=min(5, self._collection.count() or 1),
                include=["documents"],
            )
            ids = results.get("ids", [[]])[0]
            for doc_id in ids:
                self._collection.delete(ids=[doc_id])
        except Exception as e:
            log.warning("memory forget failed: %s", e)

    def _upsert(self, fact: str) -> None:
        import hashlib
        doc_id = hashlib.md5(fact.encode()).hexdigest()
        emb = self._embed(fact)
        self._collection.upsert(ids=[doc_id], documents=[fact], embeddings=[emb])

    def _extract_facts(self, text: str) -> list[str]:
        """Use claude -p to extract memorable facts from text."""
        prompt = _EXTRACT_PROMPT.format(text=text[:2000])
        try:
            result = subprocess.run(
                ["claude", "-p", prompt, "--output-format", "text"],
                capture_output=True, text=True, timeout=30,
            )
            raw = result.stdout.strip()
            # Find JSON array in output
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start >= 0 and end > start:
                facts = json.loads(raw[start:end])
                return [f for f in facts if isinstance(f, str) and f.strip()]
        except Exception as e:
            log.debug("fact extraction failed: %s", e)
        return []
