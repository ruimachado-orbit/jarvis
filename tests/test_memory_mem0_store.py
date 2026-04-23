import pytest
from unittest.mock import patch, MagicMock
from jarvis.memory.mem0_store import Mem0Store


@pytest.fixture
def mock_store(tmp_path):
    """Patch chromadb and fastembed so tests run without heavy deps."""
    mock_collection = MagicMock()
    mock_collection.count.return_value = 1
    mock_collection.query.return_value = {
        "documents": [["Rui likes TypeScript"]],
        "ids": [["abc123"]],
    }
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection

    mock_embed_model = MagicMock()
    mock_embed_model.embed.return_value = [[0.1] * 384]

    with patch("jarvis.memory.mem0_store._DEPS_OK", True), \
         patch("jarvis.memory.mem0_store.chromadb") as mock_chroma, \
         patch("jarvis.memory.mem0_store.TextEmbedding", return_value=mock_embed_model):
        mock_chroma.PersistentClient.return_value = mock_client
        store = Mem0Store(str(tmp_path / "mem0"), user_id="test_user")
        store._embed_model = mock_embed_model
        store._collection = mock_collection
        yield store, mock_collection


def test_retrieve_returns_list(mock_store, tmp_path):
    store, collection = mock_store
    results = store.retrieve("TypeScript preferences", limit=5)
    assert isinstance(results, list)
    assert "TypeScript" in results[0]


def test_store_calls_add(mock_store, tmp_path):
    store, collection = mock_store
    with patch.object(store, "_extract_facts", return_value=["prefers tabs"]), \
         patch.object(store, "_upsert") as mock_upsert:
        store.store("Rui prefers tabs over spaces", role="user")
        mock_upsert.assert_called_once_with("prefers tabs")


def test_remember_explicit(mock_store, tmp_path):
    store, collection = mock_store
    with patch.object(store, "_upsert") as mock_upsert:
        store.remember("Always use Python 3.11+")
        mock_upsert.assert_called_once_with("Always use Python 3.11+")


def test_retrieve_disabled(tmp_path):
    store = Mem0Store(str(tmp_path / "mem0"), user_id="test", enabled=False)
    results = store.retrieve("anything")
    assert results == []


def test_forget_calls_delete(mock_store, tmp_path):
    store, collection = mock_store
    store.forget("old fact")
    collection.delete.assert_called_once()
