import pytest
from unittest.mock import patch, MagicMock
from jarvis.memory.mem0_store import Mem0Store


@pytest.fixture
def mock_mem0(tmp_path):
    with patch("jarvis.memory.mem0_store.Memory") as mock_cls:
        instance = MagicMock()
        instance.search.return_value = {"results": [{"memory": "Rui likes TypeScript"}]}
        instance.add.return_value = {"results": []}
        mock_cls.return_value = instance
        yield instance


def test_retrieve_returns_list(mock_mem0, tmp_path):
    store = Mem0Store(str(tmp_path / "mem0"), user_id="test_user")
    results = store.retrieve("TypeScript preferences", limit=5)
    assert isinstance(results, list)
    assert "TypeScript" in results[0]


def test_store_calls_add(mock_mem0, tmp_path):
    store = Mem0Store(str(tmp_path / "mem0"), user_id="test_user")
    store.store("Rui prefers tabs over spaces", role="user")
    mock_mem0.add.assert_called_once()


def test_remember_explicit(mock_mem0, tmp_path):
    store = Mem0Store(str(tmp_path / "mem0"), user_id="test_user")
    store.remember("Always use Python 3.11+")
    mock_mem0.add.assert_called_once()
    call_args = str(mock_mem0.add.call_args)
    assert "Python 3.11" in call_args


def test_retrieve_disabled(tmp_path):
    with patch("jarvis.memory.mem0_store.Memory") as mock_cls:
        instance = MagicMock()
        mock_cls.return_value = instance
        store = Mem0Store(str(tmp_path / "mem0"), user_id="test", enabled=False)
        results = store.retrieve("anything")
        assert results == []
        instance.search.assert_not_called()


def test_forget_calls_delete(mock_mem0, tmp_path):
    mock_mem0.search.return_value = {"results": [{"id": "mem123", "memory": "old fact"}]}
    store = Mem0Store(str(tmp_path / "mem0"), user_id="test_user")
    store.forget("old fact")
    mock_mem0.delete.assert_called_once_with("mem123")
