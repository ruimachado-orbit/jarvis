import pytest
from unittest.mock import patch, MagicMock
from jarvis.core.config import Settings


def test_build_session_returns_components():
    from jarvis.core.session import build_session
    settings = Settings(_env_file=None, mem0_enabled=False)
    with patch("jarvis.core.session.anthropic.AsyncAnthropic"), \
         patch("jarvis.core.session.build_graph") as mock_graph:
        mock_graph.return_value = MagicMock()
        session = build_session(settings)
    assert "graph" in session
    assert "toolbox" in session
    assert "mem0" in session
    assert "client" in session


def test_build_session_mem0_disabled():
    from jarvis.core.session import build_session
    settings = Settings(_env_file=None, mem0_enabled=False)
    with patch("jarvis.core.session.anthropic.AsyncAnthropic"), \
         patch("jarvis.core.session.build_graph") as mock_graph:
        mock_graph.return_value = MagicMock()
        session = build_session(settings)
    assert session["mem0"] is None
