import pytest
from typer.testing import CliRunner
from unittest.mock import patch, AsyncMock, MagicMock
from jarvis.main import app

runner = CliRunner()


def test_ask_command():
    with patch("jarvis.main.build_session") as mock_session, \
         patch("jarvis.main.run_turn", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = ("Hello from Jarvis.", [])
        mock_session.return_value = {
            "graph": MagicMock(), "toolbox": MagicMock(),
            "mem0": None, "client": MagicMock(), "settings": MagicMock()
        }
        result = runner.invoke(app, ["ask", "what is 2+2"])
    assert result.exit_code == 0
    assert "Hello" in result.output or "Jarvis" in result.output


def test_auth_google_command_exists():
    result = runner.invoke(app, ["auth", "--help"])
    assert result.exit_code == 0
    assert "google" in result.output.lower()
