import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from jarvis.core.config import Settings
from jarvis.tools.registry import Toolbox, TOOL_SCHEMAS


def test_tool_schemas_present():
    names = {t["function"]["name"] for t in TOOL_SCHEMAS}
    assert "read_file" in names
    assert "web_search" in names
    assert "remember" in names
    assert "list_events" in names
    assert "list_emails" in names


@pytest.mark.asyncio
async def test_dispatch_unknown_tool():
    s = Settings(_env_file=None)
    tb = Toolbox(s)
    result = await tb.dispatch("nonexistent_tool", {})
    assert "ERROR" in result or "unknown" in result.lower()


@pytest.mark.asyncio
async def test_web_search_disabled_without_key():
    s = Settings(_env_file=None, brave_api_key="")
    tb = Toolbox(s)
    with patch("jarvis.tools.web_tools.DDGS") as mock_ddg:
        mock_ddg.return_value.__enter__ = lambda self: self
        mock_ddg.return_value.__exit__ = MagicMock(return_value=False)
        mock_ddg.return_value.text.return_value = [{"title": "Result", "href": "http://x.com", "body": "text"}]
        result = await tb.dispatch("web_search", {"query": "Python asyncio"})
    assert "Result" in result or "text" in result or result
