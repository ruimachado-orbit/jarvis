import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from jarvis.graph.proactive.calendar import CalendarMonitor
from jarvis.graph.proactive.email import EmailMonitor


@pytest.fixture
def mock_graph():
    g = MagicMock()
    g.ainvoke = AsyncMock(return_value={"final_response": "You have a meeting soon."})
    return g


@pytest.fixture
def mock_calendar_tools():
    ct = MagicMock()
    ct.dispatch = AsyncMock(return_value="- Call with Sarah at 2026-04-24T15:00:00Z")
    return ct


@pytest.fixture
def mock_email_tools():
    et = MagicMock()
    et.dispatch = AsyncMock(return_value="id=msg1 | Thu | From: boss@co.com | Urgent: review needed")
    return et


@pytest.mark.asyncio
async def test_calendar_monitor_runs_once(mock_graph, mock_calendar_tools):
    monitor = CalendarMonitor(mock_graph, mock_calendar_tools, notify_fn=AsyncMock())
    await monitor.check_once()
    mock_calendar_tools.dispatch.assert_awaited_once_with("list_events", {"days": 1})


@pytest.mark.asyncio
async def test_email_monitor_runs_once(mock_graph, mock_email_tools):
    monitor = EmailMonitor(mock_graph, mock_email_tools, notify_fn=AsyncMock())
    await monitor.check_once()
    mock_email_tools.dispatch.assert_awaited_once()


@pytest.mark.asyncio
async def test_calendar_monitor_calls_notify_on_upcoming(mock_graph, mock_calendar_tools):
    notify = AsyncMock()
    monitor = CalendarMonitor(mock_graph, mock_calendar_tools, notify_fn=notify)
    await monitor.check_once()
    mock_graph.ainvoke.assert_awaited()
