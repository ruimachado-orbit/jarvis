import pytest
from unittest.mock import MagicMock, patch
from jarvis.tools.calendar_tools import CalendarTools
from jarvis.tools.email_tools import EmailTools


@pytest.fixture
def mock_calendar_service():
    svc = MagicMock()
    svc.events.return_value.list.return_value.execute.return_value = {"items": [
        {
            "id": "evt1",
            "summary": "Call with Sarah",
            "start": {"dateTime": "2026-04-24T15:00:00Z"},
            "end": {"dateTime": "2026-04-24T15:30:00Z"},
            "attendees": [{"email": "sarah@example.com"}],
        }
    ]}
    return svc


@pytest.fixture
def mock_gmail_service():
    svc = MagicMock()
    svc.users.return_value.messages.return_value.list.return_value.execute.return_value = {
        "messages": [{"id": "msg1", "threadId": "thread1"}]
    }
    svc.users.return_value.messages.return_value.get.return_value.execute.return_value = {
        "id": "msg1",
        "payload": {
            "headers": [
                {"name": "Subject", "value": "Project update"},
                {"name": "From", "value": "sarah@example.com"},
                {"name": "Date", "value": "Thu, 24 Apr 2026 10:00:00 +0000"},
            ],
            "body": {"data": "SGVsbG8gUnVpIQ=="},  # base64 "Hello Rui!"
        },
    }
    return svc


@pytest.mark.asyncio
async def test_list_events(mock_calendar_service):
    ct = CalendarTools(mock_calendar_service)
    result = await ct.dispatch("list_events", {"days": 7})
    assert "Call with Sarah" in result


@pytest.mark.asyncio
async def test_list_emails(mock_gmail_service):
    et = EmailTools(mock_gmail_service, allow_send=False)
    result = await et.dispatch("list_emails", {"query": "is:unread", "limit": 5})
    assert result  # any non-empty result


@pytest.mark.asyncio
async def test_send_email_blocked_without_flag(mock_gmail_service):
    et = EmailTools(mock_gmail_service, allow_send=False)
    result = await et.dispatch("send_email", {"email_id": "msg1"})
    assert "ERROR" in result or "disabled" in result.lower()
