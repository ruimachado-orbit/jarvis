"""Google Calendar tool implementations."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from functools import partial
from typing import Any

log = logging.getLogger(__name__)


class CalendarTools:
    def __init__(self, service) -> None:
        self._svc = service

    async def dispatch(self, name: str, args: dict[str, Any]) -> str:
        try:
            loop = asyncio.get_running_loop()
            if name == "list_events":
                return await loop.run_in_executor(
                    None, partial(self._list_events, args.get("days", 7))
                )
            if name == "create_event":
                return await loop.run_in_executor(
                    None, partial(self._create_event, **args)
                )
            if name == "update_event":
                return await loop.run_in_executor(
                    None, partial(self._update_event, **args)
                )
            if name == "find_free_slot":
                return await loop.run_in_executor(
                    None, partial(self._find_free_slot, **args)
                )
            return f"ERROR: unknown calendar tool {name!r}"
        except Exception as e:
            log.exception("calendar tool %s failed", name)
            return f"ERROR: {e}"

    def _list_events(self, days: int) -> str:
        now = datetime.now(timezone.utc)
        end = now + timedelta(days=days)
        result = (
            self._svc.events()
            .list(
                calendarId="primary",
                timeMin=now.isoformat(),
                timeMax=end.isoformat(),
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
        )
        items = result.get("items", [])
        if not items:
            return f"No events in the next {days} days."
        lines = []
        for e in items:
            start = e.get("start", {}).get("dateTime") or e.get("start", {}).get("date", "")
            attendees = ", ".join(a["email"] for a in e.get("attendees", []))
            line = f"- {e.get('summary','(no title)')} at {start}"
            if attendees:
                line += f" with {attendees}"
            lines.append(line)
        return "\n".join(lines)

    def _create_event(
        self,
        title: str,
        start_time: str,
        duration_minutes: int = 60,
        attendees: list[str] | None = None,
    ) -> str:
        start = datetime.fromisoformat(start_time)
        end = start + timedelta(minutes=duration_minutes)
        body: dict[str, Any] = {
            "summary": title,
            "start": {"dateTime": start.isoformat(), "timeZone": "UTC"},
            "end": {"dateTime": end.isoformat(), "timeZone": "UTC"},
        }
        if attendees:
            body["attendees"] = [{"email": a} for a in attendees]
        event = self._svc.events().insert(calendarId="primary", body=body).execute()
        return f"Event created: {event.get('htmlLink', event.get('id'))}"

    def _update_event(self, event_id: str, **kwargs) -> str:
        event = self._svc.events().get(calendarId="primary", eventId=event_id).execute()
        if "title" in kwargs:
            event["summary"] = kwargs["title"]
        if "start_time" in kwargs:
            start = datetime.fromisoformat(kwargs["start_time"])
            dur = int(kwargs.get("duration_minutes", 60))
            event["start"] = {"dateTime": start.isoformat(), "timeZone": "UTC"}
            event["end"] = {"dateTime": (start + timedelta(minutes=dur)).isoformat(), "timeZone": "UTC"}
        updated = self._svc.events().update(
            calendarId="primary", eventId=event_id, body=event
        ).execute()
        return f"Event updated: {updated.get('summary')}"

    def _find_free_slot(self, duration_minutes: int, within_days: int = 7) -> str:
        now = datetime.now(timezone.utc)
        end = now + timedelta(days=within_days)
        result = (
            self._svc.events()
            .list(
                calendarId="primary",
                timeMin=now.isoformat(),
                timeMax=end.isoformat(),
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
        )
        busy = []
        for e in result.get("items", []):
            s = e.get("start", {}).get("dateTime")
            en = e.get("end", {}).get("dateTime")
            if s and en:
                busy.append((datetime.fromisoformat(s), datetime.fromisoformat(en)))

        slot_start = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        while slot_start < end:
            slot_end = slot_start + timedelta(minutes=duration_minutes)
            if 9 <= slot_start.hour < 18:
                overlap = any(s < slot_end and e > slot_start for s, e in busy)
                if not overlap:
                    return f"Next free {duration_minutes}-min slot: {slot_start.strftime('%A %d %b at %H:%M UTC')}"
            slot_start += timedelta(minutes=30)
        return f"No free {duration_minutes}-min slot found in the next {within_days} days."
