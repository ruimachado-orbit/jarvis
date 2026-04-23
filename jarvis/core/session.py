"""Shared runtime factory: builds all Jarvis components from Settings."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import anthropic

from jarvis.core.config import Settings
from jarvis.graph.agent import build_graph
from jarvis.memory.mem0_store import Mem0Store
from jarvis.tools.registry import Toolbox

log = logging.getLogger(__name__)


def build_session(settings: Settings) -> dict[str, Any]:
    """Build and return all shared runtime components."""
    client = anthropic.AsyncAnthropic()  # reads ANTHROPIC_API_KEY from env

    mem0: Mem0Store | None = None
    if settings.mem0_enabled:
        mem0 = Mem0Store(
            path=str(Path(settings.mem0_path).expanduser()),
            user_id="jarvis_user",
            enabled=True,
        )

    toolbox = Toolbox(settings, mem0_store=mem0)
    graph = build_graph(settings, client, toolbox, mem0)

    return {
        "client": client,
        "mem0": mem0,
        "toolbox": toolbox,
        "graph": graph,
        "settings": settings,
    }


def inject_google(session: dict[str, Any]) -> None:
    """Attempt to load Google credentials and inject calendar/email tools."""
    settings: Settings = session["settings"]
    creds_path = Path(settings.google_credentials).expanduser()
    token_path = Path(settings.google_token).expanduser()

    if not token_path.exists():
        log.info("Google token not found at %s — skipping Google tools", token_path)
        return

    try:
        from jarvis.integrations.google_auth import build_services
        from jarvis.tools.calendar_tools import CalendarTools
        from jarvis.tools.email_tools import EmailTools

        cal_svc, gmail_svc = build_services(creds_path, token_path)
        cal_tools = CalendarTools(cal_svc)
        email_tools = EmailTools(gmail_svc, allow_send=settings.allow_send_email)
        session["toolbox"].inject_google(cal_tools, email_tools)
        log.info("Google Calendar + Gmail tools injected")
    except Exception as e:
        log.warning("Could not load Google tools: %s", e)
