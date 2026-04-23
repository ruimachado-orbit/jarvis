"""Unified tool registry: schemas + dispatch for all Jarvis tools."""

from __future__ import annotations

from typing import Any

from jarvis.core.config import Settings
from jarvis.tools import web_tools
from jarvis.tools.coding import Toolbox as CodingToolbox
from jarvis.tools.coding import ToolError


class Toolbox:
    def __init__(self, settings: Settings, mem0_store=None) -> None:
        self._coding = CodingToolbox(settings)
        self._settings = settings
        self._mem0 = mem0_store
        self._calendar = None
        self._email = None

    def inject_google(self, calendar_tools, email_tools) -> None:
        self._calendar = calendar_tools
        self._email = email_tools

    async def dispatch(self, name: str, args: dict[str, Any]) -> str:
        try:
            # Coding tools
            if name == "read_file":
                return self._coding.read_file(**args)
            if name == "list_dir":
                return self._coding.list_dir(**args)
            if name == "grep":
                return self._coding.grep(**args)
            if name == "write_file":
                return self._coding.write_file(**args)
            if name == "run_shell":
                return await self._coding.run_shell(**args)

            # Web tools
            if name == "web_search":
                return await web_tools.web_search(
                    args["query"],
                    brave_api_key=self._settings.brave_api_key,
                    limit=args.get("limit", 5),
                )
            if name == "fetch_page":
                return await web_tools.fetch_page(args["url"])

            # Memory tools
            if name == "remember":
                if self._mem0:
                    self._mem0.remember(args["fact"])
                return "Remembered."
            if name == "recall":
                if self._mem0:
                    results = self._mem0.retrieve(args["query"], limit=args.get("limit", 5))
                    return "\n".join(results) or "(nothing found)"
                return "(memory not available)"
            if name == "forget":
                if self._mem0:
                    self._mem0.forget(args["fact"])
                return "Forgotten."

            # Calendar tools
            if name in {"list_events", "create_event", "update_event", "find_free_slot"}:
                if self._calendar is None:
                    return "ERROR: Google Calendar not configured. Run `jarvis auth google`."
                return await self._calendar.dispatch(name, args)

            # Email tools
            if name in {"list_emails", "read_email", "draft_reply", "send_email"}:
                if self._email is None:
                    return "ERROR: Gmail not configured. Run `jarvis auth google`."
                return await self._email.dispatch(name, args)

            # Notification
            if name == "notify":
                return f"[notify] {args.get('message', '')}"

            return f"ERROR: unknown tool {name!r}"

        except ToolError as e:
            return f"ERROR: {e}"
        except TypeError as e:
            return f"ERROR: bad arguments for {name}: {e}"
        except Exception as e:
            return f"ERROR: {e}"


TOOL_SCHEMAS: list[dict[str, Any]] = [
    # Coding
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a file from the workspace.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "start_line": {"type": "integer"},
            "end_line": {"type": "integer"},
        }, "required": ["path"]},
    }},
    {"type": "function", "function": {
        "name": "list_dir",
        "description": "List files and folders in a workspace directory.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string", "default": "."}}},
    }},
    {"type": "function", "function": {
        "name": "grep",
        "description": "Search the workspace for a regex pattern.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string", "default": "."},
            "glob": {"type": "string", "default": "*"},
        }, "required": ["pattern"]},
    }},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Overwrite a file. Gated by JARVIS_ALLOW_WRITES. Requires confirmation.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        }, "required": ["path", "content"]},
    }},
    {"type": "function", "function": {
        "name": "run_shell",
        "description": "Run a shell command. Gated by JARVIS_ALLOW_SHELL. Requires confirmation.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"},
            "timeout": {"type": "number", "default": 60},
        }, "required": ["command"]},
    }},
    # Web
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Search the web for information.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "default": 5},
        }, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "fetch_page",
        "description": "Fetch a URL and return clean extracted text.",
        "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
    }},
    # Memory
    {"type": "function", "function": {
        "name": "remember",
        "description": "Explicitly store a fact in long-term memory.",
        "parameters": {"type": "object", "properties": {"fact": {"type": "string"}}, "required": ["fact"]},
    }},
    {"type": "function", "function": {
        "name": "recall",
        "description": "Retrieve relevant memories for a query.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "default": 5},
        }, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "forget",
        "description": "Remove a fact from long-term memory.",
        "parameters": {"type": "object", "properties": {"fact": {"type": "string"}}, "required": ["fact"]},
    }},
    # Calendar
    {"type": "function", "function": {
        "name": "list_events",
        "description": "List upcoming Google Calendar events.",
        "parameters": {"type": "object", "properties": {"days": {"type": "integer", "default": 7}}},
    }},
    {"type": "function", "function": {
        "name": "create_event",
        "description": "Create a calendar event. Requires confirmation.",
        "parameters": {"type": "object", "properties": {
            "title": {"type": "string"},
            "start_time": {"type": "string", "description": "ISO 8601 datetime"},
            "duration_minutes": {"type": "integer"},
            "attendees": {"type": "array", "items": {"type": "string"}},
        }, "required": ["title", "start_time"]},
    }},
    {"type": "function", "function": {
        "name": "update_event",
        "description": "Update an existing calendar event. Requires confirmation.",
        "parameters": {"type": "object", "properties": {
            "event_id": {"type": "string"},
            "title": {"type": "string"},
            "start_time": {"type": "string"},
            "duration_minutes": {"type": "integer"},
        }, "required": ["event_id"]},
    }},
    {"type": "function", "function": {
        "name": "find_free_slot",
        "description": "Find the next available time slot in the calendar.",
        "parameters": {"type": "object", "properties": {
            "duration_minutes": {"type": "integer"},
            "within_days": {"type": "integer", "default": 7},
        }, "required": ["duration_minutes"]},
    }},
    # Email
    {"type": "function", "function": {
        "name": "list_emails",
        "description": "List emails from Gmail matching a query.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "default": "is:unread"},
            "limit": {"type": "integer", "default": 10},
        }},
    }},
    {"type": "function", "function": {
        "name": "read_email",
        "description": "Read the full content of an email by ID.",
        "parameters": {"type": "object", "properties": {"email_id": {"type": "string"}}, "required": ["email_id"]},
    }},
    {"type": "function", "function": {
        "name": "draft_reply",
        "description": "Create a draft reply to an email. Does not send.",
        "parameters": {"type": "object", "properties": {
            "email_id": {"type": "string"},
            "body": {"type": "string"},
        }, "required": ["email_id", "body"]},
    }},
    {"type": "function", "function": {
        "name": "send_email",
        "description": "Send a drafted email. Gated by JARVIS_ALLOW_SEND_EMAIL. Requires confirmation.",
        "parameters": {"type": "object", "properties": {"email_id": {"type": "string"}}, "required": ["email_id"]},
    }},
    # Notification
    {"type": "function", "function": {
        "name": "notify",
        "description": "Send a push notification via Telegram.",
        "parameters": {"type": "object", "properties": {
            "message": {"type": "string"},
            "urgency": {"type": "string", "enum": ["low", "normal", "high"], "default": "normal"},
        }, "required": ["message"]},
    }},
]
