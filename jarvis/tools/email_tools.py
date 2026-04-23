"""Gmail tool implementations."""

from __future__ import annotations

import asyncio
import base64
import logging
from functools import partial
from typing import Any

log = logging.getLogger(__name__)


class EmailTools:
    def __init__(self, service, allow_send: bool = False) -> None:
        self._svc = service
        self._allow_send = allow_send

    async def dispatch(self, name: str, args: dict[str, Any]) -> str:
        try:
            loop = asyncio.get_running_loop()
            if name == "list_emails":
                return await loop.run_in_executor(
                    None, partial(self._list_emails, args.get("query", "is:unread"), args.get("limit", 10))
                )
            if name == "read_email":
                return await loop.run_in_executor(
                    None, partial(self._read_email, args["email_id"])
                )
            if name == "draft_reply":
                return await loop.run_in_executor(
                    None, partial(self._draft_reply, args["email_id"], args["body"])
                )
            if name == "send_email":
                if not self._allow_send:
                    return "ERROR: sending email is disabled. Set JARVIS_ALLOW_SEND_EMAIL=true."
                return await loop.run_in_executor(
                    None, partial(self._send_email, args["email_id"])
                )
            return f"ERROR: unknown email tool {name!r}"
        except Exception as e:
            log.exception("email tool %s failed", name)
            return f"ERROR: {e}"

    def _list_emails(self, query: str, limit: int) -> str:
        result = (
            self._svc.users()
            .messages()
            .list(userId="me", q=query, maxResults=limit)
            .execute()
        )
        messages = result.get("messages", [])
        if not messages:
            return "No emails found."
        lines = []
        for m in messages:
            msg = self._svc.users().messages().get(
                userId="me", id=m["id"], format="metadata",
                metadataHeaders=["Subject", "From", "Date"]
            ).execute()
            headers = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}
            lines.append(
                f"id={m['id']} | {headers.get('Date','')} | From: {headers.get('From','')} | {headers.get('Subject','(no subject)')}"
            )
        return "\n".join(lines)

    def _read_email(self, email_id: str) -> str:
        msg = self._svc.users().messages().get(userId="me", id=email_id, format="full").execute()
        headers = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}
        body_data = msg.get("payload", {}).get("body", {}).get("data", "")
        body = ""
        if body_data:
            body = base64.urlsafe_b64decode(body_data + "==").decode("utf-8", errors="replace")
        return (
            f"From: {headers.get('From','')}\n"
            f"Subject: {headers.get('Subject','')}\n"
            f"Date: {headers.get('Date','')}\n\n"
            f"{body[:3000]}"
        )

    def _draft_reply(self, email_id: str, body: str) -> str:
        import email.mime.text as _mime

        original = self._svc.users().messages().get(
            userId="me", id=email_id, format="metadata",
            metadataHeaders=["Subject", "From", "Message-ID"]
        ).execute()
        headers = {h["name"]: h["value"] for h in original.get("payload", {}).get("headers", [])}

        msg = _mime.MIMEText(body)
        msg["To"] = headers.get("From", "")
        msg["Subject"] = "Re: " + headers.get("Subject", "")
        msg["In-Reply-To"] = headers.get("Message-ID", "")
        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
        draft = self._svc.users().drafts().create(
            userId="me", body={"message": {"raw": raw, "threadId": original.get("threadId")}}
        ).execute()
        return f"Draft created: id={draft['id']}"

    def _send_email(self, email_id: str) -> str:
        result = self._svc.users().drafts().send(userId="me", body={"id": email_id}).execute()
        return f"Email sent: id={result.get('id')}"
