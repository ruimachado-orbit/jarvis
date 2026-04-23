"""Google OAuth 2.0 flow. Run `jarvis auth google` to authorise."""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

SCOPES = [
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.send",
]


def get_credentials(credentials_path: Path, token_path: Path):
    """Return valid Google credentials, refreshing or re-authorising as needed."""
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    credentials_path = Path(credentials_path).expanduser()
    token_path = Path(token_path).expanduser()
    creds = None

    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not credentials_path.exists():
                raise FileNotFoundError(
                    f"Google credentials file not found: {credentials_path}\n"
                    "Download it from Google Cloud Console and set JARVIS_GOOGLE_CREDENTIALS."
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), SCOPES)
            creds = flow.run_local_server(port=0)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(creds.to_json())
        log.info("Google token saved to %s", token_path)

    return creds


def build_services(credentials_path: Path, token_path: Path):
    """Return (calendar_service, gmail_service) ready to use."""
    from googleapiclient.discovery import build

    creds = get_credentials(credentials_path, token_path)
    calendar = build("calendar", "v3", credentials=creds)
    gmail = build("gmail", "v1", credentials=creds)
    return calendar, gmail
