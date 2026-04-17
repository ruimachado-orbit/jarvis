"""Filesystem watcher that notifies Telegram on change.

Optionally runs a command (for example ``pytest``) after each change and pushes
the exit status plus a tail of the output.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from pathlib import Path

from jarvis.config import Settings
from jarvis.telegram_bot import TelegramBridge

log = logging.getLogger(__name__)

OUTPUT_TAIL = 1500


async def run_watch(
    settings: Settings,
    bridge: TelegramBridge,
    paths: Iterable[str | Path],
    command: str | None,
    debounce_ms: int,
) -> None:
    try:
        from watchfiles import awatch
    except ImportError as e:
        raise RuntimeError(
            "watchfiles is not installed. Run: pip install watchfiles"
        ) from e

    resolved = [str(Path(p).resolve()) for p in paths]
    if not resolved:
        raise ValueError("no paths to watch")

    log.info("watching: %s  command=%s", resolved, command)
    await bridge.notify(f"Jarvis watching {', '.join(resolved)}")

    async for changes in awatch(*resolved, step=debounce_ms):
        summary_lines = [f"{len(changes)} change(s) in {', '.join(resolved)}"]
        for change, path in list(changes)[:5]:
            summary_lines.append(f"  {change.name.lower()}: {path}")
        if len(changes) > 5:
            summary_lines.append(f"  ... +{len(changes) - 5} more")

        if command:
            rc, tail = await _run_command(command, cwd=str(settings.workspace))
            status = "ok" if rc == 0 else f"FAILED (exit {rc})"
            summary_lines.append(f"$ {command} -> {status}")
            if tail:
                summary_lines.append(tail)

        message = "\n".join(summary_lines)
        log.info("notify: %s", message.splitlines()[0])
        await bridge.notify(message)


async def _run_command(command: str, cwd: str) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_shell(
        command,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    stdout, _ = await proc.communicate()
    text = stdout.decode("utf-8", errors="replace")
    if len(text) > OUTPUT_TAIL:
        text = "... " + text[-OUTPUT_TAIL:]
    return proc.returncode or 0, text.rstrip()
