"""File/CI watchdog rewired as a proactive monitor node."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Callable, Awaitable

from watchfiles import awatch

log = logging.getLogger(__name__)


class WatchdogMonitor:
    def __init__(
        self,
        paths: list[str],
        notify_fn: Callable[[str], Awaitable[None]],
        command: str | None = None,
        debounce_ms: int = 400,
    ) -> None:
        self._paths = paths
        self._notify = notify_fn
        self._command = command
        self._debounce_ms = debounce_ms

    async def run_forever(self) -> None:
        log.info("watchdog started on paths: %s", self._paths)
        async for changes in awatch(*self._paths, debounce=self._debounce_ms):
            changed_files = [str(Path(c[1])) for c in changes]
            summary = f"Files changed: {', '.join(changed_files[:5])}"
            if len(changed_files) > 5:
                summary += f" (+{len(changed_files) - 5} more)"

            if self._command:
                proc = await asyncio.create_subprocess_shell(
                    self._command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=120)
                output = stdout.decode("utf-8", errors="replace")[-1000:]
                summary += f"\n`{self._command}` exit={proc.returncode}\n{output}"

            await self._notify(summary)
