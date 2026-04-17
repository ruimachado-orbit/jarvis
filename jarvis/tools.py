"""Coding tools exposed to the agent via OpenAI-style function calling.

The tool surface is deliberately small. Write and shell tools are gated by settings so
voice-only mode stays safe by default.
"""

from __future__ import annotations

import asyncio
import fnmatch
import shlex
from pathlib import Path
from typing import Any

from jarvis.config import Settings

MAX_FILE_BYTES = 200_000
MAX_GREP_MATCHES = 200
MAX_SHELL_OUTPUT = 20_000


class ToolError(Exception):
    pass


class Toolbox:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.root = settings.workspace.resolve()

    # ----- path safety -----

    def _resolve(self, path: str) -> Path:
        p = (self.root / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
        try:
            p.relative_to(self.root)
        except ValueError as e:
            raise ToolError(f"path {path!r} is outside the workspace {self.root}") from e
        return p

    # ----- tool implementations -----

    def read_file(self, path: str, start_line: int = 1, end_line: int | None = None) -> str:
        p = self._resolve(path)
        if not p.is_file():
            raise ToolError(f"{path!r} is not a file")
        if p.stat().st_size > MAX_FILE_BYTES:
            raise ToolError(f"{path!r} is larger than {MAX_FILE_BYTES} bytes; slice it with start_line/end_line")
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(1, start_line) - 1
        end = end_line if end_line is not None else len(lines)
        sliced = lines[start:end]
        return "\n".join(f"{i + start + 1:>5}\t{line}" for i, line in enumerate(sliced))

    def list_dir(self, path: str = ".") -> str:
        p = self._resolve(path)
        if not p.is_dir():
            raise ToolError(f"{path!r} is not a directory")
        entries = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name))
        out = []
        for e in entries:
            if e.name.startswith("."):
                continue
            marker = "/" if e.is_dir() else ""
            out.append(f"{e.name}{marker}")
        return "\n".join(out) or "(empty)"

    def grep(self, pattern: str, path: str = ".", glob: str = "*") -> str:
        p = self._resolve(path)
        import re

        try:
            regex = re.compile(pattern)
        except re.error as e:
            raise ToolError(f"invalid regex: {e}") from e

        matches: list[str] = []
        iterator = p.rglob(glob) if p.is_dir() else [p]
        for fp in iterator:
            if not fp.is_file():
                continue
            if fnmatch.fnmatch(fp.name, "*.pyc") or ".git/" in str(fp):
                continue
            try:
                for lineno, line in enumerate(fp.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                    if regex.search(line):
                        rel = fp.relative_to(self.root)
                        matches.append(f"{rel}:{lineno}: {line.strip()[:300]}")
                        if len(matches) >= MAX_GREP_MATCHES:
                            matches.append(f"... truncated at {MAX_GREP_MATCHES} matches")
                            return "\n".join(matches)
            except (UnicodeDecodeError, PermissionError):
                continue
        return "\n".join(matches) or "(no matches)"

    def write_file(self, path: str, content: str) -> str:
        if not self.settings.allow_writes:
            raise ToolError("writes are disabled; set JARVIS_ALLOW_WRITES=true to enable")
        p = self._resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"wrote {len(content)} bytes to {p.relative_to(self.root)}"

    async def run_shell(self, command: str, timeout: float = 60.0) -> str:
        if not self.settings.allow_shell:
            raise ToolError("shell execution is disabled; set JARVIS_ALLOW_SHELL=true to enable")
        # Deny a few obvious footguns even when shell is enabled.
        lowered = command.lower()
        for banned in ("rm -rf /", "mkfs", ":(){ :|:& };:", "dd if=/dev/zero"):
            if banned in lowered:
                raise ToolError(f"refusing to run dangerous command containing {banned!r}")

        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=str(self.root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError as e:
            proc.kill()
            raise ToolError(f"command timed out after {timeout}s") from e
        text = stdout.decode("utf-8", errors="replace")
        if len(text) > MAX_SHELL_OUTPUT:
            text = text[:MAX_SHELL_OUTPUT] + "\n... (truncated)"
        return f"[exit {proc.returncode}]\n{text}"

    # ----- dispatch -----

    async def dispatch(self, name: str, args: dict[str, Any]) -> str:
        try:
            if name == "read_file":
                return self.read_file(**args)
            if name == "list_dir":
                return self.list_dir(**args)
            if name == "grep":
                return self.grep(**args)
            if name == "write_file":
                return self.write_file(**args)
            if name == "run_shell":
                return await self.run_shell(**args)
            raise ToolError(f"unknown tool {name!r}")
        except ToolError as e:
            return f"ERROR: {e}"
        except TypeError as e:
            return f"ERROR: bad arguments for {name}: {e}"


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the workspace. Optionally slice by line range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the workspace."},
                    "start_line": {"type": "integer", "default": 1},
                    "end_line": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files and folders inside a workspace directory.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "default": "."}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search the workspace for a regex pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string", "default": "."},
                    "glob": {"type": "string", "default": "*", "description": "Filename glob, e.g. *.py"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Overwrite a file in the workspace. Gated by JARVIS_ALLOW_WRITES.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Run a shell command in the workspace. Gated by JARVIS_ALLOW_SHELL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "number", "default": 60},
                },
                "required": ["command"],
            },
        },
    },
]


def describe_command(command: str) -> str:
    """Human-readable one-liner for voice confirmation."""
    try:
        parts = shlex.split(command)
    except ValueError:
        return command
    return " ".join(parts)
