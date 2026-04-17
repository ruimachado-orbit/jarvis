from pathlib import Path

import pytest

from jarvis.config import Settings
from jarvis.tools import ToolError, Toolbox


def _toolbox(tmp_path: Path, **overrides) -> Toolbox:
    settings = Settings(workspace=tmp_path, **overrides)
    return Toolbox(settings)


def test_read_file_inside_workspace(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("world\n", encoding="utf-8")
    tb = _toolbox(tmp_path)
    out = tb.read_file("hello.txt")
    assert "world" in out


def test_read_file_rejects_escape(tmp_path: Path) -> None:
    tb = _toolbox(tmp_path)
    with pytest.raises(ToolError):
        tb.read_file("../etc/passwd")


def test_writes_disabled_by_default(tmp_path: Path) -> None:
    tb = _toolbox(tmp_path)
    with pytest.raises(ToolError):
        tb.write_file("x.txt", "y")


def test_writes_enabled_when_flag_set(tmp_path: Path) -> None:
    tb = _toolbox(tmp_path, allow_writes=True)
    tb.write_file("x.txt", "y")
    assert (tmp_path / "x.txt").read_text() == "y"
