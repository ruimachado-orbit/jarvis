"""CLI entry points for Jarvis.

Sub-commands:
    jarvis voice     — hands-free voice loop (wake word → VAD → STT → LLM → streaming TTS)
    jarvis chat      — text REPL with streaming output (no audio deps needed)
    jarvis ask       — one-shot text query
    jarvis telegram  — run only the Telegram bridge
    jarvis notify    — send a one-off Telegram notification
    jarvis watch     — watch workspace paths and push Telegram notifications
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.prompt import Prompt

from jarvis.agent import Agent
from jarvis.config import get_settings
from jarvis.llm import LLMClient
from jarvis.tools import Toolbox

app = typer.Typer(add_completion=False, help="Jarvis — voice-first local coding assistant.")
console = Console()
log = logging.getLogger("jarvis")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _build_agent() -> Agent:
    settings = get_settings()
    llm = LLMClient(settings)
    tools = Toolbox(settings)
    return Agent(settings, llm, tools)


def _install_signal_handlers(stop: asyncio.Event) -> None:
    def _sig(*_a) -> None:
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            asyncio.get_event_loop().add_signal_handler(sig, _sig)
        except NotImplementedError:  # pragma: no cover - windows
            signal.signal(sig, lambda *_: stop.set())


# ----- voice -----

@app.command()
def voice(
    stream: Optional[bool] = typer.Option(
        None,
        "--stream/--no-stream",
        help="Override JARVIS_STREAM_VOICE. Streams TTS sentence-by-sentence.",
    ),
    wake: Optional[bool] = typer.Option(
        None,
        "--wake/--no-wake",
        help="Override JARVIS_WAKE_ENABLED. Disable for always-on listening.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Hands-free voice loop. Wake word → record → transcribe → stream reply."""
    _setup_logging(verbose)
    asyncio.run(_voice_main(stream=stream, wake=wake))


async def _voice_main(stream: bool | None, wake: bool | None) -> None:
    from jarvis.audio import AudioIO
    from jarvis.streaming import speak_stream
    from jarvis.stt import STT
    from jarvis.telegram_bot import TelegramBridge
    from jarvis.tts import TTS
    from jarvis.wake import WakeWord

    settings = get_settings()
    if stream is not None:
        settings.stream_voice = stream
    if wake is not None:
        settings.wake_enabled = wake

    agent = _build_agent()
    stt = STT(settings)
    tts = TTS(settings)
    audio = AudioIO(settings)
    wake_detector = WakeWord(settings)

    bridge = TelegramBridge(settings, agent.respond)
    await bridge.start()

    stop = asyncio.Event()
    _install_signal_handlers(stop)

    mode = "streaming" if settings.stream_voice else "batch"
    wake_mode = "wake-word" if settings.wake_enabled else "always-on"
    console.print(
        f"[bold green]Jarvis online.[/] mode={mode} listening={wake_mode}. Ctrl-C to quit."
    )

    try:
        while not stop.is_set():
            if settings.wake_enabled:
                console.print(f"[dim]waiting for wake phrase '{settings.wake_word}'...[/dim]")
                await wake_detector.wait_for_wake()
                if stop.is_set():
                    break
                console.print("[yellow]yes?[/yellow]")

            console.print("[dim]listening...[/dim]")
            pcm = await audio.record_utterance()
            if pcm.size == 0:
                continue
            text = await stt.transcribe(pcm, settings.sample_rate)
            if not text:
                continue
            console.print(f"[cyan]you[/cyan]: {text}")

            if (
                not wake_detector.ready
                and settings.require_wake_word
                and not wake_detector.matches_text(text)
            ):
                console.print("[dim](no wake word in transcript, ignoring)[/dim]")
                continue

            console.print("[magenta]jarvis[/magenta]: ", end="")
            if settings.stream_voice:
                async def _sentences():
                    async for sentence in agent.respond_stream(text):
                        console.print(sentence)
                        yield sentence

                await speak_stream(_sentences(), tts, audio)
            else:
                reply = await agent.respond(text)
                console.print(reply)
                if reply:
                    pcm_out, sr = await tts.synthesize(reply)
                    await audio.play(pcm_out, sr)
    finally:
        await bridge.stop()


# ----- chat -----

@app.command()
def chat(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    """Text REPL with streaming output. No audio dependencies required."""
    _setup_logging(verbose)
    asyncio.run(_chat_main())


async def _chat_main() -> None:
    agent = _build_agent()
    console.print(
        "[bold green]Jarvis (chat).[/] Type a question. "
        "Commands: [dim]/reset[/dim], [dim]/history[/dim]. Ctrl-D to exit."
    )
    while True:
        try:
            line = Prompt.ask("[cyan]you[/cyan]")
        except (EOFError, KeyboardInterrupt):
            console.print()
            return
        line = line.strip()
        if not line:
            continue
        if line in {"/reset", "/clear"}:
            agent.reset()
            console.print("[dim]history cleared[/dim]")
            continue
        if line == "/history":
            for msg in agent.history:
                console.print(f"[dim]{msg['role']}[/dim]: {str(msg.get('content'))[:200]}")
            continue
        console.print("[magenta]jarvis[/magenta]: ", end="")
        try:
            async for sentence in agent.respond_stream(line):
                console.print(sentence + " ", end="")
            console.print()
        except Exception as e:
            console.print(f"[red]error:[/red] {e}")


# ----- ask (one-shot) -----

@app.command()
def ask(
    question: str = typer.Argument(..., help="The question to ask Jarvis."),
    json_output: bool = typer.Option(False, "--json", help="Emit raw text only."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """One-shot text query. Prints the answer and exits."""
    _setup_logging(verbose)
    asyncio.run(_ask_main(question, json_output))


async def _ask_main(question: str, raw: bool) -> None:
    agent = _build_agent()
    reply = await agent.respond(question)
    if raw:
        print(reply)
    else:
        console.print(f"[magenta]jarvis[/magenta]: {reply}")


# ----- telegram-only -----

@app.command()
def telegram(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    """Run only the Telegram bridge (no local mic)."""
    _setup_logging(verbose)
    asyncio.run(_telegram_main())


async def _telegram_main() -> None:
    from jarvis.telegram_bot import TelegramBridge

    settings = get_settings()
    agent = _build_agent()
    bridge = TelegramBridge(settings, agent.respond)
    if not bridge.enabled():
        console.print("[red]JARVIS_TELEGRAM_TOKEN is not set.[/red]")
        sys.exit(1)
    await bridge.start()
    console.print("[bold green]Telegram bridge running.[/] Ctrl-C to quit.")
    stop = asyncio.Event()
    _install_signal_handlers(stop)
    try:
        await stop.wait()
    finally:
        await bridge.stop()


# ----- one-off notification -----

@app.command()
def notify(
    message: str = typer.Argument(..., help="Text to send to the notify chat."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Send a push notification via the Telegram bot."""
    _setup_logging(verbose)
    asyncio.run(_notify_main(message))


async def _notify_main(message: str) -> None:
    from jarvis.telegram_bot import TelegramBridge

    settings = get_settings()

    async def _noop(_: str) -> str:
        return ""

    bridge = TelegramBridge(settings, _noop)
    if not bridge.enabled():
        console.print("[red]telegram disabled[/red]")
        sys.exit(1)
    await bridge.start()
    try:
        await bridge.notify(message)
        console.print("[green]sent[/green]")
    finally:
        await bridge.stop()


# ----- watch -----

@app.command()
def watch(
    paths: list[str] = typer.Argument(None, help="Paths to watch. Defaults to JARVIS_WATCH_PATHS."),
    command: Optional[str] = typer.Option(
        None,
        "--command",
        "-c",
        help="Command to run on change. Overrides JARVIS_WATCH_COMMAND.",
    ),
    debounce_ms: Optional[int] = typer.Option(
        None,
        "--debounce-ms",
        help="Debounce window in milliseconds. Overrides JARVIS_WATCH_DEBOUNCE_MS.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Watch workspace paths and push changes to Telegram."""
    _setup_logging(verbose)
    asyncio.run(_watch_main(paths, command, debounce_ms))


async def _watch_main(
    paths: list[str] | None,
    command: str | None,
    debounce_ms: int | None,
) -> None:
    from jarvis.telegram_bot import TelegramBridge
    from jarvis.watch import run_watch

    settings = get_settings()
    target_paths = paths or settings.watch_path_list
    cmd = command if command is not None else settings.watch_command
    debounce = debounce_ms if debounce_ms is not None else settings.watch_debounce_ms

    async def _noop(_: str) -> str:
        return ""

    bridge = TelegramBridge(settings, _noop)
    if not bridge.enabled():
        console.print("[red]Telegram is required for watch mode. Set JARVIS_TELEGRAM_TOKEN.[/red]")
        sys.exit(1)
    await bridge.start()
    console.print(
        f"[bold green]Watching[/] {', '.join(target_paths)} "
        f"(command={cmd!r}, debounce={debounce}ms). Ctrl-C to stop."
    )
    try:
        await run_watch(settings, bridge, target_paths, cmd, debounce)
    except KeyboardInterrupt:
        pass
    finally:
        await bridge.stop()


if __name__ == "__main__":
    app()
