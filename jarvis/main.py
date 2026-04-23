"""CLI entry points for Jarvis v0.2.

Sub-commands:
    jarvis voice      — hands-free voice loop
    jarvis chat       — text REPL
    jarvis ask        — one-shot text query
    jarvis telegram   — Telegram bridge only
    jarvis notify     — send a push notification
    jarvis watch      — watch paths, push to Telegram
    jarvis auth       — authentication subcommands
      jarvis auth google  — Google OAuth flow
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

from jarvis.core.config import get_settings, reload_settings
from jarvis.core.session import build_session, inject_google
from jarvis.graph.agent import run_turn

app = typer.Typer(add_completion=False, help="Jarvis — proactive personal AI.")
auth_app = typer.Typer(add_completion=False, help="Authentication commands.")
app.add_typer(auth_app, name="auth")
console = Console()
log = logging.getLogger("jarvis")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _install_signal_handlers(stop: asyncio.Event) -> None:
    def _sig(*_a) -> None:
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            asyncio.get_event_loop().add_signal_handler(sig, _sig)
        except NotImplementedError:
            signal.signal(sig, lambda *_: stop.set())


# ----- voice -----

@app.command()
def voice(
    stream: Optional[bool] = typer.Option(None, "--stream/--no-stream"),
    wake: Optional[bool] = typer.Option(None, "--wake/--no-wake"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Hands-free voice loop. Wake word → record → transcribe → reply."""
    _setup_logging(verbose)
    asyncio.run(_voice_main(stream=stream, wake=wake))


async def _voice_main(stream: bool | None, wake: bool | None) -> None:
    from jarvis.voice.audio import AudioIO
    from jarvis.voice.stt import STT
    from jarvis.voice.tts import TTS
    from jarvis.voice.wake import WakeWord
    from jarvis.telegram_bot import TelegramBridge
    from jarvis.graph.nodes import _build_claude_prompt, _build_system_with_memories, TOOL_CALL_RE
    import json, re

    settings = get_settings()
    if stream is not None:
        settings.stream_voice = stream
    if wake is not None:
        settings.wake_enabled = wake

    session = build_session(settings)
    inject_google(session)
    toolbox = session["toolbox"]

    stt = STT(settings)
    tts = TTS(settings)
    audio = AudioIO(settings)

    # --- wake/sleep phrases ---
    WAKE_TRIGGERS = ["jarvis on", "hey jarvis", "jarvis wake", "wake up jarvis", "jarvis, on"]
    SLEEP_TRIGGERS = ["jarvis sleep", "go to sleep", "jarvis, sleep", "goodbye jarvis", "jarvis off", "jarvis, off"]

    def _is_wake(text: str) -> bool:
        t = text.lower().strip()
        # fuzzy: any transcription containing "jarvis" wakes (covers "jarvis on", "hey jarvis", etc.)
        return "jarvis" in t

    def _is_sleep(text: str) -> bool:
        t = text.lower().strip()
        return any(trigger in t for trigger in SLEEP_TRIGGERS)

    model = settings.llm_model
    SENT_RE = re.compile(r"([.!?])\s+")
    conversation_history: list[dict] = []

    async def _ask_claude(user_text: str, sentence_cb=None) -> str:
        from jarvis.graph.agent import run_turn
        response, updated = await run_turn(
            session["graph"], user_text,
            trigger="voice", output_channel="voice",
            conversation_history=conversation_history,
            tts_callback=sentence_cb,
        )
        conversation_history.clear()
        conversation_history.extend(updated)
        return response

    # --- TTS with interrupt support ---
    _speaking = asyncio.Event()
    _interrupt = asyncio.Event()

    async def _speak(sentence: str) -> None:
        if _interrupt.is_set():
            return
        try:
            pcm_out, sr = await tts.synthesize(sentence)
            _speaking.set()
            await audio.play(pcm_out, sr)
        except Exception as e:
            log.warning("TTS error: %s", e)
        finally:
            _speaking.clear()

    async def _stream_and_speak(text: str) -> str:
        tts_tasks: list[asyncio.Task] = []
        _interrupt.clear()

        async def _on_sentence(s: str) -> None:
            if not _interrupt.is_set():
                t = asyncio.create_task(_speak(s))
                tts_tasks.append(t)

        reply = await _ask_claude(text, sentence_cb=_on_sentence)

        # wait for all queued TTS to finish (unless interrupted)
        for t in tts_tasks:
            if not _interrupt.is_set():
                await t
            else:
                t.cancel()

        return reply

    async def _respond(text: str) -> str:
        reply = await _stream_and_speak(text)
        return reply

    bridge = TelegramBridge(settings, _respond)
    await bridge.start()

    monitor_tasks = []
    if toolbox._calendar:
        from jarvis.graph.proactive.calendar import CalendarMonitor
        cal_monitor = CalendarMonitor(
            session["graph"], toolbox._calendar,
            notify_fn=bridge.notify,
            poll_seconds=settings.calendar_poll_seconds,
        )
        monitor_tasks.append(asyncio.create_task(cal_monitor.run_forever()))

    if toolbox._email:
        from jarvis.graph.proactive.email import EmailMonitor
        email_monitor = EmailMonitor(
            session["graph"], toolbox._email,
            notify_fn=bridge.notify,
            poll_seconds=settings.email_poll_seconds,
        )
        monitor_tasks.append(asyncio.create_task(email_monitor.run_forever()))

    stop = asyncio.Event()
    _install_signal_handlers(stop)

    # state: sleeping = waiting for wake phrase
    sleeping = True

    console.print("[bold green]Jarvis standing by.[/] Say [bold]'Hey Jarvis'[/] to wake me. Ctrl-C to quit.")

    try:
        while not stop.is_set():
            pcm = await audio.record_utterance()
            if pcm.size == 0:
                continue
            text = await stt.transcribe(pcm, settings.sample_rate)
            if not text:
                continue

            text = text.strip()
            console.print(f"[dim]you[/dim]: {text}")

            if sleeping:
                console.print(f"[dim](sleeping) heard: '{text}'[/dim]")
                if _is_wake(text):
                    sleeping = False
                    console.print("[bold green]Jarvis[/]: [yellow]Online, Sir.[/]")
                    await audio.play_boot_sound()
                    asyncio.ensure_future(_stream_and_speak("Welcome Sir, what are we doing now?"))
                continue

            # check interrupt: if speaking and user said something, stop audio
            if _speaking.is_set():
                _interrupt.set()
                audio.stop()

            if _is_sleep(text):
                sleeping = True
                console.print("[bold green]Jarvis[/]: [dim]Going to sleep.[/]")
                asyncio.ensure_future(_stream_and_speak("Going to sleep, Sir. Call me when you need me."))
                await audio.play_sleep_sound()
                continue

            console.print("[bold green]Jarvis[/]: ", end="")
            reply = await _respond(text)
            console.print(reply)

    finally:
        if _claude_proc and _claude_proc.returncode is None:
            _claude_proc.terminate()
        for t in monitor_tasks:
            t.cancel()
        await bridge.stop()


# ----- chat -----

@app.command()
def chat(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    """Text REPL with streaming output."""
    _setup_logging(verbose)
    asyncio.run(_chat_main())


async def _chat_main() -> None:
    settings = get_settings()
    session = build_session(settings)
    inject_google(session)
    graph = session["graph"]
    conversation_history: list = []

    console.print("[bold green]Jarvis (chat).[/] Ctrl-D to exit.")
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
            conversation_history = []
            console.print("[dim]history cleared[/dim]")
            continue
        console.print("[magenta]jarvis[/magenta]: ", end="")
        try:
            reply, conversation_history = await run_turn(
                graph, line, trigger="voice", output_channel="voice",
                conversation_history=conversation_history,
            )
            console.print(reply)
        except Exception as e:
            console.print(f"[red]error:[/red] {e}")


# ----- ask -----

@app.command()
def ask(
    question: str = typer.Argument(...),
    json_output: bool = typer.Option(False, "--json"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """One-shot text query."""
    _setup_logging(verbose)
    asyncio.run(_ask_main(question, json_output))


async def _ask_main(question: str, raw: bool) -> None:
    settings = get_settings()
    session = build_session(settings)
    inject_google(session)
    reply, _ = await run_turn(session["graph"], question, trigger="voice", output_channel="voice")
    if raw:
        print(reply)
    else:
        console.print(f"[magenta]jarvis[/magenta]: {reply}")


# ----- telegram -----

@app.command()
def telegram(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    """Run only the Telegram bridge."""
    _setup_logging(verbose)
    asyncio.run(_telegram_main())


async def _telegram_main() -> None:
    from jarvis.telegram_bot import TelegramBridge

    settings = get_settings()
    session = build_session(settings)
    inject_google(session)
    graph = session["graph"]
    conversation_history: list = []

    async def _respond(text: str) -> str:
        nonlocal conversation_history
        response, conversation_history = await run_turn(
            graph, text, trigger="voice", output_channel="telegram",
            conversation_history=conversation_history,
        )
        return response

    bridge = TelegramBridge(settings, _respond)
    if not bridge.enabled():
        console.print("[red]JARVIS_TELEGRAM_TOKEN not set.[/red]")
        sys.exit(1)
    await bridge.start()
    console.print("[bold green]Telegram bridge running.[/] Ctrl-C to quit.")
    stop = asyncio.Event()
    _install_signal_handlers(stop)
    try:
        await stop.wait()
    finally:
        await bridge.stop()


# ----- notify -----

@app.command()
def notify(
    message: str = typer.Argument(...),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Send a push notification via Telegram."""
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
    paths: list[str] = typer.Argument(None),
    command: Optional[str] = typer.Option(None, "--command", "-c"),
    debounce_ms: Optional[int] = typer.Option(None, "--debounce-ms"),
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
    from jarvis.graph.proactive.watchdog import WatchdogMonitor

    settings = get_settings()
    target_paths = paths or settings.watch_path_list
    cmd = command if command is not None else settings.watch_command
    debounce = debounce_ms if debounce_ms is not None else settings.watch_debounce_ms

    async def _noop(_: str) -> str:
        return ""

    bridge = TelegramBridge(settings, _noop)
    if not bridge.enabled():
        console.print("[red]Telegram required for watch mode.[/red]")
        sys.exit(1)
    await bridge.start()
    monitor = WatchdogMonitor(target_paths, bridge.notify, command=cmd, debounce_ms=debounce)
    console.print(f"[bold green]Watching[/] {', '.join(target_paths)}. Ctrl-C to stop.")
    try:
        await monitor.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        await bridge.stop()


# ----- auth -----

@auth_app.command("google")
def auth_google(
    credentials: str = typer.Option(None, "--credentials", help="Path to credentials.json"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Authenticate with Google (Calendar + Gmail). Opens a browser."""
    _setup_logging(verbose)
    from pathlib import Path
    from jarvis.integrations.google_auth import get_credentials

    settings = get_settings()
    creds_path = Path(credentials).expanduser() if credentials else Path(settings.google_credentials).expanduser()
    token_path = Path(settings.google_token).expanduser()

    if not creds_path.exists():
        console.print(f"[red]credentials.json not found at {creds_path}[/red]")
        console.print("Download it from Google Cloud Console → APIs & Services → Credentials.")
        raise typer.Exit(1)

    try:
        get_credentials(creds_path, token_path)
        console.print(f"[green]Google authenticated. Token saved to {token_path}[/green]")
    except Exception as e:
        console.print(f"[red]Auth failed: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
