"""CLI entry points for Jarvis.

Sub-commands:
    jarvis voice     — hands-free voice loop (default)
    jarvis chat      — text REPL (great for dev / SSH)
    jarvis telegram  — run only the Telegram bridge
    jarvis notify    — send a one-off Telegram notification
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys

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


# ----- voice -----

@app.command()
def voice(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    """Hands-free voice loop with VAD, STT, LLM, and Piper TTS."""
    _setup_logging(verbose)
    asyncio.run(_voice_main())


async def _voice_main() -> None:
    from jarvis.audio import AudioIO
    from jarvis.stt import STT
    from jarvis.telegram_bot import TelegramBridge
    from jarvis.tts import TTS

    settings = get_settings()
    agent = _build_agent()
    stt = STT(settings)
    tts = TTS(settings)
    audio = AudioIO(settings)

    bridge = TelegramBridge(settings, agent.respond)
    await bridge.start()

    stop = asyncio.Event()

    def _sig(*_a) -> None:
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            asyncio.get_event_loop().add_signal_handler(sig, _sig)
        except NotImplementedError:
            signal.signal(sig, lambda *_: stop.set())

    console.print("[bold green]Jarvis online.[/] Speak whenever you want. Ctrl-C to quit.")

    try:
        while not stop.is_set():
            console.print("[dim]listening...[/dim]")
            pcm = await audio.record_utterance()
            if pcm.size == 0:
                continue
            text = await stt.transcribe(pcm, settings.sample_rate)
            if not text:
                continue
            console.print(f"[cyan]you[/cyan]: {text}")
            if settings.require_wake_word and settings.wake_word.lower() not in text.lower():
                console.print("[dim](no wake word, ignoring)[/dim]")
                continue

            reply = await agent.respond(text)
            console.print(f"[magenta]jarvis[/magenta]: {reply}")
            if reply:
                pcm_out, sr = await tts.synthesize(reply)
                await audio.play(pcm_out, sr)
    finally:
        await bridge.stop()


# ----- chat -----

@app.command()
def chat(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    """Text REPL. No audio dependencies required."""
    _setup_logging(verbose)
    asyncio.run(_chat_main())


async def _chat_main() -> None:
    agent = _build_agent()
    console.print("[bold green]Jarvis (chat).[/] Type a question. Ctrl-D to exit.")
    while True:
        try:
            line = Prompt.ask("[cyan]you[/cyan]")
        except (EOFError, KeyboardInterrupt):
            console.print()
            return
        if not line.strip():
            continue
        if line.strip() in {"/reset", "/clear"}:
            agent.reset()
            console.print("[dim]history cleared[/dim]")
            continue
        try:
            reply = await agent.respond(line)
        except Exception as e:
            console.print(f"[red]error:[/red] {e}")
            continue
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
    try:
        await stop.wait()
    except KeyboardInterrupt:
        pass
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


if __name__ == "__main__":
    app()
