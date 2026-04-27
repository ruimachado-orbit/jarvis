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
import random
import signal
import sys
from pathlib import Path
from typing import Optional

import numpy as np

import typer
from rich.console import Console
from rich.prompt import Prompt

from jarvis.core.config import get_settings, reload_settings
from jarvis.core.session import build_session, inject_google
from jarvis.graph.agent import run_turn


def _load_env_file_vars() -> None:
    """Export non-JARVIS_-prefixed vars from .env into os.environ.

    pydantic-settings already picks up JARVIS_* for the Settings object, but
    libraries like huggingface_hub only see process env — not .env. Without
    this, running ``python -m jarvis voice`` directly (bypassing ``make dev``)
    leaves HF_TOKEN unset and gated repos fail with 401.

    Precedence: existing os.environ wins over .env values.
    """
    import os
    from dotenv import dotenv_values
    for k, v in dotenv_values(".env").items():
        if v is not None and k not in os.environ:
            os.environ[k] = v


_load_env_file_vars()


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

    # Spoken-aloud acknowledgement fired as soon as Jarvis receives a request,
    # while the LLM is still generating — keeps the user feeling heard.
    ACKNOWLEDGEMENTS = [
        "Right away, Sir.",
        "Let me check that for you, Sir.",
        "One moment, Sir.",
        "On it, Sir.",
        "Looking into it, Sir.",
        "Working on it, Sir.",
        "Certainly, Sir.",
        "At once, Sir.",
    ]
    WAKE_RESPONSE = "At your service, Sir. How may I be of assistance?"
    SLEEP_RESPONSE = "Very good, Sir. Powering down. Do call if you need me."

    # Periodic filler phrases fired during long tool-calling turns so the
    # user keeps hearing signs of life while Claude runs email/calendar ops.
    FILLERS = [
        "Still working on it, Sir.",
        "One moment more, Sir.",
        "Almost there, Sir.",
        "Bear with me, Sir.",
        "Just a bit longer, Sir.",
        "Nearly done, Sir.",
    ]

    # Pre-synth fixed phrases so the first ack/wake/sleep plays instantly
    # (no Kokoro cold-start, no per-call synth). Disk-cached across runs.
    _prewarm_list = [*ACKNOWLEDGEMENTS, *FILLERS, WAKE_RESPONSE, SLEEP_RESPONSE]
    log.info("pre-warming TTS cache (%d phrases)...", len(_prewarm_list))
    _pcm_cache: dict[str, tuple[np.ndarray, int]] = await tts.prewarm(_prewarm_list)
    log.info("TTS ready (%d phrases cached)", len(_pcm_cache))

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

    # --- TTS + interrupt ---
    _tts_queue: asyncio.Queue[str | None] = asyncio.Queue()
    _speaking = asyncio.Event()   # set while audio is physically playing
    _interrupt = asyncio.Event()  # set when user speaks over Jarvis
    # Queue for utterances captured by the interrupt watcher
    _captured: asyncio.Queue[np.ndarray] = asyncio.Queue()

    async def _tts_consumer() -> None:
        while True:
            sentence = await _tts_queue.get()
            if sentence is None:
                _tts_queue.task_done()
                break
            if _interrupt.is_set():
                _tts_queue.task_done()
                continue
            try:
                cached = _pcm_cache.get(sentence)
                if cached is not None:
                    pcm_out, sr = cached
                    log.debug("TTS cache hit: %s", sentence[:40])
                else:
                    log.debug("TTS synthesising: %s", sentence[:40])
                    pcm_out, sr = await tts.synthesize(sentence)
                log.debug("TTS playing %d samples", len(pcm_out))
                _speaking.set()
                await audio.play(pcm_out, sr)
                log.debug("TTS done")
            except Exception as e:
                console.print(f"[red]TTS failed:[/red] {e}")
                log.error("TTS consumer error: %s", e, exc_info=True)
            finally:
                _speaking.clear()
                _tts_queue.task_done()

    async def _speak_direct(text: str) -> None:
        """Synthesise and play a fixed phrase without going through the LLM."""
        try:
            cached = _pcm_cache.get(text)
            if cached is not None:
                pcm_out, sr = cached
                log.debug("speak_direct cache hit: %s", text[:40])
            else:
                log.debug("speak_direct: %s", text[:40])
                pcm_out, sr = await tts.synthesize(text)
            _speaking.set()
            await audio.play(pcm_out, sr)
            log.debug("speak_direct done")
        except Exception as e:
            console.print(f"[red]TTS failed:[/red] {e}")
            log.error("speak_direct error: %s", e, exc_info=True)
        finally:
            _speaking.clear()

    def _drain_tts_queue() -> None:
        while not _tts_queue.empty():
            try:
                _tts_queue.get_nowait()
                _tts_queue.task_done()
            except asyncio.QueueEmpty:
                break

    async def _stream_and_speak(text: str) -> str:
        _interrupt.clear()
        _drain_tts_queue()
        consumer = asyncio.create_task(_tts_consumer())

        loop = asyncio.get_running_loop()
        last_put = loop.time()

        async def _put(item: str) -> None:
            nonlocal last_put
            last_put = loop.time()
            await _tts_queue.put(item)

        # Queue an acknowledgement first so it plays while claude is thinking —
        # the consumer serialises playback, so the LLM's first sentence lands
        # right after the ack finishes.
        ack = random.choice(ACKNOWLEDGEMENTS)
        console.print(f"[dim](ack)[/dim] {ack}")
        await _put(ack)

        # Filler ticker: drop a short "still working" phrase every ~5 s of
        # queue silence while the LLM+tools run, so slow email/calendar turns
        # don't leave the user hearing dead air. Stops when claude returns.
        claude_done = asyncio.Event()

        # Shuffled queue so each cycle uses every phrase once before repeating,
        # and the cycle boundary doesn't accidentally repeat the previous filler.
        filler_queue: list[str] = []
        last_filler: str | None = None

        def _next_filler() -> str:
            nonlocal last_filler
            if not filler_queue:
                candidates = random.sample(FILLERS, len(FILLERS))
                if last_filler and candidates[0] == last_filler and len(candidates) > 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
                filler_queue.extend(candidates)
            f = filler_queue.pop(0)
            last_filler = f
            return f

        async def _filler_ticker() -> None:
            while not claude_done.is_set():
                try:
                    await asyncio.wait_for(claude_done.wait(), timeout=1.0)
                    return
                except asyncio.TimeoutError:
                    pass
                if _interrupt.is_set():
                    return
                if loop.time() - last_put >= 5.0:
                    filler = _next_filler()
                    log.debug("filler: %s", filler)
                    await _put(filler)

        ticker = asyncio.create_task(_filler_ticker())

        async def _on_sentence(s: str) -> None:
            if not _interrupt.is_set():
                await _put(s)

        try:
            reply = await _ask_claude(text, sentence_cb=_on_sentence)
        finally:
            claude_done.set()
            await ticker

        await _tts_queue.put(None)
        await consumer
        return reply

    async def _respond(text: str) -> str:
        return await _stream_and_speak(text)

    async def _mic_loop(stop_event: asyncio.Event) -> None:
        """Single mic owner. Handles both normal recording and interrupt detection.

        During silence: VAD-gate frames → emit complete utterances to _captured.
        During playback: watch for user voice → interrupt Jarvis, then collect utterance.
        """
        import sounddevice as sd
        import queue as _queue
        import webrtcvad

        sr = settings.sample_rate
        frame_ms = 30
        frame_samples = int(sr * frame_ms / 1000)
        silence_frames = max(1, settings.silence_ms // frame_ms)
        min_frames = max(1, settings.min_utterance_ms // frame_ms)
        max_frames = max(min_frames, settings.max_utterance_ms // frame_ms)

        # webrtcvad does the speech/silence classification; energy is just a
        # floor to reject dead-silent frames that VAD sometimes marks as speech.
        ENERGY_FLOOR = 0.0015  # frames below this are always non-speech
        MIN_UTT_RMS = 0.004    # reject whole utterances too quiet for STT to parse
        ECHO_MUTE_FRAMES = 17  # frames (~500ms) to discard after playback ends

        vad = webrtcvad.Vad(settings.vad_aggressiveness)
        raw_q: _queue.Queue[np.ndarray] = _queue.Queue()

        def _cb(indata, *_):
            raw_q.put(indata[:, 0].copy())

        loop = asyncio.get_running_loop()

        # State machine
        triggered = False
        trailing_silence = 0
        total_frames = 0
        collecting: list[np.ndarray] = []
        was_playing = False
        echo_mute = 0

        # settings.input_device accepts a device index ("3") or a substring
        # ("Anker PowerConf C200") — sounddevice resolves both. None → system default.
        dev: int | str | None = settings.input_device or None
        if isinstance(dev, str) and dev.isdigit():
            dev = int(dev)
        log.info("mic: opening input stream on device=%r", dev)

        with sd.InputStream(samplerate=sr, channels=1, dtype="int16",
                            blocksize=frame_samples, device=dev, callback=_cb):
            while not stop_event.is_set():
                try:
                    frame = await loop.run_in_executor(None, lambda: raw_q.get(timeout=0.1))
                except Exception:
                    continue

                if len(frame) < frame_samples:
                    continue
                frame = frame[:frame_samples]
                frame_rms = float(np.sqrt(np.mean((frame.astype(np.float32) / 32768.0) ** 2)))

                playing_now = _speaking.is_set() or audio._is_playing

                # Falling edge: playback just ended → arm echo mute window
                if was_playing and not playing_now:
                    echo_mute = ECHO_MUTE_FRAMES
                    triggered = False
                    collecting = []
                    total_frames = 0
                    trailing_silence = 0
                was_playing = playing_now

                # Discard all mic input while Jarvis is playing — he hears himself
                if playing_now:
                    continue

                # Discard echo tail right after playback
                if echo_mute > 0:
                    echo_mute -= 1
                    continue

                # VAD-gated recording. webrtcvad is designed for this; energy
                # floor just suppresses VAD misfires on dead-silent buffers.
                try:
                    vad_speech = vad.is_speech(frame.tobytes(), sr)
                except Exception:
                    vad_speech = False
                is_speech = vad_speech and frame_rms > ENERGY_FLOOR

                if triggered:
                    collecting.append(frame)
                    total_frames += 1
                    if is_speech:
                        trailing_silence = 0
                    else:
                        trailing_silence += 1
                    hit_max = total_frames >= max_frames
                    if (trailing_silence >= silence_frames and total_frames >= min_frames) \
                            or hit_max:
                        pcm = np.concatenate(collecting)
                        utt_rms = float(np.sqrt(np.mean((pcm.astype(np.float32) / 32768.0) ** 2)))
                        dur_ms = total_frames * frame_ms
                        log.info(
                            "mic: captured %dms rms=%.4f %s",
                            dur_ms, utt_rms,
                            "(hit max)" if hit_max else "(silence)",
                        )
                        triggered = False
                        collecting = []
                        total_frames = 0
                        trailing_silence = 0
                        if utt_rms >= MIN_UTT_RMS:
                            await _captured.put(pcm)
                        else:
                            log.info("mic: dropped utterance (too quiet rms=%.4f)", utt_rms)
                else:
                    if is_speech:
                        triggered = True
                        collecting = [frame]
                        total_frames = 1
                        trailing_silence = 0

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

    sleeping = True
    console.print("[bold green]Jarvis standing by.[/] Say [bold]'Hey Jarvis'[/] to wake me. Ctrl-C to quit.")

    watcher_task = asyncio.create_task(_mic_loop(stop))

    try:
        while not stop.is_set():
            pcm = await _captured.get()
            if pcm.size == 0:
                continue

            text = await stt.transcribe(pcm, settings.sample_rate)
            if not text:
                continue

            text = text.strip()
            console.print(f"[dim]you[/dim]: {text}")

            if sleeping:
                if _is_wake(text):
                    sleeping = False
                    console.print("[bold green]Jarvis[/]: [yellow]At your service, Sir.[/]")
                    await audio.play_boot_sound()
                    await _speak_direct(WAKE_RESPONSE)
                else:
                    console.print(f"[dim](sleeping) heard: '{text}'[/dim]")
                continue

            if _is_sleep(text):
                sleeping = True
                console.print("[bold green]Jarvis[/]: [dim]Powering down.[/]")
                await _speak_direct(SLEEP_RESPONSE)
                await audio.play_sleep_sound()
                continue

            console.print("[bold green]Jarvis[/]: ", end="")
            reply = await _respond(text)
            console.print(reply)

    finally:
        watcher_task.cancel()
        try:
            await watcher_task
        except asyncio.CancelledError:
            pass
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


# ----- service management (launchd) -----

_PLIST_LABEL = "ai.maiolabs.jarvis"
_PLIST_PATH = Path.home() / "Library" / "LaunchAgents" / f"{_PLIST_LABEL}.plist"
_LOG_DIR = Path.home() / "Library" / "Logs" / "Jarvis"
_JARVIS_BIN = Path(__file__).parent.parent / ".venv" / "bin" / "jarvis"
_PROJECT_DIR = Path(__file__).parent.parent


def _plist_content() -> str:
    log_out = _LOG_DIR / "jarvis.log"
    log_err = _LOG_DIR / "jarvis-error.log"
    # Inherit PATH so `claude` binary is found at runtime
    import os
    path_env = os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin:/opt/homebrew/bin")
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{_PLIST_LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{_JARVIS_BIN}</string>
        <string>voice</string>
    </array>
    <key>WorkingDirectory</key>
    <string>{_PROJECT_DIR}</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>{path_env}</string>
        <key>HOME</key>
        <string>{Path.home()}</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{log_out}</string>
    <key>StandardErrorPath</key>
    <string>{log_err}</string>
    <key>ThrottleInterval</key>
    <integer>5</integer>
</dict>
</plist>
"""


def _launchctl(*args: str) -> tuple[int, str]:
    import subprocess
    r = subprocess.run(["launchctl", *args], capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()


@app.command()
def start(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Install and start Jarvis as a macOS background service (auto-starts on login)."""
    from pathlib import Path

    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    _PLIST_PATH.write_text(_plist_content())
    console.print(f"[dim]Plist written to {_PLIST_PATH}[/dim]")

    # Unload first in case an old version is loaded
    _launchctl("unload", str(_PLIST_PATH))

    rc, out = _launchctl("load", "-w", str(_PLIST_PATH))
    if rc != 0:
        console.print(f"[red]Failed to load service:[/red] {out}")
        raise typer.Exit(1)

    console.print("[bold green]Jarvis service started.[/] It will restart automatically on login.")
    console.print(f"Logs: [dim]{_LOG_DIR}/jarvis.log[/dim]")


@app.command()
def stop() -> None:
    """Stop the Jarvis background service."""
    if not _PLIST_PATH.exists():
        console.print("[yellow]Jarvis service is not installed.[/yellow]")
        raise typer.Exit(0)

    rc, out = _launchctl("unload", "-w", str(_PLIST_PATH))
    if rc != 0:
        console.print(f"[red]Failed to stop service:[/red] {out}")
        raise typer.Exit(1)

    console.print("[bold]Jarvis service stopped.[/] Run [cyan]jarvis start[/cyan] to restart.")


@app.command()
def status() -> None:
    """Show whether the Jarvis service is running."""
    import subprocess

    if not _PLIST_PATH.exists():
        console.print("[yellow]Not installed.[/yellow] Run [cyan]jarvis start[/cyan] to install.")
        raise typer.Exit(0)

    rc, out = _launchctl("list", _PLIST_LABEL)
    if rc != 0 or "Could not find service" in out:
        console.print("[red]● Jarvis[/red] — stopped (not loaded)")
        raise typer.Exit(0)

    # Parse PID and last exit code from launchctl list output
    pid, last_exit = None, None
    for line in out.splitlines():
        line = line.strip()
        if line.startswith('"PID"'):
            pid = line.split("=")[-1].strip().rstrip(";").strip('"')
        elif line.startswith('"LastExitStatus"'):
            last_exit = line.split("=")[-1].strip().rstrip(";").strip('"')

    if pid and pid != "0":
        console.print(f"[bold green]● Jarvis[/bold green] — running (PID {pid})")
    else:
        exit_info = f", last exit {last_exit}" if last_exit and last_exit != "0" else ""
        console.print(f"[red]● Jarvis[/red] — not running{exit_info}")

    console.print(f"[dim]Logs: {_LOG_DIR}/jarvis.log[/dim]")
    # Show last 5 log lines
    log_file = _LOG_DIR / "jarvis.log"
    if log_file.exists():
        lines = log_file.read_text().splitlines()[-5:]
        if lines:
            console.print("\n[dim]— recent log —[/dim]")
            for l in lines:
                console.print(f"[dim]{l}[/dim]")


@app.command()
def restart() -> None:
    """Restart the Jarvis background service."""
    stop()
    start()


if __name__ == "__main__":
    app()
