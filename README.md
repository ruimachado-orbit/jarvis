# Jarvis

A voice-first local coding assistant. You say "hey Jarvis", it reads your codebase, explains changes, and streams the answer back through the speakers sentence by sentence. Inspired by the Hermes agent gateway pattern, it also has a Telegram bridge so it can poke you when something needs attention and accept remote chat from your phone.

Everything runs locally by default:

- **LLM**: any model served through [vLLM](https://github.com/vllm-project/vllm) over the OpenAI-compatible API. Default is `Qwen/Qwen2.5-Coder-7B-Instruct`; drop to `Qwen2.5-Coder-1.5B-Instruct` for 6 GB GPUs.
- **Wake word**: [openWakeWord](https://github.com/dscripka/openWakeWord) running the `hey_jarvis` model (falls back to STT keyword match if not installed).
- **STT**: [faster-whisper](https://github.com/SYSTRAN/faster-whisper), VAD-gated recording via `webrtcvad`.
- **TTS**: [Piper](https://github.com/rhasspy/piper), sentence-level synthesis pipelined with playback so the first words come back fast.
- **Messaging**: `python-telegram-bot` for pushing warnings and accepting remote prompts.
- **Watch**: [watchfiles](https://github.com/samuelcolvin/watchfiles) turns any path into a Telegram notification source.

```
        ┌── wake word ──┐
mic ────┤               ├── faster-whisper ── Agent ──► vLLM (Qwen2.5-Coder)
        └── VAD record ─┘                       │
                                                ├── read_file / grep / list_dir / write_file / run_shell
                                                │
                                                ├─► Piper (sentence-pipelined) ─► speakers
                                                │
                                                └─► Telegram  (notify + remote chat + watch)
```

## Quick start

```bash
make init        # venv + deps + .env + Piper voice
# edit .env (Telegram token, workspace, etc)
make dev         # spins up vLLM and the voice loop together
```

Say **"hey Jarvis"**, ask your question, and it streams the answer back.

## Commands

| Command                 | What it does                                                  |
| ----------------------- | ------------------------------------------------------------- |
| `make dev`              | Start vLLM, wait for it, then run the voice loop.             |
| `make vllm`             | Start vLLM only (uses `scripts/serve_vllm.sh`).               |
| `make voice`            | Run the hands-free voice loop (assumes vLLM is up).           |
| `make chat`             | Streaming text REPL, no mic needed.                           |
| `make ask Q="..."`      | One-shot text query, prints the answer and exits.             |
| `make telegram`         | Only the Telegram bridge (no local mic).                      |
| `make watch`            | Watch `JARVIS_WATCH_PATHS` and notify the Telegram chat.      |
| `make notify M="..."`   | Send a single push notification.                              |
| `make test`             | Run the test suite.                                           |

Every target maps to a `jarvis …` sub-command, so you can also call the CLI directly:

```bash
jarvis voice --no-wake         # disable wake word for this run
jarvis voice --no-stream       # batch TTS (wait for full reply)
jarvis ask "what does main.py do?" --json
jarvis watch ./src --command "pytest -q"
```

## vLLM serving

Defaults live in `scripts/serve_vllm.sh`. Tune for your VRAM via env vars:

```bash
JARVIS_LLM_MODEL=Qwen/Qwen2.5-Coder-1.5B-Instruct \
JARVIS_VLLM_MAX_LEN=6000 \
JARVIS_VLLM_GPU_UTIL=0.8 \
JARVIS_VLLM_ENFORCE_EAGER=true \
./scripts/serve_vllm.sh
```

Quick decision rule: if model weights take more than 50 % of free GPU memory, set `JARVIS_VLLM_ENFORCE_EAGER=true` (no CUDA graphs, ~1.5 GB more for the KV cache).

## Telegram

1. Create a bot with [@BotFather](https://t.me/BotFather); paste the token into `JARVIS_TELEGRAM_TOKEN`.
2. Message the bot, then send `/whoami` — it replies with your chat id.
3. Put that id into both `JARVIS_TELEGRAM_ALLOWED_CHATS` (access control) and `JARVIS_TELEGRAM_NOTIFY_CHAT` (push target).
4. Restart `jarvis voice` or `jarvis telegram`. From your phone: `/ask what does main.py do?` or plain text.

Use `make notify M="CI just went red"` from cron, git hooks, or `jarvis watch` to get async alerts.

## Watch mode

```bash
# Watch the whole workspace, push each change to Telegram:
jarvis watch .

# Watch src/ and run pytest on every change; push exit code + log tail:
jarvis watch ./src --command "pytest -q"

# Configure defaults via env vars:
JARVIS_WATCH_PATHS=./src,./tests
JARVIS_WATCH_COMMAND=pytest -q
JARVIS_WATCH_DEBOUNCE_MS=400
```

## Wake word

Install the optional extra (`pip install -e '.[wake]'` or the pre-baked `make install`). On first run openWakeWord fetches the `hey_jarvis` model into its cache. To swap phrases, point `JARVIS_WAKE_MODEL` at another openWakeWord model id.

Disable entirely for always-on mode:

```bash
JARVIS_WAKE_ENABLED=false jarvis voice
# or
jarvis voice --no-wake
```

If openWakeWord is not installed, the voice loop still works — set `JARVIS_REQUIRE_WAKE_WORD=true` and the transcript has to contain `JARVIS_WAKE_WORD` before a reply is produced.

## Streaming

`JARVIS_STREAM_VOICE=true` (default) runs the LLM and TTS as a pipeline: sentences are synthesised the moment they come off the model, while playback drains them in order. The first audio usually arrives before the full response is written. Override per-run with `jarvis voice --no-stream`.

## Long-term memory

Jarvis keeps two tiers of memory in a single SQLite file (`./.jarvis/memory.db` by default):

1. **Episodic log** — every user message, assistant reply, and tool result is appended row-by-row. Nothing is summarised at write time; this is the raw record.
2. **Distilled profile** — every `JARVIS_MEMORY_REFRESH_EVERY` user turns, a background task feeds the recent episodes to the LLM with a note-taking prompt and stores a short (≤1 KB) bullet list of durable facts: preferred languages, style, projects, things you like or dislike. The latest profile is folded into the system prompt so every future reply benefits from it.

The distillation prompt explicitly forbids fabrication. If nothing durable is learned the store records `(no durable facts yet)` and moves on.

Controls:

```bash
make memory-show       # jarvis memory show      — profile + last 10 episodes
make memory-refresh    # jarvis memory refresh   — force a distillation now
make memory-reset      # jarvis memory reset     — wipe everything (prompts first)
jarvis memory export --limit 1000 -o session.jsonl
```

Disable entirely with `JARVIS_MEMORY_ENABLED=false`. Point `JARVIS_MEMORY_DB` at a per-workspace path if you want project-scoped memory.

## Safety defaults

`read_file`, `list_dir`, `grep` are always on. `write_file` and `run_shell` are disabled until `JARVIS_ALLOW_WRITES=true` / `JARVIS_ALLOW_SHELL=true`. Paths resolve against `JARVIS_WORKSPACE` and escapes are rejected.

## Layout

```
jarvis/
├── agent.py          # multi-turn tool loop (respond + respond_stream)
├── audio.py          # VAD-gated recording, serial playback
├── config.py         # pydantic settings (env-backed)
├── llm.py            # OpenAI-compatible vLLM client (stream + tools)
├── main.py           # typer CLI: voice | chat | ask | telegram | notify | watch | memory
├── memory.py         # SQLite episodic log + LLM-distilled user profile
├── personality.py    # Jarvis system prompt (TTS-friendly)
├── streaming.py      # SentenceSplitter + speak_stream pipeline
├── stt.py            # faster-whisper transcriber
├── telegram_bot.py   # notify + remote chat bridge
├── tools.py          # sandboxed read/list/grep/write/shell tools
├── tts.py            # Piper sentence synthesis
├── wake.py           # openWakeWord detector (STT fallback)
└── watch.py          # watchfiles → Telegram notifier
scripts/
├── serve_vllm.sh     # vLLM flags tuned for consumer GPUs
├── wait_for_vllm.sh  # poll until the OpenAI endpoint is live
└── download_voice.sh # grab a Piper voice from HF
```

## Tests

```bash
make install
make test
```
