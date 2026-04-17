# Jarvis

A voice-first local coding assistant. You talk, it reads your codebase, explains changes, and answers back through the speakers. Inspired by the Hermes agent gateway pattern, it also has a Telegram bridge so it can poke you when something needs attention and accept remote chat from your phone.

Everything runs locally by default:

- **LLM**: any model served through [vLLM](https://github.com/vllm-project/vllm) over the OpenAI-compatible API. Default is `Qwen/Qwen2.5-Coder-7B-Instruct`; drop to `Qwen2.5-Coder-1.5B-Instruct` for 6 GB GPUs.
- **STT**: [faster-whisper](https://github.com/SYSTRAN/faster-whisper), streaming VAD-gated recording via `webrtcvad`.
- **TTS**: [Piper](https://github.com/rhasspy/piper), sentence-level synthesis so the first words come back fast.
- **Messaging**: `python-telegram-bot` for pushing warnings and accepting remote prompts.

```
mic  ──▶  VAD  ──▶  faster-whisper  ──▶  Agent  ──▶  vLLM (Qwen2.5-Coder)
                                           │
                                           ├── read_file, grep, list_dir, write_file, run_shell
                                           │
                                           ├──▶  Piper TTS  ──▶  speakers
                                           │
                                           └──▶  Telegram bridge  (notify + remote chat)
```

## Install

```bash
git clone <this repo> jarvis && cd jarvis
python -m venv .venv && source .venv/bin/activate
pip install -e .

# System deps on Linux for audio:
sudo apt install portaudio19-dev libsndfile1 ffmpeg
```

Download a Piper voice:

```bash
./scripts/download_voice.sh en_GB-alan-medium
```

Copy and edit the env file:

```bash
cp .env.example .env
# fill in JARVIS_TTS_VOICE, JARVIS_TELEGRAM_TOKEN, etc.
```

## Run vLLM

On a local GPU (adjust flags for your VRAM):

```bash
pip install vllm
JARVIS_VLLM_ENFORCE_EAGER=true ./scripts/serve_vllm.sh
```

For a 6 GB card, switch `JARVIS_LLM_MODEL=Qwen/Qwen2.5-Coder-1.5B-Instruct` and set `JARVIS_VLLM_MAX_LEN=6000`.

## Use

```bash
# Hands-free voice loop (VAD -> STT -> LLM -> TTS)
jarvis voice

# Dev-friendly text REPL, no mic/speakers needed
jarvis chat

# Only the Telegram bridge (no local mic)
jarvis telegram

# Fire a one-off push notification
jarvis notify "CI just went red on main"
```

Commands inside the REPL: `/reset` clears the conversation.

## Telegram

1. Talk to [@BotFather](https://t.me/BotFather), create a bot, paste the token into `JARVIS_TELEGRAM_TOKEN`.
2. Send the bot any message, then send `/whoami` — it replies with your chat id.
3. Put that id into both `JARVIS_TELEGRAM_ALLOWED_CHATS` (access control) and `JARVIS_TELEGRAM_NOTIFY_CHAT` (push target).
4. Restart `jarvis voice` (or `jarvis telegram`). Send `/ask what does main.py do?` from your phone.

Use `jarvis notify "<text>"` from cron, a CI hook, or another script to get push alerts from the same bot.

## Safety defaults

The `read_file`, `list_dir`, and `grep` tools are always on. `write_file` and `run_shell` are disabled by default — flip `JARVIS_ALLOW_WRITES` and `JARVIS_ALLOW_SHELL` in `.env` once you trust the setup. All file paths are sandboxed to `JARVIS_WORKSPACE`.

## Layout

```
jarvis/
├── agent.py          # multi-turn tool-calling loop + sentence streaming
├── audio.py          # VAD-gated recording, serial playback
├── config.py         # pydantic settings (env-backed)
├── llm.py            # OpenAI-compatible vLLM client
├── main.py           # typer CLI (voice / chat / telegram / notify)
├── personality.py    # Jarvis system prompt (TTS-friendly)
├── stt.py            # faster-whisper transcriber
├── telegram_bot.py   # notify + remote chat bridge
├── tools.py          # read_file / list_dir / grep / write_file / run_shell
└── tts.py            # Piper sentence synthesis
scripts/
├── serve_vllm.sh     # flags tuned for consumer GPUs
└── download_voice.sh # grab a Piper voice from HF
```

## Tests

```bash
pip install -e '.[dev]'
pytest
```
