# Jarvis Full Revision Design
**Date:** 2026-04-23
**Status:** Approved

## Overview

A full revision of Jarvis from a voice-first local coding assistant into a proactive personal AI — Iron Man style. Jarvis listens continuously, manages your calendar and email, monitors changes, and suggests courses of action without being asked. It still handles all coding tasks. Everything runs on Mac Apple Silicon with Claude API as the brain.

**Core stack:** LangGraph (orchestration) + mem0 (memory) + Claude API (LLM) + Kokoro TTS + faster-whisper STT.

---

## 1. Overall Architecture

Two runtime modes run concurrently inside a single Python process:

**Reactive mode** — always listening. Wake word → STT → LangGraph agent graph → tool calls → TTS/Telegram response.

**Proactive mode** — background asyncio loops on schedules. Calendar monitor (every 5 min), email monitor (every 10 min), notification watchdog (file/CI changes). Each loop enters the agent graph at the `think` node with a pre-built context message and exits via `respond`.

Both modes share the same agent state, mem0 memory layer, tool registry, and Claude API client.

```
┌─────────────────────────────────────────────────────┐
│                    JARVIS RUNTIME                   │
│                                                     │
│  ┌─────────────────┐    ┌────────────────────────┐  │
│  │  REACTIVE LOOP  │    │   PROACTIVE LOOPS       │  │
│  │                 │    │                        │  │
│  │  wake → STT     │    │  calendar_monitor      │  │
│  │  → agent graph  │    │  email_monitor         │  │
│  │  → TTS/Telegram │    │  notification_watchdog │  │
│  └────────┬────────┘    └──────────┬─────────────┘  │
│           └──────────┬─────────────┘                │
│                      ▼                              │
│            ┌─────────────────┐                      │
│            │  AGENT GRAPH    │                      │
│            │  (LangGraph)    │                      │
│            │  plan → act     │                      │
│            │  → observe      │                      │
│            │  → respond      │                      │
│            └────────┬────────┘                      │
│                     │                               │
│          ┌──────────┼──────────┐                    │
│          ▼          ▼          ▼                    │
│       Claude     mem0       Tools                   │
│       API        memory     registry                │
└─────────────────────────────────────────────────────┘
```

---

## 2. Memory Architecture

mem0 replaces the current SQLite episodic log + LLM distillation loop entirely. Three layers:

**Episodic** — every conversation turn stored with timestamp and session ID. Queryable by meaning via vector similarity, not just recency.

**Semantic (vector)** — facts extracted from conversations, emails, and calendar events. Examples: "Rui prefers TypeScript", "Orbit project deadline is May 15", "Sarah is the product lead". Retrieved by cosine similarity and injected into the system prompt before each turn.

**Graph (relational)** — entities and relationships extracted automatically. "Orbit project" → has_deadline → "May 15" → assigned_to → "Rui" → collaborates_with → "Sarah".

**Phase 1 backend:** mem0 local mode — Chroma (embedded vectors) + SQLite (graph). Stored in `~/.jarvis/mem0/`. Zero infrastructure, no Docker.

**Phase 2 upgrade path:** swap SQLite graph backend for Neo4j for full Cypher query support.

**Memory flow per turn:**
1. `retrieve_memory` node queries mem0 with the current message → top-K facts injected into system prompt
2. Agent turn executes
3. `store_memory` node extracts new facts and saves to mem0

**What gets remembered:**
- User preferences and habits (from conversations)
- People, projects, deadlines (from calendar + email ingestion)
- Past decisions and outcomes
- Explicitly requested facts via `remember(fact)` tool

**Storage location:** `~/.jarvis/mem0/` — outside the repo, never committed.

---

## 3. LangGraph Agent Graph

### Nodes

| Node | Responsibility |
|------|---------------|
| `route` | Classifies trigger (voice, proactive check, tool result). Decides entry path. |
| `retrieve_memory` | Queries mem0. Injects relevant context into state. |
| `think` | Claude Sonnet reasons. Outputs a response or tool call list. |
| `act` | Executes tool calls. Parallel where tools are independent. |
| `observe` | Processes tool results. Decides: think again or respond. |
| `respond` | Formats output for the right channel (voice, Telegram, silent log). |
| `store_memory` | Extracts facts from completed turn. Saves to mem0. |

### Graph flow

```
[trigger] → route → retrieve_memory → think → act → observe → think (loop max 6)
                                                              ↓
                                                           respond → store_memory
```

### Proactive subgraphs

Each runs as an asyncio background task. Enters at `think`, exits at `respond`.

**`calendar_monitor`** (every 5 min):
- Fetches events for next 24h
- Detects: new events, changes, upcoming deadlines, conflicts
- Proactive output example: "Rui, you have a call with Sarah in 25 minutes and the Orbit deck isn't shared yet. Want me to send her the link?"

**`email_monitor`** (every 10 min):
- Fetches unread emails since last check
- Classifies urgency, extracts action items
- Surfaces: emails needing reply, decisions required, FYIs

**`notification_watchdog`**:
- Existing `watchfiles` watcher rewired as a LangGraph node
- File changes, CI results → agent graph → Telegram push

### AgentState TypedDict

```python
class AgentState(TypedDict):
    trigger: str                    # "voice" | "calendar" | "email" | "watchdog"
    messages: list[BaseMessage]     # LangChain message types; translated to Anthropic format by the think node
    memories: list[str]             # retrieved mem0 facts, injected into system prompt
    tool_calls: list[ToolCall]
    tool_results: list[ToolResult]
    output_channel: str             # "voice" | "telegram" | "silent"
    final_response: str
    requires_confirmation: bool     # True for destructive tool calls (write, send, create)
    pending_action: dict | None     # action waiting for user approval before execution
```

---

## 4. Tool Registry

All destructive tools require voice or Telegram confirmation before execution. Jarvis proposes, user approves.

### Coding tools (existing, unchanged)
- `read_file(path, start_line, end_line)`
- `write_file(path, content)` — gated by `JARVIS_ALLOW_WRITES`
- `list_dir(path)`
- `grep(pattern, path, glob)`
- `run_shell(command, timeout)` — gated by `JARVIS_ALLOW_SHELL`

### Calendar tools (Google Calendar API)
- `list_events(days=7)` — upcoming events
- `create_event(title, time, duration, attendees)` — requires confirmation
- `update_event(id, **changes)` — requires confirmation
- `find_free_slot(duration_minutes, within_days)`

### Email tools (Gmail API)
- `list_emails(query, limit)`
- `read_email(id)`
- `draft_reply(id, body)` — creates draft, never sends automatically
- `send_email(id)` — gated by `JARVIS_ALLOW_SEND_EMAIL`, requires confirmation

### Web tools
- `web_search(query)` — Brave Search API, fallback to DuckDuckGo
- `fetch_page(url)` — scrape and return clean text (via trafilatura)

### Memory tools (exposed to LLM)
- `remember(fact)` — explicitly store a fact
- `recall(query)` — surface memories on demand
- `forget(fact)` — remove a stored fact

### Notification tool
- `notify(message, urgency)` — Telegram push; speaks aloud if Jarvis is active

---

## 5. Voice Pipeline & Personality

### Changes from current

| Component | Before | After |
|-----------|--------|-------|
| TTS | Piper (Linux PyPI, broken on Mac ARM) | Kokoro-82M (native Mac ARM, better quality) |
| LLM | vLLM + Qwen2.5-Coder (local) | Claude API (claude-sonnet-4-6) |
| Agent loop | Custom flat loop in `agent.py` | LangGraph graph |
| Memory | SQLite + LLM distillation | mem0 (Chroma + SQLite graph) |
| Interrupt | None | "stop" or "Jarvis" cuts audio immediately |

### STT/Wake — unchanged
- openWakeWord (`hey_jarvis`) + STT keyword fallback
- faster-whisper `base.en`, WebRTC VAD

### Personality

Short, confident, proactive. Alerts follow a fixed template:
> "Rui, [situation]. [Suggested action]. Want me to [specific action]?"

Waits for voice confirmation or Telegram reply before any destructive action.

### Output channel priority
1. **Voice** — if mic active and recent wake-word detected (user is present)
2. **Telegram** — if no recent voice activity (user is away)
3. **Silent log** — background completions that don't need attention

### Interrupt handling (new)
Dedicated audio thread monitors for "stop" or "Jarvis" keyword during TTS playback. On detection: stop audio immediately, re-enter listening state.

---

## 6. Google Integration & Auth

### Setup (one-time)
1. Create Google Cloud project, enable Calendar API + Gmail API
2. Download `credentials.json` (OAuth 2.0 client secret)
3. Run `jarvis auth google` — opens browser, approve scopes, token saved to `~/.jarvis/google_token.json`
4. All subsequent runs auto-refresh the token

### OAuth scopes
```
https://www.googleapis.com/auth/calendar.readonly
https://www.googleapis.com/auth/calendar.events
https://www.googleapis.com/auth/gmail.readonly
https://www.googleapis.com/auth/gmail.compose
https://www.googleapis.com/auth/gmail.send   # only if JARVIS_ALLOW_SEND_EMAIL=true
```

### Token storage
- `~/.jarvis/credentials.json` — OAuth client secret
- `~/.jarvis/google_token.json` — access + refresh token

Both outside the workspace root — never accidentally committed.

### Polling strategy
- Phase 1: polling (calendar 5 min, email 10 min) — simple, reliable
- Phase 2: Gmail push notifications via Google Pub/Sub for true real-time

---

## 7. Project Structure

```
jarvis/
├── core/
│   ├── config.py          # extends existing Settings with new env vars
│   ├── personality.py     # updated system prompt + proactive templates
│   └── session.py         # shared runtime (mem0, tools, llm client, graph)
│
├── graph/
│   ├── agent.py           # LangGraph graph definition (compile + run)
│   ├── nodes.py           # all node functions
│   ├── state.py           # AgentState TypedDict
│   └── proactive/
│       ├── calendar.py    # calendar_monitor subgraph + scheduler
│       ├── email.py       # email_monitor subgraph + scheduler
│       └── watchdog.py    # rewired from watch.py
│
├── memory/
│   └── mem0_store.py      # mem0 wrapper (retrieve, store, remember, forget)
│
├── tools/
│   ├── coding.py          # existing tools from tools.py
│   ├── calendar.py        # Google Calendar tool implementations
│   ├── email.py           # Gmail tool implementations
│   ├── web.py             # web_search + fetch_page
│   └── registry.py        # unified TOOL_SCHEMAS + Toolbox dispatch
│
├── integrations/
│   └── google_auth.py     # OAuth flow + token management
│
├── voice/
│   ├── audio.py           # unchanged
│   ├── stt.py             # unchanged
│   ├── tts.py             # swapped to Kokoro
│   ├── wake.py            # unchanged
│   └── streaming.py       # unchanged
│
├── telegram_bot.py        # largely unchanged
├── main.py                # CLI rewired (new `auth` command added)
└── __main__.py
```

### New env vars
```bash
# LLM — now Claude API
ANTHROPIC_API_KEY=sk-ant-...
JARVIS_LLM_MODEL=claude-sonnet-4-6

# Google
JARVIS_GOOGLE_CREDENTIALS=~/.jarvis/credentials.json
JARVIS_GOOGLE_TOKEN=~/.jarvis/google_token.json
JARVIS_ALLOW_SEND_EMAIL=false

# Web search
JARVIS_BRAVE_API_KEY=...        # optional, falls back to DuckDuckGo

# Memory
JARVIS_MEM0_PATH=~/.jarvis/mem0

# Proactive loop intervals
JARVIS_CALENDAR_POLL_SECONDS=300
JARVIS_EMAIL_POLL_SECONDS=600
```

---

## 8. Dependencies (additions to pyproject.toml)

```toml
# Core additions
"anthropic>=0.40.0"          # Claude API (replaces openai for LLM)
"langgraph>=0.2.0"           # agent orchestration
"mem0ai>=0.1.0"              # memory layer
"chromadb>=0.5.0"            # vector store (used by mem0 local)

# Google
"google-auth>=2.30.0"
"google-auth-oauthlib>=1.2.0"
"google-api-python-client>=2.130.0"

# Web tools
"trafilatura>=1.12.0"        # clean text extraction from pages
"duckduckgo-search>=6.0.0"   # fallback search

# TTS — Kokoro via ONNX (Mac ARM native, no GPU required)
"kokoro-onnx>=0.4.0"         # replaces piper-tts; ONNX runtime on Apple Silicon
"soundfile>=0.12.0"          # audio I/O for kokoro output

# Keep existing: faster-whisper, sounddevice, webrtcvad, watchfiles,
#                python-telegram-bot, pydantic-settings, typer, rich
```

---

## 9. What Is NOT Changing

- Wake word detection (openWakeWord + STT fallback)
- STT (faster-whisper + WebRTC VAD)
- Audio recording pipeline
- Telegram bridge (notify + remote chat)
- Safety model (all destructive tools gated by env flags + confirmation)
- CLI interface (`jarvis voice`, `jarvis chat`, `jarvis ask`, etc.)
- Makefile targets

---

## 10. Implementation Phases

**Phase 1 — Core rewrite (this plan):**
- LangGraph graph replaces flat agent loop
- Claude API replaces vLLM/Qwen
- mem0 local replaces SQLite memory
- Kokoro replaces Piper
- Google Calendar + Gmail tools + auth
- Web search + fetch tools
- Proactive loops (calendar + email monitors)
- Confirmation flow for destructive actions

**Phase 2 — Future:**
- Neo4j graph backend for mem0 (richer relational queries)
- Gmail push notifications via Pub/Sub (real-time vs polling)
- Home Assistant integration
- Multi-device presence detection (voice vs Telegram routing)
- Browser control (Playwright) for web automation
