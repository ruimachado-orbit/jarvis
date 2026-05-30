.PHONY: init voice chat ask telegram watch notify test auth-google memory-show memory-reset ensure-venv kill-zombies

init:
	@PY=python3; command -v $$PY >/dev/null 2>&1 || PY=python; \
	$$PY -m venv .venv && \
	.venv/bin/pip install -e '.[dev,wake]'
	cp -n .env.example .env || true
	@echo "Edit .env and set HF_TOKEN for Voxtral access, then run: make auth-google"

# Create .venv and install deps when missing (same as init, without clobbering .env).
ensure-venv:
	@if [ ! -f .env ] && [ -f .env.example ]; then \
		cp .env.example .env; \
		echo "Created .env from .env.example — set API keys as needed."; \
	fi
	@if [ ! -x .venv/bin/python ]; then \
		echo "Creating virtual environment and installing dependencies..."; \
		PY=python3; command -v $$PY >/dev/null 2>&1 || PY=python; \
		$$PY -m venv .venv && \
		.venv/bin/pip install -e '.[dev,wake]'; \
	fi

# HF_TOKEN from the environment, else first ^HF_TOKEN= line in .env (do not `source` .env — values may be dotenv-only, e.g. spaces).
load_hf_token = if [ -z "$$HF_TOKEN" ] && [ -f .env ]; then \
	HF_TOKEN=$$(grep '^HF_TOKEN=' .env 2>/dev/null | head -1 | cut -d= -f2- | tr -d '\r' | sed "s/^[\"']//;s/[\"']$$//"); \
	export HF_TOKEN; \
	fi

# Stopped (Ctrl-Z'd) jarvis voice processes hold the macOS mic + speaker
# handles and silently break the next session. Force-kill before each dev run.
kill-zombies:
	@if pgrep -f 'jarvis voice' >/dev/null 2>&1; then \
		echo "Killing lingering jarvis voice processes..."; \
		pkill -9 -f 'jarvis voice' 2>/dev/null || true; \
		sleep 0.3; \
	fi

voice: ensure-venv kill-zombies
	@$(load_hf_token); \
	if [ -z "$$HF_TOKEN" ]; then \
		echo "Set HF_TOKEN in .env (or export it) to download Voxtral model weights."; \
		echo "Get a token at: https://huggingface.co/settings/tokens"; \
		exit 1; \
	fi; \
	.venv/bin/python -m jarvis voice

chat:
	jarvis chat

ask:
	@[ "$(Q)" ] || (echo "Usage: make ask Q='your question'"; exit 1)
	jarvis ask "$(Q)"

telegram:
	jarvis telegram

watch:
	jarvis watch

notify:
	@[ "$(M)" ] || (echo "Usage: make notify M='your message'"; exit 1)
	jarvis notify "$(M)"

auth-google:
	jarvis auth google

test:
	pytest tests/ -v --tb=short --ignore=tests/test_memory.py --ignore=tests/test_tools_safety.py --ignore=tests/test_streaming_pipeline.py --ignore=tests/test_agent_streaming.py

memory-show:
	@echo "Memory is now managed by mem0. Check ~/.jarvis/mem0/"

memory-reset:
	rm -rf ~/.jarvis/mem0/
	@echo "mem0 store cleared"
