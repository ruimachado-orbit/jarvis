# Jarvis — voice-first local coding assistant.
#
# One-shot setup:   make init
# Run everything:   make dev         (vLLM + voice loop)
# Dev bits:         make chat | make ask Q="..." | make telegram | make watch
#
# Variables (override on the command line, e.g. `make voice VOICE=en_US-lessac-medium`):
#   VOICE   Piper voice id to download/use. Default: en_GB-alan-medium.
#   PYTHON  Python interpreter. Default: python.

SHELL := /bin/bash
PYTHON ?= python
VENV ?= .venv
VOICE ?= en_GB-alan-medium
VLLM_WAIT ?= 300

# Activate the venv implicitly by prepending its bin to PATH for every recipe.
export PATH := $(VENV)/bin:$(PATH)

.PHONY: help install init env voice-model vllm wait-vllm voice chat ask telegram notify watch test lint clean dev memory-show memory-refresh memory-reset

help:
	@echo "Common targets:"
	@echo "  make init        create venv, install deps, create .env, download voice"
	@echo "  make dev         start vLLM + hands-free voice loop (Ctrl-C stops both)"
	@echo "  make vllm        start vLLM only"
	@echo "  make voice       start the voice loop (vLLM must be running)"
	@echo "  make chat        text REPL"
	@echo "  make ask Q=...   one-shot text query"
	@echo "  make telegram    run only the Telegram bridge"
	@echo "  make watch       watch workspace and notify Telegram on changes"
	@echo "  make notify M=.. push a message to the notify chat"
	@echo "  make memory-show      show profile + recent episodes"
	@echo "  make memory-refresh   force profile distillation now"
	@echo "  make memory-reset     wipe the memory DB (asks to confirm)"
	@echo "  make test        run pytest"

# ----- setup -----

install:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -e '.[dev,wake]'

env:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env from .env.example — edit it before running."; \
	else \
		echo ".env already exists; leaving it alone."; \
	fi

voice-model:
	@mkdir -p voices
	@if [ ! -f voices/$(VOICE).onnx ]; then \
		./scripts/download_voice.sh $(VOICE); \
	else \
		echo "voices/$(VOICE).onnx already present."; \
	fi

init: install env voice-model
	@echo ""
	@echo "Jarvis is ready. Next steps:"
	@echo "  1. Edit .env (Telegram token, voice path, workspace)."
	@echo "  2. Run 'make dev' to launch vLLM + the voice loop."

# ----- run -----

vllm:
	./scripts/serve_vllm.sh

wait-vllm:
	./scripts/wait_for_vllm.sh $(VLLM_WAIT)

voice:
	jarvis voice

chat:
	jarvis chat

ask:
	@if [ -z "$$Q" ]; then echo "Usage: make ask Q=\"your question\""; exit 2; fi
	jarvis ask "$$Q"

telegram:
	jarvis telegram

notify:
	@if [ -z "$$M" ]; then echo "Usage: make notify M=\"message\""; exit 2; fi
	jarvis notify "$$M"

watch:
	jarvis watch

memory-show:
	jarvis memory show

memory-refresh:
	jarvis memory refresh

memory-reset:
	jarvis memory reset

test:
	pytest -q

lint:
	ruff check jarvis tests

clean:
	rm -rf $(VENV) dist build *.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +

# ----- orchestration -----
# `make dev` starts vLLM in the background, waits for its OpenAI endpoint to
# answer, then runs the voice loop in the foreground. Ctrl-C (or exit) tears
# down both via the trap.

dev:
	@trap 'echo; echo "Stopping dev..."; kill 0' INT TERM EXIT; \
	echo "Starting vLLM..."; \
	./scripts/serve_vllm.sh > .vllm.log 2>&1 & \
	VLLM_PID=$$!; \
	./scripts/wait_for_vllm.sh $(VLLM_WAIT); \
	echo "vLLM PID=$$VLLM_PID — tailing to .vllm.log"; \
	jarvis voice; \
	wait
