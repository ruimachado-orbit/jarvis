.PHONY: init dev voice chat ask telegram watch notify test auth-google memory-show memory-reset

init:
	python -m venv .venv
	.venv/bin/pip install -e '.[dev,wake]'
	cp -n .env.example .env || true
	@echo "Edit .env and set ANTHROPIC_API_KEY, then run: make auth-google"

dev: voice

voice:
	jarvis voice

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
