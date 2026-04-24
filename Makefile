.PHONY: init dev voice chat ask telegram watch notify test auth-google memory-show memory-reset csm-install

# Auto-load .env if present
-include .env
export

init:
	python -m venv .venv
	.venv/bin/pip install -e '.[dev,wake]'
	cp -n .env.example .env || true
	@echo "Edit .env and set ANTHROPIC_API_KEY, then run: make auth-google"

csm-install:
	@echo "Installing CSM 1B TTS (Sesame)..."
	@if [ -z "$$HF_TOKEN" ]; then echo "Set HF_TOKEN first in .env: HF_TOKEN=hf_..."; exit 1; fi
	.venv/bin/pip install torch torchaudio
	.venv/bin/pip install transformers>=4.52.1 moshi tokenizers huggingface_hub torchtune silentcipher
	.venv/bin/huggingface-cli login --token "$$HF_TOKEN"
	@if [ ! -f jarvis/voice/csm_generator.py ]; then \
		git clone --depth 1 https://github.com/SesameAILabs/csm.git /tmp/csm && \
		cp /tmp/csm/generator.py jarvis/voice/csm_generator.py && \
		cp /tmp/csm/models.py jarvis/voice/csm_models.py && \
		cp /tmp/csm/watermarking.py jarvis/voice/csm_watermarking.py && \
		rm -rf /tmp/csm; \
	fi
	@echo "CSM installed."

dev:
	@if [ ! -f jarvis/voice/csm_generator.py ]; then \
		echo "CSM not installed. Running make csm-install first..."; \
		$(MAKE) csm-install; \
	fi
	@if [ -z "$$HF_TOKEN" ]; then \
		echo "Set HF_TOKEN in .env: HF_TOKEN=hf_..."; \
		exit 1; \
	fi
	@.venv/bin/python -c "import transformers, moshi, torchtune, silentcipher" 2>/dev/null || \
		(echo "CSM Python deps missing, installing..." && $(MAKE) csm-install)
	.venv/bin/python -m jarvis voice

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
