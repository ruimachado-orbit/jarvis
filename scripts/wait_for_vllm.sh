#!/usr/bin/env bash
# Poll the vLLM OpenAI endpoint until it reports ready.
set -euo pipefail

BASE="${JARVIS_LLM_BASE_URL:-http://localhost:8000/v1}"
MAX_WAIT="${1:-300}"

echo "Waiting for vLLM at ${BASE} (up to ${MAX_WAIT}s)..."
end=$(( $(date +%s) + MAX_WAIT ))
while true; do
  if curl -sf "${BASE}/models" > /dev/null 2>&1; then
    echo "vLLM is ready."
    exit 0
  fi
  if [[ $(date +%s) -ge ${end} ]]; then
    echo "Timed out waiting for vLLM." >&2
    exit 1
  fi
  sleep 2
done
