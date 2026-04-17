#!/usr/bin/env bash
# Start vLLM with flags tuned for consumer GPUs.
# Quick decision rule: if model weights exceed 50% of free VRAM, add --enforce-eager.

set -euo pipefail

MODEL="${JARVIS_LLM_MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"
PORT="${JARVIS_VLLM_PORT:-8000}"
MAX_LEN="${JARVIS_VLLM_MAX_LEN:-8192}"
GPU_UTIL="${JARVIS_VLLM_GPU_UTIL:-0.85}"
EAGER_FLAG=""
if [[ "${JARVIS_VLLM_ENFORCE_EAGER:-false}" == "true" ]]; then
  EAGER_FLAG="--enforce-eager"
fi

echo "Serving ${MODEL} on :${PORT} (max_len=${MAX_LEN}, gpu_util=${GPU_UTIL})"
exec vllm serve "${MODEL}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --max-model-len "${MAX_LEN}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --enable-prefix-caching \
  --served-model-name "${MODEL}" \
  ${EAGER_FLAG}
