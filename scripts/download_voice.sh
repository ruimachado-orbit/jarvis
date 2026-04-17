#!/usr/bin/env bash
# Download a Piper voice into ./voices.
# Usage: ./scripts/download_voice.sh [voice_id]
# Default: en_GB-alan-medium. Browse catalogue: https://huggingface.co/rhasspy/piper-voices

set -euo pipefail

VOICE="${1:-en_GB-alan-medium}"
LANG_CODE="${VOICE%%-*}"
LOCALE="${VOICE%%-*}"
# Piper voice paths on HF follow: <locale>/<speaker>/<quality>/<voice>.onnx
# Parse voice id "en_GB-alan-medium" -> locale=en_GB speaker=alan quality=medium
IFS='-' read -r LOCALE SPEAKER QUALITY <<< "${VOICE}"
LANG_SHORT="${LOCALE%%_*}"
BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/${LANG_SHORT}/${LOCALE}/${SPEAKER}/${QUALITY}"

mkdir -p voices
echo "Downloading ${VOICE}..."
curl -L "${BASE}/${VOICE}.onnx" -o "voices/${VOICE}.onnx"
curl -L "${BASE}/${VOICE}.onnx.json" -o "voices/${VOICE}.onnx.json"
echo "Saved to voices/${VOICE}.onnx"
echo "Set JARVIS_TTS_VOICE=./voices/${VOICE}.onnx in your .env"
