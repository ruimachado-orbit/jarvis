# Voxtral Migration Guide

## Overview

Jarvis has been completely migrated from separate STT (faster-whisper) and TTS (CSM/Kokoro) engines to **Mistral Voxtral-Mini-4B-Realtime** — a unified 4B-parameter model that handles both speech-to-text and text-to-speech.

## What Changed

### Removed Components
- ❌ `jarvis/voice/stt.py` (faster-whisper)
- ❌ `jarvis/voice/tts.py` (CSM 1B + Kokoro fallback)
- ❌ `kokoro/` directory (ONNX model files)
- ❌ Separate STT and TTS configuration options

### Added Components
- ✅ `jarvis/voice/voxtral.py` (unified STT+TTS engine)
- ✅ Single model for both transcription and synthesis
- ✅ Optimized for Apple Silicon (MPS) with bf16 precision
- ✅ Disk-based phrase caching for instant acknowledgements

## Configuration Changes

### Old `.env` (removed)
```bash
# STT
JARVIS_STT_MODEL=base.en
JARVIS_STT_DEVICE=auto
JARVIS_STT_COMPUTE_TYPE=int8

# TTS
HF_TOKEN=hf_...
JARVIS_TTS_ENGINE=csm
JARVIS_TTS_VOICE=0
```

### New `.env` (simplified)
```bash
# Voice — Voxtral-Mini-4B-Realtime (unified STT+TTS)
HF_TOKEN=hf_...
JARVIS_STT_LANGUAGE=en
JARVIS_TTS_SPEED=1.0
JARVIS_TTS_MAX_AUDIO_MS=15000
```

## Dependencies

### Old
```toml
"faster-whisper>=1.0.3"
"kokoro-onnx>=0.4.0"
```

### New
```toml
"transformers>=4.48.0"
"torch>=2.5.0"
"torchaudio>=2.5.0"
"accelerate>=0.34.0"
```

## Migration Steps

1. **Update dependencies**
   ```bash
   pip install -e '.[dev,wake]'
   ```

2. **Get Hugging Face access**
   - Create HF account: https://huggingface.co/join
   - Create token: https://huggingface.co/settings/tokens (read access)
   - Request access: https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602
   - Add to `.env`: `HF_TOKEN=hf_...`

3. **Update `.env` settings**
   - Remove: `JARVIS_STT_MODEL`, `JARVIS_STT_DEVICE`, `JARVIS_STT_COMPUTE_TYPE`, `JARVIS_TTS_ENGINE`, `JARVIS_TTS_VOICE`
   - Keep: `JARVIS_STT_LANGUAGE`, `JARVIS_TTS_SPEED`, `JARVIS_TTS_MAX_AUDIO_MS`

4. **Test the migration**
   ```bash
   make voice
   ```

   First run downloads ~8GB model weights to `~/.cache/huggingface/hub/`.

## Performance

### Hardware Optimization

**Apple Silicon (MPS) — Recommended**
- M1/M2/M3 with 16GB+ unified memory
- Uses `bf16` precision for optimal speed
- ~500ms-1s TTS latency (warm)
- ~200ms STT latency

**CUDA**
- RTX 3060 12GB or better
- Uses `bf16` precision
- Similar latency to MPS

**CPU Fallback**
- Uses `fp32` precision
- 5-10s TTS latency (not recommended for real-time)

### Caching

Voxtral pre-warms common phrases (acknowledgements, fillers) at startup:
- "Right away, Sir."
- "One moment, Sir."
- "At your service, Sir."
- etc.

These are cached to `~/.cache/jarvis/voxtral/` as `.npz` files for instant playback (<10ms).

## Benefits

1. **Single Model**: One 4B model replaces two separate models (faster-whisper + CSM/Kokoro)
2. **Real-time Optimized**: Designed for low-latency speech-to-speech
3. **Simplified Config**: Fewer settings to tune
4. **Better Integration**: Native transformers support, no ONNX conversion needed
5. **Future-Proof**: Can leverage speech-to-speech mode (direct audio→audio) for even lower latency

## Rollback (if needed)

If you need to revert temporarily:

```bash
git checkout HEAD~1 -- jarvis/voice/
pip install faster-whisper kokoro-onnx
```

Then restore old `.env` settings.

## Troubleshooting

### "HF_TOKEN not set"
Add your Hugging Face token to `.env`: `HF_TOKEN=hf_...`

### "Cannot access mistralai/Voxtral-Mini-4B-Realtime-2602"
Request access at the model page. May take a few hours for approval.

### "Out of memory"
- Close other apps to free unified memory (macOS)
- Reduce `JARVIS_TTS_MAX_AUDIO_MS` to generate shorter clips
- Use CPU fallback (slow): `PYTORCH_ENABLE_MPS_FALLBACK=1`

### "Slow synthesis"
First run is always slow (downloads model). Subsequent runs should be faster.
If still slow, check:
- MPS is available: `python -c "import torch; print(torch.backends.mps.is_available())"`
- Not using CPU fallback

## Code Changes

If you've customized Jarvis voice code:

- Replace `from jarvis.voice.stt import STT` → `from jarvis.voice.voxtral import VoxtralEngine`
- Replace `from jarvis.voice.tts import TTS` → `from jarvis.voice.voxtral import VoxtralEngine`
- Replace `stt = STT(settings)` → `voxtral = VoxtralEngine(settings)`
- Replace `tts = TTS(settings)` → (same voxtral instance)
- Replace `await stt.transcribe(...)` → `await voxtral.transcribe(...)`
- Replace `await tts.synthesize(...)` → `await voxtral.synthesize(...)`
- Replace `await tts.prewarm(...)` → `await voxtral.prewarm(...)`

Both STT and TTS are now unified in a single `VoxtralEngine` instance.
