# Voxtral Installation Quick Start

## 1. Install Dependencies

```bash
cd /Users/ruimachado/Code/jarvis
pip install -e '.[dev,wake]'
```

This installs:
- `transformers>=4.48.0` (Hugging Face Transformers)
- `torch>=2.5.0` (PyTorch with MPS support)
- `torchaudio>=2.5.0` (Audio processing)
- `accelerate>=0.34.0` (Model loading optimization)

## 2. Get Hugging Face Access

### Create Token
1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Name: `jarvis-voxtral`
4. Type: **Read**
5. Copy the token (starts with `hf_...`)

### Request Model Access
1. Go to https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602
2. Click "Request access"
3. Wait for approval (usually instant, may take a few hours)

## 3. Configure Environment

Edit `.env`:
```bash
# Voice — Voxtral-Mini-4B-Realtime (unified STT+TTS)
HF_TOKEN=hf_your_token_here
JARVIS_STT_LANGUAGE=en
JARVIS_TTS_SPEED=1.0
JARVIS_TTS_MAX_AUDIO_MS=15000
```

**Remove these old settings if present:**
```bash
# Remove:
JARVIS_STT_MODEL=...
JARVIS_STT_DEVICE=...
JARVIS_STT_COMPUTE_TYPE=...
JARVIS_TTS_ENGINE=...
JARVIS_TTS_VOICE=...
```

## 4. Test Installation

```bash
make voice
```

**First run:**
- Downloads ~8GB model to `~/.cache/huggingface/hub/`
- Takes 3-5 minutes depending on internet speed
- Loads model into memory (~6GB unified memory on macOS)

**Subsequent runs:**
- Loads from cache (3-5 seconds)
- Ready to use immediately

## Expected Output

```
17:20:45 INFO jarvis: pre-warming Voxtral TTS cache (16 phrases)...
17:20:45 INFO jarvis.voice.voxtral: Loading Voxtral-Mini-4B from mistralai/Voxtral-Mini-4B-Realtime-2602 on device=mps (first run downloads ~8GB)
17:20:48 INFO jarvis.voice.voxtral: Voxtral-Mini-4B ready on mps (dtype=torch.bfloat16)
17:20:49 INFO jarvis: Voxtral TTS ready (16 phrases cached)
Jarvis standing by. Say 'Hey Jarvis' to wake me. Ctrl-C to quit.
```

## Verify Hardware Acceleration

```bash
python3 -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

Expected: `MPS available: True` (on Apple Silicon)

## Troubleshooting

### "HF_TOKEN not set"
- Add `HF_TOKEN=hf_...` to `.env`
- OR: `export HF_TOKEN=hf_...` before running

### "Cannot access repository"
- Request access at https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602
- Wait for approval email
- Try again after approval

### "MPS available: False"
- You're on Intel Mac or Linux → will use CUDA or CPU
- CPU is slow (~5-10s per synthesis)
- Recommended: Use on Apple Silicon Mac

### "Out of memory"
- Close other apps (need ~6GB free unified memory)
- Check Activity Monitor → Memory
- Restart Mac if memory pressure is high

### Downloads are slow
- Model is 8GB, will take time on slow connections
- Download happens once, then cached
- Can manually download to speed up:
  ```bash
  huggingface-cli download mistralai/Voxtral-Mini-4B-Realtime-2602
  ```

## Verify Installation

```bash
python3 << 'EOF'
from jarvis.voice.voxtral import VoxtralEngine
from jarvis.core.config import Settings
import asyncio
import numpy as np

settings = Settings(_env_file=".env")
voxtral = VoxtralEngine(settings)

# Test TTS
async def test():
    pcm, sr = await voxtral.synthesize("Hello, Sir.")
    print(f"✓ TTS: generated {len(pcm)} samples at {sr}Hz")
    
    # Test STT
    audio = np.random.randint(-1000, 1000, 16000, dtype=np.int16)
    text = await voxtral.transcribe(audio, 16000)
    print(f"✓ STT: transcribed to '{text}'")

asyncio.run(test())
print("✓ Voxtral ready!")
EOF
```

## Performance

### Your Hardware (Apple M-series)
- **First synthesis**: 2-3s (model loading)
- **Warm synthesis**: 500ms-1s
- **Cached phrases**: <10ms
- **Transcription**: 200-400ms

### Expected Latency
```
User speaks → VAD (30-700ms) → STT (200-400ms) → LLM (1-3s) → TTS (500ms) → Playback
Total: ~2-5 seconds from speech to response
```

Cached acknowledgements ("Right away, Sir.") play in <10ms while LLM processes.

## Next Steps

Once installation works:

1. **Test wake word**: Say "Hey Jarvis" and ask a question
2. **Check logs**: Look for "Voxtral-Mini-4B ready on mps"
3. **Verify latency**: First response will be slower (model loading)
4. **Monitor memory**: Activity Monitor → should see ~6GB used by Python

## Rollback

If you need to revert:

```bash
git revert HEAD
pip install faster-whisper kokoro-onnx
# Restore old .env settings
```

## Support

- Model page: https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602
- Migration guide: `docs/VOXTRAL_MIGRATION.md`
- Implementation details: `VOXTRAL_IMPLEMENTATION.md`
