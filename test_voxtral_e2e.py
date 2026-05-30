#!/usr/bin/env python3
"""End-to-end Voxtral testing script.

Tests:
1. Model loading and device detection
2. TTS synthesis (text → audio)
3. STT transcription (audio → text)
4. Phrase pre-warming and caching
5. Performance benchmarks
"""

import asyncio
import time
from pathlib import Path

import numpy as np

from jarvis.core.config import Settings
from jarvis.voice.voxtral import VoxtralEngine


def print_header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


async def test_1_device_detection():
    """Test 1: Device detection"""
    print_header("Test 1: Device Detection")

    from jarvis.voice.voxtral import _get_device
    device = _get_device()

    print(f"✓ Detected device: {device}")

    import torch
    if device == "mps":
        print(f"✓ MPS available: {torch.backends.mps.is_available()}")
    elif device == "cuda":
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"⚠️  Using CPU (slow)")

    return device


async def test_2_tts_synthesis(voxtral: VoxtralEngine):
    """Test 2: TTS Synthesis"""
    print_header("Test 2: TTS Synthesis")

    test_phrases = [
        "Hello, Sir. Jarvis is online.",
        "The weather is sunny today.",
        "I am ready to assist you.",
    ]

    for i, phrase in enumerate(test_phrases, 1):
        print(f"\n[{i}/{len(test_phrases)}] Synthesizing: '{phrase}'")

        start = time.time()
        pcm, sr = await voxtral.synthesize(phrase)
        elapsed = time.time() - start

        duration_s = len(pcm) / sr

        print(f"  ✓ Generated {len(pcm)} samples at {sr}Hz")
        print(f"  ✓ Audio duration: {duration_s:.2f}s")
        print(f"  ✓ Synthesis time: {elapsed:.2f}s")
        print(f"  ✓ Real-time factor: {duration_s/elapsed:.2f}x")

        # Verify audio properties
        assert isinstance(pcm, np.ndarray), "PCM should be numpy array"
        assert pcm.dtype == np.float32, "PCM should be float32"
        assert sr == 24000, "Sample rate should be 24kHz"
        assert len(pcm) > 0, "Audio should not be empty"

        # Check audio energy
        rms = float(np.sqrt(np.mean(pcm ** 2)))
        print(f"  ✓ Audio RMS: {rms:.4f}")
        assert rms > 0.001, "Audio should have energy"


async def test_3_stt_transcription(voxtral: VoxtralEngine):
    """Test 3: STT Transcription"""
    print_header("Test 3: STT Transcription")

    # Generate synthetic audio (sine wave) to test the pipeline
    print("\n[1/2] Testing with synthetic audio...")
    sample_rate = 16000
    duration = 1.0  # 1 second
    freq = 440  # A4 note

    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = (np.sin(2 * np.pi * freq * t) * 5000).astype(np.int16)

    start = time.time()
    text = await voxtral.transcribe(audio, sample_rate)
    elapsed = time.time() - start

    print(f"  ✓ Transcription time: {elapsed:.2f}s")
    print(f"  ✓ Transcribed text: '{text}'")
    print(f"  ℹ️  (Synthetic audio may not produce meaningful text)")

    # Test low-energy rejection
    print("\n[2/2] Testing low-energy rejection...")
    quiet_audio = np.random.randint(-10, 10, 16000, dtype=np.int16)

    start = time.time()
    text = await voxtral.transcribe(quiet_audio, sample_rate)
    elapsed = time.time() - start

    print(f"  ✓ Transcription time: {elapsed:.2f}s")
    assert text == "", "Low-energy audio should be rejected"
    print(f"  ✓ Correctly rejected low-energy audio")


async def test_4_phrase_caching(voxtral: VoxtralEngine):
    """Test 4: Phrase Pre-warming and Caching"""
    print_header("Test 4: Phrase Caching")

    test_phrases = [
        "Right away, Sir.",
        "One moment, Sir.",
        "At your service, Sir.",
    ]

    print(f"\n[1/3] Pre-warming {len(test_phrases)} phrases...")
    start = time.time()
    cache = await voxtral.prewarm(test_phrases)
    elapsed = time.time() - start

    print(f"  ✓ Pre-warming completed in {elapsed:.2f}s")
    print(f"  ✓ Cached {len(cache)} phrases")

    # Verify cache contents
    for phrase in test_phrases:
        assert phrase in cache, f"Phrase '{phrase}' should be in cache"
        pcm, sr = cache[phrase]
        assert isinstance(pcm, np.ndarray), "Cached PCM should be numpy array"
        assert sr == 24000, "Cached sample rate should be 24kHz"
        print(f"  ✓ '{phrase}' → {len(pcm)} samples")

    # Test cache hit (should be instant)
    print(f"\n[2/3] Testing cache retrieval...")
    for phrase in test_phrases:
        start = time.time()
        pcm, sr = cache[phrase]
        elapsed = time.time() - start
        print(f"  ✓ '{phrase}' retrieved in {elapsed*1000:.1f}ms")

    # Verify disk cache
    print(f"\n[3/3] Verifying disk cache...")
    from jarvis.voice.voxtral import _PHRASE_CACHE_DIR
    cache_files = list(_PHRASE_CACHE_DIR.glob("*.npz"))
    print(f"  ✓ Found {len(cache_files)} cached files in {_PHRASE_CACHE_DIR}")


async def test_5_performance_benchmark(voxtral: VoxtralEngine):
    """Test 5: Performance Benchmarks"""
    print_header("Test 5: Performance Benchmarks")

    # Warm-up (first call loads the model)
    print("\n[Warm-up] Loading model...")
    start = time.time()
    await voxtral.synthesize("Warm up.")
    warmup_time = time.time() - start
    print(f"  ✓ First synthesis (model loading): {warmup_time:.2f}s")

    # Benchmark TTS
    print("\n[TTS Benchmark] Running 5 synthesis iterations...")
    test_text = "The quick brown fox jumps over the lazy dog."
    times = []

    for i in range(5):
        start = time.time()
        pcm, sr = await voxtral.synthesize(test_text)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.3f}s")

    avg_time = sum(times) / len(times)
    print(f"\n  ✓ Average TTS time: {avg_time:.3f}s")
    print(f"  ✓ Min: {min(times):.3f}s, Max: {max(times):.3f}s")

    # Benchmark STT
    print("\n[STT Benchmark] Running 3 transcription iterations...")
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = (np.sin(2 * np.pi * 440 * t) * 5000).astype(np.int16)

    times = []
    for i in range(3):
        start = time.time()
        text = await voxtral.transcribe(audio, sample_rate)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.3f}s")

    avg_time = sum(times) / len(times)
    print(f"\n  ✓ Average STT time: {avg_time:.3f}s")
    print(f"  ✓ Min: {min(times):.3f}s, Max: {max(times):.3f}s")


async def main():
    """Run all tests"""
    print(f"\n{'#'*60}")
    print(f"#  Voxtral End-to-End Test Suite")
    print(f"{'#'*60}")

    # Load settings
    print("\nLoading settings...")
    settings = Settings(_env_file=".env")
    print(f"✓ Settings loaded")
    print(f"  - STT Language: {settings.stt_language}")
    print(f"  - TTS Speed: {settings.tts_speed}")
    print(f"  - Max Audio MS: {settings.tts_max_audio_ms}")

    # Initialize Voxtral
    print("\nInitializing VoxtralEngine...")
    voxtral = VoxtralEngine(settings)
    print(f"✓ VoxtralEngine initialized")

    try:
        # Run tests
        device = await test_1_device_detection()
        await test_2_tts_synthesis(voxtral)
        await test_3_stt_transcription(voxtral)
        await test_4_phrase_caching(voxtral)
        await test_5_performance_benchmark(voxtral)

        # Summary
        print_header("Test Summary")
        print(f"✓ All tests passed!")
        print(f"✓ Device: {device}")
        print(f"✓ Voxtral is ready for production use")
        print(f"\nNext steps:")
        print(f"  1. Run: make voice")
        print(f"  2. Say: 'Hey Jarvis'")
        print(f"  3. Ask a question and listen for the response")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
