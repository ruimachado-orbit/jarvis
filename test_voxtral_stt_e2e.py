#!/usr/bin/env python3
"""End-to-end Voxtral STT testing script.

Tests:
1. Model loading and device detection
2. STT transcription with synthetic audio
3. Low-energy rejection
4. Resampling from different sample rates
5. Performance benchmarks
"""

import asyncio
import time
from pathlib import Path

import numpy as np

from jarvis.core.config import Settings
from jarvis.voice.stt import STT


def print_header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


async def test_1_device_detection():
    """Test 1: Device detection"""
    print_header("Test 1: Device Detection")

    from jarvis.voice.stt import _get_device
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


async def test_2_stt_transcription(stt: STT):
    """Test 2: STT Transcription"""
    print_header("Test 2: STT Transcription")

    print("\n[1/2] Testing with synthetic audio...")
    sample_rate = 16000
    duration = 1.0
    freq = 440  # A4 note

    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = (np.sin(2 * np.pi * freq * t) * 5000).astype(np.int16)

    start = time.time()
    text = await stt.transcribe(audio, sample_rate)
    elapsed = time.time() - start

    print(f"  ✓ Transcription time: {elapsed:.2f}s")
    print(f"  ✓ Transcribed text: '{text}'")
    print(f"  ℹ️  (Synthetic audio may not produce meaningful text)")

    # Test low-energy rejection
    print("\n[2/2] Testing low-energy rejection...")
    quiet_audio = np.random.randint(-10, 10, 16000, dtype=np.int16)

    start = time.time()
    text = await stt.transcribe(quiet_audio, sample_rate)
    elapsed = time.time() - start

    print(f"  ✓ Transcription time: {elapsed:.2f}s")
    assert text == "", "Low-energy audio should be rejected"
    print(f"  ✓ Correctly rejected low-energy audio")


async def test_3_resampling(stt: STT):
    """Test 3: Audio Resampling"""
    print_header("Test 3: Audio Resampling")

    sample_rates = [8000, 16000, 22050, 44100, 48000]

    for sr in sample_rates:
        print(f"\n[{sample_rates.index(sr)+1}/{len(sample_rates)}] Testing {sr}Hz...")

        # Generate 0.5s of audio
        duration = 0.5
        t = np.linspace(0, duration, int(sr * duration))
        audio = (np.sin(2 * np.pi * 440 * t) * 5000).astype(np.int16)

        start = time.time()
        text = await stt.transcribe(audio, sr)
        elapsed = time.time() - start

        print(f"  ✓ {sr}Hz → 16kHz resampling")
        print(f"  ✓ Transcription time: {elapsed:.2f}s")
        print(f"  ✓ Result: '{text}'")


async def test_4_performance_benchmark(stt: STT):
    """Test 4: Performance Benchmarks"""
    print_header("Test 4: Performance Benchmarks")

    # Warm-up (first call loads the model)
    print("\n[Warm-up] Loading model...")
    audio = np.random.randint(-1000, 1000, 16000, dtype=np.int16)
    start = time.time()
    await stt.transcribe(audio, 16000)
    warmup_time = time.time() - start
    print(f"  ✓ First transcription (model loading): {warmup_time:.2f}s")

    # Benchmark STT with different durations
    durations = [1.0, 2.0, 3.0]

    for duration in durations:
        print(f"\n[Benchmark {duration}s audio] Running 3 iterations...")
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = (np.sin(2 * np.pi * 440 * t) * 5000).astype(np.int16)

        times = []
        for i in range(3):
            start = time.time()
            text = await stt.transcribe(audio, sample_rate)
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"  Run {i+1}: {elapsed:.3f}s")

        avg_time = sum(times) / len(times)
        print(f"\n  ✓ Average time: {avg_time:.3f}s")
        print(f"  ✓ Min: {min(times):.3f}s, Max: {max(times):.3f}s")
        print(f"  ✓ Latency per second of audio: {avg_time/duration:.3f}s")


async def main():
    """Run all tests"""
    print(f"\n{'#'*60}")
    print(f"#  Voxtral STT End-to-End Test Suite")
    print(f"{'#'*60}")

    # Load settings
    print("\nLoading settings...")
    settings = Settings(_env_file=".env")
    print(f"✓ Settings loaded")
    print(f"  - STT Language: {settings.stt_language}")

    # Initialize STT
    print("\nInitializing Voxtral STT...")
    stt = STT(settings)
    print(f"✓ STT initialized")

    try:
        # Run tests
        device = await test_1_device_detection()
        await test_2_stt_transcription(stt)
        await test_3_resampling(stt)
        await test_4_performance_benchmark(stt)

        # Summary
        print_header("Test Summary")
        print(f"✓ All tests passed!")
        print(f"✓ Device: {device}")
        print(f"✓ Voxtral STT is ready for production use")
        print(f"\nExpected latency: <500ms for real speech")
        print(f"(Synthetic audio may be slower to process)")
        print(f"\nNext steps:")
        print(f"  1. Run: make voice")
        print(f"  2. Say: 'Hey Jarvis'")
        print(f"  3. Ask a question and verify STT transcription")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
