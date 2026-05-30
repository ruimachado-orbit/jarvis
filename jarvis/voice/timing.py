"""Timing instrumentation for latency analysis."""
import time
import logging
from contextlib import contextmanager

log = logging.getLogger(__name__)

_timings = {}

@contextmanager
def measure(step: str):
    """Context manager to measure elapsed time for a step."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = (time.perf_counter() - start) * 1000  # ms
        _timings[step] = elapsed
        log.info(f"⏱️  {step}: {elapsed:.0f}ms")

def reset_timings():
    """Clear all timing measurements."""
    global _timings
    _timings = {}

def get_timings():
    """Return all timing measurements."""
    return _timings.copy()

def log_summary():
    """Log a summary of all timing measurements."""
    if not _timings:
        return

    total = sum(_timings.values())
    log.info("=" * 60)
    log.info("LATENCY BREAKDOWN:")
    for step, ms in _timings.items():
        pct = (ms / total * 100) if total > 0 else 0
        log.info(f"  {step:30s} {ms:6.0f}ms ({pct:4.1f}%)")
    log.info(f"  {'TOTAL':30s} {total:6.0f}ms")
    log.info("=" * 60)
