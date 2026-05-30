#!/usr/bin/env python3
"""Test Cmd+Shift+S keyboard shortcut listener."""

import asyncio
from pynput import keyboard

async def test_keyboard():
    print("Testing Cmd+Shift+S listener...")
    print("Press Cmd+Shift+S to trigger (or Ctrl-C to quit)")
    print()

    toggle_count = [0]

    def on_activate():
        toggle_count[0] += 1
        print(f"✓ Cmd+Shift+S pressed! (count: {toggle_count[0]})")

    listener = keyboard.GlobalHotKeys({
        '<cmd>+<shift>+s': on_activate,
        '<cmd>+<shift>+S': on_activate,
    })

    listener.start()

    try:
        await asyncio.sleep(30)  # Test for 30 seconds
        print(f"\nTest complete. Cmd+Shift+S was pressed {toggle_count[0]} times")
    except KeyboardInterrupt:
        print(f"\nTest cancelled. Cmd+Shift+S was pressed {toggle_count[0]} times")
    finally:
        listener.stop()

if __name__ == "__main__":
    asyncio.run(test_keyboard())
