# Keyboard Shortcut: Cmd+Shift+S Wake/Sleep Toggle

## Feature

Press **Cmd+Shift+S** once to toggle Jarvis between wake and sleep states:
- **First press (sleeping)**: Wakes up Jarvis → "At your service, Sir."
- **Second press (awake)**: Puts Jarvis to sleep → "Very good, Sir. Powering down."

This provides a quick keyboard alternative to voice commands ("Hey Jarvis" / "Jarvis sleep").

## Requirements

- **macOS only** (uses `pynput` global hotkey listener)
- **Accessibility permissions** required (see setup below)

## Setup (First Time Only)

### Grant Accessibility Permissions

macOS requires apps to have Accessibility permissions to monitor global keyboard shortcuts.

1. **Run Jarvis once**:
   ```bash
   make voice
   ```

2. **macOS will prompt** for Accessibility access:
   - Click "Open System Settings"
   - OR manually go to: **System Settings → Privacy & Security → Accessibility**

3. **Grant access**:
   - Find "Terminal" (or "iTerm", "Claude Code", etc. depending on what you're running from)
   - Toggle it **ON**

4. **Restart Jarvis**:
   ```bash
   make voice
   ```

The shortcut will now work!

## Usage

### Voice Control (Always Works)
```
You: "Hey Jarvis"                    → Jarvis wakes up
You: "What is the weather today?"    → Jarvis responds
You: "Jarvis sleep"                  → Jarvis goes to sleep
```

### Keyboard Control (After Setup)
```
Press: Cmd+Shift+S                   → Jarvis wakes up
You: "What is the weather today?"    → Jarvis responds  
Press: Cmd+Shift+S                   → Jarvis goes to sleep
```

### Combined Usage
```
You: "Hey Jarvis"                    → Jarvis wakes up
You: "What is the time?"             → Jarvis responds
Press: Cmd+Shift+S                   → Jarvis goes to sleep (no need to say "Jarvis sleep")
```

## Behavior

- **Debounced**: Pressing Cmd+Shift+S multiple times quickly only registers once (500ms debounce)
- **Audio feedback**: Plays boot/sleep sounds just like voice commands
- **Visual feedback**: Shows "⌨️  Cmd+Shift+S: Waking up" / "⌨️  Cmd+Shift+S: Powering down" in terminal
- **Non-blocking**: Doesn't interfere with voice commands or ongoing responses

## Technical Details

### Implementation
- Uses `pynput.keyboard.GlobalHotKeys` for cross-platform hotkey monitoring
- Runs in background asyncio task alongside voice loop
- Thread-safe via `asyncio.Event` for state toggle signaling
- Gracefully handles missing Accessibility permissions (logs warning, continues)

### Dependencies
```toml
"pynput>=1.7.6"  # Keyboard monitoring
```

### Code Location
- Main implementation: `jarvis/main.py::_keyboard_listener()`
- Integration: `jarvis/main.py::_voice_main()` (line ~460)

## Troubleshooting

### "This process is not trusted!"
**Cause**: Accessibility permissions not granted

**Fix**:
1. Go to **System Settings → Privacy & Security → Accessibility**
2. Find your terminal app (Terminal, iTerm, VS Code, etc.)
3. Toggle it **ON**
4. Restart Jarvis

### Shortcut Not Working
**Check**:
1. Are you on macOS? (Only supported platform)
2. Did you grant Accessibility permissions?
3. Is another app using Cmd+Shift+S? (Unlikely - less common than Cmd+J)

**Test**:
```bash
python test_keyboard.py
# Press Cmd+Shift+S a few times
# Should see: "✓ Cmd+Shift+S pressed! (count: 1)" etc.
```

### Want to Change the Shortcut?
Edit `jarvis/main.py`, line ~88:
```python
with keyboard.GlobalHotKeys({
    '<cmd>+<shift>+k': on_activate,  # Change to Cmd+Shift+K
    '<ctrl>+<shift>+s': on_activate,  # Or Ctrl+Shift+S
}) as listener:
```

Common alternatives:
- `<cmd>+<shift>+k` — Cmd+Shift+K
- `<ctrl>+<shift>+s` — Ctrl+Shift+S
- `<cmd>+<alt>+s` — Cmd+Option+S (macOS)
- `<cmd>+j` — Cmd+J (simpler, but may conflict with IDEs)

## Disable Keyboard Shortcut

If you don't want the keyboard shortcut (voice-only mode):

**Option 1**: Remove pynput
```bash
pip uninstall pynput
```

**Option 2**: Edit `jarvis/main.py` (line ~521):
```python
# Keyboard shortcut handler (Cmd+J to toggle wake/sleep)
keyboard_task = None
# if sys.platform == "darwin":  # Comment out this line
#     keyboard_task = asyncio.create_task(_keyboard_listener(toggle_event, stop))
```

## Security & Privacy

- **No data sent**: All keyboard monitoring is local
- **Limited scope**: Only monitors Cmd+J, not all keystrokes
- **Open source**: Full implementation visible in `jarvis/main.py`
- **Standard permissions**: Uses macOS Accessibility API (same as tools like Alfred, Raycast, etc.)
