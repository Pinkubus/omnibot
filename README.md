# Macro Maker

A powerful Windows automation tool with a GUI for creating and executing macros with advanced control flow.

## Features

- **GUI-based macro creation** - Visual interface to build automation sequences
- **Control flow** - IF/ELSE/ENDIF statements with image recognition
- **Keyboard automation** - Press, hold, and release keys
- **Mouse automation** - Click, hold, move, and release with position capture
- **Image recognition** - IF IMAGE and FIND IMAGE commands
- **Repeat loops** - Execute blocks of commands multiple times
- **System-wide hotkeys** - Trigger macros from anywhere with custom key combinations
- **Persistent storage** - Macros saved between sessions
- **Repeat execution** - Run macros 1-N times or infinitely (0)

## Installation

1. Install Python 3.8 or higher
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the application:
```bash
python macro_maker.py
```

### Creating a Macro

1. Click "New Macro" and enter a name
2. Add commands using the buttons:
   - **Key commands**: Press, Hold, Release
   - **Mouse commands**: Click, Move (captures mouse position)
   - **Control flow**: IF, ELSE, ENDIF
   - **Image commands**: IF IMAGE, FIND IMAGE (captures screen)
   - **Loops**: REPEAT, END REPEAT
   - **Timing**: Delay

3. Set a hotkey with "Set Hotkey" button
4. Specify repeat count (0 = infinite)
5. Click "Run Macro" or use the hotkey

### Command Examples

- **Key Press**: Simulates pressing and releasing a key (e.g., "enter", "a", "ctrl")
- **Mouse Click**: Captures position, then clicks there when executed
- **IF/ELSE/ENDIF**: Conditional execution
- **IF IMAGE**: Checks if image appears on screen
- **FIND IMAGE**: Finds and moves mouse to image
- **REPEAT**: Loops commands (e.g., REPEAT 5 times)

### Tips

- Use delays between commands to ensure proper execution
- Image recognition requires good lighting and consistent screen conditions
- Hotkeys work system-wide when the app is running
- Set repeat to 0 for infinite loops (use Stop button to halt)

## File Structure

- `macro_maker.py` - Main application
- `macros.json` - Saved macros (auto-generated)
- `images/` - Captured images for recognition (auto-generated)

## Requirements

See `requirements.txt` for dependencies.
