# Omnibot Setup Guide

This guide explains how to set up Omnibot on a new computer.

## Prerequisites

- **Windows OS** - Omnibot is designed for Windows (uses pyautogui, keyboard library, and Windows-specific features)
- **Python 3.8 or higher** - [Download Python](https://www.python.org/downloads/)

## Installation Steps

### 1. Clone or Download the Repository

```bash
git clone https://github.com/yourusername/omnibot.git
cd omnibot
```

Or download and extract the ZIP file to your desired location.

### 2. Install Python Dependencies

Open a terminal (PowerShell or Command Prompt) in the Omnibot directory and run:

```bash
pip install -r requirements.txt
```

This will install:
- `pyautogui` - For mouse/keyboard automation and screen capture
- `keyboard` - For hardware-level key presses and system-wide hotkeys
- `Pillow` (PIL) - For image processing and recognition
- `opencv-python` - For advanced image recognition
- `numpy` - Required by OpenCV

### 3. Run the Application

#### Option 1: Using the Batch File (Windows)
```bash
run.bat
```

#### Option 2: Direct Python Execution
```bash
python Omnibot.py
```

### 4. First Run

On first launch, Omnibot will:
- Create a `macros.json` file to store your macros
- Create an `images/` folder for captured screenshots
- Initialize the GUI with default settings

## Directory Structure

```
Omnibot/
├── Omnibot.py           # Main application file
├── requirements.txt     # Python dependencies
├── run.bat             # Windows batch file to launch app
├── activate.bat        # Virtual environment activation (if used)
├── README.md           # General documentation
├── SETUP.md            # This setup guide
├── macros.json         # Saved macros (created on first run)
└── images/             # Captured images (created as needed)
```

## Permissions and Security

### Administrator Rights
Some automation features may require administrator privileges:
- System-wide hotkeys
- Controlling applications running as administrator
- Certain keyboard/mouse operations

To run with administrator rights:
- Right-click `run.bat` → "Run as administrator"
- Or right-click `python.exe` → "Run as administrator", then run the script

### Windows Defender / Antivirus
Automation tools may trigger antivirus warnings because they:
- Monitor keyboard/mouse input
- Control system input
- Capture screenshots

Add Omnibot to your antivirus exclusions if needed.

### Firewall
Omnibot runs locally and does not require internet access or open ports.

## Keyboard Library Requirements

The `keyboard` library requires:
- Windows OS (uses Windows API for hardware-level key simulation)
- Administrator privileges for system-wide hotkeys
- May need to run as admin on first use to register hotkeys

## Troubleshooting

### "ModuleNotFoundError" - Missing Dependencies
```bash
pip install -r requirements.txt
```

### Hotkeys Not Working
- Run Omnibot as administrator
- Check if another application is using the same hotkey
- Try a different key combination

### Image Recognition Not Working
- Ensure captured images match current screen resolution
- Check lighting and screen conditions remain consistent
- Adjust confidence threshold (default: 0.8)
- Images are stored in `images/` folder

### Commands Not Executing
- Add delays between commands (Delay or Delay (ms))
- Check if target applications require administrator rights
- Verify mouse/keyboard positions haven't changed

### "Access Denied" Errors
- Run as administrator
- Check Windows security settings
- Verify antivirus isn't blocking automation

## Features Overview

- **Drag & Drop** - Reorder actions by dragging
- **Copy/Cut/Paste** - Ctrl+C, Ctrl+X, Ctrl+V in action list
- **Keyboard Navigation** - Arrow keys, Page Up/Down, Home/End
- **Quick Edit** - Double-click or press Enter on an action
- **Context Menu** - Right-click on actions for quick access
- **Undo/Redo** - Ctrl+Z, Ctrl+Y
- **Type-Ahead Search** - In key selector dropdown
- **Visual Drop Indicator** - Blue line shows where dragged items will drop

## Configuration Files

### macros.json
Stores all macros, commands, and settings. Backup this file to save your work.

### images/
Contains captured screenshots for image recognition. Keep these files if you want to reuse image recognition patterns.

## Updating

To update Omnibot:
1. Pull latest changes: `git pull origin main`
2. Update dependencies: `pip install -r requirements.txt --upgrade`
3. Restart the application

## Uninstalling

1. Close Omnibot
2. Delete the Omnibot directory
3. Optionally uninstall Python packages:
   ```bash
   pip uninstall pyautogui keyboard Pillow opencv-python numpy
   ```

## Support

For issues, questions, or contributions, visit the GitHub repository.
