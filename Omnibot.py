import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import json
import os
import threading
import time
from pathlib import Path
import pyautogui
import keyboard
from pynput import mouse as pynput_mouse
from PIL import ImageGrab, Image
import numpy as np
import cv2
import ctypes
from ctypes import wintypes

# Disable pyautogui failsafe
pyautogui.FAILSAFE = False


class MacroCommand:
    """Base class for all macro commands"""
    def __init__(self, command_type):
        self.command_type = command_type
    
    def execute(self, context):
        """Execute the command"""
        pass
    
    def to_dict(self):
        """Convert command to dictionary for saving"""
        return {"type": self.command_type}
    
    @staticmethod
    def from_dict(data):
        """Create command from dictionary"""
        cmd_type = data.get("type")
        if cmd_type == "key_press":
            return KeyPressCommand(data.get("key"))
        elif cmd_type == "key_hold":
            return KeyHoldCommand(data.get("key"))
        elif cmd_type == "key_release":
            return KeyReleaseCommand(data.get("key"))
        elif cmd_type == "mouse_click":
            return MouseClickCommand(
                data.get("button"), 
                data.get("x"), 
                data.get("y"),
                data.get("mode", "absolute"),
                data.get("offset_x", 0),
                data.get("offset_y", 0)
            )
        elif cmd_type == "mouse_hold":
            return MouseHoldCommand(data.get("button"), data.get("x"), data.get("y"))
        elif cmd_type == "mouse_release":
            return MouseReleaseCommand(data.get("button"), data.get("x"), data.get("y"))
        elif cmd_type == "mouse_move":
            return MouseMoveCommand(data.get("x"), data.get("y"))
        elif cmd_type == "if_statement":
            return IfStatementCommand(data.get("condition"))
        elif cmd_type == "else_statement":
            return ElseStatementCommand()
        elif cmd_type == "endif_statement":
            return EndIfStatementCommand()
        elif cmd_type == "if_image":
            return IfImageCommand(
                data.get("image_path"),
                data.get("confidence", 0.8),
                data.get("move_to_image", False),
                data.get("image_paths", None)
            )
        elif cmd_type == "find_image":
            return FindImageCommand(data.get("image_path"), data.get("confidence", 0.8))
        elif cmd_type == "repeat":
            return RepeatCommand(data.get("count", 1))
        elif cmd_type == "end_repeat":
            return EndRepeatCommand()
        elif cmd_type == "delay":
            return DelayCommand(data.get("seconds", 0.1))
        elif cmd_type == "delay_ms":
            return DelayMsCommand(data.get("milliseconds", 100))
        elif cmd_type == "wait_for_window":
            return WaitForWindowCommand(data.get("window_pattern", "*"), data.get("timeout", 30))
        elif cmd_type == "message":
            return MessageCommand(
                data.get("text", ""),
                data.get("always_on_top", False),
                data.get("always_focused", False)
            )
        return None
    
    def __str__(self):
        return self.command_type.upper()


class KeyPressCommand(MacroCommand):
    def __init__(self, key):
        super().__init__("key_press")
        self.key = key
    
    def execute(self, context):
        keyboard.press_and_release(self.key)
    
    def to_dict(self):
        return {"type": self.command_type, "key": self.key}
    
    def __str__(self):
        return f"KEY PRESS: {self.key}"


class KeyHoldCommand(MacroCommand):
    def __init__(self, key):
        super().__init__("key_hold")
        self.key = key
    
    def execute(self, context):
        keyboard.press(self.key)
    
    def to_dict(self):
        return {"type": self.command_type, "key": self.key}
    
    def __str__(self):
        return f"KEY HOLD: {self.key}"


class KeyReleaseCommand(MacroCommand):
    def __init__(self, key):
        super().__init__("key_release")
        self.key = key
    
    def execute(self, context):
        keyboard.release(self.key)
    
    def to_dict(self):
        return {"type": self.command_type, "key": self.key}
    
    def __str__(self):
        return f"KEY RELEASE: {self.key}"


class MouseClickCommand(MacroCommand):
    def __init__(self, button, x, y, mode="absolute", offset_x=0, offset_y=0):
        super().__init__("mouse_click")
        self.button = button
        self.x = x
        self.y = y
        self.mode = mode
        self.offset_x = offset_x
        self.offset_y = offset_y
    
    def execute(self, context):
        if self.mode == "in_place":
            # Click at current mouse position
            current_x, current_y = pyautogui.position()
            pyautogui.click(current_x, current_y, button=self.button)
        elif self.mode == "offset":
            # Click at current position + offset
            current_x, current_y = pyautogui.position()
            pyautogui.click(current_x + self.offset_x, current_y + self.offset_y, button=self.button)
        else:
            # Click at absolute position (default)
            pyautogui.click(self.x, self.y, button=self.button)
    
    def to_dict(self):
        return {
            "type": self.command_type, 
            "button": self.button, 
            "x": self.x, 
            "y": self.y,
            "mode": self.mode,
            "offset_x": self.offset_x,
            "offset_y": self.offset_y
        }
    
    def __str__(self):
        if self.mode == "in_place":
            return f"MOUSE CLICK: {self.button} (in place)"
        elif self.mode == "offset":
            return f"MOUSE CLICK: {self.button} (offset {self.offset_x:+d}, {self.offset_y:+d})"
        else:
            return f"MOUSE CLICK: {self.button} at ({self.x}, {self.y})"


class MouseHoldCommand(MacroCommand):
    def __init__(self, button, x, y):
        super().__init__("mouse_hold")
        self.button = button
        self.x = x
        self.y = y
    
    def execute(self, context):
        pyautogui.moveTo(self.x, self.y)
        pyautogui.mouseDown(button=self.button)
    
    def to_dict(self):
        return {"type": self.command_type, "button": self.button, "x": self.x, "y": self.y}
    
    def __str__(self):
        return f"MOUSE HOLD: {self.button} at ({self.x}, {self.y})"


class MouseReleaseCommand(MacroCommand):
    def __init__(self, button, x, y):
        super().__init__("mouse_hold")
        self.button = button
        self.x = x
        self.y = y
    
    def execute(self, context):
        pyautogui.moveTo(self.x, self.y)
        pyautogui.mouseUp(button=self.button)
    
    def to_dict(self):
        return {"type": self.command_type, "button": self.button, "x": self.x, "y": self.y}
    
    def __str__(self):
        return f"MOUSE RELEASE: {self.button} at ({self.x}, {self.y})"


class MouseMoveCommand(MacroCommand):
    def __init__(self, x, y):
        super().__init__("mouse_move")
        self.x = x
        self.y = y
    
    def execute(self, context):
        pyautogui.moveTo(self.x, self.y)
    
    def to_dict(self):
        return {"type": self.command_type, "x": self.x, "y": self.y}
    
    def __str__(self):
        return f"MOUSE MOVE: to ({self.x}, {self.y})"


class IfStatementCommand(MacroCommand):
    def __init__(self, condition):
        super().__init__("if_statement")
        self.condition = condition
    
    def execute(self, context):
        # Control flow handled by executor
        pass
    
    def to_dict(self):
        return {"type": self.command_type, "condition": self.condition}
    
    def __str__(self):
        return f"IF: {self.condition}"


class ElseStatementCommand(MacroCommand):
    def __init__(self):
        super().__init__("else_statement")
    
    def execute(self, context):
        # Control flow handled by executor
        pass
    
    def __str__(self):
        return "ELSE"


class EndIfStatementCommand(MacroCommand):
    def __init__(self):
        super().__init__("endif_statement")
    
    def execute(self, context):
        # Control flow handled by executor
        pass
    
    def __str__(self):
        return "ENDIF"


class IfImageCommand(MacroCommand):
    def __init__(self, image_path, confidence=0.8, move_to_image=False, image_paths=None):
        super().__init__("if_image")
        # Support both single image (legacy) and multiple images
        if image_paths is None:
            self.image_paths = [{"path": image_path, "confidence": confidence}]
        else:
            self.image_paths = image_paths
        self.image_path = image_path  # Keep for backwards compatibility
        self.confidence = confidence  # Keep for backwards compatibility
        self.move_to_image = move_to_image
    
    def execute(self, context):
        # Returns True/False for if statement
        return self.find_image()
    
    def find_image(self):
        # Try to find any of the images
        for img_info in self.image_paths:
            try:
                location = pyautogui.locateOnScreen(img_info["path"], confidence=img_info["confidence"])
                if location:
                    if self.move_to_image:
                        x, y = pyautogui.center(location)
                        pyautogui.moveTo(x, y)
                    return True
            except Exception as e:
                print(f"Error finding image {img_info['path']}: {e}")
        return False
    
    def to_dict(self):
        return {
            "type": self.command_type, 
            "image_path": self.image_path, 
            "confidence": self.confidence, 
            "move_to_image": self.move_to_image,
            "image_paths": self.image_paths
        }
    
    def get_image_name(self):
        """Get just the filename without path"""
        if len(self.image_paths) == 1:
            return Path(self.image_paths[0]["path"]).stem
        else:
            return f"{Path(self.image_paths[0]['path']).stem} +{len(self.image_paths)-1} more"
    
    def __str__(self):
        move_indicator = " [MOVE]" if self.move_to_image else ""
        return f"IF IMAGE{move_indicator}: {self.get_image_name()}"


class FindImageCommand(MacroCommand):
    def __init__(self, image_path, confidence=0.8):
        super().__init__("find_image")
        self.image_path = image_path
        self.confidence = confidence
    
    def execute(self, context):
        try:
            location = pyautogui.locateOnScreen(self.image_path, confidence=self.confidence)
            if location:
                x, y = pyautogui.center(location)
                pyautogui.moveTo(x, y)
                return True
            return False
        except Exception as e:
            print(f"Error finding image: {e}")
            return False
    
    def to_dict(self):
        return {"type": self.command_type, "image_path": self.image_path, "confidence": self.confidence}
    
    def get_image_name(self):
        """Get just the filename without path"""
        return Path(self.image_path).stem
    
    def __str__(self):
        return f"FIND IMAGE: {self.get_image_name()} (conf: {self.confidence})"


class RepeatCommand(MacroCommand):
    def __init__(self, count):
        super().__init__("repeat")
        self.count = count
    
    def execute(self, context):
        # Control flow handled by executor
        pass
    
    def to_dict(self):
        return {"type": self.command_type, "count": self.count}
    
    def __str__(self):
        return f"REPEAT: {self.count} times"


class EndRepeatCommand(MacroCommand):
    def __init__(self):
        super().__init__("end_repeat")
    
    def execute(self, context):
        # Control flow handled by executor
        pass
    
    def __str__(self):
        return "END REPEAT"


class DelayCommand(MacroCommand):
    def __init__(self, seconds):
        super().__init__("delay")
        self.seconds = seconds
    
    def execute(self, context):
        time.sleep(self.seconds)
    
    def to_dict(self):
        return {"type": self.command_type, "seconds": self.seconds}
    
    def __str__(self):
        return f"DELAY: {self.seconds}s"


class DelayMsCommand(MacroCommand):
    def __init__(self, milliseconds):
        super().__init__("delay_ms")
        self.milliseconds = milliseconds
    
    def execute(self, context):
        time.sleep(self.milliseconds / 1000.0)
    
    def to_dict(self):
        return {"type": self.command_type, "milliseconds": self.milliseconds}
    
    def __str__(self):
        return f"DELAY: {self.milliseconds}ms"


class WaitForWindowCommand(MacroCommand):
    def __init__(self, window_pattern, timeout=30):
        super().__init__("wait_for_window")
        self.window_pattern = window_pattern
        self.timeout = timeout
    
    def execute(self, context):
        import re
        # Convert wildcard pattern to regex
        pattern = self.window_pattern.replace('*', '.*')
        regex = re.compile(pattern, re.IGNORECASE)
        
        user32 = ctypes.windll.user32
        EnumWindows = user32.EnumWindows
        EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int))
        GetWindowText = user32.GetWindowTextW
        GetWindowTextLength = user32.GetWindowTextLengthW
        IsWindowVisible = user32.IsWindowVisible
        
        found = False
        start_time = time.time()
        
        def check_window(hwnd, lParam):
            nonlocal found
            if IsWindowVisible(hwnd):
                length = GetWindowTextLength(hwnd)
                buff = ctypes.create_unicode_buffer(length + 1)
                GetWindowText(hwnd, buff, length + 1)
                if buff.value and regex.search(buff.value):
                    found = True
                    return False  # Stop enumeration
            return True  # Continue enumeration
        
        # Poll for window until found or timeout
        while not found and (time.time() - start_time) < self.timeout:
            EnumWindows(EnumWindowsProc(check_window), 0)
            if not found:
                time.sleep(0.1)  # Check every 100ms
        
        if not found:
            raise TimeoutError(f"Window matching '{self.window_pattern}' not found within {self.timeout} seconds")
    
    def to_dict(self):
        return {"type": self.command_type, "window_pattern": self.window_pattern, "timeout": self.timeout}
    
    def __str__(self):
        return f"WAIT FOR WINDOW: {self.window_pattern} (timeout: {self.timeout}s)"


class MessageCommand(MacroCommand):
    def __init__(self, text, always_on_top=False, always_focused=False):
        super().__init__("message")
        self.text = text
        self.always_on_top = always_on_top
        self.always_focused = always_focused
    
    def execute(self, context):
        # Add small delay to ensure window appears centered
        time.sleep(0.05)
        
        # Show message in a non-blocking way
        root = tk.Tk()
        root.withdraw()
        
        # Small delay to let root window initialize
        time.sleep(0.05)
        
        if self.always_on_top or self.always_focused:
            # Create a custom toplevel for always on top / always focused
            msg_window = tk.Toplevel(root)
            msg_window.title("Macro Message")
            
            msg_window.geometry("400x150")
            
            # Center the message
            screen_width = msg_window.winfo_screenwidth()
            screen_height = msg_window.winfo_screenheight()
            x = (screen_width - 400) // 2
            y = (screen_height - 150) // 2
            msg_window.geometry(f"400x150+{x}+{y}")
            
            # Dismiss function
            def dismiss(event=None):
                try:
                    msg_window.destroy()
                    root.destroy()
                except:
                    pass
                return "break"  # Prevent default key behavior
            
            # Message text
            msg_label = tk.Label(msg_window, text=self.text, wraplength=350, justify=tk.CENTER)
            msg_label.pack(pady=20)
            
            # OK button
            ok_button = tk.Button(msg_window, text="OK", command=dismiss, width=10)
            ok_button.pack(pady=10)
            
            # Bind spacebar, enter, and escape keys to dismiss
            msg_window.bind("<space>", dismiss)
            msg_window.bind("<Return>", dismiss)
            msg_window.bind("<KeyPress-space>", dismiss)
            msg_window.bind("<KeyPress-Return>", dismiss)
            msg_window.bind('<Escape>', dismiss)
            
            # Force window to appear and get its handle
            msg_window.update_idletasks()
            msg_window.update()
            
            # Set focus on window itself
            msg_window.focus_set()
            
            # Always set topmost first
            msg_window.attributes('-topmost', True)
            msg_window.attributes('-topmost', False)
            msg_window.attributes('-topmost', True)
            
            # Use Windows API to force window to topmost position
            def force_to_front():
                try:
                    # Get window handle
                    hwnd = msg_window.winfo_id()
                    
                    # Define constants
                    HWND_TOPMOST = -1
                    HWND_NOTOPMOST = -2
                    SWP_NOMOVE = 0x0002
                    SWP_NOSIZE = 0x0001
                    SWP_SHOWWINDOW = 0x0040
                    SW_RESTORE = 9
                    
                    # Simulate Alt key press/release to allow SetForegroundWindow
                    VK_MENU = 0x12
                    KEYEVENTF_EXTENDEDKEY = 0x0001
                    KEYEVENTF_KEYUP = 0x0002
                    
                    ctypes.windll.user32.keybd_event(VK_MENU, 0, KEYEVENTF_EXTENDEDKEY, 0)
                    ctypes.windll.user32.keybd_event(VK_MENU, 0, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0)
                    
                    # Small sleep to let Alt key release register
                    time.sleep(0.01)
                    
                    # Show and restore window
                    ctypes.windll.user32.ShowWindow(hwnd, SW_RESTORE)
                    
                    # Set to topmost, then not topmost, then topmost again (flashing technique)
                    ctypes.windll.user32.SetWindowPos(
                        hwnd, HWND_TOPMOST,
                        0, 0, 0, 0,
                        SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW
                    )
                    
                    # Flash the window to get attention
                    ctypes.windll.user32.FlashWindow(hwnd, True)
                    
                    # Bring to foreground
                    ctypes.windll.user32.BringWindowToTop(hwnd)
                    ctypes.windll.user32.SetForegroundWindow(hwnd)
                    ctypes.windll.user32.SetActiveWindow(hwnd)
                    ctypes.windll.user32.SetFocus(hwnd)
                    
                    # Send an extra Escape key to clear any menu state
                    VK_ESCAPE = 0x1B
                    ctypes.windll.user32.keybd_event(VK_ESCAPE, 0, 0, 0)
                    ctypes.windll.user32.keybd_event(VK_ESCAPE, 0, KEYEVENTF_KEYUP, 0)
                    
                except Exception as e:
                    print(f"Error setting window topmost: {e}")
            
            # Call immediately
            force_to_front()
            msg_window.lift()
            msg_window.focus_force()
            
            # Call again after a short delay to ensure it sticks and focus is ready
            msg_window.after(100, lambda: [force_to_front(), msg_window.focus_force()])
            
            # Handle always focused - keep bringing window to front
            if self.always_focused:
                # Make window modal and grab all input
                msg_window.grab_set()
                msg_window.focus_set()
                
                def keep_focus():
                    try:
                        if msg_window.winfo_exists():
                            hwnd = msg_window.winfo_id()
                            ctypes.windll.user32.SetForegroundWindow(hwnd)
                            msg_window.focus_force()
                            msg_window.lift()
                            msg_window.after(50, keep_focus)
                    except:
                        pass
                keep_focus()
                
                # Bind focus events to force focus back
                def on_focus_out(event):
                    try:
                        if msg_window.winfo_exists():
                            hwnd = msg_window.winfo_id()
                            ctypes.windll.user32.SetForegroundWindow(hwnd)
                            msg_window.focus_force()
                            msg_window.lift()
                    except:
                        pass
                
                msg_window.bind("<FocusOut>", on_focus_out)
            else:
                # Even without always_focused, ensure window appears on top initially
                msg_window.after(10, lambda: msg_window.focus_force())
            
            msg_window.wait_window()
        else:
            # Create custom window for simple messages too to ensure centering
            msg_window = tk.Toplevel(root)
            msg_window.title("Macro Message")
            
            # Message text
            msg_label = tk.Label(msg_window, text=self.text, wraplength=350, justify=tk.CENTER, padx=20, pady=20)
            msg_label.pack(pady=20)
            
            # OK button
            def dismiss():
                msg_window.destroy()
                root.destroy()
            
            ok_button = tk.Button(msg_window, text="OK", command=dismiss, width=10)
            ok_button.pack(pady=10)
            
            # Bind Enter, Space, and Escape to dismiss
            msg_window.bind("<Return>", lambda e: dismiss())
            msg_window.bind("<space>", lambda e: dismiss())
            msg_window.bind('<Escape>', lambda e: dismiss())
            
            # Update to get actual size
            msg_window.update_idletasks()
            
            # Center the window
            width = msg_window.winfo_width()
            height = msg_window.winfo_height()
            screen_width = msg_window.winfo_screenwidth()
            screen_height = msg_window.winfo_screenheight()
            x = (screen_width - width) // 2
            y = (screen_height - height) // 2
            msg_window.geometry(f"+{x}+{y}")
            
            # Focus and bring to top
            msg_window.attributes('-topmost', True)
            msg_window.focus_force()
            msg_window.lift()
            
            # Re-center and focus after a short delay
            def recenter_and_focus():
                w = msg_window.winfo_width()
                h = msg_window.winfo_height()
                sw = msg_window.winfo_screenwidth()
                sh = msg_window.winfo_screenheight()
                new_x = (sw - w) // 2
                new_y = (sh - h) // 2
                msg_window.geometry(f"+{new_x}+{new_y}")
                msg_window.focus_force()
                msg_window.lift()
            
            msg_window.after(50, recenter_and_focus)
            msg_window.after(100, recenter_and_focus)
            
            msg_window.wait_window()
    
    def to_dict(self):
        return {"type": self.command_type, "text": self.text, "always_on_top": self.always_on_top, "always_focused": self.always_focused}
    
    def __str__(self):
        indicators = []
        if self.always_on_top:
            indicators.append("TOP")
        if self.always_focused:
            indicators.append("FOCUS")
        indicator_str = f" [{', '.join(indicators)}]" if indicators else ""
        return f"MESSAGE{indicator_str}: {self.text[:50]}{'...' if len(self.text) > 50 else ''}"


class ClipboardClearCommand(MacroCommand):
    def __init__(self):
        super().__init__("clipboard_clear")
    
    def execute(self, context):
        import pyperclip
        pyperclip.copy('')  # Clear clipboard
    
    def to_dict(self):
        return {"type": self.command_type}
    
    def __str__(self):
        return "CLIPBOARD: Clear"


class ClipboardSetCommand(MacroCommand):
    def __init__(self, value):
        super().__init__("clipboard_set")
        self.value = value
    
    def execute(self, context):
        import pyperclip
        pyperclip.copy(self.value)
    
    def to_dict(self):
        return {"type": self.command_type, "value": self.value}
    
    def __str__(self):
        return f"CLIPBOARD: Set to '{self.value[:30]}{'...' if len(self.value) > 30 else ''}'"


class ClipboardIncrementCommand(MacroCommand):
    def __init__(self):
        super().__init__("clipboard_increment")
    
    def execute(self, context):
        import pyperclip
        try:
            current = pyperclip.paste()
            # Try to convert to int and increment
            num = int(current)
            pyperclip.copy(str(num + 1))
        except:
            pass  # If not a number, do nothing
    
    def to_dict(self):
        return {"type": self.command_type}
    
    def __str__(self):
        return "CLIPBOARD: Increment"


class ClipboardCopyCommand(MacroCommand):
    def __init__(self):
        super().__init__("clipboard_copy")
    
    def execute(self, context):
        pyautogui.hotkey('ctrl', 'c')
        time.sleep(0.1)  # Small delay for copy to complete
    
    def to_dict(self):
        return {"type": self.command_type}
    
    def __str__(self):
        return "CLIPBOARD: Copy (Ctrl+C)"


class ClipboardPasteCommand(MacroCommand):
    def __init__(self):
        super().__init__("clipboard_paste")
    
    def execute(self, context):
        pyautogui.hotkey('ctrl', 'v')
        time.sleep(0.1)  # Small delay for paste to complete
    
    def to_dict(self):
        return {"type": self.command_type}
    
    def __str__(self):
        return "CLIPBOARD: Paste (Ctrl+V)"


class Macro:
    def __init__(self, name):
        self.name = name
        self.commands = []
        self.hotkey = None
        self.window_geometry = None  # Store window position/size as "widthxheight+x+y"
    
    def add_command(self, command):
        self.commands.append(command)
    
    def remove_command(self, index):
        if 0 <= index < len(self.commands):
            self.commands.pop(index)
    
    def to_dict(self):
        return {
            "name": self.name,
            "hotkey": self.hotkey,
            "window_geometry": self.window_geometry,
            "commands": [cmd.to_dict() for cmd in self.commands]
        }
    
    @staticmethod
    def from_dict(data):
        macro = Macro(data.get("name", "Unnamed"))
        macro.hotkey = data.get("hotkey")
        macro.window_geometry = data.get("window_geometry")
        for cmd_data in data.get("commands", []):
            cmd = MacroCommand.from_dict(cmd_data)
            if cmd:
                macro.add_command(cmd)
        return macro


class MacroExecutor:
    def __init__(self):
        self.running = False
        self.stop_flag = False
    
    def execute(self, macro, repeat_count=1):
        """Execute a macro with repeat support"""
        self.running = True
        self.stop_flag = False
        
        if repeat_count == 0:
            # Infinite loop
            while not self.stop_flag:
                self._run_commands(macro.commands)
        else:
            for _ in range(repeat_count):
                if self.stop_flag:
                    break
                self._run_commands(macro.commands)
        
        self.running = False
    
    def _run_commands(self, commands):
        """Run a list of commands with control flow support"""
        context = {}
        i = 0
        
        while i < len(commands) and not self.stop_flag:
            cmd = commands[i]
            
            if isinstance(cmd, IfStatementCommand):
                # Simple boolean condition evaluation
                condition_result = eval(cmd.condition) if cmd.condition else False
                if not condition_result:
                    # Skip to ELSE or ENDIF
                    i = self._skip_to_else_or_endif(commands, i)
                    continue
            
            elif isinstance(cmd, IfImageCommand):
                if not cmd.execute(context):
                    # Skip to ELSE or ENDIF
                    i = self._skip_to_else_or_endif(commands, i)
                    continue
            
            elif isinstance(cmd, ElseStatementCommand):
                # If we reach ELSE, we executed the IF block, skip to ENDIF
                i = self._skip_to_endif(commands, i)
                continue
            
            elif isinstance(cmd, RepeatCommand):
                # Find matching END REPEAT
                end_repeat_idx = self._find_end_repeat(commands, i)
                if end_repeat_idx != -1:
                    repeat_commands = commands[i+1:end_repeat_idx]
                    for _ in range(cmd.count):
                        if self.stop_flag:
                            break
                        self._run_commands(repeat_commands)
                    i = end_repeat_idx
            
            elif not isinstance(cmd, (EndIfStatementCommand, EndRepeatCommand)):
                cmd.execute(context)
            
            i += 1
    
    def _skip_to_else_or_endif(self, commands, start_idx):
        """Skip to ELSE or ENDIF statement"""
        depth = 0
        for i in range(start_idx + 1, len(commands)):
            cmd = commands[i]
            if isinstance(cmd, (IfStatementCommand, IfImageCommand)):
                depth += 1
            elif isinstance(cmd, ElseStatementCommand) and depth == 0:
                return i
            elif isinstance(cmd, EndIfStatementCommand):
                if depth == 0:
                    return i
                depth -= 1
        return len(commands)
    
    def _skip_to_endif(self, commands, start_idx):
        """Skip to ENDIF statement"""
        depth = 0
        for i in range(start_idx + 1, len(commands)):
            cmd = commands[i]
            if isinstance(cmd, (IfStatementCommand, IfImageCommand)):
                depth += 1
            elif isinstance(cmd, EndIfStatementCommand):
                if depth == 0:
                    return i
                depth -= 1
        return len(commands)
    
    def _find_end_repeat(self, commands, start_idx):
        """Find matching END REPEAT"""
        depth = 0
        for i in range(start_idx + 1, len(commands)):
            cmd = commands[i]
            if isinstance(cmd, RepeatCommand):
                depth += 1
            elif isinstance(cmd, EndRepeatCommand):
                if depth == 0:
                    return i
                depth -= 1
        return -1
    
    def stop(self):
        """Stop the currently running macro"""
        self.stop_flag = True


class MacroMakerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Macro Maker")
        self.root.geometry("1000x600")  # Increased width for better button spacing
        
        self.macros = {}
        self.current_macro = None
        self.executor = MacroExecutor()
        self.hotkey_handlers = {}
        self.data_file = Path("macros.json")
        self.thumbnail_cache = {}  # Cache for image thumbnails
        self.selected_index = None  # Track selected command
        self.click_timer = None  # Timer for double-click detection
        self.copied_command = None  # Store copied command
        
        # Undo/Redo stacks
        self.undo_stack = []
        self.redo_stack = []
        
        self.setup_ui()
        self.load_macros()
        self.start_hotkey_listener()
    
    def setup_ui(self):
        # Top frame for macro selection
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.pack(fill=tk.X)
        
        ttk.Label(top_frame, text="Macro:").pack(side=tk.LEFT, padx=5)
        
        self.macro_var = tk.StringVar()
        self.macro_dropdown = ttk.Combobox(top_frame, textvariable=self.macro_var, width=30)
        self.macro_dropdown.pack(side=tk.LEFT, padx=5)
        self.macro_dropdown.bind("<<ComboboxSelected>>", self.on_macro_selected)
        
        ttk.Label(top_frame, text="Repeat:").pack(side=tk.LEFT, padx=5)
        self.repeat_var = tk.StringVar(value="1")
        self.repeat_entry = ttk.Entry(top_frame, textvariable=self.repeat_var, width=10)
        self.repeat_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(top_frame, text="New Macro", command=self.new_macro).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="Delete Macro", command=self.delete_macro).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="Set Hotkey", command=self.set_hotkey).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="Set Position", command=self.set_window_position).pack(side=tk.LEFT, padx=5)
        
        # Middle frame for command list
        middle_frame = ttk.Frame(self.root, padding="10")
        middle_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(middle_frame, text="Macro Steps:").pack(anchor=tk.W)
        
        # Canvas-based list with scrollbar for image thumbnails
        list_frame = ttk.Frame(middle_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.command_canvas = tk.Canvas(list_frame, yscrollcommand=scrollbar.set, bg="white", highlightthickness=1)
        self.command_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.command_canvas.yview)
        
        # Frame inside canvas to hold command items
        self.command_frame = ttk.Frame(self.command_canvas)
        self.canvas_frame_id = self.command_canvas.create_window((0, 0), window=self.command_frame, anchor="nw")
        
        # Enable drag and drop reordering
        self.drag_data = {"index": None, "y": 0, "widget": None, "started": False, "time": 0}
        self.drop_indicator = None  # Visual indicator for drop location
        
        # Bind scrolling
        self.command_frame.bind("<Configure>", lambda e: self.command_canvas.configure(scrollregion=self.command_canvas.bbox("all")))
        self.command_canvas.bind("<MouseWheel>", self._on_mousewheel)
        
        # Bind undo/redo shortcuts
        self.root.bind("<Control-z>", lambda e: self.undo())
        self.root.bind("<Control-Z>", lambda e: self.redo())  # Ctrl+Shift+Z
        
        # Bind F2 to focus command list
        self.root.bind("<F2>", lambda e: self.command_canvas.focus_set())
        
        # Bind arrow keys for navigation when canvas is focused
        self.command_canvas.bind("<Up>", self.navigate_up)
        self.command_canvas.bind("<Down>", self.navigate_down)
        self.command_canvas.bind("<Home>", self.navigate_home)
        self.command_canvas.bind("<End>", self.navigate_end)
        self.command_canvas.bind("<Prior>", self.navigate_page_up)  # Page Up
        self.command_canvas.bind("<Next>", self.navigate_page_down)  # Page Down
        self.command_canvas.bind("<Shift-F10>", self.show_context_menu_keyboard)
        self.command_canvas.bind("<Return>", self.open_selected_command)
        self.command_canvas.bind("<Control-c>", self.copy_command)
        self.command_canvas.bind("<Control-x>", self.cut_command)
        self.command_canvas.bind("<Control-v>", self.paste_command)
        self.command_canvas.bind("<Delete>", lambda e: self.remove_command())
        
        # Also bind to root window for global shortcuts
        self.root.bind("<Control-c>", self.copy_command)
        self.root.bind("<Control-x>", self.cut_command)
        self.root.bind("<Control-v>", self.paste_command)
        
        # Command buttons
        btn_frame = ttk.Frame(middle_frame)
        btn_frame.pack(fill=tk.X)
        
        ttk.Button(btn_frame, text="Add Key Press", command=self.add_key_press).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Add Key Hold", command=self.add_key_hold).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Add Key Release", command=self.add_key_release).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Add Mouse Click", command=self.add_mouse_click).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Add Mouse Click (Absolute)", command=self.add_mouse_click_absolute).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Add Mouse Move", command=self.add_mouse_move).pack(side=tk.LEFT, padx=2)
        
        btn_frame2 = ttk.Frame(middle_frame)
        btn_frame2.pack(fill=tk.X, pady=2)
        
        ttk.Button(btn_frame2, text="Add IF", command=self.add_if).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame2, text="Add ELSE", command=self.add_else).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame2, text="Add ENDIF", command=self.add_endif).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame2, text="Add IF IMAGE", command=self.add_if_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame2, text="Add FIND IMAGE", command=self.add_find_image).pack(side=tk.LEFT, padx=2)
        
        btn_frame3 = ttk.Frame(middle_frame)
        btn_frame3.pack(fill=tk.X, pady=2)
        
        ttk.Button(btn_frame3, text="Add REPEAT", command=self.add_repeat).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame3, text="Add END REPEAT", command=self.add_end_repeat).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame3, text="Add Delay", command=self.add_delay).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame3, text="Add Delay (ms)", command=self.add_delay_ms).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame3, text="Add Message", command=self.add_message).pack(side=tk.LEFT, padx=2)
        
        btn_frame4 = ttk.Frame(middle_frame)
        btn_frame4.pack(fill=tk.X, pady=2)
        
        ttk.Button(btn_frame4, text="Wait For Window", command=self.add_wait_for_window).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame4, text="Remove Selected", command=self.remove_command).pack(side=tk.LEFT, padx=2)
        
        # Bottom frame for execution
        bottom_frame = ttk.Frame(self.root, padding="10")
        bottom_frame.pack(fill=tk.X)
        
        ttk.Button(bottom_frame, text="Run Macro", command=self.run_macro).pack(side=tk.LEFT, padx=5)
        ttk.Button(bottom_frame, text="Stop Macro", command=self.stop_macro).pack(side=tk.LEFT, padx=5)
        
        self.status_label = ttk.Label(bottom_frame, text="Ready")
        self.status_label.pack(side=tk.LEFT, padx=20)
    
    def update_command_list(self):
        """Update the command list with thumbnails for image commands"""
        # Clear existing widgets
        for widget in self.command_frame.winfo_children():
            widget.destroy()
        
        if not self.current_macro:
            return
        
        # Calculate nesting level for each command
        nesting_levels = []
        current_level = 0
        in_else = False
        
        for cmd in self.current_macro.commands:
            if isinstance(cmd, (IfImageCommand, RepeatCommand)):
                nesting_levels.append(current_level)
                current_level += 1
                in_else = False
            elif isinstance(cmd, ElseStatementCommand):
                # ELSE is at same level as IF
                nesting_levels.append(current_level - 1)
                in_else = True
            elif isinstance(cmd, (EndIfStatementCommand, EndRepeatCommand)):
                current_level = max(0, current_level - 1)
                nesting_levels.append(current_level)
                in_else = False
            else:
                nesting_levels.append(current_level)
        
        # Create a frame for each command
        for idx, cmd in enumerate(self.current_macro.commands):
            # Determine background color based on selection
            bg_color = "lightblue" if idx == self.selected_index else "white"
            
            # Calculate indentation based on nesting level
            indent = nesting_levels[idx] * 20  # 20 pixels per nesting level
            
            cmd_frame = tk.Frame(self.command_frame, relief=tk.RAISED, borderwidth=1, bg=bg_color)
            cmd_frame.pack(fill=tk.X, padx=(2 + indent, 2), pady=2)
            cmd_frame._command_index = idx
            
            # Check if this is an image command
            if isinstance(cmd, (IfImageCommand, FindImageCommand)):
                # Create horizontal layout: thumbnail | text
                thumb_label = self._create_thumbnail(cmd_frame, cmd.image_path)
                if thumb_label:
                    thumb_label.pack(side=tk.LEFT, padx=5, pady=2)
                    thumb_label.config(bg=bg_color)
                
                text_label = tk.Label(cmd_frame, text=str(cmd), anchor=tk.W, bg=bg_color)
                text_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
            else:
                # Regular command - just text
                text_label = tk.Label(cmd_frame, text=str(cmd), anchor=tk.W, bg=bg_color)
                text_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Bind events for selection and dragging (double-click must be bound first)
            cmd_frame.bind("<Double-1>", self.on_edit_command)
            cmd_frame.bind("<Button-1>", self.on_command_click)
            cmd_frame.bind("<B1-Motion>", self.on_drag_motion)
            cmd_frame.bind("<ButtonRelease-1>", self.on_drag_release)
            cmd_frame.bind("<Button-3>", self.show_context_menu)  # Right-click
            
            # Also bind to child widgets
            for child in cmd_frame.winfo_children():
                child.bind("<Double-1>", self.on_edit_command)
                child.bind("<Button-1>", self.on_command_click)
                child.bind("<B1-Motion>", self.on_drag_motion)
                child.bind("<ButtonRelease-1>", self.on_drag_release)
                child.bind("<Button-3>", self.show_context_menu)  # Right-click
        
        # Update scroll region
        self.command_frame.update_idletasks()
        self.command_canvas.configure(scrollregion=self.command_canvas.bbox("all"))
    
    def _on_mousewheel(self, event):
        """Handle mousewheel scrolling"""
        self.command_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    def save_state(self):
        """Save current state to undo stack"""
        if not self.current_macro:
            return
        
        # Deep copy the current commands list
        state = {
            'macro_name': self.macro_var.get(),
            'commands': [cmd.to_dict() for cmd in self.current_macro.commands],
            'selected_index': self.selected_index
        }
        self.undo_stack.append(state)
        
        # Limit undo stack to 50 items
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)
        
        # Clear redo stack when new action is performed
        self.redo_stack.clear()
    
    def undo(self):
        """Undo last action"""
        if not self.undo_stack or not self.current_macro:
            return
        
        # Save current state to redo stack
        current_state = {
            'macro_name': self.macro_var.get(),
            'commands': [cmd.to_dict() for cmd in self.current_macro.commands],
            'selected_index': self.selected_index
        }
        self.redo_stack.append(current_state)
        
        # Restore previous state
        state = self.undo_stack.pop()
        self.restore_state(state)
    
    def redo(self):
        """Redo last undone action"""
        if not self.redo_stack or not self.current_macro:
            return
        
        # Save current state to undo stack
        current_state = {
            'macro_name': self.macro_var.get(),
            'commands': [cmd.to_dict() for cmd in self.current_macro.commands],
            'selected_index': self.selected_index
        }
        self.undo_stack.append(current_state)
        
        # Restore redo state
        state = self.redo_stack.pop()
        self.restore_state(state)
    
    def restore_state(self, state):
        """Restore a saved state"""
        # Reconstruct commands from dictionaries
        self.current_macro.commands = []
        for cmd_dict in state['commands']:
            cmd = self.command_from_dict(cmd_dict)
            if cmd:
                self.current_macro.commands.append(cmd)
        
        self.selected_index = state['selected_index']
        self.update_command_list()
        self.save_macros()
    
    def command_from_dict(self, data):
        """Recreate command object from dictionary"""
        cmd_type = data.get('type')
        
        if cmd_type == 'key_press':
            return KeyPressCommand(data['key'])
        elif cmd_type == 'key_hold':
            return KeyHoldCommand(data['key'])
        elif cmd_type == 'key_release':
            return KeyReleaseCommand(data['key'])
        elif cmd_type == 'mouse_click':
            return MouseClickCommand(
                data['button'], 
                data['x'], 
                data['y'],
                data.get('mode', 'absolute'),
                data.get('offset_x', 0),
                data.get('offset_y', 0)
            )
        elif cmd_type == 'mouse_move':
            return MouseMoveCommand(data['x'], data['y'])
        elif cmd_type == 'if_statement':
            return IfStatementCommand(data['condition'])
        elif cmd_type == 'else_statement':
            return ElseStatementCommand()
        elif cmd_type == 'endif_statement':
            return EndIfStatementCommand()
        elif cmd_type == 'if_image':
            return IfImageCommand(
                data['image_path'], 
                data.get('confidence', 0.8), 
                data.get('move_to_image', False),
                data.get('image_paths', None)
            )
        elif cmd_type == 'find_image':
            return FindImageCommand(data['image_path'], data.get('confidence', 0.8))
        elif cmd_type == 'message':
            return MessageCommand(data['text'], data.get('always_on_top', False), data.get('always_focused', False))
        elif cmd_type == 'repeat':
            return RepeatCommand(data['count'])
        elif cmd_type == 'end_repeat':
            return EndRepeatCommand()
        elif cmd_type == 'delay':
            return DelayCommand(data['seconds'])
        elif cmd_type == 'clipboard_clear':
            return ClipboardClearCommand()
        elif cmd_type == 'clipboard_set':
            return ClipboardSetCommand(data['value'])
        elif cmd_type == 'clipboard_increment':
            return ClipboardIncrementCommand()
        elif cmd_type == 'clipboard_copy':
            return ClipboardCopyCommand()
        elif cmd_type == 'clipboard_paste':
            return ClipboardPasteCommand()
        
        return None
    
    def on_command_click(self, event):
        """Handle command click - select or start drag"""
        if not self.current_macro:
            return
        
        # Find which command frame was clicked
        widget = event.widget
        while widget and not hasattr(widget, '_command_index'):
            widget = widget.master
            if widget == self.command_frame:
                return
        
        if widget and hasattr(widget, '_command_index'):
            cmd_index = widget._command_index
            
            # Check if this is a double-click (clicking same item within 300ms)
            if (self.click_timer is not None and 
                hasattr(self, '_last_click_index') and 
                self._last_click_index == cmd_index):
                # This is a double-click - cancel timer and open edit
                self.root.after_cancel(self.click_timer)
                self.click_timer = None
                self.on_edit_command_direct(cmd_index)
                return
            
            # Cancel any pending single-click action
            if self.click_timer is not None:
                self.root.after_cancel(self.click_timer)
            
            # Store click info and schedule single-click action
            self._last_click_index = cmd_index
            self.click_timer = self.root.after(300, lambda: self.handle_single_click(cmd_index, event.y_root, widget))
    
    def handle_single_click(self, cmd_index, y_root, widget):
        """Handle confirmed single click after delay"""
        self.click_timer = None
        
        # Update selection
        self.selected_index = cmd_index
        self.update_command_list()
        
        # Initialize drag data
        self.drag_data["index"] = cmd_index
        self.drag_data["y"] = y_root
        self.drag_data["widget"] = widget
        self.drag_data["started"] = False
    
    def on_edit_command_direct(self, index):
        """Open edit dialog directly by index"""
        if not self.current_macro or index < 0 or index >= len(self.current_macro.commands):
            return
        
        # Clear drag data to prevent interference
        self.drag_data["started"] = False
        self.drag_data["index"] = None
        
        cmd = self.current_macro.commands[index]
        
        # Handle different command types
        if isinstance(cmd, (KeyPressCommand, KeyHoldCommand, KeyReleaseCommand)):
            new_key = self.show_key_selector("Edit Key")
            if new_key:
                self.save_state()
                cmd.key = new_key
                self.update_command_list()
                self.save_macros()
        
        elif isinstance(cmd, DelayCommand):
            new_delay = simpledialog.askfloat("Edit Delay", "Enter delay in seconds:", initialvalue=cmd.seconds)
            if new_delay is not None:
                self.save_state()
                cmd.seconds = new_delay
                self.update_command_list()
                self.save_macros()
        
        elif isinstance(cmd, DelayMsCommand):
            new_delay = simpledialog.askinteger("Edit Delay (ms)", "Enter delay in milliseconds:", initialvalue=cmd.milliseconds)
            if new_delay is not None:
                self.save_state()
                cmd.milliseconds = new_delay
                self.update_command_list()
                self.save_macros()
        
        elif isinstance(cmd, WaitForWindowCommand):
            new_pattern = simpledialog.askstring("Edit Wait For Window", "Enter window title pattern (use * as wildcard):", initialvalue=cmd.window_pattern)
            if new_pattern:
                new_timeout = simpledialog.askinteger("Edit Timeout", "Enter timeout in seconds:", initialvalue=cmd.timeout)
                if new_timeout is not None:
                    self.save_state()
                    cmd.window_pattern = new_pattern
                    cmd.timeout = new_timeout
                    self.update_command_list()
                    self.save_macros()
        
        elif isinstance(cmd, MessageCommand):
            self.edit_message_command(cmd)
        
        elif isinstance(cmd, RepeatCommand):
            new_count = simpledialog.askinteger("Edit Repeat", "Enter repeat count:", initialvalue=cmd.count)
            if new_count is not None:
                self.save_state()
                cmd.count = new_count
                self.update_command_list()
                self.save_macros()
        
        elif isinstance(cmd, (IfImageCommand, FindImageCommand)):
            self.edit_image_command(cmd, index)
    
    def navigate_up(self, event):
        """Navigate to previous command in list"""
        if not self.current_macro or not self.current_macro.commands:
            return
        
        if self.selected_index is None:
            # Select last item
            self.selected_index = len(self.current_macro.commands) - 1
        elif self.selected_index > 0:
            self.selected_index -= 1
        
        self.update_command_list()
        self.scroll_to_selected()
        return "break"  # Prevent default scrolling
    
    def navigate_down(self, event):
        """Navigate to next command in list"""
        if not self.current_macro or not self.current_macro.commands:
            return
        
        if self.selected_index is None:
            # Select first item
            self.selected_index = 0
        elif self.selected_index < len(self.current_macro.commands) - 1:
            self.selected_index += 1
        
        self.update_command_list()
        self.scroll_to_selected()
        return "break"  # Prevent default scrolling
    
    def navigate_home(self, event):
        """Navigate to first command in list"""
        if not self.current_macro or not self.current_macro.commands:
            return
        
        self.selected_index = 0
        self.update_command_list()
        self.scroll_to_selected()
        return "break"
    
    def navigate_end(self, event):
        """Navigate to last command in list"""
        if not self.current_macro or not self.current_macro.commands:
            return
        
        self.selected_index = len(self.current_macro.commands) - 1
        self.update_command_list()
        self.scroll_to_selected()
        return "break"
    
    def navigate_page_up(self, event):
        """Navigate up by ~10 commands"""
        if not self.current_macro or not self.current_macro.commands:
            return
        
        if self.selected_index is None:
            self.selected_index = len(self.current_macro.commands) - 1
        else:
            self.selected_index = max(0, self.selected_index - 10)
        
        self.update_command_list()
        self.scroll_to_selected()
        return "break"
    
    def navigate_page_down(self, event):
        """Navigate down by ~10 commands"""
        if not self.current_macro or not self.current_macro.commands:
            return
        
        if self.selected_index is None:
            self.selected_index = 0
        else:
            self.selected_index = min(len(self.current_macro.commands) - 1, self.selected_index + 10)
        
        self.update_command_list()
        self.scroll_to_selected()
        return "break"
    
    def scroll_to_selected(self):
        """Scroll canvas to show selected command"""
        if self.selected_index is None:
            return
        
        # Find the selected frame
        for child in self.command_frame.winfo_children():
            if hasattr(child, '_command_index') and child._command_index == self.selected_index:
                # Get frame position relative to command_frame
                frame_y = child.winfo_y()
                frame_height = child.winfo_height()
                
                # Get canvas viewport
                canvas_height = self.command_canvas.winfo_height()
                
                # Calculate scroll position to center the frame
                scroll_region = self.command_canvas.bbox("all")
                if scroll_region:
                    total_height = scroll_region[3] - scroll_region[1]
                    if total_height > canvas_height:
                        # Calculate fraction to scroll
                        scroll_pos = (frame_y + frame_height / 2 - canvas_height / 2) / total_height
                        scroll_pos = max(0, min(1, scroll_pos))  # Clamp between 0 and 1
                        self.command_canvas.yview_moveto(scroll_pos)
                break
    
    def open_selected_command(self, event):
        """Open edit dialog for selected command when Enter is pressed"""
        if self.selected_index is not None:
            self.on_edit_command_direct(self.selected_index)
        return "break"
    
    def show_context_menu_keyboard(self, event):
        """Show context menu via keyboard (Shift+F10)"""
        if not self.current_macro or self.selected_index is None:
            return
        
        # Find the selected frame to get its position
        for child in self.command_frame.winfo_children():
            if hasattr(child, '_command_index') and child._command_index == self.selected_index:
                # Calculate screen position for menu
                frame_x = child.winfo_rootx()
                frame_y = child.winfo_rooty() + child.winfo_height() // 2
                
                # Create a mock event with the position
                class MockEvent:
                    def __init__(self, x, y):
                        self.x_root = x
                        self.y_root = y
                        self.widget = child
                
                self.show_context_menu(MockEvent(frame_x, frame_y))
                break
        
        return "break"
    
    def show_context_menu(self, event):
        """Show right-click context menu"""
        if not self.current_macro:
            return
        
        # Find which command frame was clicked
        widget = event.widget
        while widget and not hasattr(widget, '_command_index'):
            widget = widget.master
            if widget == self.command_frame:
                return
        
        if widget and hasattr(widget, '_command_index'):
            self.selected_index = widget._command_index
            self.update_command_list()
        
        # Create context menu
        menu = tk.Menu(self.root, tearoff=0)
        
        # Mouse commands submenu
        mouse_menu = tk.Menu(menu, tearoff=0)
        mouse_menu.add_command(label="Click", underline=0, command=lambda: self.insert_command_at_selection("mouse_click"))
        mouse_menu.add_command(label="Click (Absolute)", underline=7, command=lambda: self.insert_command_at_selection("mouse_click_absolute"))
        mouse_menu.add_command(label="Move", underline=0, command=lambda: self.insert_command_at_selection("mouse_move"))
        menu.add_cascade(label="Mouse Commands", underline=0, menu=mouse_menu)
        
        # Keyboard commands submenu
        keyboard_menu = tk.Menu(menu, tearoff=0)
        keyboard_menu.add_command(label="Press", underline=0, command=lambda: self.insert_command_at_selection("key_press"))
        keyboard_menu.add_command(label="Hold", underline=0, command=lambda: self.insert_command_at_selection("key_hold"))
        keyboard_menu.add_command(label="Release", underline=0, command=lambda: self.insert_command_at_selection("key_release"))
        menu.add_cascade(label="Keyboard Commands", underline=0, menu=keyboard_menu)
        
        # Image commands submenu
        image_menu = tk.Menu(menu, tearoff=0)
        image_menu.add_command(label="IF IMAGE", underline=0, command=lambda: self.insert_command_at_selection("if_image"))
        image_menu.add_command(label="FIND IMAGE", underline=0, command=lambda: self.insert_command_at_selection("find_image"))
        menu.add_cascade(label="Image Commands", underline=0, menu=image_menu)
        
        # Control flow commands submenu
        flow_menu = tk.Menu(menu, tearoff=0)
        flow_menu.add_command(label="IF Statement", underline=0, command=lambda: self.insert_command_at_selection("if_statement"))
        flow_menu.add_command(label="ELSE", underline=0, command=lambda: self.insert_command_at_selection("else"))
        flow_menu.add_command(label="END IF", underline=4, command=lambda: self.insert_command_at_selection("end_if"))
        flow_menu.add_command(label="REPEAT", underline=0, command=lambda: self.insert_command_at_selection("repeat"))
        flow_menu.add_command(label="END REPEAT", underline=4, command=lambda: self.insert_command_at_selection("end_repeat"))
        menu.add_cascade(label="Control Flow", underline=0, menu=flow_menu)
        
        # Clipboard commands submenu
        clipboard_menu = tk.Menu(menu, tearoff=0)
        clipboard_menu.add_command(label="Clear", underline=0, command=lambda: self.insert_command_at_selection("clipboard_clear"))
        clipboard_menu.add_command(label="Set", underline=0, command=lambda: self.insert_command_at_selection("clipboard_set"))
        clipboard_menu.add_command(label="Increment", underline=0, command=lambda: self.insert_command_at_selection("clipboard_increment"))
        clipboard_menu.add_command(label="cOpy", underline=1, command=lambda: self.insert_command_at_selection("clipboard_copy"))
        clipboard_menu.add_command(label="Paste", underline=0, command=lambda: self.insert_command_at_selection("clipboard_paste"))
        menu.add_cascade(label="Clipboard Commands", underline=0, menu=clipboard_menu)
        
        # Timing/Utility commands submenu
        timing_menu = tk.Menu(menu, tearoff=0)
        timing_menu.add_command(label="Delay (seconds)", underline=0, command=lambda: self.insert_command_at_selection("delay"))
        timing_menu.add_command(label="Delay (Milliseconds)", underline=7, command=lambda: self.insert_command_at_selection("delay_ms"))
        timing_menu.add_command(label="Wait for Window", underline=0, command=lambda: self.insert_command_at_selection("wait_for_window"))
        timing_menu.add_command(label="Message", underline=0, command=lambda: self.insert_command_at_selection("message"))
        menu.add_cascade(label="Timing/Utility", underline=0, menu=timing_menu)
        
        # Add separator and Copy/Paste/Delete options
        menu.add_separator()
        menu.add_command(label="Copy", underline=0, command=self.copy_command, accelerator="Ctrl+C")
        menu.add_command(label="Paste", underline=0, command=self.paste_command, accelerator="Ctrl+V")
        menu.add_separator()
        menu.add_command(label="Delete", underline=0, command=self.remove_command)
        
        # Show menu at cursor position
        menu.tk_popup(event.x_root, event.y_root)
    
    def insert_command_at_selection(self, command_type):
        """Insert a command below the currently selected command"""
        if not self.current_macro:
            messagebox.showwarning("Warning", "Please select a macro first")
            return
        
        # Calculate insert position
        insert_index = (self.selected_index + 1) if self.selected_index is not None else len(self.current_macro.commands)
        
        self.save_state()
        
        # Handle different command types
        if command_type == "mouse_click":
            button = simpledialog.askstring("Mouse Click", "Enter button (left/right/middle):", initialvalue="left")
            if button:
                mode_result = self.show_click_mode_dialog()
                if mode_result:
                    if mode_result == "in_place":
                        self.current_macro.commands.insert(insert_index, MouseClickCommand(button, 0, 0, mode="in_place"))
                    elif mode_result == "offset":
                        offset_x = simpledialog.askinteger("X Offset", "Enter X offset (pixels, can be negative):", initialvalue=0)
                        if offset_x is not None:
                            offset_y = simpledialog.askinteger("Y Offset", "Enter Y offset (pixels, can be negative):", initialvalue=0)
                            if offset_y is not None:
                                self.current_macro.commands.insert(insert_index, MouseClickCommand(button, 0, 0, mode="offset", offset_x=offset_x, offset_y=offset_y))
                            else:
                                return
                        else:
                            return
                else:
                    return
        
        elif command_type == "mouse_click_absolute":
            button = simpledialog.askstring("Mouse Click (Absolute)", "Enter button (left/right/middle):", initialvalue="left")
            if button:
                messagebox.showinfo(
                    "Capture Position", 
                    "Move your mouse to the desired click location and press F2 to capture the coordinates."
                )
                
                captured_pos = {"x": None, "y": None}
                
                def on_f2_press(event):
                    if event.name == 'f2':
                        captured_pos["x"], captured_pos["y"] = pyautogui.position()
                        keyboard.unhook_all()
                
                keyboard.on_press(on_f2_press)
                
                timeout = 30
                start_time = time.time()
                while captured_pos["x"] is None and (time.time() - start_time) < timeout:
                    time.sleep(0.1)
                
                keyboard.unhook_all()
                
                if captured_pos["x"] is not None:
                    self.current_macro.commands.insert(insert_index, MouseClickCommand(button, captured_pos["x"], captured_pos["y"], mode="absolute"))
                else:
                    messagebox.showwarning("Timeout", "Position capture timed out. No command was added.")
                    return
        
        elif command_type == "mouse_move":
            x, y = self.get_mouse_position()
            if x is not None:
                self.current_macro.commands.insert(insert_index, MouseMoveCommand(x, y))
        
        elif command_type == "key_press":
            key = self.show_key_selector("Key Press")
            if key:
                self.current_macro.commands.insert(insert_index, KeyPressCommand(key))
        
        elif command_type == "key_hold":
            key = self.show_key_selector("Key Hold")
            if key:
                self.current_macro.commands.insert(insert_index, KeyHoldCommand(key))
        
        elif command_type == "key_release":
            key = self.show_key_selector("Key Release")
            if key:
                self.current_macro.commands.insert(insert_index, KeyReleaseCommand(key))
        
        elif command_type == "if_image":
            self.add_if_image_at_index(insert_index)
            return  # Method handles update internally
        
        elif command_type == "find_image":
            self.add_find_image_at_index(insert_index)
            return  # Method handles update internally
        
        elif command_type == "clipboard_clear":
            self.current_macro.commands.insert(insert_index, ClipboardClearCommand())
        
        elif command_type == "clipboard_set":
            value = simpledialog.askstring("Set Clipboard", "Enter value to set clipboard to:")
            if value is not None:
                self.current_macro.commands.insert(insert_index, ClipboardSetCommand(value))
        
        elif command_type == "clipboard_increment":
            self.current_macro.commands.insert(insert_index, ClipboardIncrementCommand())
        
        elif command_type == "clipboard_copy":
            self.current_macro.commands.insert(insert_index, ClipboardCopyCommand())
        
        elif command_type == "clipboard_paste":
            self.current_macro.commands.insert(insert_index, ClipboardPasteCommand())
        
        elif command_type == "delay":
            seconds = simpledialog.askfloat("Delay", "Enter delay in seconds:")
            if seconds is not None:
                self.current_macro.commands.insert(insert_index, DelayCommand(seconds))
        
        elif command_type == "delay_ms":
            milliseconds = simpledialog.askinteger("Delay (ms)", "Enter delay in milliseconds:")
            if milliseconds is not None:
                self.current_macro.commands.insert(insert_index, DelayMsCommand(milliseconds))
        
        elif command_type == "wait_for_window":
            pattern = simpledialog.askstring("Wait For Window", "Enter window title pattern (use * as wildcard):")
            if pattern:
                timeout = simpledialog.askinteger("Timeout", "Enter timeout in seconds:", initialvalue=30)
                if timeout is not None:
                    self.current_macro.commands.insert(insert_index, WaitForWindowCommand(pattern, timeout))
        
        elif command_type == "message":
            text = simpledialog.askstring("Message", "Enter message text:")
            if text:
                always_on_top = messagebox.askyesno("Always On Top", "Should the message window be always on top?")
                always_focused = messagebox.askyesno("Always Focused", "Should the message window remain focused?")
                self.current_macro.commands.insert(insert_index, MessageCommand(text, always_on_top, always_focused))
        
        elif command_type == "if_statement":
            condition = simpledialog.askstring("IF Statement", "Enter condition (e.g., clipboard == 'value'):")
            if condition:
                self.current_macro.commands.insert(insert_index, IfStatementCommand(condition))
        
        elif command_type == "else":
            self.current_macro.commands.insert(insert_index, ElseStatementCommand())
        
        elif command_type == "end_if":
            self.current_macro.commands.insert(insert_index, EndIfStatementCommand())
        
        elif command_type == "repeat":
            count = simpledialog.askinteger("REPEAT", "Enter number of repetitions:")
            if count is not None:
                self.current_macro.commands.insert(insert_index, RepeatCommand(count))
        
        elif command_type == "end_repeat":
            self.current_macro.commands.insert(insert_index, EndRepeatCommand())
        
        # Select the newly inserted command
        self.selected_index = insert_index
        self.update_command_list()
        self.save_macros()
    
    def on_drag_start(self, event):
        """Start dragging a command"""
        if not self.current_macro:
            return
        
        # Find which command frame was clicked
        widget = event.widget
        # Traverse up to find the command frame
        while widget and not hasattr(widget, '_command_index'):
            widget = widget.master
            if widget == self.command_frame:
                return
        
        if widget and hasattr(widget, '_command_index'):
            self.drag_data["index"] = widget._command_index
            self.drag_data["y"] = event.y_root
            self.drag_data["widget"] = widget
    
    def on_drag_motion(self, event):
        """Handle dragging motion"""
        if self.drag_data["index"] is None or not self.drag_data.get("widget"):
            return
        
        # Calculate delta and move widget
        delta = event.y_root - self.drag_data["y"]
        
        # Only start drag if moved more than 10 pixels
        if not self.drag_data.get("started", False) and abs(delta) > 10:
            self.drag_data["started"] = True
        
        if abs(delta) > 5:  # Minimum movement threshold
            self.drag_data["y"] = event.y_root
        
        # Show drop indicator if dragging has started
        if self.drag_data.get("started", False):
            self.update_drop_indicator(event)
    
    def update_drop_indicator(self, event):
        """Update visual indicator showing where item will be dropped"""
        if not self.current_macro:
            return
        
        # Remove old indicator if exists
        if self.drop_indicator:
            self.drop_indicator.destroy()
            self.drop_indicator = None
        
        # Calculate position relative to canvas
        y_pos = event.y_root - self.command_canvas.winfo_rooty()
        canvas_y = self.command_canvas.canvasy(y_pos)
        
        # Find which command frame the mouse is over
        target_widget = None
        target_index = None
        for child in self.command_frame.winfo_children():
            if hasattr(child, '_command_index'):
                child_y = child.winfo_y()
                child_height = child.winfo_height()
                if canvas_y >= child_y and canvas_y <= child_y + child_height:
                    target_widget = child
                    target_index = child._command_index
                    break
        
        # Create drop indicator line
        if target_widget and target_index != self.drag_data["index"]:
            # Determine if we're in top or bottom half of the target
            target_y = target_widget.winfo_y()
            target_height = target_widget.winfo_height()
            middle_y = target_y + target_height / 2
            
            if canvas_y < middle_y:
                # Insert above
                indicator_y = target_y
            else:
                # Insert below
                indicator_y = target_y + target_height
            
            # Create a visible line indicator
            self.drop_indicator = tk.Frame(self.command_frame, bg="blue", height=3)
            self.drop_indicator.place(x=0, y=indicator_y, relwidth=1.0)
    
    def on_drag_release(self, event):
        """Handle end of drag"""
        # Remove drop indicator
        if self.drop_indicator:
            self.drop_indicator.destroy()
            self.drop_indicator = None
        
        if not self.current_macro or self.drag_data["index"] is None:
            self.drag_data["index"] = None
            self.drag_data["widget"] = None
            self.drag_data["started"] = False
            return
        
        # Only perform reorder if drag actually started
        if self.drag_data.get("started", False):
            drag_index = self.drag_data["index"]
            
            # Find target index based on Y position
            y_pos = event.y_root - self.command_canvas.winfo_rooty()
            canvas_y = self.command_canvas.canvasy(y_pos)
            
            # Find which command frame the mouse is over
            target_index = drag_index
            for child in self.command_frame.winfo_children():
                if hasattr(child, '_command_index'):
                    child_y = child.winfo_y()
                    child_height = child.winfo_height()
                    if canvas_y >= child_y and canvas_y <= child_y + child_height:
                        target_index = child._command_index
                        break
            
            # Swap commands
            if target_index != drag_index and 0 <= target_index < len(self.current_macro.commands):
                self.save_state()  # Save state before reordering
                self.current_macro.commands[drag_index], self.current_macro.commands[target_index] = \
                    self.current_macro.commands[target_index], self.current_macro.commands[drag_index]
                
                # Update selected_index to follow the moved command
                if self.selected_index == drag_index:
                    self.selected_index = target_index
                elif self.selected_index == target_index:
                    self.selected_index = drag_index
                
                # Update display
                self.update_command_list()
                self.save_macros()
        
        self.drag_data["index"] = None
        self.drag_data["widget"] = None
        self.drag_data["started"] = False
    
    def on_macro_selected(self, event=None):
        """Handle macro selection from dropdown"""
        macro_name = self.macro_var.get()
        if macro_name in self.macros:
            self.current_macro = self.macros[macro_name]
            self.update_command_list()
            if self.current_macro.hotkey:
                self.status_label.config(text=f"Hotkey: {self.current_macro.hotkey}")
            else:
                self.status_label.config(text="Ready")
            
            # Restore window position if saved
            if self.current_macro.window_geometry:
                try:
                    self.root.geometry(self.current_macro.window_geometry)
                except:
                    pass
    
    def new_macro(self):
        """Create a new macro"""
        name = simpledialog.askstring("New Macro", "Enter macro name:")
        if name and name not in self.macros:
            self.macros[name] = Macro(name)
            self.update_macro_list()
            self.macro_var.set(name)
            self.on_macro_selected()
            self.save_macros()
        elif name in self.macros:
            messagebox.showerror("Error", "Macro with this name already exists")
    
    def delete_macro(self):
        """Delete the current macro"""
        if self.current_macro:
            if messagebox.askyesno("Confirm", f"Delete macro '{self.current_macro.name}'?"):
                # Remove hotkey handler
                if self.current_macro.hotkey and self.current_macro.hotkey in self.hotkey_handlers:
                    try:
                        keyboard.remove_hotkey(self.hotkey_handlers[self.current_macro.hotkey])
                        del self.hotkey_handlers[self.current_macro.hotkey]
                    except:
                        pass
                
                del self.macros[self.current_macro.name]
                self.current_macro = None
                self.update_macro_list()
                self.update_command_list()
                self.save_macros()
    
    def update_macro_list(self):
        """Update the dropdown with available macros"""
        self.macro_dropdown['values'] = list(self.macros.keys())
    
    def ensure_dialog_focused(self, dialog):
        """Ensure dialog is focused and remains focused"""
        dialog.focus_force()
        dialog.lift()
        dialog.attributes('-topmost', True)
        # Reapply focus after short delays to ensure it sticks (before 100ms)
        dialog.after(20, lambda: dialog.focus_force())
        dialog.after(40, lambda: [dialog.focus_force(), dialog.lift()])
        dialog.after(60, lambda: dialog.focus_force())
    
    def set_hotkey(self):
        """Set hotkey for current macro"""
        if not self.current_macro:
            messagebox.showwarning("Warning", "Please select a macro first")
            return
        
        # Create dialog to capture hotkey
        dialog = tk.Toplevel(self.root)
        dialog.title("Set Hotkey")
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(dialog, text="Press the key combination you want to use:").pack(pady=20)
        
        hotkey_label = ttk.Label(dialog, text="Waiting...", font=("Arial", 12, "bold"))
        hotkey_label.pack(pady=10)
        
        captured_keys = []
        
        def on_key(event):
            key = event.name
            if key not in captured_keys:
                captured_keys.append(key)
                hotkey_label.config(text=" + ".join(captured_keys))
        
        keyboard.hook(on_key)
        
        # Center the dialog
        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        screen_width = dialog.winfo_screenwidth()
        screen_height = dialog.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        dialog.geometry(f"{width}x{height}+{x}+{y}")
        
        self.ensure_dialog_focused(dialog)
        
        def on_ok():
            keyboard.unhook_all()
            if captured_keys:
                hotkey = "+".join(captured_keys)
                
                # Remove old hotkey
                if self.current_macro.hotkey and self.current_macro.hotkey in self.hotkey_handlers:
                    try:
                        keyboard.remove_hotkey(self.hotkey_handlers[self.current_macro.hotkey])
                        del self.hotkey_handlers[self.current_macro.hotkey]
                    except:
                        pass
                
                # Set new hotkey
                self.current_macro.hotkey = hotkey
                self.register_hotkey(self.current_macro)
                self.save_macros()
                self.status_label.config(text=f"Hotkey set: {hotkey}")
            dialog.destroy()
        
        def on_cancel():
            keyboard.unhook_all()
            dialog.destroy()
        
        ok_btn = ttk.Button(dialog, text="OK", command=on_ok)
        ok_btn.pack(side=tk.LEFT, padx=50, pady=10)
        ttk.Button(dialog, text="Cancel", command=on_cancel).pack(side=tk.RIGHT, padx=50, pady=10)
        
        # Bind Escape to cancel
        dialog.bind('<Escape>', lambda e: on_cancel())
        
        # Re-center after a short delay
        def recenter():
            dialog.update_idletasks()
            w = dialog.winfo_width()
            h = dialog.winfo_height()
            sw = dialog.winfo_screenwidth()
            sh = dialog.winfo_screenheight()
            new_x = (sw - w) // 2
            new_y = (sh - h) // 2
            dialog.geometry(f"{w}x{h}+{new_x}+{new_y}")
            dialog.focus_force()
            dialog.lift()
            # Refocus OK button after recentering
            ok_btn.focus_set()
        
        dialog.after(10, recenter)
        dialog.after(30, recenter)
    
    def register_hotkey(self, macro):
        """Register a hotkey for a macro"""
        if macro.hotkey:
            try:
                handler = keyboard.add_hotkey(macro.hotkey, lambda m=macro: self.trigger_macro(m))
                self.hotkey_handlers[macro.hotkey] = handler
            except Exception as e:
                print(f"Error registering hotkey: {e}")
    
    def trigger_macro(self, macro):
        """Triggered when hotkey is pressed"""
        self.macro_var.set(macro.name)
        self.current_macro = macro
        self.update_command_list()
        if not self.executor.running:
            threading.Thread(target=self.run_macro, daemon=True).start()
    
    def show_key_selector(self, title):
        """Show a dropdown dialog to select a keyboard key"""
        # Comprehensive list of all keyboard keys
        all_keys = [
            # Letters
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
            'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
            # Numbers
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            # Numpad
            'num0', 'num1', 'num2', 'num3', 'num4', 'num5', 'num6', 'num7', 'num8', 'num9',
            'num lock', 'num /', 'num *', 'num -', 'num +', 'num enter', 'num .', 
            # Function keys
            'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12',
            'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20', 'f21', 'f22', 'f23', 'f24',
            # Special keys
            'enter', 'return', 'tab', 'space', 'backspace', 'delete', 'esc', 'escape',
            # Navigation
            'up', 'down', 'left', 'right', 'home', 'end', 'page up', 'page down',
            'insert', 'print screen', 'scroll lock', 'pause',
            # Modifiers
            'shift', 'left shift', 'right shift',
            'ctrl', 'left ctrl', 'right ctrl', 'control', 'left control', 'right control',
            'alt', 'left alt', 'right alt',
            'win', 'left win', 'right win', 'windows', 'left windows', 'right windows',
            'caps lock', 'menu', 'apps',
            # Symbols
            '-', '=', '[', ']', '\\', ';', "'", ',', '.', '/',
            '`', '~', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '+',
            '{', '}', '|', ':', '"', '<', '>', '?',
        ]
        
        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.transient(self.root)
        dialog.grab_set()
        
        result = {'key': None}
        
        tk.Label(dialog, text="Select or type to search for a key:", font=("Arial", 10)).pack(pady=10, padx=20)
        
        # Create combobox with all keys
        key_var = tk.StringVar()
        combo = ttk.Combobox(dialog, textvariable=key_var, values=all_keys, width=30, state='readonly')
        combo.pack(pady=10, padx=20)
        
        # Type-ahead search state
        search_state = {
            'last_key': '',
            'last_time': 0,
            'current_index': -1
        }
        
        def on_key_press(event):
            import time
            
            # Ignore special keys
            if len(event.char) != 1 or not event.char.isalnum() and event.char not in "'-. ":
                return
            
            char = event.char.lower()
            current_time = time.time()
            
            # If same key pressed within 0.8 seconds, move to next match
            if char == search_state['last_key'] and (current_time - search_state['last_time']) < 0.8:
                # Find next match after current index
                start_index = search_state['current_index'] + 1
            else:
                # New search, start from beginning
                start_index = 0
                search_state['current_index'] = -1
            
            # Search for matching key
            found = False
            for i in range(start_index, len(all_keys)):
                if all_keys[i].lower().startswith(char):
                    combo.current(i)
                    search_state['current_index'] = i
                    found = True
                    break
            
            # If no match found after current position, wrap to beginning
            if not found and start_index > 0:
                for i in range(0, start_index):
                    if all_keys[i].lower().startswith(char):
                        combo.current(i)
                        search_state['current_index'] = i
                        found = True
                        break
            
            search_state['last_key'] = char
            search_state['last_time'] = current_time
            
            return "break"  # Prevent default typing behavior
        
        def on_ok():
            if key_var.get():
                result['key'] = key_var.get()
            dialog.destroy()
        
        def on_cancel():
            dialog.destroy()
        
        # Bind key press for type-ahead search
        combo.bind('<KeyPress>', on_key_press)
        
        # Bind Enter to OK
        combo.bind('<Return>', lambda e: on_ok())
        combo.bind('<Escape>', lambda e: on_cancel())
        
        btn_frame = tk.Frame(dialog)
        btn_frame.pack(pady=10)
        ok_btn = ttk.Button(btn_frame, text="OK", command=on_ok)
        ok_btn.pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=on_cancel).pack(side=tk.LEFT, padx=5)
        
        # Center the dialog
        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        screen_width = dialog.winfo_screenwidth()
        screen_height = dialog.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        dialog.geometry(f"{width}x{height}+{x}+{y}")
        
        self.ensure_dialog_focused(dialog)
        
        # Re-center and focus
        def recenter():
            dialog.update_idletasks()
            w = dialog.winfo_width()
            h = dialog.winfo_height()
            sw = dialog.winfo_screenwidth()
            sh = dialog.winfo_screenheight()
            new_x = (sw - w) // 2
            new_y = (sh - h) // 2
            dialog.geometry(f"{w}x{h}+{new_x}+{new_y}")
            dialog.focus_force()
            dialog.lift()
        
        def focus_combo():
            combo.focus_set()
        
        dialog.after(10, recenter)
        dialog.after(30, recenter)
        dialog.after(80, focus_combo)
        
        dialog.wait_window()
        return result['key']
    
    def add_key_press(self):
        """Add key press command"""
        if not self.current_macro:
            messagebox.showwarning("Warning", "Please select a macro first")
            return
        
        key = self.show_key_selector("Key Press")
        if key:
            self.save_state()
            self.current_macro.add_command(KeyPressCommand(key))
            self.selected_index = len(self.current_macro.commands) - 1
            self.update_command_list()
            self.save_macros()
    
    def add_key_hold(self):
        """Add key hold command"""
        if not self.current_macro:
            messagebox.showwarning("Warning", "Please select a macro first")
            return
        
        key = self.show_key_selector("Key Hold")
        if key:
            self.save_state()
            self.current_macro.add_command(KeyHoldCommand(key))
            self.selected_index = len(self.current_macro.commands) - 1
            self.update_command_list()
            self.save_macros()
    
    def add_key_release(self):
        """Add key release command"""
        if not self.current_macro:
            messagebox.showwarning("Warning", "Please select a macro first")
            return
        
        key = self.show_key_selector("Key Release")
        if key:
            self.save_state()
            self.current_macro.add_command(KeyReleaseCommand(key))
            self.selected_index = len(self.current_macro.commands) - 1
            self.update_command_list()
            self.save_macros()
    
    def capture_screen_region(self):
        """Capture a screen region by clicking and dragging"""
        # Create fullscreen transparent window for region selection
        selection_window = tk.Toplevel(self.root)
        selection_window.attributes('-fullscreen', True)
        selection_window.attributes('-alpha', 0.3)
        selection_window.attributes('-topmost', True)
        selection_window.configure(bg='black')
        
        canvas = tk.Canvas(selection_window, cursor='cross', bg='black', highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)
        
        coords = {'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0, 'rect': None, 'done': False}
        
        def on_press(event):
            coords['x1'] = event.x
            coords['y1'] = event.y
            if coords['rect']:
                canvas.delete(coords['rect'])
            coords['rect'] = canvas.create_rectangle(coords['x1'], coords['y1'], coords['x1'], coords['y1'], outline='red', width=2)
        
        def on_drag(event):
            coords['x2'] = event.x
            coords['y2'] = event.y
            if coords['rect']:
                canvas.coords(coords['rect'], coords['x1'], coords['y1'], coords['x2'], coords['y2'])
        
        def on_release(event):
            coords['x2'] = event.x
            coords['y2'] = event.y
            coords['done'] = True
            selection_window.destroy()
        
        canvas.bind('<ButtonPress-1>', on_press)
        canvas.bind('<B1-Motion>', on_drag)
        canvas.bind('<ButtonRelease-1>', on_release)
        
        # Wait for selection
        selection_window.wait_window()
        
        if coords['done']:
            # Ensure coordinates are in correct order
            x1 = min(coords['x1'], coords['x2'])
            y1 = min(coords['y1'], coords['y2'])
            x2 = max(coords['x1'], coords['x2'])
            y2 = max(coords['y1'], coords['y2'])
            
            # Check if region is valid
            if x2 - x1 > 5 and y2 - y1 > 5:
                # Capture the selected region
                time.sleep(0.2)  # Small delay to ensure window is closed
                screenshot = ImageGrab.grab(bbox=(x1, y1, x2, y2))
                return screenshot
        
        return None
    
    def show_click_mode_dialog(self):
        """Show dialog to choose click mode: in place or offset"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Click Mode")
        dialog.transient(self.root)
        
        tk.Label(dialog, text="Click in place/offset?", font=("Arial", 10)).pack(pady=20)
        
        result = {"mode": None}
        
        def choose_in_place(event=None):
            result["mode"] = "in_place"
            dialog.destroy()
            return "break"
        
        def choose_offset(event=None):
            result["mode"] = "offset"
            dialog.destroy()
            return "break"
        
        btn_frame = tk.Frame(dialog)
        btn_frame.pack(pady=10)
        
        in_place_btn = tk.Button(btn_frame, text="In Place", command=choose_in_place, width=12)
        in_place_btn.pack(side=tk.LEFT, padx=5)
        
        offset_btn = tk.Button(btn_frame, text="Offset", command=choose_offset, width=12)
        offset_btn.pack(side=tk.LEFT, padx=5)
        
        # Bind keyboard shortcuts
        dialog.bind("<Return>", choose_in_place)  # Enter selects In Place (default)
        dialog.bind("<space>", choose_in_place)   # Space selects In Place (default)
        dialog.bind("1", choose_in_place)
        dialog.bind("2", choose_offset)
        dialog.bind('<Escape>', lambda e: dialog.destroy())
        
        # Now that all widgets are packed, center the dialog
        dialog.update_idletasks()
        
        # Get actual dialog size after widgets are laid out
        width = dialog.winfo_reqwidth()
        height = dialog.winfo_reqheight()
        
        # Get screen dimensions
        screen_width = dialog.winfo_screenwidth()
        screen_height = dialog.winfo_screenheight()
        
        # Calculate center position
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        
        # Set geometry with calculated position
        dialog.geometry(f"+{x}+{y}")
        
        # Grab focus and bring to front
        dialog.grab_set()
        self.ensure_dialog_focused(dialog)
        
        # Force reposition after window appears (Windows workaround)
        def reposition_and_focus():
            dialog.update_idletasks()
            # Recalculate in case window size changed
            w = dialog.winfo_width()
            h = dialog.winfo_height()
            sw = dialog.winfo_screenwidth()
            sh = dialog.winfo_screenheight()
            new_x = (sw - w) // 2
            new_y = (sh - h) // 2
            dialog.geometry(f"+{new_x}+{new_y}")
            dialog.focus_force()
            dialog.lift()
            # Refocus first button after recentering
            in_place_btn.focus_set()
        
        dialog.after(10, reposition_and_focus)
        dialog.after(30, reposition_and_focus)
        
        dialog.wait_window()
        return result["mode"]
    
    def get_mouse_position(self):
        """Capture mouse position"""
        position = {"x": None, "y": None, "captured": False}
        listener_ref = {"listener": None}
        
        def on_click(x, y, button, pressed):
            if pressed:
                position["x"] = x
                position["y"] = y
                position["captured"] = True
                # Stop the listener
                if listener_ref["listener"]:
                    listener_ref["listener"].stop()
                return False  # Stop listener
        
        # Show message BEFORE starting listener
        result = messagebox.showinfo("Capture Mouse Position", 
                                      "Click OK, then click where you want the mouse action to occur.\n\n"
                                      "The next click will be captured.")
        
        if result:
            # Wait a moment for user to move away from OK button
            time.sleep(0.5)
            
            # Start listener
            listener_ref["listener"] = pynput_mouse.Listener(on_click=on_click)
            listener_ref["listener"].start()
            listener_ref["listener"].join()  # Wait for click
        
        return (position["x"], position["y"]) if position["captured"] else (None, None)
    
    def add_mouse_click(self):
        """Add mouse click command (in place or offset)"""
        if not self.current_macro:
            messagebox.showwarning("Warning", "Please select a macro first")
            return
        
        button = simpledialog.askstring("Mouse Click", "Enter button (left/right/middle):", initialvalue="left")
        if button:
            # Show in-place/offset dialog
            mode_result = self.show_click_mode_dialog()
            if mode_result:
                self.save_state()
                if mode_result == "in_place":
                    self.current_macro.add_command(MouseClickCommand(button, 0, 0, mode="in_place"))
                elif mode_result == "offset":
                    offset_x = simpledialog.askinteger("X Offset", "Enter X offset (pixels, can be negative):", initialvalue=0)
                    if offset_x is not None:
                        offset_y = simpledialog.askinteger("Y Offset", "Enter Y offset (pixels, can be negative):", initialvalue=0)
                        if offset_y is not None:
                            self.current_macro.add_command(MouseClickCommand(button, 0, 0, mode="offset", offset_x=offset_x, offset_y=offset_y))
                        else:
                            return
                    else:
                        return
                self.selected_index = len(self.current_macro.commands) - 1
                self.update_command_list()
                self.save_macros()
    
    def add_mouse_click_absolute(self):
        """Add mouse click at absolute position using F2 to capture"""
        if not self.current_macro:
            messagebox.showwarning("Warning", "Please select a macro first")
            return
        
        button = simpledialog.askstring("Mouse Click (Absolute)", "Enter button (left/right/middle):", initialvalue="left")
        if button:
            # Show instructions
            messagebox.showinfo(
                "Capture Position", 
                "Move your mouse to the desired click location and press F2 to capture the coordinates."
            )
            
            # Set up F2 capture
            captured_pos = {"x": None, "y": None}
            
            def on_f2_press(event):
                if event.name == 'f2':
                    captured_pos["x"], captured_pos["y"] = pyautogui.position()
                    keyboard.unhook_all()
            
            keyboard.on_press(on_f2_press)
            
            # Wait for F2 press (with timeout)
            timeout = 30  # 30 seconds
            start_time = time.time()
            while captured_pos["x"] is None and (time.time() - start_time) < timeout:
                time.sleep(0.1)
                self.root.update()  # Keep UI responsive
            
            keyboard.unhook_all()
            
            if captured_pos["x"] is not None:
                self.save_state()
                self.current_macro.add_command(MouseClickCommand(button, captured_pos["x"], captured_pos["y"], mode="absolute"))
                self.selected_index = len(self.current_macro.commands) - 1
                self.update_command_list()
                self.save_macros()
            else:
                messagebox.showwarning("Timeout", "Position capture timed out. No command was added.")
    
    def add_mouse_move(self):
        """Add mouse move command"""
        if not self.current_macro:
            messagebox.showwarning("Warning", "Please select a macro first")
            return
        
        x, y = self.get_mouse_position()
        if x is not None:
            self.save_state()
            self.current_macro.add_command(MouseMoveCommand(x, y))
            self.selected_index = len(self.current_macro.commands) - 1
            self.update_command_list()
            self.save_macros()
    
    def add_if(self):
        """Add IF statement"""
        if not self.current_macro:
            messagebox.showwarning("Warning", "Please select a macro first")
            return
        
        condition = simpledialog.askstring("IF Statement", "Enter condition (e.g., True, False):")
        if condition is not None:
            self.save_state()
            self.current_macro.add_command(IfStatementCommand(condition))
            self.selected_index = len(self.current_macro.commands) - 1
            self.update_command_list()
            self.save_macros()
    
    def add_else(self):
        """Add ELSE statement"""
        if not self.current_macro:
            messagebox.showwarning("Warning", "Please select a macro first")
            return
        
        self.save_state()
        self.current_macro.add_command(ElseStatementCommand())
        self.selected_index = len(self.current_macro.commands) - 1
        self.update_command_list()
        self.save_macros()
    
    def add_endif(self):
        """Add ENDIF statement"""
        if not self.current_macro:
            messagebox.showwarning("Warning", "Please select a macro first")
            return
        
        self.save_state()
        self.current_macro.add_command(EndIfStatementCommand())
        self.selected_index = len(self.current_macro.commands) - 1
        self.update_command_list()
        self.save_macros()
    
    def add_if_image(self):
        """Add IF IMAGE command"""
        if not self.current_macro:
            messagebox.showwarning("Warning", "Please select a macro first")
            return
        
        image_name = simpledialog.askstring("Image Name", "Enter name for this image:")
        if not image_name:
            return
        
        # Capture screen region
        screenshot = self.capture_screen_region()
        if screenshot:
            image_path = f"images/{image_name}.png"
            os.makedirs("images", exist_ok=True)
            screenshot.save(image_path)
            
            # Show preview
            self.show_image_preview(screenshot, "Captured Image")
            
            # Create dialog for options
            dialog = tk.Toplevel(self.root)
            dialog.title("IF IMAGE Options")
            dialog.transient(self.root)
            dialog.grab_set()
            
            # Confidence slider
            conf_frame = ttk.Frame(dialog)
            conf_frame.pack(pady=10)
            ttk.Label(conf_frame, text="Confidence:").pack(side=tk.LEFT, padx=5)
            conf_var = tk.DoubleVar(value=0.8)
            conf_scale = ttk.Scale(conf_frame, from_=0.1, to=1.0, variable=conf_var, orient=tk.HORIZONTAL, length=150)
            conf_scale.pack(side=tk.LEFT, padx=5)
            conf_label = ttk.Label(conf_frame, text="0.80")
            conf_label.pack(side=tk.LEFT, padx=5)
            
            def update_conf_label(event=None):
                conf_label.config(text=f"{conf_var.get():.2f}")
            conf_scale.bind("<Motion>", update_conf_label)
            conf_scale.bind("<ButtonRelease-1>", update_conf_label)
            
            # Move to image checkbox
            move_var = tk.BooleanVar(value=False)
            tk.Checkbutton(dialog, text="Move mouse to image if found", variable=move_var).pack(pady=10)
            
            # Buttons
            def save_options():
                self.save_state()
                self.current_macro.add_command(IfImageCommand(image_path, conf_var.get(), move_var.get()))
                self.selected_index = len(self.current_macro.commands) - 1
                self.update_command_list()
                self.save_macros()
                dialog.destroy()
            
            btn_frame = tk.Frame(dialog)
            btn_frame.pack(pady=20)
            ok_btn = tk.Button(btn_frame, text="OK", command=save_options, width=10)
            ok_btn.pack(side=tk.LEFT, padx=5)
            tk.Button(btn_frame, text="Cancel", command=dialog.destroy, width=10).pack(side=tk.LEFT, padx=5)
            
            # Bind Escape to cancel
            dialog.bind('<Escape>', lambda e: dialog.destroy())
            
            # Center the dialog
            dialog.update_idletasks()
            width = dialog.winfo_width()
            height = dialog.winfo_height()
            screen_width = dialog.winfo_screenwidth()
            screen_height = dialog.winfo_screenheight()
            x = (screen_width - width) // 2
            y = (screen_height - height) // 2
            dialog.geometry(f"{width}x{height}+{x}+{y}")
            
            self.ensure_dialog_focused(dialog)
            
            # Re-center after a short delay
            def recenter():
                dialog.update_idletasks()
                w = dialog.winfo_width()
                h = dialog.winfo_height()
                sw = dialog.winfo_screenwidth()
                sh = dialog.winfo_screenheight()
                new_x = (sw - w) // 2
                new_y = (sh - h) // 2
                dialog.geometry(f"{w}x{h}+{new_x}+{new_y}")
                dialog.focus_force()
                dialog.lift()
                # Refocus OK button after recentering
                ok_btn.focus_set()
            
            dialog.after(10, recenter)
            dialog.after(30, recenter)
    
    def add_if_image_at_index(self, insert_index):
        """Add IF IMAGE command at specific index"""
        image_name = simpledialog.askstring("Image Name", "Enter name for this image:")
        if not image_name:
            return
        
        screenshot = self.capture_screen_region()
        if screenshot:
            image_path = f"images/{image_name}.png"
            os.makedirs("images", exist_ok=True)
            screenshot.save(image_path)
            
            self.show_image_preview(screenshot, "Captured Image")
            
            dialog = tk.Toplevel(self.root)
            dialog.title("IF IMAGE Options")
            dialog.transient(self.root)
            dialog.grab_set()
            
            conf_frame = ttk.Frame(dialog)
            conf_frame.pack(pady=10)
            ttk.Label(conf_frame, text="Confidence:").pack(side=tk.LEFT, padx=5)
            conf_var = tk.DoubleVar(value=0.8)
            conf_scale = ttk.Scale(conf_frame, from_=0.1, to=1.0, variable=conf_var, orient=tk.HORIZONTAL, length=150)
            conf_scale.pack(side=tk.LEFT, padx=5)
            conf_label = ttk.Label(conf_frame, text="0.80")
            conf_label.pack(side=tk.LEFT, padx=5)
            
            def update_conf_label(event=None):
                conf_label.config(text=f"{conf_var.get():.2f}")
            conf_scale.bind("<Motion>", update_conf_label)
            conf_scale.bind("<ButtonRelease-1>", update_conf_label)
            
            move_var = tk.BooleanVar(value=False)
            tk.Checkbutton(dialog, text="Move mouse to image if found", variable=move_var).pack(pady=10)
            
            def save_options():
                self.current_macro.commands.insert(insert_index, IfImageCommand(image_path, conf_var.get(), move_var.get()))
                self.update_command_list()
                self.save_macros()
                dialog.destroy()
            
            btn_frame = tk.Frame(dialog)
            btn_frame.pack(pady=20)
            ok_btn = tk.Button(btn_frame, text="OK", command=save_options, width=10)
            ok_btn.pack(side=tk.LEFT, padx=5)
            tk.Button(btn_frame, text="Cancel", command=dialog.destroy, width=10).pack(side=tk.LEFT, padx=5)
            
            # Bind Escape to cancel
            dialog.bind('<Escape>', lambda e: dialog.destroy())
            
            # Center the dialog
            dialog.update_idletasks()
            width = dialog.winfo_width()
            height = dialog.winfo_height()
            screen_width = dialog.winfo_screenwidth()
            screen_height = dialog.winfo_screenheight()
            x = (screen_width - width) // 2
            y = (screen_height - height) // 2
            dialog.geometry(f"{width}x{height}+{x}+{y}")
            
            self.ensure_dialog_focused(dialog)
            
            # Re-center after a short delay
            def recenter():
                dialog.update_idletasks()
                w = dialog.winfo_width()
                h = dialog.winfo_height()
                sw = dialog.winfo_screenwidth()
                sh = dialog.winfo_screenheight()
                new_x = (sw - w) // 2
                new_y = (sh - h) // 2
                dialog.geometry(f"{w}x{h}+{new_x}+{new_y}")
                dialog.focus_force()
                dialog.lift()
                # Refocus OK button after recentering
                ok_btn.focus_set()
            
            dialog.after(10, recenter)
            dialog.after(30, recenter)
    
    def add_find_image_at_index(self, insert_index):
        """Add FIND IMAGE command at specific index"""
        image_name = simpledialog.askstring("Image Name", "Enter name for this image:")
        if not image_name:
            return
        
        screenshot = self.capture_screen_region()
        if screenshot:
            image_path = f"images/{image_name}.png"
            os.makedirs("images", exist_ok=True)
            screenshot.save(image_path)
            
            self.show_image_preview(screenshot, "Captured Image")
            
            confidence = simpledialog.askfloat("Confidence", "Enter confidence (0.0-1.0):", initialvalue=0.8)
            if confidence is not None:
                self.current_macro.commands.insert(insert_index, FindImageCommand(image_path, confidence))
                self.update_command_list()
                self.save_macros()
    
    def add_find_image(self):
        """Add FIND IMAGE command"""
        if not self.current_macro:
            messagebox.showwarning("Warning", "Please select a macro first")
            return
        
        image_name = simpledialog.askstring("Image Name", "Enter name for this image:")
        if not image_name:
            return
        
        # Capture screen region
        screenshot = self.capture_screen_region()
        if screenshot:
            image_path = f"images/{image_name}.png"
            os.makedirs("images", exist_ok=True)
            screenshot.save(image_path)
            
            # Show preview
            self.show_image_preview(screenshot, "Captured Image")
            
            confidence = simpledialog.askfloat("Confidence", "Enter confidence (0.0-1.0):", initialvalue=0.8)
            if confidence is not None:
                self.save_state()
                self.current_macro.add_command(FindImageCommand(image_path, confidence))
                self.selected_index = len(self.current_macro.commands) - 1
                self.update_command_list()
                self.save_macros()
    
    def add_message(self):
        """Add message command"""
        if not self.current_macro:
            messagebox.showwarning("Warning", "Please select a macro first")
            return
        
        # Create dialog for message options
        dialog = tk.Toplevel(self.root)
        dialog.title("Add Message")
        dialog.transient(self.root)
        dialog.grab_set()
        
        tk.Label(dialog, text="Enter message to display:").pack(pady=10)
        
        text_var = tk.StringVar()
        text_entry = tk.Entry(dialog, textvariable=text_var, width=50)
        text_entry.pack(pady=5)
        text_entry.focus()
        
        always_on_top_var = tk.BooleanVar(value=False)
        tk.Checkbutton(dialog, text="Always on top", variable=always_on_top_var).pack(pady=5)
        
        always_focused_var = tk.BooleanVar(value=False)
        tk.Checkbutton(dialog, text="Always focused", variable=always_focused_var).pack(pady=5)
        
        def save_message():
            text = text_var.get()
            if text:
                self.save_state()
                self.current_macro.add_command(MessageCommand(text, always_on_top_var.get(), always_focused_var.get()))
                self.selected_index = len(self.current_macro.commands) - 1
                self.update_command_list()
                self.save_macros()
                dialog.destroy()
            else:
                messagebox.showwarning("Warning", "Please enter a message")
        
        btn_frame = tk.Frame(dialog)
        btn_frame.pack(pady=20)
        tk.Button(btn_frame, text="OK", command=save_message, width=10).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Cancel", command=dialog.destroy, width=10).pack(side=tk.LEFT, padx=5)
        
        # Bind Enter key to save and Escape to cancel
        text_entry.bind("<Return>", lambda e: save_message())
        dialog.bind('<Escape>', lambda e: dialog.destroy())
        
        # Center the dialog
        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        screen_width = dialog.winfo_screenwidth()
        screen_height = dialog.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        dialog.geometry(f"{width}x{height}+{x}+{y}")
        
        self.ensure_dialog_focused(dialog)
        
        # Re-center after a short delay
        def recenter():
            dialog.update_idletasks()
            w = dialog.winfo_width()
            h = dialog.winfo_height()
            sw = dialog.winfo_screenwidth()
            sh = dialog.winfo_screenheight()
            new_x = (sw - w) // 2
            new_y = (sh - h) // 2
            dialog.geometry(f"{w}x{h}+{new_x}+{new_y}")
            dialog.focus_force()
            dialog.lift()
        
        # Focus text field after all dialog focusing is done
        def focus_text():
            text_entry.focus_set()
            text_entry.select_range(0, tk.END)
            text_entry.icursor(tk.END)
        
        dialog.after(10, recenter)
        dialog.after(50, recenter)
        dialog.after(250, focus_text)
    
    def add_repeat(self):
        """Add REPEAT command"""
        if not self.current_macro:
            messagebox.showwarning("Warning", "Please select a macro first")
            return
        
        count = simpledialog.askinteger("Repeat", "Enter repeat count:")
        if count is not None:
            self.save_state()
            self.current_macro.add_command(RepeatCommand(count))
            self.selected_index = len(self.current_macro.commands) - 1
            self.update_command_list()
            self.save_macros()
    
    def add_end_repeat(self):
        """Add END REPEAT command"""
        if not self.current_macro:
            messagebox.showwarning("Warning", "Please select a macro first")
            return
        
        self.save_state()
        self.current_macro.add_command(EndRepeatCommand())
        self.selected_index = len(self.current_macro.commands) - 1
        self.update_command_list()
        self.save_macros()
    
    def add_delay(self):
        """Add delay command"""
        if not self.current_macro:
            messagebox.showwarning("Warning", "Please select a macro first")
            return
        
        seconds = simpledialog.askfloat("Delay", "Enter delay in seconds:")
        if seconds is not None:
            self.save_state()
            self.current_macro.add_command(DelayCommand(seconds))
            self.selected_index = len(self.current_macro.commands) - 1
            self.update_command_list()
            self.save_macros()
    
    def add_delay_ms(self):
        """Add delay (milliseconds) command"""
        if not self.current_macro:
            messagebox.showwarning("Warning", "Please select a macro first")
            return
        
        milliseconds = simpledialog.askinteger("Delay (ms)", "Enter delay in milliseconds:")
        if milliseconds is not None:
            self.save_state()
            self.current_macro.add_command(DelayMsCommand(milliseconds))
            self.selected_index = len(self.current_macro.commands) - 1
            self.update_command_list()
            self.save_macros()
    
    def add_wait_for_window(self):
        """Add wait for window command"""
        if not self.current_macro:
            messagebox.showwarning("Warning", "Please select a macro first")
            return
        
        pattern = simpledialog.askstring("Wait For Window", "Enter window title pattern (use * as wildcard):")
        if pattern:
            timeout = simpledialog.askinteger("Timeout", "Enter timeout in seconds:", initialvalue=30)
            if timeout is not None:
                self.save_state()
                self.current_macro.add_command(WaitForWindowCommand(pattern, timeout))
                self.selected_index = len(self.current_macro.commands) - 1
                self.update_command_list()
                self.save_macros()
    
    def copy_command(self, event=None):
        """Copy selected command"""
        if not self.current_macro:
            return "break"
        
        if self.selected_index is not None and 0 <= self.selected_index < len(self.current_macro.commands):
            cmd = self.current_macro.commands[self.selected_index]
            # Store a deep copy of the command
            self.copied_command = cmd.to_dict()
        return "break"
    
    def cut_command(self, event=None):
        """Cut selected command (copy + delete)"""
        if not self.current_macro:
            return "break"
        
        if self.selected_index is not None and 0 <= self.selected_index < len(self.current_macro.commands):
            # Copy the command
            cmd = self.current_macro.commands[self.selected_index]
            self.copied_command = cmd.to_dict()
            
            # Delete the command
            self.save_state()
            deleted_index = self.selected_index
            self.current_macro.remove_command(self.selected_index)
            
            # Select the command above the deleted one
            if deleted_index > 0:
                self.selected_index = deleted_index - 1
            elif len(self.current_macro.commands) > 0:
                self.selected_index = 0
            else:
                self.selected_index = None
            
            self.update_command_list()
            self.save_macros()
        return "break"
    
    def paste_command(self, event=None):
        """Paste command below selected command"""
        if not self.current_macro:
            return "break"
        
        if self.copied_command is None:
            return "break"
        
        # Determine insert position (below selected or at end)
        insert_index = (self.selected_index + 1) if self.selected_index is not None else len(self.current_macro.commands)
        
        # Create command from dict
        cmd = MacroCommand.from_dict(self.copied_command)
        if cmd:
            self.save_state()
            self.current_macro.commands.insert(insert_index, cmd)
            self.selected_index = insert_index
            self.update_command_list()
            self.save_macros()
        return "break"
    
    def remove_command(self):
        """Remove selected command"""
        if not self.current_macro:
            messagebox.showwarning("Warning", "Please select a macro first")
            return
        
        # Check if a command is selected
        if self.selected_index is not None and 0 <= self.selected_index < len(self.current_macro.commands):
            self.save_state()  # Save state before modification
            deleted_index = self.selected_index
            self.current_macro.remove_command(self.selected_index)
            
            # Select the command above the deleted one
            if deleted_index > 0:
                self.selected_index = deleted_index - 1
            elif len(self.current_macro.commands) > 0:
                self.selected_index = 0
            else:
                self.selected_index = None
            
            self.update_command_list()
            self.save_macros()
        else:
            messagebox.showinfo("Info", "No commands to remove")
    
    def run_macro(self):
        """Run the current macro"""
        if not self.current_macro:
            messagebox.showwarning("Warning", "Please select a macro first")
            return
        
        if self.executor.running:
            messagebox.showwarning("Warning", "A macro is already running")
            return
        
        try:
            repeat_count = int(self.repeat_var.get())
        except ValueError:
            repeat_count = 1
        
        self.status_label.config(text="Running...")
        
        def run():
            self.executor.execute(self.current_macro, repeat_count)
            self.status_label.config(text="Ready")
        
        threading.Thread(target=run, daemon=True).start()
    
    def stop_macro(self):
        """Stop the running macro"""
        if self.executor.running:
            self.executor.stop()
            self.status_label.config(text="Stopped")
    
    def save_macros(self):
        """Save all macros to file"""
        data = {
            "macros": {name: macro.to_dict() for name, macro in self.macros.items()},
            "last_selected": self.current_macro.name if self.current_macro else None
        }
        with open(self.data_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_macros(self):
        """Load macros from file"""
        if self.data_file.exists():
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    
                    # Handle both old and new format
                    if "macros" in data:
                        # New format with last_selected
                        macros_data = data["macros"]
                        last_selected = data.get("last_selected")
                    else:
                        # Old format - all keys are macros
                        macros_data = data
                        last_selected = None
                    
                    for name, macro_data in macros_data.items():
                        macro = Macro.from_dict(macro_data)
                        self.macros[name] = macro
                        self.register_hotkey(macro)
                    
                    self.update_macro_list()
                    
                    # Restore last selected macro
                    if last_selected and last_selected in self.macros:
                        self.macro_var.set(last_selected)
                        self.on_macro_selected()
            except Exception as e:
                print(f"Error loading macros: {e}")
    
    def _create_thumbnail(self, parent, image_path):
        """Create a thumbnail label for an image"""
        try:
            # Check cache first
            if image_path in self.thumbnail_cache:
                thumb_label = tk.Label(parent, image=self.thumbnail_cache[image_path], bg="white")
                return thumb_label
            
            # Load and create thumbnail
            img = Image.open(image_path)
            img.thumbnail((40, 40), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            import io
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            
            photo = tk.PhotoImage(data=img_byte_arr.getvalue())
            self.thumbnail_cache[image_path] = photo  # Cache it
            
            thumb_label = tk.Label(parent, image=photo, bg="white", borderwidth=1, relief=tk.SOLID)
            return thumb_label
        except Exception as e:
            print(f"Error creating thumbnail: {e}")
            # Return a placeholder
            placeholder = tk.Label(parent, text="[IMG]", bg="lightgray", width=5, height=2)
            return placeholder
    
    def show_image_preview(self, image, title="Image Preview"):
        """Show a preview of the captured image"""
        preview_window = tk.Toplevel(self.root)
        preview_window.title(title)
        preview_window.transient(self.root)
        preview_window.grab_set()
        
        # Resize image to fit preview window (max 400x400)
        img = image.copy()
        img.thumbnail((400, 400), Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        from tkinter import PhotoImage
        import io
        
        # Save to bytes and load as PhotoImage
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # Create label with image
        photo = tk.PhotoImage(data=img_byte_arr.getvalue())
        label = tk.Label(preview_window, image=photo)
        label.image = photo  # Keep a reference
        label.pack(padx=10, pady=10)
        
        # Add OK button
        ok_btn = ttk.Button(preview_window, text="OK", command=preview_window.destroy)
        ok_btn.pack(pady=10)
        
        # Update and center the window
        preview_window.update_idletasks()
        width = preview_window.winfo_width()
        height = preview_window.winfo_height()
        screen_width = preview_window.winfo_screenwidth()
        screen_height = preview_window.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        preview_window.geometry(f"+{x}+{y}")
        
        # Focus and bring to top
        preview_window.attributes('-topmost', True)
        preview_window.focus_force()
        preview_window.lift()
        
        # Re-center and focus after a short delay
        def recenter_and_focus():
            w = preview_window.winfo_width()
            h = preview_window.winfo_height()
            sw = preview_window.winfo_screenwidth()
            sh = preview_window.winfo_screenheight()
            new_x = (sw - w) // 2
            new_y = (sh - h) // 2
            preview_window.geometry(f"+{new_x}+{new_y}")
            preview_window.focus_force()
            preview_window.lift()
            # Refocus OK button after recentering
            ok_btn.focus_set()
        
        preview_window.after(50, recenter_and_focus)
        preview_window.after(100, recenter_and_focus)
        
        # Wait for window to close before continuing
        preview_window.wait_window()
    
    def on_edit_command(self, event):
        """Handle double-click to edit command"""
        if not self.current_macro:
            return
        
        # Clear drag data to prevent interference
        self.drag_data["started"] = False
        self.drag_data["index"] = None
        
        # Find which command frame was clicked
        widget = event.widget
        while widget and not hasattr(widget, '_command_index'):
            widget = widget.master
            if widget == self.command_frame:
                return
        
        if not widget or not hasattr(widget, '_command_index'):
            return
        
        index = widget._command_index
        if index < 0 or index >= len(self.current_macro.commands):
            return
        
        cmd = self.current_macro.commands[index]
        
        # Handle different command types
        if isinstance(cmd, (KeyPressCommand, KeyHoldCommand, KeyReleaseCommand)):
            new_key = self.show_key_selector("Edit Key")
            if new_key:
                self.save_state()
                cmd.key = new_key
                self.update_command_list()
                self.save_macros()
        
        elif isinstance(cmd, DelayCommand):
            new_delay = simpledialog.askfloat("Edit Delay", "Enter delay in seconds:", initialvalue=cmd.seconds)
            if new_delay is not None:
                self.save_state()
                cmd.seconds = new_delay
                self.update_command_list()
                self.save_macros()
        
        elif isinstance(cmd, MessageCommand):
            self.edit_message_command(cmd)
        
        elif isinstance(cmd, RepeatCommand):
            new_count = simpledialog.askinteger("Edit Repeat", "Enter repeat count:", initialvalue=cmd.count)
            if new_count is not None:
                self.save_state()
                cmd.count = new_count
                self.update_command_list()
                self.save_macros()
        
        elif isinstance(cmd, (IfImageCommand, FindImageCommand)):
            # Create dialog for editing image command
            print(f"DEBUG: About to call edit_image_command")
            try:
                self.edit_image_command(cmd, index)
                print(f"DEBUG: edit_image_command completed")
            except Exception as e:
                print(f"DEBUG: Exception in edit_image_command: {e}")
                import traceback
                traceback.print_exc()
        
        return "break"  # Prevent event propagation
    
    def edit_message_command(self, cmd):
        """Edit a message command"""
        # Create dialog for editing message
        dialog = tk.Toplevel(self.root)
        dialog.title("Edit Message")
        dialog.transient(self.root)
        dialog.grab_set()
        
        tk.Label(dialog, text="Enter message:").pack(pady=10)
        
        text_var = tk.StringVar(value=cmd.text)
        text_entry = tk.Entry(dialog, textvariable=text_var, width=50)
        text_entry.pack(pady=5)
        text_entry.focus()
        
        always_on_top_var = tk.BooleanVar(value=cmd.always_on_top)
        tk.Checkbutton(dialog, text="Always on top", variable=always_on_top_var).pack(pady=5)
        
        always_focused_var = tk.BooleanVar(value=cmd.always_focused)
        tk.Checkbutton(dialog, text="Always focused", variable=always_focused_var).pack(pady=5)
        
        def save_changes():
            text = text_var.get()
            if text:
                self.save_state()
                cmd.text = text
                cmd.always_on_top = always_on_top_var.get()
                cmd.always_focused = always_focused_var.get()
                self.update_command_list()
                self.save_macros()
                dialog.destroy()
            else:
                messagebox.showwarning("Warning", "Please enter a message")
        
        btn_frame = tk.Frame(dialog)
        btn_frame.pack(pady=20)
        tk.Button(btn_frame, text="OK", command=save_changes, width=10).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Cancel", command=dialog.destroy, width=10).pack(side=tk.LEFT, padx=5)
        
        # Bind Enter key to save and Escape to cancel
        text_entry.bind("<Return>", lambda e: save_changes())
        dialog.bind('<Escape>', lambda e: dialog.destroy())
        
        # Center the dialog
        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        screen_width = dialog.winfo_screenwidth()
        screen_height = dialog.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        dialog.geometry(f"{width}x{height}+{x}+{y}")
        
        self.ensure_dialog_focused(dialog)
        
        # Re-center after a short delay
        def recenter():
            dialog.update_idletasks()
            w = dialog.winfo_width()
            h = dialog.winfo_height()
            sw = dialog.winfo_screenwidth()
            sh = dialog.winfo_screenheight()
            new_x = (sw - w) // 2
            new_y = (sh - h) // 2
            dialog.geometry(f"{w}x{h}+{new_x}+{new_y}")
            dialog.focus_force()
            dialog.lift()
        
        # Focus text field after all dialog focusing is done
        def focus_text():
            text_entry.focus_set()
            text_entry.select_range(0, tk.END)
            text_entry.icursor(tk.END)
        
        dialog.after(10, recenter)
        dialog.after(30, recenter)
        dialog.after(80, focus_text)
    
    def edit_image_command(self, cmd, index):
        """Edit an image command with preview and multiple images support"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Edit Image Command")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # For IF IMAGE commands, show list of images
        if isinstance(cmd, IfImageCommand):
            tk.Label(dialog, text="Images (will search for any of these):", font=("Arial", 10, "bold")).pack(pady=10)
            
            # Create scrollable frame for images
            list_frame = ttk.Frame(dialog)
            list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            canvas = tk.Canvas(list_frame, height=300)
            scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
            # Display each image with its settings
            image_frames = []
            
            def refresh_image_list():
                # Clear existing frames
                for widget in scrollable_frame.winfo_children():
                    widget.destroy()
                image_frames.clear()
                
                for idx, img_info in enumerate(cmd.image_paths):
                    frame = ttk.Frame(scrollable_frame, relief="solid", borderwidth=1)
                    frame.pack(fill=tk.X, padx=5, pady=5)
                    
                    # Try to show thumbnail
                    try:
                        img = Image.open(img_info["path"])
                        img.thumbnail((100, 100), Image.Resampling.LANCZOS)
                        import io
                        img_byte_arr = io.BytesIO()
                        img.save(img_byte_arr, format='PNG')
                        img_byte_arr.seek(0)
                        photo = tk.PhotoImage(data=img_byte_arr.getvalue())
                        label = tk.Label(frame, image=photo)
                        label.image = photo
                        label.pack(side=tk.LEFT, padx=5)
                    except:
                        tk.Label(frame, text="[No Image]", width=15).pack(side=tk.LEFT, padx=5)
                    
                    info_frame = ttk.Frame(frame)
                    info_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
                    
                    tk.Label(info_frame, text=Path(img_info["path"]).stem, font=("Arial", 9, "bold")).pack(anchor="w")
                    tk.Label(info_frame, text=f"Confidence: {img_info['confidence']:.2f}").pack(anchor="w")
                    
                    btn_frame = ttk.Frame(frame)
                    btn_frame.pack(side=tk.RIGHT, padx=5)
                    
                    def make_edit_func(img_idx):
                        def edit_img():
                            # Create mini dialog to edit confidence
                            edit_dlg = tk.Toplevel(dialog)
                            edit_dlg.title("Edit Image Settings")
                            edit_dlg.transient(dialog)
                            edit_dlg.grab_set()
                            
                            tk.Label(edit_dlg, text="Confidence:").pack(pady=10)
                            conf_var = tk.DoubleVar(value=cmd.image_paths[img_idx]["confidence"])
                            conf_scale = ttk.Scale(edit_dlg, from_=0.1, to=1.0, variable=conf_var, orient=tk.HORIZONTAL, length=200)
                            conf_scale.pack(pady=5)
                            conf_label = tk.Label(edit_dlg, text=f"{cmd.image_paths[img_idx]['confidence']:.2f}")
                            conf_label.pack()
                            
                            def update_conf_label(event=None):
                                conf_label.config(text=f"{conf_var.get():.2f}")
                            conf_scale.bind("<Motion>", update_conf_label)
                            conf_scale.bind("<ButtonRelease-1>", update_conf_label)
                            
                            def save_conf():
                                cmd.image_paths[img_idx]["confidence"] = conf_var.get()
                                # Update primary confidence if editing first image
                                if img_idx == 0:
                                    cmd.confidence = conf_var.get()
                                self.save_state()
                                self.save_macros()
                                edit_dlg.destroy()
                                refresh_image_list()
                            
                            save_btn = ttk.Button(edit_dlg, text="Save", command=save_conf)
                            save_btn.pack(pady=10)
                            
                            # Center and focus the dialog
                            edit_dlg.update_idletasks()
                            width = edit_dlg.winfo_width()
                            height = edit_dlg.winfo_height()
                            screen_width = edit_dlg.winfo_screenwidth()
                            screen_height = edit_dlg.winfo_screenheight()
                            x = (screen_width - width) // 2
                            y = (screen_height - height) // 2
                            edit_dlg.geometry(f"{width}x{height}+{x}+{y}")
                            self.ensure_dialog_focused(edit_dlg)
                            
                            # Re-center after a short delay
                            def recenter_edit():
                                edit_dlg.update_idletasks()
                                w = edit_dlg.winfo_width()
                                h = edit_dlg.winfo_height()
                                sw = edit_dlg.winfo_screenwidth()
                                sh = edit_dlg.winfo_screenheight()
                                new_x = (sw - w) // 2
                                new_y = (sh - h) // 2
                                edit_dlg.geometry(f"{w}x{h}+{new_x}+{new_y}")
                                edit_dlg.focus_force()
                                edit_dlg.lift()
                                # Refocus Save button after recentering
                                save_btn.focus_set()
                            
                            edit_dlg.after(10, recenter_edit)
                            edit_dlg.after(30, recenter_edit)
                        return edit_img
                    
                    def make_remove_func(img_idx):
                        def remove_img():
                            if len(cmd.image_paths) > 1:
                                cmd.image_paths.pop(img_idx)
                                # Update primary image_path if removing first image
                                if img_idx == 0 and cmd.image_paths:
                                    cmd.image_path = cmd.image_paths[0]["path"]
                                    cmd.confidence = cmd.image_paths[0]["confidence"]
                                self.save_state()
                                self.save_macros()
                                refresh_image_list()
                            else:
                                messagebox.showwarning("Warning", "Cannot remove the last image")
                        return remove_img
                    
                    ttk.Button(btn_frame, text="Edit", command=make_edit_func(idx), width=8).pack(pady=2)
                    ttk.Button(btn_frame, text="Remove", command=make_remove_func(idx), width=8).pack(pady=2)
                    
                    image_frames.append(frame)
            
            refresh_image_list()
            
            # Add new image button
            def add_new_image():
                # Custom dialog for image name
                name_dlg = tk.Toplevel(dialog)
                name_dlg.title("Image Name")
                name_dlg.transient(dialog)
                name_dlg.grab_set()
                
                tk.Label(name_dlg, text="Enter name for this image:").pack(pady=10, padx=20)
                name_var = tk.StringVar()
                entry = tk.Entry(name_dlg, textvariable=name_var, width=30)
                entry.pack(pady=5, padx=20)
                
                result = {'confirmed': False}
                
                def on_ok():
                    result['confirmed'] = True
                    name_dlg.destroy()
                
                def on_cancel():
                    name_dlg.destroy()
                
                entry.bind('<Return>', lambda e: on_ok())
                entry.bind('<Escape>', lambda e: on_cancel())
                
                btn_frame = tk.Frame(name_dlg)
                btn_frame.pack(pady=10)
                ttk.Button(btn_frame, text="OK", command=on_ok).pack(side=tk.LEFT, padx=5)
                ttk.Button(btn_frame, text="Cancel", command=on_cancel).pack(side=tk.LEFT, padx=5)
                
                # Center and focus
                name_dlg.update_idletasks()
                width = name_dlg.winfo_width()
                height = name_dlg.winfo_height()
                screen_width = name_dlg.winfo_screenwidth()
                screen_height = name_dlg.winfo_screenheight()
                x = (screen_width - width) // 2
                y = (screen_height - height) // 2
                name_dlg.geometry(f"{width}x{height}+{x}+{y}")
                self.ensure_dialog_focused(name_dlg)
                
                # Re-center after a short delay
                def recenter_name():
                    name_dlg.update_idletasks()
                    w = name_dlg.winfo_width()
                    h = name_dlg.winfo_height()
                    sw = name_dlg.winfo_screenwidth()
                    sh = name_dlg.winfo_screenheight()
                    new_x = (sw - w) // 2
                    new_y = (sh - h) // 2
                    name_dlg.geometry(f"{w}x{h}+{new_x}+{new_y}")
                    name_dlg.focus_force()
                    name_dlg.lift()
                
                # Focus text field after all dialog focusing is done
                def focus_entry():
                    entry.focus_set()
                    entry.select_range(0, tk.END)
                    entry.icursor(tk.END)
                
                name_dlg.after(10, recenter_name)
                name_dlg.after(30, recenter_name)
                name_dlg.after(80, focus_entry)
                
                name_dlg.wait_window()
                
                if not result['confirmed'] or not name_var.get():
                    return
                
                image_name = name_var.get()
                screenshot = self.capture_screen_region()
                if screenshot:
                    image_path = f"images/{image_name}.png"
                    os.makedirs("images", exist_ok=True)
                    screenshot.save(image_path)
                    self.show_image_preview(screenshot, "Captured Image")
                    
                    # Custom dialog for confidence
                    conf_dlg = tk.Toplevel(dialog)
                    conf_dlg.title("Confidence")
                    conf_dlg.transient(dialog)
                    conf_dlg.grab_set()
                    
                    tk.Label(conf_dlg, text="Enter confidence (0.0-1.0):").pack(pady=10, padx=20)
                    conf_var = tk.DoubleVar(value=0.8)
                    conf_scale = ttk.Scale(conf_dlg, from_=0.1, to=1.0, variable=conf_var, orient=tk.HORIZONTAL, length=200)
                    conf_scale.pack(pady=5, padx=20)
                    conf_label = tk.Label(conf_dlg, text="0.80")
                    conf_label.pack(pady=5)
                    
                    def update_conf_label(event=None):
                        conf_label.config(text=f"{conf_var.get():.2f}")
                    conf_scale.bind("<Motion>", update_conf_label)
                    conf_scale.bind("<ButtonRelease-1>", update_conf_label)
                    
                    conf_result = {'confirmed': False}
                    
                    def on_conf_ok():
                        conf_result['confirmed'] = True
                        conf_dlg.destroy()
                    
                    def on_conf_cancel():
                        conf_dlg.destroy()
                    
                    btn_frame = tk.Frame(conf_dlg)
                    btn_frame.pack(pady=10)
                    ok_btn = ttk.Button(btn_frame, text="OK", command=on_conf_ok)
                    ok_btn.pack(side=tk.LEFT, padx=5)
                    ttk.Button(btn_frame, text="Cancel", command=on_conf_cancel).pack(side=tk.LEFT, padx=5)
                    
                    # Bind Escape to cancel
                    conf_dlg.bind('<Escape>', lambda e: on_conf_cancel())
                    
                    # Center and focus
                    conf_dlg.update_idletasks()
                    width = conf_dlg.winfo_width()
                    height = conf_dlg.winfo_height()
                    screen_width = conf_dlg.winfo_screenwidth()
                    screen_height = conf_dlg.winfo_screenheight()
                    x = (screen_width - width) // 2
                    y = (screen_height - height) // 2
                    conf_dlg.geometry(f"{width}x{height}+{x}+{y}")
                    self.ensure_dialog_focused(conf_dlg)
                    
                    # Re-center after a short delay
                    def recenter_conf():
                        conf_dlg.update_idletasks()
                        w = conf_dlg.winfo_width()
                        h = conf_dlg.winfo_height()
                        sw = conf_dlg.winfo_screenwidth()
                        sh = conf_dlg.winfo_screenheight()
                        new_x = (sw - w) // 2
                        new_y = (sh - h) // 2
                        conf_dlg.geometry(f"{w}x{h}+{new_x}+{new_y}")
                        conf_dlg.focus_force()
                        conf_dlg.lift()
                        # Refocus OK button after recentering
                        ok_btn.focus_set()
                    
                    conf_dlg.after(10, recenter_conf)
                    conf_dlg.after(30, recenter_conf)
                    
                    conf_dlg.wait_window()
                    
                    if conf_result['confirmed']:
                        cmd.image_paths.append({"path": image_path, "confidence": conf_var.get()})
                        # Update primary image_path and confidence for backwards compatibility
                        if not cmd.image_path or len(cmd.image_paths) == 1:
                            cmd.image_path = cmd.image_paths[0]["path"]
                            cmd.confidence = cmd.image_paths[0]["confidence"]
                        self.save_state()
                        self.save_macros()
                        refresh_image_list()
            
            ttk.Button(dialog, text="Add Additional Image", command=add_new_image).pack(pady=10)
            
            # Move to image checkbox
            move_var = tk.BooleanVar(value=cmd.move_to_image)
            tk.Checkbutton(dialog, text="Move mouse to image if found", variable=move_var).pack(pady=10)
            
            # Save/Cancel buttons
            btn_frame = ttk.Frame(dialog)
            btn_frame.pack(pady=20)
            
            def save_changes():
                self.save_state()
                cmd.move_to_image = move_var.get()
                # Update primary image_path and confidence for backwards compatibility
                if cmd.image_paths:
                    cmd.image_path = cmd.image_paths[0]["path"]
                    cmd.confidence = cmd.image_paths[0]["confidence"]
                self.update_command_list()
                self.save_macros()
                dialog.destroy()
            
            save_btn = ttk.Button(btn_frame, text="Save", command=save_changes)
            save_btn.pack(side=tk.LEFT, padx=5)
            ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
            
            # Bind Escape to cancel
            dialog.bind('<Escape>', lambda e: dialog.destroy())
            
            # Center the dialog
            dialog.update_idletasks()
            width = dialog.winfo_width()
            height = dialog.winfo_height()
            screen_width = dialog.winfo_screenwidth()
            screen_height = dialog.winfo_screenheight()
            x = (screen_width - width) // 2
            y = (screen_height - height) // 2
            dialog.geometry(f"{width}x{height}+{x}+{y}")
            
            self.ensure_dialog_focused(dialog)
            
            # Re-center after a short delay
            def recenter():
                dialog.update_idletasks()
                w = dialog.winfo_width()
                h = dialog.winfo_height()
                sw = dialog.winfo_screenwidth()
                sh = dialog.winfo_screenheight()
                new_x = (sw - w) // 2
                new_y = (sh - h) // 2
                dialog.geometry(f"{w}x{h}+{new_x}+{new_y}")
                dialog.focus_force()
                dialog.lift()
                # Refocus Save button after recentering
                save_btn.focus_set()
            
            dialog.after(10, recenter)
            dialog.after(30, recenter)
        
        else:
            # FIND IMAGE command - single image only
            try:
                img = Image.open(cmd.image_path)
                img.thumbnail((400, 400), Image.Resampling.LANCZOS)
                
                import io
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
                
                photo = tk.PhotoImage(data=img_byte_arr.getvalue())
                label = tk.Label(dialog, image=photo)
                label.image = photo
                label.pack(padx=10, pady=10)
            except:
                tk.Label(dialog, text="Image not found", fg="red").pack(pady=10)
            
            # Confidence slider
            conf_frame = ttk.Frame(dialog)
            conf_frame.pack(pady=10)
            
            ttk.Label(conf_frame, text="Confidence:").pack(side=tk.LEFT, padx=5)
            conf_var = tk.DoubleVar(value=cmd.confidence)
            conf_scale = ttk.Scale(conf_frame, from_=0.1, to=1.0, variable=conf_var, orient=tk.HORIZONTAL, length=200)
            conf_scale.pack(side=tk.LEFT, padx=5)
            conf_label = ttk.Label(conf_frame, text=f"{cmd.confidence:.2f}")
            conf_label.pack(side=tk.LEFT, padx=5)
            
            def update_label(event=None):
                conf_label.config(text=f"{conf_var.get():.2f}")
            
            conf_scale.bind("<Motion>", update_label)
            conf_scale.bind("<ButtonRelease-1>", update_label)
            
            # Buttons
            btn_frame = ttk.Frame(dialog)
            btn_frame.pack(pady=20)
            
            def recapture():
                screenshot = self.capture_screen_region()
                if screenshot:
                    self.save_state()
                    screenshot.save(cmd.image_path)
                    # Clear thumbnail cache to force refresh
                    if cmd.image_path in self.thumbnail_cache:
                        del self.thumbnail_cache[cmd.image_path]
                    messagebox.showinfo("Success", "Image updated!")
                    dialog.destroy()
                    self.update_command_list()
                    self.save_macros()
            
            def save_changes():
                self.save_state()
                cmd.confidence = conf_var.get()
                self.update_command_list()
                self.save_macros()
                dialog.destroy()
            
            ttk.Button(btn_frame, text="Recapture Image", command=recapture).pack(side=tk.LEFT, padx=5)
            ttk.Button(btn_frame, text="Save", command=save_changes).pack(side=tk.LEFT, padx=5)
            ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
    
    def set_window_position(self):
        """Save current window position and size for the current macro"""
        if not self.current_macro:
            messagebox.showwarning("Warning", "Please select a macro first")
            return
        
        # Get current window geometry
        geometry = self.root.geometry()
        self.current_macro.window_geometry = geometry
        self.save_macros()
        messagebox.showinfo("Success", f"Window position saved for '{self.current_macro.name}'\n{geometry}")
    
    def start_hotkey_listener(self):
        """Start listening for hotkeys"""
        # Keyboard library handles this automatically
        pass


def main():
    root = tk.Tk()
    app = MacroMakerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
