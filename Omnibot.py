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

# Debug mode for image matching - set to True to save debug artifacts
DEBUG_IMAGE = False
DEBUG_IMAGE_DIR = "debug_images"

# ============================================================================
# DPI AWARENESS (Windows) - Call early to fix coordinate mismatches
# ============================================================================
def set_dpi_awareness():
    """Make the process DPI-aware on Windows to fix coordinate mismatches."""
    try:
        # Try per-monitor DPI awareness (Windows 8.1+)
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        print("[DPI] Set per-monitor DPI awareness (level 2)")
    except AttributeError:
        try:
            # Fallback to basic DPI awareness (Windows Vista+)
            ctypes.windll.user32.SetProcessDPIAware()
            print("[DPI] Set basic DPI awareness")
        except Exception as e:
            print(f"[DPI] Could not set DPI awareness: {e}")
    except Exception as e:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
            print("[DPI] Fallback: Set basic DPI awareness")
        except Exception as e2:
            print(f"[DPI] Could not set DPI awareness: {e}, {e2}")

# Call DPI awareness at module load time
set_dpi_awareness()

# ============================================================================
# IMAGE MATCHING HELPER FUNCTIONS
# ============================================================================

def get_dpi_scale_factor():
    """Detect DPI scale factor by comparing pyautogui screen size vs actual grab size."""
    try:
        pag_size = pyautogui.size()
        # Grab a small region and check actual pixel dimensions
        grab = ImageGrab.grab(bbox=(0, 0, 100, 100))
        grab_size = grab.size
        sx = grab_size[0] / 100.0
        sy = grab_size[1] / 100.0
        if abs(sx - 1.0) > 0.01 or abs(sy - 1.0) > 0.01:
            print(f"[DPI] Detected scale factor: sx={sx:.2f}, sy={sy:.2f}")
        return sx, sy
    except Exception as e:
        print(f"[DPI] Could not detect scale factor: {e}")
        return 1.0, 1.0

def normalize_region(region):
    """
    Normalize region to (left, top, width, height) format for pyautogui.
    
    Input can be:
    - (x1, y1, x2, y2) bbox format -> convert to (left, top, width, height)
    - (left, top, width, height) -> pass through
    - list or tuple
    - None -> return None
    
    Returns tuple of ints or None.
    """
    if region is None:
        return None
    
    # Coerce to list of ints
    try:
        region = [int(r) for r in region]
    except (TypeError, ValueError):
        print(f"[Region] Invalid region format: {region}")
        return None
    
    if len(region) != 4:
        print(f"[Region] Region must have 4 elements: {region}")
        return None
    
    x1, y1, x2_or_w, y2_or_h = region
    
    # Detect if this is bbox format (x2 > x1 and y2 > y1 with both > reasonable threshold)
    # If x2_or_w and y2_or_h are both larger than x1, y1 respectively and > 100, likely bbox
    if x2_or_w > x1 and y2_or_h > y1 and x2_or_w > 100 and y2_or_h > 100:
        # Looks like bbox (x1, y1, x2, y2) -> convert to (left, top, width, height)
        width = x2_or_w - x1
        height = y2_or_h - y1
        result = (x1, y1, width, height)
        if DEBUG_IMAGE:
            print(f"[Region] Converted bbox {region} -> ltwh {result}")
        return result
    else:
        # Assume already (left, top, width, height)
        return tuple(region)

def load_template(path):
    """
    Load template image and return (numpy array in BGR, width, height).
    Returns (None, 0, 0) on failure.
    """
    try:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[Template] Failed to load: {path}")
            return None, 0, 0
        h, w = img.shape[:2]
        return img, w, h
    except Exception as e:
        print(f"[Template] Error loading {path}: {e}")
        return None, 0, 0

def screenshot_region(left, top, width, height):
    """
    Capture a screenshot of the specified region.
    Returns numpy array in BGR format, or None on failure.
    """
    try:
        # Ensure integer coordinates
        left, top, width, height = int(left), int(top), int(width), int(height)
        
        # Clamp to valid values
        if width <= 0 or height <= 0:
            print(f"[Screenshot] Invalid dimensions: {width}x{height}")
            return None
        
        # Use PIL ImageGrab
        bbox = (left, top, left + width, top + height)
        grab = ImageGrab.grab(bbox=bbox)
        
        # Convert to numpy BGR
        img = np.array(grab)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    except Exception as e:
        print(f"[Screenshot] Error capturing region ({left},{top},{width},{height}): {e}")
        return None

def similarity_score(patch, template):
    """
    Compute similarity between patch and template using normalized cross-correlation.
    Both must be same size numpy arrays in BGR format.
    Returns float in range [0, 1] where 1 is perfect match.
    """
    if patch is None or template is None:
        return 0.0
    
    try:
        # Ensure same size
        if patch.shape != template.shape:
            # Resize patch to template size
            patch = cv2.resize(patch, (template.shape[1], template.shape[0]))
        
        # Convert to grayscale for comparison
        patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        
        # Use template matching with normalized cross-correlation
        # Since they're same size, we get a single value
        result = cv2.matchTemplate(patch_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        score = float(result[0, 0])
        
        # Normalize to 0-1 range (TM_CCOEFF_NORMED can be -1 to 1)
        score = (score + 1.0) / 2.0
        return score
    except Exception as e:
        print(f"[Similarity] Error computing similarity: {e}")
        return 0.0

def save_debug_artifacts(template_path, template_img, patch_img, cx, cy, location, score, attempt):
    """Save debug images to disk for analysis."""
    try:
        os.makedirs(DEBUG_IMAGE_DIR, exist_ok=True)
        timestamp = int(time.time() * 1000)
        base_name = Path(template_path).stem
        
        # Save template
        if template_img is not None:
            cv2.imwrite(f"{DEBUG_IMAGE_DIR}/{timestamp}_{base_name}_template.png", template_img)
        
        # Save patch under cursor
        if patch_img is not None:
            cv2.imwrite(f"{DEBUG_IMAGE_DIR}/{timestamp}_{base_name}_patch_attempt{attempt}.png", patch_img)
        
        # Save annotated screenshot of region around match
        try:
            # Capture area around the match
            grab_left = max(0, cx - 200)
            grab_top = max(0, cy - 200)
            context_img = screenshot_region(grab_left, grab_top, 400, 400)
            if context_img is not None:
                # Draw rectangle where we think match is
                rel_cx = cx - grab_left
                rel_cy = cy - grab_top
                if template_img is not None:
                    th, tw = template_img.shape[:2]
                    pt1 = (int(rel_cx - tw/2), int(rel_cy - th/2))
                    pt2 = (int(rel_cx + tw/2), int(rel_cy + th/2))
                    cv2.rectangle(context_img, pt1, pt2, (0, 255, 0), 2)
                # Draw crosshair at center
                cv2.drawMarker(context_img, (int(rel_cx), int(rel_cy)), (0, 0, 255), 
                              cv2.MARKER_CROSS, 20, 2)
                cv2.imwrite(f"{DEBUG_IMAGE_DIR}/{timestamp}_{base_name}_context_attempt{attempt}.png", context_img)
        except Exception as e:
            print(f"[Debug] Error saving context image: {e}")
        
        # Log info
        print(f"[Debug] Saved artifacts to {DEBUG_IMAGE_DIR}/ (attempt {attempt}, score={score:.3f})")
    except Exception as e:
        print(f"[Debug] Error saving debug artifacts: {e}")

def verify_cursor_over_template(cx, cy, template_path, tolerance=0.7):
    """
    Verify that the cursor is positioned over the template image.
    
    Args:
        cx, cy: Expected center position
        template_path: Path to template image
        tolerance: Minimum similarity score to consider a match (0-1)
    
    Returns:
        (verified: bool, score: float, patch: numpy array or None)
    """
    template, tw, th = load_template(template_path)
    if template is None:
        print(f"[Verify] Could not load template: {template_path}")
        return False, 0.0, None
    
    # Calculate region under cursor that corresponds to template bbox
    left = int(cx - tw / 2)
    top = int(cy - th / 2)
    
    # Capture the patch
    patch = screenshot_region(left, top, tw, th)
    if patch is None:
        print(f"[Verify] Could not capture patch at ({left},{top},{tw},{th})")
        return False, 0.0, None
    
    # Compute similarity
    score = similarity_score(patch, template)
    verified = score >= tolerance
    
    if DEBUG_IMAGE:
        print(f"[Verify] Template: {tw}x{th}, Center: ({cx},{cy}), Score: {score:.3f}, Verified: {verified}")
    
    return verified, score, patch

def find_image_cv2(template_path, confidence, search_region=None):
    """
    Alternative image finding using cv2.matchTemplate directly.
    More reliable than pyautogui.locateOnScreen in some cases.
    
    Returns (found: bool, cx: int, cy: int, match_rect: tuple) or (False, 0, 0, None)
    """
    template, tw, th = load_template(template_path)
    if template is None:
        return False, 0, 0, None
    
    try:
        # Capture search region or full screen
        if search_region:
            region = normalize_region(search_region)
            if region:
                screenshot = screenshot_region(*region)
                offset_x, offset_y = region[0], region[1]
            else:
                screenshot = np.array(ImageGrab.grab())
                screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
                offset_x, offset_y = 0, 0
        else:
            screenshot = np.array(ImageGrab.grab())
            screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
            offset_x, offset_y = 0, 0
        
        if screenshot is None:
            return False, 0, 0, None
        
        # Run template matching
        result = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # Check if match is good enough
        # Convert confidence to TM_CCOEFF_NORMED threshold (it returns -1 to 1)
        threshold = confidence * 2 - 1  # Map 0.5-1.0 to 0-1 in CCOEFF space
        threshold = max(0.3, confidence - 0.2)  # More lenient threshold
        
        if max_val >= threshold:
            # Found match
            match_left = max_loc[0] + offset_x
            match_top = max_loc[1] + offset_y
            cx = match_left + tw // 2
            cy = match_top + th // 2
            match_rect = (match_left, match_top, tw, th)
            
            if DEBUG_IMAGE:
                print(f"[CV2] Match found: score={max_val:.3f}, center=({cx},{cy})")
            
            return True, cx, cy, match_rect
        else:
            if DEBUG_IMAGE:
                print(f"[CV2] No match: best score={max_val:.3f} < threshold={threshold:.3f}")
            return False, 0, 0, None
            
    except Exception as e:
        print(f"[CV2] Error in template matching: {e}")
        return False, 0, 0, None

def find_image_with_verification(image_path, confidence, move_to_image, search_region=None, 
                                  max_attempts=5, verify_tolerance=0.6):
    """
    Find image with verification and retry logic.
    
    Args:
        image_path: Path to template image
        confidence: Confidence threshold for initial match
        move_to_image: Whether to move cursor to the found location
        search_region: Optional (x1,y1,x2,y2) or (left,top,w,h) region to search
        max_attempts: Maximum retry attempts
        verify_tolerance: Minimum similarity for verification
    
    Returns:
        (found: bool, final_center: tuple or None)
    """
    template, tw, th = load_template(image_path)
    if template is None:
        print(f"[Find] Could not load template: {image_path}")
        return False, None
    
    # Normalize region once
    region = normalize_region(search_region)
    
    # Log diagnostics
    if DEBUG_IMAGE:
        print(f"[Find] Template: {image_path} ({tw}x{th})")
        print(f"[Find] Confidence: {confidence}, Move: {move_to_image}, Region: {region}")
        sx, sy = get_dpi_scale_factor()
        print(f"[Find] DPI scale: ({sx:.2f}, {sy:.2f})")
    
    best_score = 0.0
    best_center = None
    
    for attempt in range(1, max_attempts + 1):
        if DEBUG_IMAGE:
            print(f"[Find] Attempt {attempt}/{max_attempts}")
        
        # Try pyautogui first
        try:
            if region:
                location = pyautogui.locateOnScreen(image_path, confidence=confidence, region=region)
            else:
                location = pyautogui.locateOnScreen(image_path, confidence=confidence)
        except Exception as e:
            print(f"[Find] pyautogui error: {e}")
            location = None
        
        # If pyautogui failed, try cv2 method
        if location is None:
            found, cx, cy, match_rect = find_image_cv2(image_path, confidence, search_region)
            if found:
                location = match_rect  # Use as pseudo-location
            else:
                if DEBUG_IMAGE:
                    print(f"[Find] No candidate found on attempt {attempt}")
                continue
        else:
            cx, cy = pyautogui.center(location)
            cx, cy = int(cx), int(cy)
        
        if DEBUG_IMAGE:
            print(f"[Find] Candidate center: ({cx}, {cy})")
            print(f"[Find] Location box: {location}")
        
        # Move cursor if requested
        if move_to_image:
            pyautogui.moveTo(cx, cy)
            time.sleep(0.05)  # Small delay for cursor to settle
            
            # Get actual cursor position
            actual_x, actual_y = pyautogui.position()
            if DEBUG_IMAGE:
                print(f"[Find] Cursor after moveTo: ({actual_x}, {actual_y})")
                if abs(actual_x - cx) > 2 or abs(actual_y - cy) > 2:
                    print(f"[Find] WARNING: Cursor drift detected! Expected ({cx},{cy}), got ({actual_x},{actual_y})")
        
        # Verify the match
        verified, score, patch = verify_cursor_over_template(cx, cy, image_path, verify_tolerance)
        
        if DEBUG_IMAGE:
            save_debug_artifacts(image_path, template, patch, cx, cy, location, score, attempt)
        
        if score > best_score:
            best_score = score
            best_center = (cx, cy)
        
        if verified:
            print(f"[Find] SUCCESS: Verified match at ({cx},{cy}) with score {score:.3f}")
            return True, (cx, cy)
        else:
            print(f"[Find] Verification failed: score={score:.3f} < tolerance={verify_tolerance}")
            
            # Try local refinement if we have a candidate
            if score > 0.3 and attempt < max_attempts:
                # Search in neighborhood around current position
                refined_found, ref_cx, ref_cy, _ = find_image_cv2(
                    image_path, confidence * 0.9, 
                    search_region=(cx - 100, cy - 100, cx + 100, cy + 100)
                )
                if refined_found:
                    if DEBUG_IMAGE:
                        print(f"[Find] Refinement found better position: ({ref_cx}, {ref_cy})")
                    cx, cy = ref_cx, ref_cy
                    
                    if move_to_image:
                        pyautogui.moveTo(cx, cy)
                        time.sleep(0.05)
                    
                    verified2, score2, patch2 = verify_cursor_over_template(cx, cy, image_path, verify_tolerance)
                    if verified2:
                        print(f"[Find] SUCCESS after refinement: ({cx},{cy}) score={score2:.3f}")
                        return True, (cx, cy)
    
    # Exhausted attempts
    print(f"[Find] FAILED after {max_attempts} attempts. Best score: {best_score:.3f} at {best_center}")
    
    # If we got close, return best position anyway (for backwards compatibility)
    if best_score > verify_tolerance * 0.8 and best_center:
        print(f"[Find] Returning best candidate despite not fully verified")
        if move_to_image:
            pyautogui.moveTo(best_center[0], best_center[1])
        return True, best_center
    
    return False, None


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
        elif cmd_type == "keyboard_sequence":
            return KeyboardSequenceCommand(data.get("sequence", []))
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
                data.get("image_paths", None),
                data.get("search_region", None)
            )
        elif cmd_type == "find_image":
            return FindImageCommand(data.get("image_path"), data.get("confidence", 0.8), data.get("search_region", None))
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
        elif cmd_type == "sound":
            cmd = SoundCommand(data.get("sound_type", "beep"))
            cmd.custom_sound = data.get("custom_sound")
            return cmd
        elif cmd_type == "clipboard_clear":
            return ClipboardClearCommand()
        elif cmd_type == "clipboard_set":
            return ClipboardSetCommand(data.get("value", ""))
        elif cmd_type == "clipboard_increment":
            return ClipboardIncrementCommand()
        elif cmd_type == "clipboard_copy":
            return ClipboardCopyCommand()
        elif cmd_type == "clipboard_paste":
            return ClipboardPasteCommand()
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


class KeyboardSequenceCommand(MacroCommand):
    def __init__(self, sequence=None):
        super().__init__("keyboard_sequence")
        self.sequence = sequence or []  # List of (key, action) tuples
    
    def execute(self, context):
        for key, action in self.sequence:
            if action == "press":
                keyboard.press_and_release(key)
            elif action == "hold":
                keyboard.press(key)
            elif action == "release":
                keyboard.release(key)
    
    def to_dict(self):
        return {"type": self.command_type, "sequence": self.sequence}
    
    def __str__(self):
        if not self.sequence:
            return "KEYBOARD SEQUENCE: (empty)"
        actions = [f"{key} {action}" for key, action in self.sequence]
        return f"KEYBOARD: {', '.join(actions)}"


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
    def __init__(self, image_path, confidence=0.8, move_to_image=False, image_paths=None, search_region=None):
        super().__init__("if_image")
        # Support both single image (legacy) and multiple images
        if image_paths is None:
            self.image_paths = [{"path": image_path, "confidence": confidence}]
        else:
            self.image_paths = image_paths
        self.image_path = image_path  # Keep for backwards compatibility
        self.confidence = confidence  # Keep for backwards compatibility
        self.move_to_image = move_to_image
        self.search_region = search_region  # (x1, y1, x2, y2) or None for full screen
    
    def execute(self, context):
        # Returns True/False for if statement
        result = self.find_image(context)
        return result
    
    def find_image(self, context=None):
        """
        Find any of the configured images on screen.
        Uses verification to ensure cursor lands on correct spot.
        """
        if context is None:
            context = {}
        
        # Try to find any of the images
        for img_info in self.image_paths:
            try:
                image_path = img_info["path"]
                confidence = img_info.get("confidence", self.confidence)
                
                # Use the new verified find function
                found, center = find_image_with_verification(
                    image_path=image_path,
                    confidence=confidence,
                    move_to_image=self.move_to_image,
                    search_region=self.search_region,
                    max_attempts=5,
                    verify_tolerance=0.55  # Slightly lower tolerance for flexibility
                )
                
                if found:
                    # Store match info in context for debugging
                    context["last_image_match"] = {
                        "template_path": image_path,
                        "center": center,
                        "confidence": confidence,
                        "move_to_image": self.move_to_image
                    }
                    return True
                    
            except Exception as e:
                print(f"[IfImage] Error finding image {img_info.get('path', 'unknown')}: {e}")
                import traceback
                traceback.print_exc()
        
        return False
    
    def to_dict(self):
        return {
            "type": self.command_type, 
            "image_path": self.image_path, 
            "confidence": self.confidence, 
            "move_to_image": self.move_to_image,
            "image_paths": self.image_paths,
            "search_region": self.search_region
        }
    
    def get_image_name(self):
        """Get just the filename without path"""
        if len(self.image_paths) == 1:
            return Path(self.image_paths[0]["path"]).stem
        else:
            return f"{Path(self.image_paths[0]['path']).stem} +{len(self.image_paths)-1} more"
    
    def __str__(self):
        move_indicator = " [MOVE]" if self.move_to_image else ""
        region_indicator = " [REGION]" if self.search_region else ""
        return f"IF IMAGE{move_indicator}{region_indicator}: {self.get_image_name()}"


class FindImageCommand(MacroCommand):
    def __init__(self, image_path, confidence=0.8, search_region=None):
        super().__init__("find_image")
        self.image_path = image_path
        self.confidence = confidence
        self.search_region = search_region  # (x1, y1, x2, y2) or None for full screen
    
    def execute(self, context):
        """
        Find image and move cursor to it with verification.
        """
        try:
            # Use the new verified find function
            found, center = find_image_with_verification(
                image_path=self.image_path,
                confidence=self.confidence,
                move_to_image=True,  # FindImageCommand always moves
                search_region=self.search_region,
                max_attempts=5,
                verify_tolerance=0.55
            )
            
            if found and context is not None:
                context["last_image_match"] = {
                    "template_path": self.image_path,
                    "center": center,
                    "confidence": self.confidence
                }
            
            return found
            
        except Exception as e:
            print(f"[FindImage] Error finding image: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def to_dict(self):
        return {"type": self.command_type, "image_path": self.image_path, "confidence": self.confidence, "search_region": self.search_region}
    
    def get_image_name(self):
        """Get just the filename without path"""
        return Path(self.image_path).stem
    
    def __str__(self):
        region_indicator = " [REGION]" if self.search_region else ""
        return f"FIND IMAGE{region_indicator}: {self.get_image_name()} (conf: {self.confidence})"


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


class SoundCommand(MacroCommand):
    def __init__(self, sound_type="beep"):
        super().__init__("sound")
        self.sound_type = sound_type  # "beep", "asterisk", "exclamation", "hand", "question", "custom"
        self.custom_sound = None  # Path to custom sound file if sound_type == "custom"
    
    def execute(self, context):
        if self.sound_type == "beep":
            import winsound
            winsound.Beep(800, 200)  # Frequency 800Hz, duration 200ms
        elif self.sound_type == "asterisk":
            import winsound
            winsound.MessageBeep(winsound.MB_ICONASTERISK)
        elif self.sound_type == "exclamation":
            import winsound
            winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
        elif self.sound_type == "hand":
            import winsound
            winsound.MessageBeep(winsound.MB_ICONHAND)
        elif self.sound_type == "question":
            import winsound
            winsound.MessageBeep(winsound.MB_ICONQUESTION)
        elif self.sound_type == "custom" and self.custom_sound:
            import winsound
            try:
                winsound.PlaySound(self.custom_sound, winsound.SND_FILENAME)
            except:
                # Fallback to beep if custom sound fails
                winsound.Beep(800, 200)
    
    def to_dict(self):
        return {"type": self.command_type, "sound_type": self.sound_type, "custom_sound": self.custom_sound}
    
    def __str__(self):
        if self.sound_type == "custom" and self.custom_sound:
            return f"SOUND: {self.sound_type} ({self.custom_sound})"
        return f"SOUND: {self.sound_type}"


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
    
    def execute(self, macro, repeat_count=1, progress_callback=None):
        """Execute a macro with repeat support"""
        self.running = True
        self.stop_flag = False
        
        if repeat_count == 0:
            # Infinite loop
            while not self.stop_flag:
                self._run_commands(macro.commands, progress_callback, 0)
        else:
            for _ in range(repeat_count):
                if self.stop_flag:
                    break
                self._run_commands(macro.commands, progress_callback, 0)
        
        self.running = False
    
    def _run_commands(self, commands, progress_callback=None, base_index=0):
        """Run a list of commands with control flow support"""
        context = {}
        i = 0
        
        while i < len(commands) and not self.stop_flag:
            cmd = commands[i]
            
            # Call progress callback with global index
            if progress_callback:
                progress_callback(base_index + i)
            
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
                        self._run_commands(repeat_commands, progress_callback, base_index + i + 1)
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
        
        self.command_canvas = tk.Canvas(list_frame, yscrollcommand=scrollbar.set, bg="white", highlightthickness=1, takefocus=1)
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
        
        # Bind left-click on canvas to focus and deselect
        self.command_canvas.bind("<Button-1>", self.on_canvas_click)
        
        # Bind right-click to canvas and frame for context menu
        self.command_canvas.bind("<Button-3>", self.show_context_menu)
        self.command_frame.bind("<Button-3>", self.show_context_menu)
        
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
        # Bind Ctrl+Arrow/Home/End for moving commands
        self.command_canvas.bind("<Control-Up>", self.move_selected_up)
        self.command_canvas.bind("<Control-Down>", self.move_selected_down)
        self.command_canvas.bind("<Control-Home>", self.move_selected_to_start)
        self.command_canvas.bind("<Control-End>", self.move_selected_to_end)
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
        self.root.bind("<Delete>", lambda e: self.remove_command())
        
        # Command buttons
        btn_frame = ttk.Frame(middle_frame)
        btn_frame.pack(fill=tk.X)
        
        ttk.Button(btn_frame, text="Add Keyboard Action", command=self.add_keyboard_sequence).pack(side=tk.LEFT, padx=2)
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
        ttk.Button(btn_frame3, text="Add Sound", command=self.add_sound).pack(side=tk.LEFT, padx=2)
        
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
    
    def highlight_command(self, index):
        """Highlight the command at the given index during execution"""
        # Reset all frames to default color
        for frame in self.command_frame.winfo_children():
            if hasattr(frame, '_command_index'):
                frame.config(bg="white")
                for child in frame.winfo_children():
                    child.config(bg="white")
        
        # Highlight the current command if index is valid
        if index >= 0 and 0 <= index < len(self.command_frame.winfo_children()):
            frames = self.command_frame.winfo_children()
            for frame in frames:
                if hasattr(frame, '_command_index') and frame._command_index == index:
                    frame.config(bg="dodgerblue")
                    for child in frame.winfo_children():
                        child.config(bg="dodgerblue")
                    # Scroll to make the highlighted command visible
                    self.command_canvas.update_idletasks()
                    y = frame.winfo_y()
                    canvas_bbox = self.command_canvas.bbox("all")
                    if canvas_bbox:
                        total_height = canvas_bbox[3]
                        if total_height > 0:
                            fraction = y / total_height
                            self.command_canvas.yview_moveto(fraction)
                    break
    
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
                data.get('image_paths', None),
                data.get('search_region', None)
            )
        elif cmd_type == 'find_image':
            return FindImageCommand(data['image_path'], data.get('confidence', 0.8), data.get('search_region', None))
        elif cmd_type == 'message':
            return MessageCommand(data['text'], data.get('always_on_top', False), data.get('always_focused', False))
        elif cmd_type == 'repeat':
            return RepeatCommand(data['count'])
        elif cmd_type == 'end_repeat':
            return EndRepeatCommand()
        elif cmd_type == 'delay':
            return DelayCommand(data['seconds'])
        elif cmd_type == 'delay_ms':
            return DelayMsCommand(data.get('milliseconds', 100))
        elif cmd_type == 'wait_for_window':
            return WaitForWindowCommand(data.get('window_pattern', '*'), data.get('timeout', 30))
        elif cmd_type == 'sound':
            cmd = SoundCommand(data.get('sound_type', 'beep'))
            cmd.custom_sound = data.get('custom_sound')
            return cmd
        elif cmd_type == 'keyboard_sequence':
            return KeyboardSequenceCommand(data.get('sequence', []))
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
    
    def on_canvas_click(self, event):
        """Handle click on canvas (empty area)"""
        # Check if click was on a command frame
        widget = self.command_canvas.winfo_containing(event.x_root, event.y_root)
        if widget:
            # Traverse up to see if it's a command frame
            while widget and widget != self.command_frame:
                if hasattr(widget, '_command_index'):
                    return  # Let the command click handler deal with it
                widget = widget.master
        
        # Click was on empty area - deselect and focus canvas
        self.selected_index = None
        self.update_command_list()
        self.command_canvas.focus_set()
    
    def handle_single_click(self, cmd_index, y_root, widget):
        """Handle confirmed single click after delay"""
        self.click_timer = None
        
        # Update selection
        self.selected_index = cmd_index
        self.update_command_list()
        # Focus the command list panel (same as pressing F2)
        try:
            self.command_canvas.focus_set()
        except:
            pass
        
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

    def move_selected_up(self, event):
        """Move the selected command up by one"""
        if not self.current_macro or self.selected_index is None:
            return "break"
        if self.selected_index > 0:
            self.save_state()
            idx = self.selected_index
            self.current_macro.commands[idx-1], self.current_macro.commands[idx] = (
                self.current_macro.commands[idx], self.current_macro.commands[idx-1]
            )
            self.selected_index = idx - 1
            self.update_command_list()
            self.save_macros()
            self.scroll_to_selected()
        return "break"

    def move_selected_down(self, event):
        """Move the selected command down by one"""
        if not self.current_macro or self.selected_index is None:
            return "break"
        if self.selected_index < len(self.current_macro.commands) - 1:
            self.save_state()
            idx = self.selected_index
            self.current_macro.commands[idx+1], self.current_macro.commands[idx] = (
                self.current_macro.commands[idx], self.current_macro.commands[idx+1]
            )
            self.selected_index = idx + 1
            self.update_command_list()
            self.save_macros()
            self.scroll_to_selected()
        return "break"

    def move_selected_to_start(self, event):
        """Move the selected command to the start of the macro"""
        if not self.current_macro or self.selected_index is None:
            return "break"
        if self.selected_index > 0:
            self.save_state()
            cmd = self.current_macro.commands.pop(self.selected_index)
            self.current_macro.commands.insert(0, cmd)
            self.selected_index = 0
            self.update_command_list()
            self.save_macros()
            self.scroll_to_selected()
        return "break"

    def move_selected_to_end(self, event):
        """Move the selected command to the end of the macro"""
        if not self.current_macro or self.selected_index is None:
            return "break"
        if self.selected_index < len(self.current_macro.commands) - 1:
            self.save_state()
            cmd = self.current_macro.commands.pop(self.selected_index)
            self.current_macro.commands.append(cmd)
            self.selected_index = len(self.current_macro.commands) - 1
            self.update_command_list()
            self.save_macros()
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
        if not self.current_macro:
            return
        
        # Create a mock event with the position
        class MockEvent:
            def __init__(self, x, y, widget):
                self.x_root = x
                self.y_root = y
                self.widget = widget
        
        # If there's a selected frame, position menu at that frame
        if self.selected_index is not None:
            for child in self.command_frame.winfo_children():
                if hasattr(child, '_command_index') and child._command_index == self.selected_index:
                    # Calculate screen position for menu
                    frame_x = child.winfo_rootx()
                    frame_y = child.winfo_rooty() + child.winfo_height() // 2
                    self.show_context_menu(MockEvent(frame_x, frame_y, child))
                    break
        else:
            # No selection - show menu at canvas position
            canvas_x = self.command_canvas.winfo_rootx() + 20
            canvas_y = self.command_canvas.winfo_rooty() + 20
            self.show_context_menu(MockEvent(canvas_x, canvas_y, self.command_canvas))
        
        return "break"
    
    def show_context_menu(self, event):
        """Show right-click context menu"""
        if not self.current_macro:
            return
        
        # Find which command frame was clicked
        widget = event.widget
        clicked_command = None
        while widget and not hasattr(widget, '_command_index'):
            widget = widget.master
            if widget == self.command_frame:
                break
        
        if widget and hasattr(widget, '_command_index'):
            clicked_command = widget._command_index
        
        # Set selection based on what was clicked
        if clicked_command is not None:
            self.selected_index = clicked_command
            self.update_command_list()
        # If clicked on empty space, don't change selection
        
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
        keyboard_menu.add_command(label="Sequence", underline=0, command=lambda: self.insert_command_at_selection("keyboard_sequence"))
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
        timing_menu.add_command(label="Sound", underline=0, command=lambda: self.insert_command_at_selection("sound"))
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
        
        # Delegate to appropriate add_ method with insert_index
        command_map = {
            "mouse_click": self.add_mouse_click,
            "mouse_click_absolute": self.add_mouse_click_absolute,
            "mouse_move": self.add_mouse_move,
            "key_press": self.add_key_press,
            "key_hold": self.add_key_hold,
            "key_release": self.add_key_release,
            "keyboard_sequence": self.add_keyboard_sequence,
            "if_statement": self.add_if,
            "else": self.add_else,
            "end_if": self.add_endif,
            "repeat": self.add_repeat,
            "end_repeat": self.add_end_repeat,
            "delay": self.add_delay,
            "delay_ms": self.add_delay_ms,
            "sound": self.add_sound,
        }
        
        # Special handlers for commands with custom logic
        if command_type == "if_image":
            self.add_if_image_at_index(insert_index)
        elif command_type == "find_image":
            self.add_find_image_at_index(insert_index)
        elif command_type == "clipboard_clear":
            self.add_clipboard_clear(insert_index)
        elif command_type == "clipboard_set":
            self.add_clipboard_set(insert_index)
        elif command_type == "clipboard_increment":
            self.add_clipboard_increment(insert_index)
        elif command_type == "clipboard_copy":
            self.add_clipboard_copy(insert_index)
        elif command_type == "clipboard_paste":
            self.add_clipboard_paste(insert_index)
        elif command_type == "wait_for_window":
            self.add_wait_for_window_at_index(insert_index)
        elif command_type == "message":
            self.add_message_at_index(insert_index)
        elif command_type in command_map:
            command_map[command_type](insert_index)
    
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
        
        # Center the dialog after packing all widgets
        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        screen_width = dialog.winfo_screenwidth()
        screen_height = dialog.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        dialog.geometry(f"{width}x{height}+{x}+{y}")
        
        self.ensure_dialog_focused(dialog)
    
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
    
    def show_sound_selector(self):
        """Show dialog to select a sound type"""
        sound_types = [
            ("beep", "Simple beep sound"),
            ("asterisk", "Asterisk (system sound)"),
            ("exclamation", "Exclamation (system sound)"),
            ("hand", "Hand (error sound)"),
            ("question", "Question (system sound)"),
            ("custom", "Custom sound file...")
        ]
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Sound")
        dialog.transient(self.root)
        dialog.grab_set()
        
        result = {'command': None}
        
        tk.Label(dialog, text="Choose a sound to play:", font=("Arial", 10)).pack(pady=10, padx=20)
        
        # Sound type selection
        sound_var = tk.StringVar(value="beep")
        
        def play_sound(sound_type):
            """Play a preview of the selected sound type"""
            if sound_type == "beep":
                import winsound
                winsound.Beep(800, 200)
            elif sound_type == "asterisk":
                import winsound
                winsound.MessageBeep(winsound.MB_ICONASTERISK)
            elif sound_type == "exclamation":
                import winsound
                winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
            elif sound_type == "hand":
                import winsound
                winsound.MessageBeep(winsound.MB_ICONHAND)
            elif sound_type == "question":
                import winsound
                winsound.MessageBeep(winsound.MB_ICONQUESTION)
            elif sound_type == "custom":
                # For custom, try to play the selected file, or play a default beep
                custom_path = custom_entry.get().strip()
                if custom_path:
                    try:
                        import winsound
                        winsound.PlaySound(custom_path, winsound.SND_FILENAME)
                    except:
                        # Fallback to beep
                        winsound.Beep(800, 200)
                else:
                    # No file selected, play default beep
                    import winsound
                    winsound.Beep(800, 200)
        
        for sound_type, description in sound_types:
            # Create a frame for each sound option
            option_frame = ttk.Frame(dialog)
            option_frame.pack(fill=tk.X, padx=20, pady=2)
            
            # Radio button
            ttk.Radiobutton(
                option_frame, 
                text=f"{sound_type.title()}: {description}", 
                variable=sound_var, 
                value=sound_type
            ).pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            # Play button
            play_btn = ttk.Button(
                option_frame, 
                text="▶", 
                width=3, 
                command=lambda st=sound_type: play_sound(st)
            )
            play_btn.pack(side=tk.RIGHT)
        
        # Custom sound file entry (initially hidden)
        custom_frame = ttk.Frame(dialog)
        custom_label = ttk.Label(custom_frame, text="Sound file path:")
        custom_entry = ttk.Entry(custom_frame, width=40)
        custom_browse = ttk.Button(custom_frame, text="Browse...", command=lambda: self.browse_sound_file(custom_entry))
        
        def on_sound_change(*args):
            if sound_var.get() == "custom":
                custom_frame.pack(pady=5, padx=20, fill=tk.X)
                custom_entry.focus_set()
            else:
                custom_frame.pack_forget()
        
        sound_var.trace_add("write", on_sound_change)
        
        custom_label.pack(side=tk.LEFT)
        custom_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        custom_browse.pack(side=tk.RIGHT)
        
        def on_ok():
            sound_type = sound_var.get()
            custom_sound = None
            if sound_type == "custom":
                custom_sound = custom_entry.get().strip()
                if not custom_sound:
                    messagebox.showwarning("Warning", "Please select a sound file")
                    return
            
            cmd = SoundCommand(sound_type)
            cmd.custom_sound = custom_sound
            result['command'] = cmd
            dialog.destroy()
        
        def on_cancel():
            dialog.destroy()
        
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
        
        dialog.wait_window()
        return result['command']
    
    def browse_sound_file(self, entry):
        """Browse for a sound file"""
        from tkinter import filedialog
        filename = filedialog.askopenfilename(
            title="Select Sound File",
            filetypes=[
                ("Sound files", "*.wav *.mp3 *.ogg"),
                ("WAV files", "*.wav"),
                ("MP3 files", "*.mp3"),
                ("OGG files", "*.ogg"),
                ("All files", "*.*")
            ]
        )
        if filename:
            entry.delete(0, tk.END)
            entry.insert(0, filename)
    
    def show_keyboard_sequence_dialog(self):
        """Show dialog to create a keyboard sequence command"""
        # Comprehensive list of all keyboard keys (same as show_key_selector)
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
        
        actions = ['press', 'hold', 'release']
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Keyboard Sequence")
        dialog.transient(self.root)
        dialog.grab_set()
        
        result = {'sequence': []}
        
        tk.Label(dialog, text="Build your keyboard sequence:", font=("Arial", 10)).pack(pady=10, padx=20)
        
        # Frame for the sequence items
        sequence_frame = ttk.Frame(dialog)
        sequence_frame.pack(pady=5, padx=20, fill=tk.BOTH, expand=True)
        
        # List to hold the row frames
        row_frames = []
        current_row = 0  # Track which row has focus
        
        def add_sequence_row(key='', action='press'):
            """Add a new row for key-action pair"""
            row_frame = ttk.Frame(sequence_frame)
            row_frame.pack(fill=tk.X, pady=2)
            
            # Key dropdown
            key_var = tk.StringVar(value=key)
            key_combo = ttk.Combobox(row_frame, textvariable=key_var, values=all_keys, width=15, state='readonly')
            key_combo.pack(side=tk.LEFT, padx=2)
            
            # Type-ahead search state for this combo
            search_state = {
                'last_key': '',
                'last_time': 0,
                'current_index': -1
            }
            
            def on_key_press(event):
                import time
                
                # Ignore special keys
                if len(event.char) != 1 or not event.char.isalnum() and event.char not in "'-.":
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
                        key_combo.current(i)
                        search_state['current_index'] = i
                        found = True
                        break
                
                # If no match found after current position, wrap to beginning
                if not found and start_index > 0:
                    for i in range(0, start_index):
                        if all_keys[i].lower().startswith(char):
                            key_combo.current(i)
                            search_state['current_index'] = i
                            found = True
                            break
                
                search_state['last_key'] = char
                search_state['last_time'] = current_time
                
                return "break"  # Prevent default typing behavior
            
            # Bind key press for type-ahead search
            key_combo.bind('<KeyPress>', on_key_press)
            
            # Action selection (using buttons for left/right as requested)
            action_var = tk.StringVar(value=action)
            
            def select_action(new_action):
                action_var.set(new_action)
                update_action_buttons()
            
            action_frame = ttk.Frame(row_frame)
            action_frame.pack(side=tk.LEFT, padx=5)
            
            left_btn = ttk.Button(action_frame, text="◀", width=3, command=lambda: select_action(actions[(actions.index(action_var.get()) - 1) % len(actions)]))
            action_label = ttk.Label(action_frame, textvariable=action_var, width=8, anchor='center')
            right_btn = ttk.Button(action_frame, text="▶", width=3, command=lambda: select_action(actions[(actions.index(action_var.get()) + 1) % len(actions)]))
            
            left_btn.pack(side=tk.LEFT)
            action_label.pack(side=tk.LEFT)
            right_btn.pack(side=tk.LEFT)
            
            def update_action_buttons():
                current = action_var.get()
                idx = actions.index(current)
                # Enable/disable based on position
                left_btn.config(state='normal')
                right_btn.config(state='normal')
            
            update_action_buttons()
            
            # Focus tracking
            def on_combo_focus(event):
                current_row = row_frames.index((row_frame, key_var, action_var, key_combo))
            
            key_combo.bind('<FocusIn>', on_combo_focus)
            
            row_frames.append((row_frame, key_var, action_var, key_combo))
            return row_frame
            """Remove a row"""
            for i, (rf, kv, av, kc) in enumerate(row_frames):
                if rf == row_frame:
                    rf.destroy()
                    del row_frames[i]
                    break
        
        # Add initial row
        add_sequence_row()
        
        # Focus the first combo
        if row_frames:
            _, _, _, first_combo = row_frames[0]
            dialog.after(100, lambda: first_combo.focus_set())
        
        # Add key button
        def add_and_focus():
            add_sequence_row()
            if row_frames:
                _, _, _, last_combo = row_frames[-1]
                last_combo.focus_set()
        
        ttk.Button(dialog, text="Add Key", command=add_and_focus).pack(pady=5)
        
        # Keyboard navigation
        def change_action(direction):
            if row_frames:
                _, _, action_var, _ = row_frames[current_row]
                current_action = action_var.get()
                idx = actions.index(current_action)
                new_idx = (idx + direction) % len(actions)
                action_var.set(actions[new_idx])
        
        def move_focus(direction):
            if row_frames:
                nonlocal current_row
                current_row = (current_row + direction) % len(row_frames)
                _, _, _, key_combo = row_frames[current_row]
                key_combo.focus_set()
        
        # Bind arrow keys for navigation
        dialog.bind('<Left>', lambda e: change_action(-1))
        dialog.bind('<Right>', lambda e: change_action(1))
        dialog.bind('<Up>', lambda e: move_focus(-1))
        dialog.bind('<Down>', lambda e: move_focus(1))
        
        def on_ok():
            sequence = []
            for _, key_var, action_var, _ in row_frames:
                key = key_var.get()
                action = action_var.get()
                if key:
                    sequence.append((key, action))
            if sequence:
                result['sequence'] = sequence
                dialog.destroy()
            else:
                messagebox.showwarning("Warning", "Please add at least one key action")
        
        def on_cancel():
            dialog.destroy()
        
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
        
        dialog.wait_window()
        return result['sequence']
    
    def add_key_press(self, insert_index=None):
        """Add key press command"""
        if not self.current_macro:
            messagebox.showwarning("Warning", "Please select a macro first")
            return
        
        key = self.show_key_selector("Key Press")
        if key:
            self.save_state()
            if insert_index is not None:
                self.current_macro.commands.insert(insert_index, KeyPressCommand(key))
                self.selected_index = insert_index
            else:
                self.current_macro.add_command(KeyPressCommand(key))
                self.selected_index = len(self.current_macro.commands) - 1
            self.update_command_list()
            self.save_macros()
    
    def add_key_hold(self, insert_index=None):
        """Add key hold command"""
        if not self.current_macro:
            messagebox.showwarning("Warning", "Please select a macro first")
            return
        
        key = self.show_key_selector("Key Hold")
        if key:
            self.save_state()
            if insert_index is not None:
                self.current_macro.commands.insert(insert_index, KeyHoldCommand(key))
                self.selected_index = insert_index
            else:
                self.current_macro.add_command(KeyHoldCommand(key))
                self.selected_index = len(self.current_macro.commands) - 1
            self.update_command_list()
            self.save_macros()
    
    def add_key_release(self, insert_index=None):
        """Add key release command"""
        if not self.current_macro:
            messagebox.showwarning("Warning", "Please select a macro first")
            return
        
        key = self.show_key_selector("Key Release")
        if key:
            self.save_state()
            if insert_index is not None:
                self.current_macro.commands.insert(insert_index, KeyReleaseCommand(key))
                self.selected_index = insert_index
            else:
                self.current_macro.add_command(KeyReleaseCommand(key))
                self.selected_index = len(self.current_macro.commands) - 1
            self.update_command_list()
            self.save_macros()
    
    def add_keyboard_sequence(self, insert_index=None):
        """Add keyboard sequence command"""
        if not self.current_macro:
            messagebox.showwarning("Warning", "Please select a macro first")
            return
        
        sequence = self.show_keyboard_sequence_dialog()
        if sequence:
            self.save_state()
            if insert_index is not None:
                self.current_macro.commands.insert(insert_index, KeyboardSequenceCommand(sequence))
                self.selected_index = insert_index
            else:
                self.current_macro.add_command(KeyboardSequenceCommand(sequence))
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
                return screenshot, (x1, y1, x2, y2)
        
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
    
    def add_mouse_click(self, insert_index=None):
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
                cmd = None
                if mode_result == "in_place":
                    cmd = MouseClickCommand(button, 0, 0, mode="in_place")
                elif mode_result == "offset":
                    offset_x = simpledialog.askinteger("X Offset", "Enter X offset (pixels, can be negative):", initialvalue=0)
                    if offset_x is not None:
                        offset_y = simpledialog.askinteger("Y Offset", "Enter Y offset (pixels, can be negative):", initialvalue=0)
                        if offset_y is not None:
                            cmd = MouseClickCommand(button, 0, 0, mode="offset", offset_x=offset_x, offset_y=offset_y)
                        else:
                            return
                    else:
                        return
                
                if cmd:
                    if insert_index is not None:
                        self.current_macro.commands.insert(insert_index, cmd)
                        self.selected_index = insert_index
                    else:
                        self.current_macro.add_command(cmd)
                        self.selected_index = len(self.current_macro.commands) - 1
                    self.update_command_list()
                    self.save_macros()
    
    def add_mouse_click_absolute(self, insert_index=None):
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
                cmd = MouseClickCommand(button, captured_pos["x"], captured_pos["y"], mode="absolute")
                if insert_index is not None:
                    self.current_macro.commands.insert(insert_index, cmd)
                    self.selected_index = insert_index
                else:
                    self.current_macro.add_command(cmd)
                    self.selected_index = len(self.current_macro.commands) - 1
                self.update_command_list()
                self.save_macros()
            else:
                messagebox.showwarning("Timeout", "Position capture timed out. No command was added.")
    
    def add_mouse_move(self, insert_index=None):
        """Add mouse move command"""
        if not self.current_macro:
            messagebox.showwarning("Warning", "Please select a macro first")
            return
        
        x, y = self.get_mouse_position()
        if x is not None:
            self.save_state()
            cmd = MouseMoveCommand(x, y)
            if insert_index is not None:
                self.current_macro.commands.insert(insert_index, cmd)
                self.selected_index = insert_index
            else:
                self.current_macro.add_command(cmd)
                self.selected_index = len(self.current_macro.commands) - 1
            self.update_command_list()
            self.save_macros()
    
    def add_if(self, insert_index=None):
        """Add IF statement"""
        if not self.current_macro:
            messagebox.showwarning("Warning", "Please select a macro first")
            return
        
        condition = simpledialog.askstring("IF Statement", "Enter condition (e.g., clipboard == 'value'):")
        if condition:
            self.save_state()
            cmd = IfStatementCommand(condition)
            if insert_index is not None:
                self.current_macro.commands.insert(insert_index, cmd)
                # Auto-insert ENDIF right after IF
                self.current_macro.commands.insert(insert_index + 1, EndIfStatementCommand())
                self.selected_index = insert_index
            else:
                self.current_macro.add_command(cmd)
                # Auto-insert ENDIF right after IF
                self.current_macro.commands.insert(len(self.current_macro.commands), EndIfStatementCommand())
                self.selected_index = len(self.current_macro.commands) - 2
            self.update_command_list()
            self.save_macros()
    
    def add_else(self, insert_index=None):
        """Add ELSE statement"""
        if not self.current_macro:
            messagebox.showwarning("Warning", "Please select a macro first")
            return
        
        self.save_state()
        cmd = ElseStatementCommand()
        if insert_index is not None:
            self.current_macro.commands.insert(insert_index, cmd)
            # Auto-insert ENDIF right after ELSE
            self.current_macro.commands.insert(insert_index + 1, EndIfStatementCommand())
            self.selected_index = insert_index
        else:
            self.current_macro.add_command(cmd)
            # Auto-insert ENDIF right after ELSE
            self.current_macro.commands.insert(len(self.current_macro.commands), EndIfStatementCommand())
            self.selected_index = len(self.current_macro.commands) - 2
        self.update_command_list()
        self.save_macros()
    
    def add_endif(self, insert_index=None):
        """Add ENDIF statement"""
        if not self.current_macro:
            messagebox.showwarning("Warning", "Please select a macro first")
            return
        
        self.save_state()
        cmd = EndIfStatementCommand()
        if insert_index is not None:
            self.current_macro.commands.insert(insert_index, cmd)
            self.selected_index = insert_index
        else:
            self.current_macro.add_command(cmd)
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
        result = self.capture_screen_region()
        if result:
            screenshot, _ = result
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
            
            # Search within area checkbox
            search_area_var = tk.BooleanVar(value=False)
            tk.Checkbutton(dialog, text="Search within area", variable=search_area_var).pack(pady=10)
            
            # Buttons
            def save_options():
                search_region = None
                if search_area_var.get():
                    # Capture search area
                    messagebox.showinfo("Search Area", "Click and drag to select the area to search within.")
                    result = self.capture_screen_region()
                    if result:
                        search_screenshot, search_region = result
                    else:
                        messagebox.showwarning("Warning", "No search area selected. Searching entire screen.")
                        search_region = None
                
                self.save_state()
                self.current_macro.add_command(IfImageCommand(image_path, conf_var.get(), move_var.get(), None, search_region))
                # Auto-insert ENDIF right after IF IMAGE
                self.current_macro.commands.insert(len(self.current_macro.commands), EndIfStatementCommand())
                # Select IF IMAGE (not ENDIF)
                self.selected_index = len(self.current_macro.commands) - 2
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
        
        result = self.capture_screen_region()
        if result:
            screenshot, _ = result
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
                # Auto-insert ENDIF right after IF IMAGE
                self.current_macro.commands.insert(insert_index + 1, EndIfStatementCommand())
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
        
        result = self.capture_screen_region()
        if result:
            screenshot, _ = result
            image_path = f"images/{image_name}.png"
            os.makedirs("images", exist_ok=True)
            screenshot.save(image_path)
            
            self.show_image_preview(screenshot, "Captured Image")
            
            confidence = simpledialog.askfloat("Confidence", "Enter confidence (0.0-1.0):", initialvalue=0.8)
            if confidence is not None:
                self.save_state()
                self.current_macro.commands.insert(insert_index, FindImageCommand(image_path, confidence))
                self.selected_index = insert_index
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
        result = self.capture_screen_region()
        if result:
            screenshot, _ = result
            image_path = f"images/{image_name}.png"
            os.makedirs("images", exist_ok=True)
            screenshot.save(image_path)
            
            # Show preview
            self.show_image_preview(screenshot, "Captured Image")
            
            # Create dialog for options
            dialog = tk.Toplevel(self.root)
            dialog.title("FIND IMAGE Options")
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
            
            # Search within area checkbox
            search_area_var = tk.BooleanVar(value=False)
            tk.Checkbutton(dialog, text="Search within area", variable=search_area_var).pack(pady=10)
            
            # Buttons
            def save_options():
                search_region = None
                if search_area_var.get():
                    # Capture search area
                    messagebox.showinfo("Search Area", "Click and drag to select the area to search within.")
                    result = self.capture_screen_region()
                    if result:
                        search_screenshot, search_region = result
                    else:
                        messagebox.showwarning("Warning", "No search area selected. Searching entire screen.")
                        search_region = None
                
                self.save_state()
                self.current_macro.add_command(FindImageCommand(image_path, conf_var.get(), search_region))
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
    
    def add_repeat(self, insert_index=None):
        """Add REPEAT command"""
        if not self.current_macro:
            messagebox.showwarning("Warning", "Please select a macro first")
            return
        
        count = simpledialog.askinteger("REPEAT", "Enter number of repetitions:")
        if count is not None:
            self.save_state()
            cmd = RepeatCommand(count)
            if insert_index is not None:
                self.current_macro.commands.insert(insert_index, cmd)
                self.selected_index = insert_index
            else:
                self.current_macro.add_command(cmd)
                self.selected_index = len(self.current_macro.commands) - 1
            self.update_command_list()
            self.save_macros()
    
    def add_end_repeat(self, insert_index=None):
        """Add END REPEAT command"""
        if not self.current_macro:
            messagebox.showwarning("Warning", "Please select a macro first")
            return
        
        self.save_state()
        cmd = EndRepeatCommand()
        if insert_index is not None:
            self.current_macro.commands.insert(insert_index, cmd)
            self.selected_index = insert_index
        else:
            self.current_macro.add_command(cmd)
            self.selected_index = len(self.current_macro.commands) - 1
        self.update_command_list()
        self.save_macros()
    
    def add_delay(self, insert_index=None):
        """Add delay command"""
        if not self.current_macro:
            messagebox.showwarning("Warning", "Please select a macro first")
            return
        
        seconds = simpledialog.askfloat("Delay", "Enter delay in seconds:")
        if seconds is not None:
            self.save_state()
            cmd = DelayCommand(seconds)
            if insert_index is not None:
                self.current_macro.commands.insert(insert_index, cmd)
                self.selected_index = insert_index
            else:
                self.current_macro.add_command(cmd)
                self.selected_index = len(self.current_macro.commands) - 1
            self.update_command_list()
            self.save_macros()
    
    def add_delay_ms(self, insert_index=None):
        """Add delay (milliseconds) command"""
        if not self.current_macro:
            messagebox.showwarning("Warning", "Please select a macro first")
            return
        
        milliseconds = simpledialog.askinteger("Delay (ms)", "Enter delay in milliseconds:")
        if milliseconds is not None:
            self.save_state()
            cmd = DelayMsCommand(milliseconds)
            if insert_index is not None:
                self.current_macro.commands.insert(insert_index, cmd)
                self.selected_index = insert_index
            else:
                self.current_macro.add_command(cmd)
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
    
    def add_wait_for_window_at_index(self, insert_index):
        """Add wait for window command at specific index"""
        if not self.current_macro:
            messagebox.showwarning("Warning", "Please select a macro first")
            return
        
        pattern = simpledialog.askstring("Wait For Window", "Enter window title pattern (use * as wildcard):")
        if pattern:
            timeout = simpledialog.askinteger("Timeout", "Enter timeout in seconds:", initialvalue=30)
            if timeout is not None:
                self.save_state()
                self.current_macro.commands.insert(insert_index, WaitForWindowCommand(pattern, timeout))
                self.selected_index = insert_index
                self.update_command_list()
                self.save_macros()
    
    def add_message_at_index(self, insert_index):
        """Add message command at specific index"""
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
                self.current_macro.commands.insert(insert_index, MessageCommand(text, always_on_top_var.get(), always_focused_var.get()))
                self.selected_index = insert_index
                self.update_command_list()
                self.save_macros()
                dialog.destroy()
            else:
                messagebox.showwarning("Warning", "Please enter a message")
        
        btn_frame = tk.Frame(dialog)
        btn_frame.pack(pady=20)
        tk.Button(btn_frame, text="OK", command=save_message, width=10).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Cancel", command=dialog.destroy, width=10).pack(side=tk.LEFT, padx=5)
        
        text_entry.bind("<Return>", lambda e: save_message())
        dialog.bind('<Escape>', lambda e: dialog.destroy())
        
        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        screen_width = dialog.winfo_screenwidth()
        screen_height = dialog.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        dialog.geometry(f"{width}x{height}+{x}+{y}")
        self.ensure_dialog_focused(dialog)
    
    def add_clipboard_clear(self, insert_index=None):
        """Add clipboard clear command"""
        if not self.current_macro:
            messagebox.showwarning("Warning", "Please select a macro first")
            return
        
        self.save_state()
        cmd = ClipboardClearCommand()
        if insert_index is not None:
            self.current_macro.commands.insert(insert_index, cmd)
            self.selected_index = insert_index
        else:
            self.current_macro.add_command(cmd)
            self.selected_index = len(self.current_macro.commands) - 1
        self.update_command_list()
        self.save_macros()
    
    def add_clipboard_set(self, insert_index=None):
        """Add clipboard set command"""
        if not self.current_macro:
            messagebox.showwarning("Warning", "Please select a macro first")
            return
        
        value = simpledialog.askstring("Clipboard Set", "Enter value to copy to clipboard:")
        if value is not None:
            self.save_state()
            cmd = ClipboardSetCommand(value)
            if insert_index is not None:
                self.current_macro.commands.insert(insert_index, cmd)
                self.selected_index = insert_index
            else:
                self.current_macro.add_command(cmd)
                self.selected_index = len(self.current_macro.commands) - 1
            self.update_command_list()
            self.save_macros()
    
    def add_clipboard_increment(self, insert_index=None):
        """Add clipboard increment command"""
        if not self.current_macro:
            messagebox.showwarning("Warning", "Please select a macro first")
            return
        
        self.save_state()
        cmd = ClipboardIncrementCommand()
        if insert_index is not None:
            self.current_macro.commands.insert(insert_index, cmd)
            self.selected_index = insert_index
        else:
            self.current_macro.add_command(cmd)
            self.selected_index = len(self.current_macro.commands) - 1
        self.update_command_list()
        self.save_macros()
    
    def add_clipboard_copy(self, insert_index=None):
        """Add clipboard copy command"""
        if not self.current_macro:
            messagebox.showwarning("Warning", "Please select a macro first")
            return
        
        self.save_state()
        cmd = ClipboardCopyCommand()
        if insert_index is not None:
            self.current_macro.commands.insert(insert_index, cmd)
            self.selected_index = insert_index
        else:
            self.current_macro.add_command(cmd)
            self.selected_index = len(self.current_macro.commands) - 1
        self.update_command_list()
        self.save_macros()
    
    def add_clipboard_paste(self, insert_index=None):
        """Add clipboard paste command"""
        if not self.current_macro:
            messagebox.showwarning("Warning", "Please select a macro first")
            return
        
        self.save_state()
        cmd = ClipboardPasteCommand()
        if insert_index is not None:
            self.current_macro.commands.insert(insert_index, cmd)
            self.selected_index = insert_index
        else:
            self.current_macro.add_command(cmd)
            self.selected_index = len(self.current_macro.commands) - 1
        self.update_command_list()
        self.save_macros()
    
    def add_sound(self, insert_index=None):
        """Add sound command"""
        if not self.current_macro:
            messagebox.showwarning("Warning", "Please select a macro first")
            return
        
        sound_cmd = self.show_sound_selector()
        if sound_cmd:
            self.save_state()
            if insert_index is not None:
                self.current_macro.commands.insert(insert_index, sound_cmd)
                self.selected_index = insert_index
            else:
                self.current_macro.add_command(sound_cmd)
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
        
        def progress_callback(index):
            self.root.after(0, lambda: self.highlight_command(index))
        
        def run():
            self.executor.execute(self.current_macro, repeat_count, progress_callback)
            self.root.after(0, lambda: self.highlight_command(-1))  # Reset highlight
            self.status_label.config(text="Ready")
        
        threading.Thread(target=run, daemon=True).start()
    
    def stop_macro(self):
        """Stop the running macro"""
        if self.executor.running:
            self.executor.stop()
            self.highlight_command(-1)  # Reset highlight
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
                result = self.capture_screen_region()
                if result:
                    screenshot, _ = result
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
                result = self.capture_screen_region()
                if result:
                    screenshot, _ = result
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
