import time
import os
import pyautogui
import numpy as np
import platform

if platform.system() == "Windows":
    import pygetwindow as gw
    import win32gui
    import win32con
elif platform.system() == "Linux":
    from Xlib import display
    import mss


def get_window_rect_linux(window_name):
    """Gets the window position and size on Linux using Xlib."""
    d = display.Display()
    root = d.screen().root
    window_id = None

    # Iterate over all windows to find the matching name
    for window in root.query_tree().children:
        name = window.get_wm_name()
        if name and window_name.lower() in name.lower():
            window_id = window.id
            geom = window.get_geometry()
            return (geom.x, geom.y, geom.width, geom.height)

    return None

def get_window_rect_windows(window_name):
    """Gets the window position and size on Windows."""
    windows = gw.getWindowsWithTitle(window_name)
    if windows:
        window = windows[0]  # Take the first matching window
        rect = win32gui.GetWindowRect(window._hWnd)
        left, top, right, bottom = rect
        width, height = right - left, bottom - top
        return (left, top, width, height)
    return None

def take_screenshot(window_name, save_path="screenshot.png"):
    """Takes a screenshot of the specified window on Windows or Linux."""
    system = platform.system()
    rect = None

    if system == "Windows":
        rect = get_window_rect_windows(window_name)
        if not rect:
            print("Window not found!")
            return None, None
        x, y, width, height = rect
        win32gui.SetForegroundWindow(win32gui.FindWindow(None, window_name))  # Bring window to front
        pyautogui.screenshot(save_path, region=(x, y, width, height))

    elif system == "Linux":
        rect = get_window_rect_linux(window_name)
        if not rect:
            print("Window not found!")
            return None, None
        x, y, width, height = rect
        with mss.mss() as sct:
            screenshot = sct.grab({"top": y, "left": x, "width": width, "height": height})
            mss.tools.to_png(screenshot.rgb, screenshot.size, output=save_path)

    else:
        print(f"Unsupported platform: {system}")
        return None, None

    print(f"Screenshot saved to {save_path}")
    return save_path, rect

# Example usage
screenshot_path, position = take_screenshot("Plants vs. Zombies" if platform.system() == "Windows" else "Terminal", "screenshot.png")

if screenshot_path and position:
    print(f"Screenshot saved at: {screenshot_path}")
    print(f"Window position: X={position[0]}, Y={position[1]}, Width={position[2]}, Height={position[3]}")
