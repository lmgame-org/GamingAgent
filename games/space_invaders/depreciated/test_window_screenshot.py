import pywinctl
import pyautogui
import time

def get_game_window():
    all_windows = pywinctl.getAllTitles()
    print(f"[DEBUG] All windows: {all_windows}")

    for title in all_windows:
        if "Space Invaders" in title:  
            game_window = pywinctl.getWindowsWithTitle(title)[0]  
            x, y, width, height = game_window.left, game_window.top, game_window.width, game_window.height
            print(f"[INFO] 发现 Space Invaders 窗口: 位置({x}, {y}), 大小({width}x{height})")
            return x, y, width, height

    print("[ERROR] Can't find Space Invaders window！")
    return None

time.sleep(3)

game_window = get_game_window()

if game_window:
    x, y, width, height = game_window

    screenshot = pyautogui.screenshot(region=(x, y, width, height))
    screenshot_path = "games/space_invaders/test_screenshot.png"
    screenshot.save(screenshot_path)
    print(f"[INFO] Screenshot saved: {screenshot_path}")

