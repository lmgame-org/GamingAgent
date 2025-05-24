import time
import os
import pyautogui
import argparse
import re
import mss
import mss.tools
import webbrowser
from collections import deque
from tools.utils import encode_image
from tools.serving.api_providers import anthropic_completion, openai_completion, gemini_completion

# Boxxel specialized system prompt
system_prompt = (
    "You are an expert AI agent specialized in playing Boxxel, a puzzle-solving game. "
    "Your goal is to push the light green square onto the final destination, marked by the dark green square, while avoiding obstacles. "
    "You are represented by a square (outer is dark green and inner is light green and). "
    "If the box is against a wall or in a corner where it cannot be pushed further, you may need to restart. "
    
    "### Boxxel Rules ###\n"
    "1. The game takes place on a 4*4 grid-based level.\n"
    "2. You are represented by a square (outer is dark green and inner is light green and).\n"
    "3. Your movement is controlled by the arrow keys (up, down, left, right).\n"
    "4. Before making any move, first analyze the positions of the three key squares:\n"
    "   - **Your position (two-colored square)**\n"
    "   - **The light green square's position**\n"
    "   - **The dark green square's position (goal)**\n"
    "5. You can push the light green square, but only if you are correctly positioned:\n"
    "   - If you are **below** the light green square, moving **up** will push it up.\n"
    "   - If you are **above** the light green square, moving **down** will push it down.\n"
    "   - If you are **left** of the light green square, moving **right** will push it right.\n"
    "   - If you are **right** of the light green square, moving **left** will push it left.\n"
    "6. The light green square cannot be moved in any other way; you must be in the correct adjacent position to push it.\n"
    "7. The objective is to push the light green square onto the dark green square.\n"
    "8. If the box gets stuck (e.g., against a wall or in a corner where it cannot move), restart using 'Z'.\n"
    "9. When the puzzle is solved, press 'X' to advance to the next level.\n"
    "10. The game starts from level 0.\n\n"

    "### Strategy ###\n"
    "1. Before making a move, first analyze:\n"
    "   - Where you are relative to the light green square.\n"
    "   - Where the light green square is relative to the dark green square.\n"
    "   - If moving in a certain direction will push the box correctly or cause a deadlock.\n"
    "2. Plan moves carefully to minimize unnecessary steps.\n"
    "3. Avoid pushing the box into corners or against walls unless necessary.\n"
    "4. Use the 'X' key only when the light green square is on the dark green square.\n"
    "5. Use 'Z' (restart) only when there is no possible way to complete the level.\n"
    "6. If stuck, move around and attempt to approach the box from another angle.\n"
    "7. Prioritize moves that bring the box closer to the destination without blocking future moves.\n\n"

    "### Output Format ###\n"
    "Before deciding on a move, always output the positions of the three squares and then the action.\n"
    "Format:\n"
    "\"analysis: My position (x, y), Box position (x, y), Goal position (x, y).\"\n"
    "\"move: <direction>, thought: <brief reasoning>\"\n\n"

    "Directions: 'up', 'down', 'left', 'right', 'restart', 'next' (for advancing).\n\n"

    "### Example Outputs ###\n"
    "analysis: My position (3,2), Box position (3,3), Goal position (5,3).\n"
    "move: up, thought: I am below the box, so moving up will push it closer to the target.\n\n"
    
    "analysis: My position (4,4), Box position (4,3), Goal position (2,3).\n"
    "move: left, thought: I am to the right of the box, so moving left will push it left.\n\n"

    "analysis: My position (2,5), Box position (3,5), Goal position (3,2).\n"
    "move: restart, thought: The box is stuck against a wall, restarting is necessary.\n"
)




import subprocess

def get_chrome_bounds():
    """ Uses AppleScript to get Chrome window bounds. """
    script = '''
    tell application "Google Chrome"
        set winBounds to bounds of window 1
        return winBounds
    end tell
    '''
    result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)

    if result.returncode == 0:
        bounds = list(map(int, result.stdout.strip().split(", ")))
        return {"left": bounds[0], "top": bounds[1], "width": bounds[2] - bounds[0], "height": bounds[3] - bounds[1]}
    else:
        print("[ERROR] Could not get Chrome window bounds!")
        return None

def capture_screenshot():
    os.makedirs("cache/boxxel", exist_ok=True)
    screenshot_path = "cache/boxxel/boxxel_screenshot.png"

    bounds = get_chrome_bounds()
    if not bounds:
        print("[ERROR] Unable to capture Chrome window!")
        return None

    with mss.mss() as sct:
        screenshot = sct.grab(bounds)
        mss.tools.to_png(screenshot.rgb, screenshot.size, output=screenshot_path)

    print(f"[INFO] Screenshot saved to {screenshot_path}")
    return screenshot_path



# LLM-based move decision
def get_best_move(api_provider, model_name, move_history):
    screenshot_path = capture_screenshot()
    base64_image = encode_image(screenshot_path)
    history_text = "\n".join(
        [f"{i+1}. {entry['move']} -> {entry['thought']}" for i, entry in enumerate(move_history)]
    ) if move_history else "No previous moves."

    user_prompt = (
        f"Here is the current Boxxel level (in base64). Analyze carefully.\n"
        f"Recent moves:\n{history_text}\n\n"
        f"Provide your best move in EXACTLY this format:\n"
        f"move: <direction>, thought: <brief reasoning>\n\n"
        f"Directions: 'up', 'down', 'left', 'right', 'restart', 'next' (for advancing)."
    )

    start_time = time.time()
    if api_provider == "anthropic":
        response = anthropic_completion(system_prompt, model_name, base64_image, user_prompt)
    elif api_provider == "openai":
        response = openai_completion(system_prompt, model_name, base64_image, user_prompt)
    elif api_provider == "gemini":
        response = gemini_completion(system_prompt, model_name, base64_image, user_prompt)
    else:
        raise NotImplementedError(f"API provider '{api_provider}' is not supported.")
    latency = time.time() - start_time
    print(f"[INFO] LLM Response Latency: {latency:.2f}s")
    print(f"[LLM raw response]: {response}")

    match = re.search(r'move:\s*(\w+),\s*thought:\s*(.*)', response, re.IGNORECASE)
    if match:
        move = match.group(1).strip()
        thought = match.group(2).strip()
        return move, thought
    else:
        print("[WARNING] Could not parse a valid move from the LLM response.")
        return None, "Failed to parse move."

# Perform movement with PyAutoGUI
def perform_move(move):
    key_map = {
        "up": "up",
        "down": "down",
        "left": "left",
        "right": "right",
        "next": "x",
        "start": "x",
        "restart": "z"
    }
    if move in key_map:
        pyautogui.press(key_map[move])
        print(f"Performed move: {move}")
    else:
        print(f"[WARNING] Invalid move: {move}")

# Main loop
def main():
    parser = argparse.ArgumentParser(description="Boxxel AI Agent")
    parser.add_argument("--api_provider", type=str, default="openai", help="API provider to use.")
    parser.add_argument("--model_name", type=str, default="gpt-4-turbo", help="LLM model name.")
    parser.add_argument("--loop_interval", type=float, default=3, help="Time in seconds between moves.")
    parser.add_argument("--game_url", type=str, default="https://wasm4.org/play/sokoban/#", help="Boxxel game URL.")
    parser.add_argument("--auto_open", action="store_true", help="Automatically open the game URL.")
    args = parser.parse_args()

    print("Starting Boxxel AI Agent.")
    print(f"API Provider: {args.api_provider}, Model Name: {args.model_name}")
    if args.auto_open:
        webbrowser.open(args.game_url)
        time.sleep(2)
    print("Opened Boxxel page. Ready to play.")

    move_history = deque(maxlen=5)
    try:
        while True:
            move, thought = get_best_move(args.api_provider, args.model_name, list(move_history))
            if move is not None:
                move_history.append({"move": move, "thought": thought})
                perform_move(move)
            else:
                print("[INFO] No valid move parsed. Skipping this round.")
            time.sleep(args.loop_interval)
    except KeyboardInterrupt:
        print("\nStopped by user.")

if __name__ == "__main__":
    main()
