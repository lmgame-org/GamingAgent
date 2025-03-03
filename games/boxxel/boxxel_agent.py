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
    "You are an expert AI agent specialized in playing Sokoban, a classic puzzle game. "
    "Your goal is to push all boxes onto the designated dock locations while avoiding deadlocks. "
    "You control a worker that can move up, down, left, and right, but can only push boxes if positioned correctly. "
    "If a box is against a wall with no way to reposition it, you may need to restart. "
    "The game requires careful planning to avoid blocking yourself. "
    "\n\n### Sokoban Rules ###\n"
    "1. The game takes place on a grid-based level.\n"
    "2. The level consists of different elements:\n"
    "   - wall (#) (impassable).\n"
    "   - dock (.) (where boxes must be placed).\n"
    "   - worker (@) (your position).\n"
    "   - box ($) (movable by the worker).\n"
    "   - box on dock (*) (goal achieved).\n"
    "3. You can push boxes, but only if there is an empty floor space or dock behind them.\n"
    "4. Boxes must be placed onto dock squares to complete the level.\n"
    "5. If a box is stuck in a corner or against a wall with no way to move it, restart the level.\n"

    "\n### Strategy ###\n"
    "1. Analyze the level layout before making a move.\n"
    "2. Plan moves carefully to avoid pushing boxes into corners or against walls where they cannot be retrieved.\n"
    "3. Use efficient movement patterns to solve the level in the fewest moves possible.\n"
    "4. If you make an irreversible mistake, restart the level using 'R'.\n"
    "5. If you want to undo the last move, press 'D' to unmove.\n"

    "\n### Output Format ###\n"
    "Before deciding on a move, always output the positions of the key elements and then the action.\n"
    "Format:\n"
    "analysis: Worker position (x, y), Box positions [(x1, y1), (x2, y2), ...], Dock positions [(dx1, dy1), (dx2, dy2), ...], Walls [(wx1, wy1), (wx2, wy2), ...].\n"
    "move: <direction>, thought: <brief reasoning>\n\n"
    "Directions: 'up', 'down', 'left', 'right', 'restart', 'unmove' (undo the last move).\n\n"

    "### Current initial Level Map ###\n"
    "#######\n"
    "#.@ # #\n"
    "#$* $ #\n"
    "#   $ #\n"
    "# ..  #\n"
    "#  *  #\n"
    "#######"
)




def capture_screenshot():
    screen_width, screen_height = pyautogui.size()
    region = (0, 0, screen_width // 64 * 9, screen_height // 64 * 20)
    

    os.makedirs("cache/boxxel", exist_ok=True)
    screenshot_path = "cache/boxxel/boxxel_screenshot.png"

    screenshot = pyautogui.screenshot(region=region)
    screenshot.save(screenshot_path)

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
        f"Directions: 'up', 'down', 'left', 'right'."
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
        "restart": 'R',
        "unmove": 'D'
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
    args = parser.parse_args()

    print("Starting Boxxel AI Agent.")
    print(f"API Provider: {args.api_provider}, Model Name: {args.model_name}")

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
