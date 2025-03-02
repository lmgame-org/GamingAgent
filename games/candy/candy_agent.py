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

# Candy Crush specialized system prompt
system_prompt = (
    "You are an expert AI agent specialized in playing a Candy Crush–style match-3 game. "
    "Your goal is to find the best candy swap that forms matches of 3 or more candies, maximizing combos. "
    "Always ensure the move actually creates a valid match. "

    "### Candy Crash Rules ###\n"
    "1. The game is on an 8x8 grid.\n"
    "2. A move is swapping two adjacent candies.\n"
    "3. A move is valid if it creates a match of 3 or more identical candies.\n"
    "4. Matches are removed, new candies fall, combos can form.\n\n"

    "### Strategy ###\n"
    "1. Seek matches of 4 or 5 first.\n"
    "2. If no big match, do a normal 3-match.\n"
    "3. Consider future combo potential.\n\n"

    "### Output Format ###\n"
    "Use the exact format:\n"
    "\"swap: (<row1>, <col1>) <-> (<row2>, <col2>), thought: <brief reasoning>\"\n\n"
    "Coordinates are 0-based indices from top-left. Swaps must be between adjacent cells."
)

# Screen capture
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

def capture_screenshot():
    os.makedirs("cache/candy", exist_ok=True)
    screenshot_path = "cache/candy/candy_screenshot.png"

    # Set up Chrome with a headless option (no GUI)
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--window-size=1600,768")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get("http://127.0.0.1:5500/games/candy-crush/index.html")

    # Wait for the page to fully load
    time.sleep(3)

    # Take a screenshot of the full browser
    driver.save_screenshot(screenshot_path)
    driver.quit()

    return screenshot_path


# LLM-based move decision
def get_best_move(api_provider, model_name, move_history):
    """
    1. Capture screenshot
    2. Encode & send to LLM
    3. Parse out the best swap (row1,col1) <-> (row2,col2)
    4. Return that plus any 'thought'
    """
    # 1) Capture
    screenshot_path = capture_screenshot()
    base64_image = encode_image(screenshot_path)

    # 2) Build user prompt, referencing recent move history if desired
    history_text = "\n".join(
        [f"{i+1}. {entry['move']} -> {entry['thought']}" for i, entry in enumerate(move_history)]
    ) if move_history else "No previous moves."

    user_prompt = (
    f"Here is the current Candy Crush board (in base64). Analyze carefully.\n"
    f"Recent moves:\n{history_text}\n\n"
    f"Provide your best move in EXACTLY this format:\n"
    f"swap: (<row1>, <col1>) <-> (<row2>, <col2>), thought: <brief reasoning>\n\n"
    f"For example:\n"
    f"swap: (3, 5) <-> (4, 5), thought: This swap creates a vertical match of three green candies, triggering a cascade.\n\n"
    f"Do NOT use any other format. Your response should be a single line following the example above."
)


    # 3) Call the LLM
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

    # 4) Parse with regex
    match = re.search(
        r'swap:\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*<->\s*\(\s*(\d+)\s*,\s*(\d+)\s*\),\s*thought:\s*(.*)',
        response,
        re.IGNORECASE
    )

    if match:
        r1 = int(match.group(1))
        c1 = int(match.group(2))
        r2 = int(match.group(3))
        c2 = int(match.group(4))
        thought = match.group(5).strip()
        return (r1, c1, r2, c2), thought
    else:
        print("[WARNING] Could not parse a valid swap from the LLM response.")
        return None, "Failed to parse swap."

# Perform the swap on screen via PyAutoGUI
def perform_swap(r1, c1, r2, c2):
    """
    Converts (row,col) in the 8×8 grid to screen coordinates and performs
    a click-drag with PyAutoGUI.
    """
    # The top-left of the .grid from getBoundingClientRect()
    grid_offset_x = 488
    grid_offset_y = 58
    
    # Each of the 8 columns/rows is ~71.25 px (570 / 8)
    cell_size = 570 / 8.0  # ~71.25
    
    def cell_center(row, col):
        # For the center of a cell, add half a cell to the offset
        x = grid_offset_x + (col + 0.5) * cell_size
        y = grid_offset_y + (row + 0.5) * cell_size
        return (x, y)
    
    # Calculate the actual mouse coordinates
    start_x, start_y = cell_center(r1, c1)
    end_x,   end_y   = cell_center(r2, c2)
    
    print(f"Dragging from cell ({r1},{c1}) to ({r2},{c2})...")
    print(f"Screen coords: ({start_x:.2f}, {start_y:.2f}) -> ({end_x:.2f}, {end_y:.2f})")
    
    # Perform the click-drag
    pyautogui.moveTo(start_x, start_y, duration=0.2)
    pyautogui.mouseDown()
    pyautogui.moveTo(end_x, end_y, duration=0.2)
    pyautogui.mouseUp()


# Main loop
def main():
    parser = argparse.ArgumentParser(description="Candy Crash LLM AI Agent (No pygetwindow)")
    parser.add_argument("--api_provider", type=str, default="openai",
                        help="API provider to use: anthropic, openai, gemini, etc.")
    parser.add_argument("--model_name", type=str, default="gpt-4-turbo",
                        help="LLM model name.")
    parser.add_argument("--loop_interval", type=float, default=3.0,
                        help="Time in seconds to wait between moves.")
    parser.add_argument("--game_url", type=str, default="http://127.0.0.1:5500/games/candy-crush/index.html",
                        help="Local URL to open the Candy Crush game.")
    parser.add_argument("--auto_open", action="store_true",
                        help="If set, automatically open the game URL in a browser.")

    args = parser.parse_args()

    print("Starting Candy Crash AI Agent.")
    print(f"API Provider: {args.api_provider}, Model Name: {args.model_name}")

    url = "http://127.0.0.1:5500/games/candy-crush/index.html"
    webbrowser.open(url)
    time.sleep(3)
    print("Opened Candy Crush page. Now ready to do more...")

    move_history = deque(maxlen=4)

    try:
        while True:
            move_tuple, thought = get_best_move(
                api_provider=args.api_provider,
                model_name=args.model_name,
                move_history=list(move_history)
            )

            if move_tuple is not None:
                r1, c1, r2, c2 = move_tuple
                move_str = f"swap: ({r1}, {c1}) <-> ({r2}, {c2})"
                move_history.append({"move": move_str, "thought": thought})

                print(f"Move: {move_str}")
                print(f"Thought: {thought}")

                perform_swap(r1, c1, r2, c2)
            else:
                print("[INFO] No valid move parsed. Skipping this round.")

            time.sleep(args.loop_interval)

    except KeyboardInterrupt:
        print("\nStopped by user.")

if __name__ == "__main__":
    main()
