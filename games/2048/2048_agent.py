import time
import os
import pyautogui
import argparse
import numpy as np
from tools.utils import encode_image, log_output
from tools.serving.api_providers import anthropic_completion, openai_completion, gemini_completion
import subprocess
import multiprocessing
import re
import pygetwindow as gw
import mss
# System prompt for LLM
system_prompt = (  
    "You are an expert AI agent specialized in playing the 2048 game with advanced strategic reasoning. "  
    "Your primary goal is to achieve the highest possible tile value by strategically merging tiles. "  
    "Prioritize keeping the highest-value tile in a corner while maintaining an open board for flexibility. "  
    "Avoid moves that result in an early game over by ensuring sufficient space for future moves. "  
      
    "### 2048 Game Rules ### "  
    "1. The game is played on a 4×4 grid, where numbered tiles slide in one of four directions: 'up', 'down', 'left', or 'right'. "  
    "2. **Only two adjacent tiles with the same number can merge.** Tiles do not merge across empty spaces. "  
    "3. **Merging is directional:**\n"  
    "   - In a row: Tiles can merge when moved 'left' or 'right'.\n"  
    "   - In a column: Tiles can merge when moved 'up' or 'down'.\n"  
    "4. A move is only valid if at least one tile can slide or merge; otherwise, the move is ignored. "  
    "5. After every move, a new tile (usually 2 or 4) appears at a random empty spot on the board. "  
    "6. The game ends when no valid moves are possible (i.e., the board is full, and no merges can be made). "  
      
    "### Strategy Guidelines ### "  
    "1. Always aim to keep the highest tile in one of the corners (preferably bottom-left or bottom-right). "  
    "2. Prioritize moves that merge tiles to create higher values while preserving board flexibility. "  
    "3. Avoid disrupting tile alignment unless necessary to create merges. "  
    "4. Do not make moves that trap high-value tiles or severely limit future options. "  
    "5. If the board state appears 'frozen' (very few possible moves), try a new and safe move that prevents game over. "  
      
    "### Decision Output ### "  
    "Analyze the game state from the provided image and determine the optimal move. "  
    "If the board state looks frozen or close to a deadlock, prioritize a move that increases flexibility. "  
    "Provide your response in a single string format as follows:\n\n"
    
    '"move: "<up, right, left, or down>", thought: "<a brief summary of why this move was chosen>""\n\n'

    "Ensure that:\n"
    "- The 'move' field contains only one of the four valid directions.\n"
    "- The 'thought' field provides a brief explanation (few words) of why the move is the best choice.\n"
)


# WIDTH, HEIGHT = 800, 700
# WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))

# def capture_screenshot():
#     """
#     Captures the full screen, saves it, and returns the image path.
#     """
#     os.makedirs("cache", exist_ok=True)
#     screenshot_path = "cache/2048_screenshot.png"

#     # Capture full screen
#     screen_width, screen_height = pyautogui.size()
    
#     region = (0, 0, screen_width, screen_height)
#     screenshot = pyautogui.screenshot(region=region)
#     screenshot.save(screenshot_path)

#     return screenshot_path

def get_pygame_window_position():
    """
    Attempts to find the 2048 pygame window and return its position.
    """
    windows = gw.getWindowsWithTitle("pygame")  # Adjust title if needed

    if windows:
        window = windows[0]
        return window.left, window.top, window.width, window.height
    else:
        print("2048 window not found!")
        return None

def capture_screenshot():
    """
    Captures the pygame window dynamically based on its detected position.
    """
    os.makedirs("cache/2048", exist_ok=True)
    screenshot_path = "cache/2048/2048_screenshot.png"

    position = get_pygame_window_position()
    if position:
        left, top, width, height = position
        with mss.mss() as sct:
            monitor = {"top": top, "left": left, "width": width, "height": height}
            screenshot = sct.grab(monitor)
            mss.tools.to_png(screenshot.rgb, screenshot.size, output=screenshot_path)

    return screenshot_path
from collections import deque

def get_best_move(system_prompt, api_provider, model_name, move_history):
    """
    Takes a screenshot, sends it to the LLM, and extracts the best move and reasoning,
    considering the previous four moves and thoughts.
    """
    screenshot_path = capture_screenshot()
    base64_image = encode_image(screenshot_path)

    # Format the move history
    history_prompt = "\n".join(
        [f"{i+1}. move: {entry['move']}, thought: {entry['thought']}" for i, entry in enumerate(move_history)]
    ) if move_history else "No previous moves."

    move_prompt = (
    f"Your last four moves and thoughts:\n{history_prompt}\n\n"
    "Analyze the 2048 game state from the image and determine the best move: 'up', 'right', 'left', or 'down'.\n"
    "Avoid repeating mistakes and prioritize flexible, strategic moves that maximize tile merging and board control.\n\n"
    
    "### Move Evaluation ###\n"
    "1. **Check if the move is possible**: A move is only valid if at least one tile can slide or merge.\n"
    "2. **Ignore invalid moves**: If a direction does not allow movement (i.e., no space or different numbers that cannot merge), do not consider it.\n"
    "3. **Avoid repeating past moves** *only if* the last four moves contained just 1-2 unique directions. If the previous four moves were diverse, focus on selecting the optimal move based on board state.\n\n"
    
    "### 2048 Game Rules & Strategy ###\n"
    "1. Keep the **highest-value tile in a corner**.\n"
    "2. **Prioritize merging** over unnecessary movement.\n"
    "3. **Avoid moves that limit future flexibility**.\n"
    "4. If a deadlock is likely, **try an alternative strategy**.\n\n"
    
    "### Decision-Making & Adaptation ###\n"
    "1. If your thought is **similar to previous ones**, you might be repeating a mistake. Try a different approach.\n"
    "2. If the last four moves include only **1-2 unique directions**, switch to a different move to avoid getting stuck.\n"
    "3. If the last four moves were already varied, **focus on making the best possible move** rather than forcing a different one.\n\n"
    
    "Provide your response in the strict format: move: \"<direction>\", thought: \"<brief reasoning>\"."
    )

    start_time = time.time()

    if api_provider == "anthropic":
        response = anthropic_completion(system_prompt, model_name, base64_image, move_prompt)
    elif api_provider == "openai":
        response = openai_completion(system_prompt, model_name, base64_image, move_prompt)
    elif api_provider == "gemini":
        response = gemini_completion(system_prompt, model_name, base64_image, move_prompt)
    else:
        raise NotImplementedError(f"API provider '{api_provider}' is not supported.")

    latency = time.time() - start_time
    print(f"[INFO] LLM Response Latency: {latency:.2f}s")
    
    # Regular expression to extract move and thought
    match = re.search(r'move:\s*"?(up|down|left|right)"?,\s*thought:\s*"([^"]+)"', response, re.IGNORECASE)

    if match:
        move = match.group(1).strip().lower()  # Extract the move (up, down, left, right)
        thought = match.group(2).strip()  # Extract the reasoning
    else:
        print(f"[WARNING] Unexpected response format: {response}")
        move, thought = "unknown", "Failed to extract reasoning."

    return move, thought

def main():
    """
    Runs a single AI worker for 2048 in a loop without concurrency,
    keeping track of the last four moves.
    """
    parser = argparse.ArgumentParser(description="2048 LLM AI Agent (Single Worker)")
    parser.add_argument("--api_provider", type=str, default="openai",
                        help="API provider to use (anthropic, openai, gemini).")
    parser.add_argument("--model_name", type=str, default="gpt-4-turbo",
                        help="Model name.")
    parser.add_argument("--loop_interval", type=float, default=0.5,
                        help="Time in seconds between moves.")

    args = parser.parse_args()

    print(f"Starting 2048 AI Agent...")
    print(f"API Provider: {args.api_provider}, Model Name: {args.model_name}")

    move_history = deque(maxlen=4)  # Store the last 4 moves

    try:
        while True:
            move, thought = get_best_move(system_prompt, args.api_provider, args.model_name, list(move_history))
            move_history.append({"move": move, "thought": thought})  # Add move to history

            if move in ["up", "right", "left", "down"]:
                pyautogui.press(move)
                print(f"Executed move: {move}")
                print(f"Thought: {thought}")  # Print the reasoning for the move
            else:
                print(f"Invalid move received: {move}, Thought: {thought}")

            time.sleep(args.loop_interval)  # Delay before next move

    except KeyboardInterrupt:
        print("\nGame interrupted by user. Exiting...")

if __name__ == "__main__":
    main()
