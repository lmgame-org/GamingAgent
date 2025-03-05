import time
import os
import pyautogui
import numpy as np

from tools.utils import encode_image, log_output
from tools.serving.api_providers import anthropic_completion, openai_completion, gemini_completion
import re
import json

CACHE_DIR = "cache/boxxel"

def load_matrix(filename='game_state.json'):
    filename = os.path.join(CACHE_DIR, filename)
    """Load the game matrix from a JSON file."""
    if not os.path.exists(filename):
        return None
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading matrix: {e}")
        return None
    
def matrix_to_string(matrix):
    """Convert a 2D list matrix into a string with each row on a new line."""
    # If each element is already a string or you want a space between them:
    return "\n".join(" ".join(str(cell) for cell in row) for row in matrix)


def log_move_and_thought(move, thought, latency):
    """
    Logs the move and thought process into a log file inside the cache directory.
    """
    log_file_path = os.path.join(CACHE_DIR, "boxxel_moves.log")
    
    log_entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Move: {move}, Thought: {thought}, Latency: {latency:.2f} sec\n"
    
    try:
        with open(log_file_path, "a") as log_file:
            log_file.write(log_entry)
    except Exception as e:
        print(f"[ERROR] Failed to write log entry: {e}")

def boxxel_read_worker(system_prompt, api_provider, model_name, image_path):
    base64_image = encode_image(image_path)
    api_provider = "openai"
    model_name = "gpt-4o"
    matrix = load_matrix()
    if matrix is not None:
        board_str = matrix_to_string(matrix)
    else:
        board_str = "No board available."
    # Construct prompt for LLM
    prompt = (
        "Extract the Boxxel board layout from the provided layout.\n\n"
        
        "### Current Game Layout ###\n"
        f"{board_str}\n\n"

        "### Key Elements ###\n"
        "- `#`: Walls (impassable obstacles)\n"
        "- `@`: Worker (player character)\n"
        "- `$`: Box (movable object)\n"
        "- `?`: Dock (goal position for boxes)\n"
        "- `*`: Box on a dock (correctly placed)\n"
        "- ` `: Floor (empty walkable space)\n\n"
        
        "### Task ###\n"
        "Use the given board layout to identify and recognize each item based on the provided symbols.\n"
        "Strictly format the output as: **ID: item type (row, column)**.\n\n"

        "Each row should reflect the board layout.\n"
        "Example format: \n1: wall (0, 0) | 2: docker (0, 1)| 3: player (0, 2)... \n8: empty (1,0) | 9: dock (1, 1)| 10: empty (1, 2) "
    
    )

    
    
    # Call LLM API based on provider
    if api_provider == "anthropic":
        response = anthropic_completion(system_prompt, model_name, base64_image, prompt)
    elif api_provider == "openai":
        response = openai_completion(system_prompt, model_name, base64_image, prompt)
    elif api_provider == "gemini":
        response = gemini_completion(system_prompt, model_name, base64_image, prompt)
    else:
        raise NotImplementedError(f"API provider: {api_provider} is not supported.")
    
    # Process response and format as structured board output
    structured_board = response.strip()
    
    # Generate final text output
    final_output = "\nBoxxel Board Representation:\n" + structured_board

    return final_output


def boxxel_worker(system_prompt, api_provider, model_name, prev_response=""):
    """
    1) Captures a screenshot of the current game state.
    2) Calls an LLM to generate PyAutoGUI code for the next move.
    3) Logs latency and the generated code.
    """
    # Capture a screenshot of the current game state.
    screen_width, screen_height = pyautogui.size()
    region = (0, 0, screen_width // 64 * 10, screen_height // 64 * 25)
    
    screenshot = pyautogui.screenshot(region=region)

    # Save the screenshot directly in the cache directory.
    os.makedirs("cache/boxxel", exist_ok=True)
    screenshot_path = "cache/boxxel/boxxel_screenshot.png"

    screenshot.save(screenshot_path)

    table = boxxel_read_worker(system_prompt, api_provider, model_name, screenshot_path)

    prompt = (
    "### Previous Lessons Learned###"
    
    "You are an expert AI agent specialized in solving Sokoban puzzles optimally. Your objective is to push all boxes onto their designated dock locations while minimizing unnecessary moves and avoiding deadlocks."
    "Before making a move, analyze the entire puzzle layout. Plan the next 5 steps by considering all possible paths for each box, ensuring they remain maneuverable when necessary to reach their dock locations. Identify potential deadlocks early and prioritize moves that maintain overall solvability. However, note that temporarily blocking a box may sometimes be necessary to progress, so focus on the broader strategy rather than ensuring all boxes are always movable at every step."
    "You control a worker who can move in four directions (up, down, left, right). You can push boxes if positioned correctly but cannot pull them. Be mindful of walls and corners, as getting a box irreversibly stuck may require a restart. Optimize for efficiency while maintaining flexibility in your approach."

    f"Here is your previous response: {prev_response}. Please evaluate your plan and thought about whether we should correct or adjust.\n"
    f"Here is the current layout of the Boxxel board:\n{table}\n\n"

    "Some reflection try to avoid:"
    "1. Vertical Stacking Error: occurred when planned to push the box from (3,3) into (4,4), which, when aligned with the box at (3,4), would have created an immovable vertical block."
    "2. Phantom Deadlock Error: occurred avoided pushing the box at (2,3) downward into (3,3)—a move that is safe thanks to ample space in (4,3) and beyond."
    "3. Corner Lock Error: Pushing the box from (3,2) into (3,1) created a deadlock by confining the box in a corner against the walls at (3,0) and (4,1)."
    "4. Path Obstruction Error:  By pushing the box from (2,2) into (2,1), lead to a deadlock by blocking the critical corridor needed to reposition box (3,2), where the layout now has the box at (2,1) obstructing access amid the surrounding wall and floor configuration."
    "5. Neglected Dock Priority Error: by not prioritizing pushing the workable box at (4,3) to its end dock and instead moving the box from (2,2) to (2,3), I created a configuration where any subsequent vertical push into (3,3) results in an irreversible stacking deadlock against the wall, locking the layout in a general"
    "6. Final Dock Saturation Error: pushing the box from (5,6) directly onto its dock at (5,7) without leaving any free adjacent floor space in the narrow corridor bordered by walls resulted in a deadlock layout where no maneuvering room remained for later boxes."

    "### Output Format ###\n"
    "move: up/down/left/right/restart/unmove, thought: <brief reasoning>\n\n"
    "Example output: move: right, thought: Positioning the player to access other boxes and docks for future moves."
    )




    base64_image = encode_image(screenshot_path)
    base64_image = None
    start_time = time.time()

    print(f"Calling {model_name} api...")
    # Call the LLM API based on the selected provider.
    if api_provider == "anthropic":
        response = anthropic_completion(system_prompt, model_name, base64_image, prompt)
    elif api_provider == "openai":
        response = openai_completion(system_prompt, model_name, base64_image, prompt)
    elif api_provider == "gemini":
        response = gemini_completion(system_prompt, model_name, base64_image, prompt)
    else:
        raise NotImplementedError(f"API provider: {api_provider} is not supported.")

    latency = time.time() - start_time

    print(f"check response: {response}")

    match = re.search(r'move:\s*(\w+),\s*thought:\s*(.*)', response, re.IGNORECASE)
    move = match.group(1).strip()
    thought = match.group(2).strip()

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

    if move is not None:
        perform_move(move)
        # Log move and thought
        log_output(
            "boxxel_worker",

            f"[INFO] Move executed: ({move}) | Thought: {thought} | Latency: {latency:.2f} sec",
            "boxxel"
        )

    return response