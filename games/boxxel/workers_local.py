import time
import os
import pyautogui
import numpy as np

from tools.utils import encode_image, log_output, get_annotate_img
from tools.serving.api_providers import anthropic_completion, anthropic_text_completion, openai_completion, openai_text_reasoning_completion, gemini_completion, gemini_text_completion, deepseek_text_reasoning_completion
import re
import json

CACHE_DIR = "cache/sokoban"

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

def matrix_to_text_table(matrix):
    """Convert a 2D list matrix into a structured text table."""
    header = "ID  | Item Type    | Position"
    line_separator = "-" * len(header)
    
    item_map = {
        '#': 'Wall',
        '@': 'Worker',
        '$': 'Box',
        '?': 'Dock',
        '*': 'Box on Dock',
        ' ': 'Empty'
    }
    
    table_rows = [header, line_separator]
    item_id = 1
    
    for row_idx, row in enumerate(matrix):
        for col_idx, cell in enumerate(row):
            item_type = item_map.get(cell, 'Unknown')
            table_rows.append(f"{item_id:<3} | {item_type:<12} | ({col_idx}, {row_idx})")
            item_id += 1
    
    return "\n".join(table_rows)


def matrix_to_string(matrix):
    """Convert a 2D list matrix into a string with each row on a new line."""
    # If each element is already a string or you want a space between them:
    return "\n".join(" ".join(str(cell) for cell in row) for row in matrix)


def log_move_and_thought(move, thought, latency):
    """
    Logs the move and thought process into a log file inside the cache directory.
    """
    log_file_path = os.path.join(CACHE_DIR, "sokoban_moves.log")
    
    log_entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Move: {move}, Thought: {thought}, Latency: {latency:.2f} sec\n"
    
    try:
        with open(log_file_path, "a") as log_file:
            log_file.write(log_entry)
    except Exception as e:
        print(f"[ERROR] Failed to write log entry: {e}")

def sokoban_read_worker(system_prompt, api_provider, model_name, image_path):
    base64_image = encode_image(image_path)
    matrix = load_matrix()
    if matrix is not None:
        board_str = matrix_to_text_table(matrix)
    else:
        board_str = "No board available."
    return board_str

def sokoban_worker(system_prompt, api_provider, model_name, 
    prev_response="", 
    thinking=True, 
    modality="vision-text",
    level=1,
    crop_left=0, 
    crop_right=0, 
    crop_top=0, 
    crop_bottom=0, 
    ):
    """
    1) Captures a screenshot of the current game state.
    2) Calls an LLM to generate PyAutoGUI code for the next move.
    3) Logs latency and the generated code.
    """
    # Capture a screenshot of the current game state.
    # Save the screenshot directly in the cache directory.
    assert modality in ["text-only", "vision-text"], f"modality {modality} is not supported."

    os.makedirs("cache/sokoban", exist_ok=True)
    screenshot_path = "cache/sokoban/sokoban_screenshot.png"

    levels_dim_path = os.path.join(CACHE_DIR, "levels_dim.json")
    with open(levels_dim_path, "r") as f:
        levels_dims = json.load(f)

    # Extract rows/cols for the specified level
    level_key = f"level_{level}"
    if level_key not in levels_dims:
        raise ValueError(f"No dimension info found for {level_key} in {levels_dim_path}")

    grid_rows = levels_dims[level_key]["rows"]
    grid_cols = levels_dims[level_key]["cols"]

    annotate_image_path, grid_annotation_path, annotate_cropped_image_path = get_annotate_img(screenshot_path, crop_left=crop_left, crop_right=crop_right, crop_top=crop_top, crop_bottom=crop_bottom, grid_rows=grid_rows, grid_cols=grid_cols, cache_dir=CACHE_DIR)

    table = sokoban_read_worker(system_prompt, api_provider, model_name, screenshot_path)

    #print(f"-------------- TABLE --------------\n{table}\n")
    #print(f"-------------- prev response --------------\n{prev_response}\n")

    prompt = (
    "## Previous Lessons Learned\n"
    "- The Sokoban board is structured as a list matrix with coordinated positions: (column_index, row_index).\n"
    "- You control a worker who can move in four directions (up along row index, down along row index, left along column index, right along column index) in a 2D Sokoban game. "
    "You can push boxes if positioned correctly but cannot pull them. "
    "Be mindful of walls and corners, as getting a box irreversibly stuck may require a restart.\n"
    "- You are an expert AI agent specialized in solving Sokoban puzzles optimally." 
    "Consider relationship among boxes, you can run the Rolling Stone algorithm: Iterative Deepening A* (IDA*) algorithm to find an optimal path.\n"
    "- Before leaving a box. Consider if it will be become a road block for future boxes.\n"
    "- Before making a move, re-analyze the entire puzzle layout. "
    "Plan the next 1 to 5 steps by considering all possible paths for each box, "
    "ensuring they will have a viable step-by-step path to reach their dock locations.\n"
    "- After a box reaches a dock location. Reconsider if the dock location is optimal, or it should be repositioned to another dock location.\n"
    "- Identify potential deadlocks early and prioritize moves that maintain overall solvability. "
    "However, note that temporarily blocking a box may sometimes be necessary to progress, "
    "so focus on the broader strategy rather than ensuring all boxes are always movable at every step.\n"

    "## Potential Errors to avoid:\n"
    "1. Vertical Stacking Error: stacked boxes can't not be moved from the stacked direction and can become road block.\n"
    "2. Phantom Deadlock Error: boxes pushed to the walls will very likely get pushed to corners and result in deadlocks.\n"
    "3. Box Accessibility Error: Consider the spacial relationship between the worker and the current box. Push it in a way that the worker can access it later to move it to a dock location.\n"
    "3. Corner Lock Error: boxes get pushed to corners will not be able to get out.\n"
    "4. Path Obstruction Error: a box blocks your way to reach other boxes and make progress to the game.\n"
    "5. Final Dock Saturation Error: choose which box goes to which dock wisely.\n"

    f"Here is your previous response: {prev_response}. Please evaluate your plan and thought about whether we should correct or adjust.\n"
    "Here is the current layout of the Sokoban board:\n"
    f"{table}\n\n"

    "### Output Format:\n"
    "move: up/down/left/right, thought: <brief reasoning>\n\n"
    "Example output: move: right, thought: Positioning the player to access other boxes and docks for future moves."
    )

    base64_image = encode_image(annotate_cropped_image_path)
    if "o3-mini" in model_name:
        base64_image = None
    start_time = time.time()

    print(f"Calling {model_name} API...")
    # Call the LLM API based on the selected provider.
    if api_provider == "anthropic" and modality=="text-only":
        response = anthropic_text_completion(system_prompt, model_name, prompt, thinking)
    elif api_provider == "anthropic":
        response = anthropic_completion(system_prompt, model_name, base64_image, prompt, thinking)
    elif api_provider == "openai" and "o3" in model_name and modality=="text-only":
        response = openai_text_reasoning_completion(system_prompt, model_name, prompt)
    elif api_provider == "openai":
        response = openai_completion(system_prompt, model_name, base64_image, prompt)
    elif api_provider == "gemini" and modality=="text-only":
        response = gemini_text_completion(system_prompt, model_name, prompt)
    elif api_provider == "gemini":
        response = gemini_completion(system_prompt, model_name, base64_image, prompt)
    elif api_provider == "deepseek":
        response = deepseek_text_reasoning_completion(system_prompt, model_name, prompt)
    else:
        raise NotImplementedError(f"API provider: {api_provider} is not supported.")

    latency = time.time() - start_time

    pattern = r'move:\s*(\w+),\s*thought:\s*(.*)'
    matches = re.findall(pattern, response, re.IGNORECASE)

    move_thought_list = []
    # Loop through every move in the order they appear
    for move, thought in matches:
        move = move.strip().lower()
        thought = thought.strip()

        action_pair = {"move": move, "thought": thought}
        move_thought_list.append(action_pair)

        # Log move and thought
        log_output(
            "sokoban_worker",
            f"[INFO] Move executed: ({move}) | Thought: {thought} | Latency: {latency:.2f} sec",
            "sokoban",
            mode="a",
        )

    # response
    return move_thought_list