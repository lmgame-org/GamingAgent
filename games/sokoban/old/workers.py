import time
import os
import pyautogui
import numpy as np
import concurrent.futures

from tools.utils import encode_image, log_output, get_annotate_img
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

def boxxel_read_worker(system_prompt, api_provider, model_name, image_path, level=1):
    base64_image = encode_image(image_path)
    api_provider = "anthropic"
    model_name = "claude-3-7-sonnet-20250219"
    # api_provider = "openai"
    # model_name = "gpt-4-turbo"
    # matrix = load_matrix()
    # if matrix is not None:
    #     board_str = matrix_to_string(matrix)
    # else:
    #     board_str = "No board available."
    # Construct prompt for LLM
    if level == 1:
        level_info = "12, 34, 31, and 53 are docks"
    elif level == 2:
        level_info = "35, 44, and 53 are docks."
    prompt = (
        f"Current level information: {level_info}"
        '''
    Extract the Sokoban board layout from the provided image.  
    Use the existing unique IDs in the image to identify each game element.  
    For each ID, recognize the corresponding element based on color and shape.
    The ID is labeld at center-left in the block. Be carefully think of which block belongs to which id.

    Strictly format the output as: **ID: element type (row, column)**.  
    Each row should reflect the board layout.
    The number of boxes should match the number of Docks.  

    ### Recognized Elements:
    - **Wall**: Red brick block.
    - **Floor**: Dark gray tiled block.
    - **Worker**: Character with a red head and white shirt.
    - **Box**: Brown/gold block with a cross pattern.
    - **Dock** (Goal position): A **white dot** on a floor tile. This white dot is on the center of the floor and the white dot between two digits
    - **Box on Dock**: A box correctly positioned on a dock, appearing **gray with a cross on a dock tile**.
    - **Background**: Pixelated, diagonal, striped, gold-brown texture.

    ### Example Format:
    1: Wall (0, 0) | 2: Floor (0, 1) | 3: Worker (0, 2)  
    4: Box (1, 0) | 5: Dock (1, 1) | 6: Box on Dock (1, 2)  
    7: Background (2, 0)  

    Ensure the format remains consistent and strictly adheres to the example layout.
    '''   )

    
    
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


def api_call(system_prompt, api_provider, model_name, base64_image, prompt, thinking=False):
    if api_provider == "anthropic":
        response = anthropic_completion(system_prompt, model_name, base64_image, prompt, thinking=thinking)
    elif api_provider == "openai":
        response = openai_completion(system_prompt, model_name, base64_image, prompt, temperature=1)
    elif api_provider == "gemini":
        response = gemini_completion(system_prompt, model_name, base64_image, prompt)
    else:
        raise NotImplementedError(f"API provider: {api_provider} is not supported.")
    
    return response

def boxxel_evaluator(system_prompt, api_provider, model_name, table, solution):
    prompt = (
        f"Sokoban table: {table} Please carefully review the sokoban puzzle in the text table"
        f"Please utilize sokoban rules to verify the generated sequences of boxes are valid solutions.{solution}"
        "They shouldn't cause deadlocks. They should be a valid path. No conflict or obstacles are on the path."
        "Please generate your verification result and reflection about why this solution is not valid."
        "Carefully consider worker and box occupy two blocks so when worker move block to right and to left, worker should move to the position left to or right to box and this position should not be wall."
        "Simulate worker's path to see whether it is valid and whether box and worker's position are carefully considered."
    )

    response = api_call(system_prompt, api_provider, model_name, base64_image=None, prompt=prompt, thinking=True)

    return response


def boxxel_worker(system_prompt, api_provider, model_name, prev_response="", level = 1):
    """
    1) Captures a screenshot of the current game state.
    2) Calls an LLM to generate PyAutoGUI code for the next move.
    3) Logs latency and the generated code.
    """
    # Capture a screenshot of the current game state.
    screen_width, screen_height = pyautogui.size()
    region = (0, 0, screen_width, screen_height)
    
    screenshot = pyautogui.screenshot(region=region)

    # Save the screenshot directly in the cache directory.
    os.makedirs("cache/boxxel", exist_ok=True)
    screenshot_path = "cache/boxxel/boxxel_screenshot.png"


    screenshot.save(screenshot_path)
    if level == 2:
        _, _, annotated_cropped_image_path = get_annotate_img(screenshot_path, crop_left=225, crop_right=1570, crop_top=365, crop_bottom=460, grid_rows=9, grid_cols=9, cache_dir=CACHE_DIR)
    elif level == 1:
        _, _, annotated_cropped_image_path = get_annotate_img(screenshot_path, crop_left=255, crop_right=1630, crop_top=400, crop_bottom=520, grid_rows=8, grid_cols=8, cache_dir=CACHE_DIR)
    table = boxxel_read_worker(system_prompt, api_provider, model_name, annotated_cropped_image_path, level = level)
    print(table)
    prompt = (
        f"This is previous reflection: {prev_response}"
        f"Here is the current layout of the Boxxel board:\n{table}\n\n"
        f"Please analyze the board and generate the shortest sequence of numeric IDs "
        f"representing the worker's movement path to push all boxes onto docks. "
        f"Ensure all moves follow Sokoban rules, avoiding walls and deadlocks. "
        "Try to use **Rolling Stone Solver** method to solve this problem and generate a sequence of action for each box."
        "write code to solve this puzzle"
    )


    

    base64_image = encode_image(annotated_cropped_image_path)
    start_time = time.time()

    print(f"Calling {model_name} api...")
    # Call the LLM API based on the selected provider.
    response = api_call(system_prompt, api_provider, model_name, base64_image, prompt, thinking=False)
    print(response)
    reflection = boxxel_evaluator(system_prompt, api_provider, model_name, table, response)


    latency = time.time() - start_time

    # match = re.search(r'actions:\s*([\w\s,]+);\s*thought:\s*(.*)', response, re.IGNORECASE)

    # if match:
    #     actions = [action.strip().lower() for action in match.group(1).split(",")]
    #     thought = match.group(2).strip()
    #     print(actions)
    #     print(thought)

    # for move in actions:
    #     print(f"Executing: {move}")
        
    #     # Perform the move
    #     pyautogui.keyDown(move)
    #     time.sleep(0.01)
    #     pyautogui.keyUp(move)

    # final_output = f"Generated Solution:\n{response}\n\nReflection:\n{reflection}"
    
    return reflection

    # match = re.search(r'move:\s*(\w+)\s*(\d*)\s*;\s*thought:\s*(.*?);\s*planning:\s*(.*)', response, re.IGNORECASE)


    # if match:
    #     move = match.group(1).strip().lower()  # Direction (up, down, left, right)
    #     step_count = match.group(2).strip()  # Step count (if provided)
    #     thought = match.group(3).strip()
    #     planning = match.group(4).strip()

    #     print(f"Move: {move} {step_count}")
    #     print(f"Thought: {thought}")
    #     print(f"Planning: {planning}")
    #     # Default step count to 1 if empty
    #     step_count = int(step_count) if step_count.isdigit() else 1

    #     print(f"Perform next move: {move} {step_count} times")

    #     # Perform the move using multiple key presses
    #     for _ in range(step_count):
    #         pyautogui.keyDown(move)
    #         time.sleep(0.01)
    #         pyautogui.keyUp(move)

    #     # Log the move
    #     log_output(
    #         "boxxel_worker",
    #         f"[INFO] Move executed: ({move} {step_count}) | Thought: {thought} | Latency: {latency:.2f} sec",
    #         "boxxel"
    #     )

    # else:
    #     print("[ERROR] Failed to parse response. Check LLM output format.")

    # return response
    # match = re.search(r'move:\s*(\w+),\s*thought:\s*(.*?),\s*planning:\s*(.*)', response, re.IGNORECASE)
    # move = match.group(1).strip()
    # thought = match.group(2).strip()
    # planning = match.group(3).strip()


    # print(f"perform next move:{move}")
    # pyautogui.press(move.lower().strip())

    # # def perform_move(move):
    # #     key_map = {
    # #         "up": "up",
    # #         "down": "down",
    # #         "left": "left",
    # #         "right": "right",
    # #         "restart": 'R',
    # #         "unmove": 'D'
    # #     }
    # #     if move in key_map:
    # #         pyautogui.press(key_map[move])
    # #         print(f"Performed move: {move}")
    # #     else:
    # #         print(f"[WARNING] Invalid move: {move}")

    # # if move is not None:
    # #     perform_move(move)
    # #     # Log move and thought
    # #     log_output(
    # #         "boxxel_worker",

    # #         f"[INFO] Move executed: ({move}) | Thought: {thought} | Latency: {latency:.2f} sec",
    # #         "boxxel"
    # #     )

    # return response