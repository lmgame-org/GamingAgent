import time
import os
import pyautogui
import numpy as np

from tools.utils import encode_image, log_output, extract_python_code, get_annotate_img
from tools.serving.api_providers import anthropic_completion, openai_completion, gemini_completion, anthropic_text_completion, gemini_text_completion, openai_text_reasoning_completion, deepseek_text_reasoning_completion

cache_dir = "cache/candy_crush"

import time
import os
import pyautogui
import numpy as np

from tools.utils import encode_image, log_output, extract_python_code, get_annotate_img
from tools.serving.api_providers import anthropic_completion, openai_completion, gemini_completion
import re
import json

CACHE_DIR = "cache/candy_crush"

def log_move_and_thought(move, thought, latency):
    """
    Logs the move and thought process into a log file inside the cache directory.
    """
    log_file_path = os.path.join(CACHE_DIR, "candy_crush_moves.log")
    
    log_entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Move: {move}, Thought: {thought}, Latency: {latency:.2f} sec\n"
    
    try:
        with open(log_file_path, "a") as log_file:
            log_file.write(log_entry)
    except Exception as e:
        print(f"[ERROR] Failed to write log entry: {e}")

def candy_crush_read_worker(system_prompt, api_provider, model_name, image_path, modality, thinking):
    base64_image = encode_image(image_path)
    
    # Construct prompt for LLM
    prompt = (
        "Extract the Candy Crush board layout from the provided image. "
        "Use the existing unique IDs in the image to identify each candy type. "
        "For each ID, recognize the corresponding candy based on color and shape. "
        "Strictly format the output as: **ID: candy type (row, column)**. "
        "Each row should reflect the board layout. "
        "Example format: \n1: blue sphere candy (0, 0) | 2: green square candy (0, 1)| 3: red bean candy (0, 2)... \n8: orange jelly candy (1,0) | 9: yellow teardrop candy (1, 1)| 10: purple cluster candy (1, 2) "
    )
    
    # Call LLM API based on provider
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
    
    # Process response and format as structured board output
    structured_board = response.strip()
    
    # Generate final text output
    final_output = "\nCandy Crush Board Representation:\n" + structured_board

    return final_output


def candy_crush_worker(system_prompt, api_provider, model_name, modality, thinking, crop_left=700, crop_right=800, crop_top=300, crop_bottom=300, grid_rows=7, grid_cols=7, prev_response=""):
    """
    Worker function for short-term (1 second) control in Candy Crush.
    1) Captures a screenshot of the current Candy Crush game state.
    2) Calls an LLM to generate PyAutoGUI code for the next move.
    3) Logs latency and the generated code.
    """


    # Capture a screenshot of the current game state.
    screen_width, screen_height = pyautogui.size()
    region = (0, 0, screen_width, screen_height)
    screenshot = pyautogui.screenshot(region=region)

    # Save the screenshot directly in the cache directory.
    os.makedirs(cache_dir, exist_ok=True)
    screenshot_path = os.path.join(cache_dir, "screenshot.png")

    screenshot.save(screenshot_path)

    annotate_image_path, grid_annotation_path, annotate_cropped_image_path = get_annotate_img(screenshot_path, crop_left=crop_left, crop_right=crop_right, crop_top=crop_top, crop_bottom=crop_bottom, grid_rows=grid_rows, grid_cols=grid_cols, cache_dir=CACHE_DIR)

    candy_crush_text_table = candy_crush_read_worker(system_prompt, "anthropic", "claude-3-7-sonnet-20250219", annotate_cropped_image_path, modality="vision-text", thinking=False)

    prompt = (
        f"Here is the layout of the Candy Crush board:\n\n"
        f"{candy_crush_text_table}\n\n"
        "Please carefully analyze the candy crush table corresponding to the input image. Figure out next best move."
        f"Previous response: {prev_response}. Use previous responses as references, explore new move different from previous moves, and find additional three-match opportunities.\n\n"
        "Please generate next move for this candy crush game."
        "## STRICT OUTPUT FORMAT ##\n"
        "- Respond in this format:\n"
        '  **move: "(U, M)", thought: "(explaination)"**\n\n'
        "U and M are unique ids for candies. You can reason with coordinate but finally output U and M corresponding ids."
    )
    

    base64_image = encode_image(annotate_cropped_image_path)
    start_time = time.time()

    print(f"Calling {model_name} api...")
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

    # Extract the move (X, Y) and thought from LLM response
    move_match = re.search(r'move:\s*"\((\d+),\s*(\d+)\)"', response)
    thought_match = re.search(r'thought:\s*"(.*?)"', response)

    if not move_match or not thought_match:
        log_output("candy_crush_worker", f"[ERROR] Invalid LLM response: {response}", "candy_crush")
        return

    id_1, id_2 = int(move_match.group(1)), int(move_match.group(2))
    thought_text = thought_match.group(1)

    print(f"Extracted move: ({id_1}, {id_2})")
    print(f"LLM Thought Process: {thought_text}")

    log_move_and_thought(f"({id_1}, {id_2})", thought_text, latency)

    # Read the grid annotations to find coordinates
    try:
        with open(grid_annotation_path, "r") as file:
            grid_data = json.load(file)
    except Exception as e:
        log_output("candy_crush_worker", f"[ERROR] Failed to read grid annotations: {e}", "candy_crush")
        return

    # Find coordinates for the extracted IDs
    pos_1 = next((entry for entry in grid_data if entry['id'] == id_1), None)
    pos_2 = next((entry for entry in grid_data if entry['id'] == id_2), None)

    if not pos_1 or not pos_2:
        log_output("candy_crush_worker", f"[ERROR] IDs not found in grid: {id_1}, {id_2}", "candy_crush")
        return

    x1, y1 = pos_1["x"], pos_1["y"]
    x2, y2 = pos_2["x"], pos_2["y"]
    print(f"Swapping ({id_1} -> {x1}, {y1}) with ({id_2} -> {x2}, {y2})")

    # Perform the swap using PyAutoGUI
    pyautogui.moveTo(x1, y1, duration=0.2)
    pyautogui.mouseDown()
    pyautogui.moveTo(x2, y2, duration=0.2)
    pyautogui.mouseUp()

    # Log move and thought
    log_output(
        "candy_crush_worker",
        f"[INFO] Move executed: ({id_1}, {id_2}) | Thought: {thought_text} | Latency: {latency:.2f} sec",
        "candy_crush"
    )

    return response
