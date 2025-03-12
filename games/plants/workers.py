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

from tools.utils import encode_image, log_output, get_annotate_img, take_screenshot
from tools.serving.api_providers import anthropic_completion, anthropic_text_completion, openai_completion, openai_text_reasoning_completion, gemini_completion, gemini_text_completion, deepseek_text_reasoning_completion
import re
import json

CACHE_DIR = "cache/plants"
os.makedirs(CACHE_DIR, exist_ok=True)

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


def plants_state_read_worker(system_prompt, api_provider, model_name, image_path, modality="vision-text", thinking=False):
    base64_image = encode_image(image_path)
    print(f"Using {model_name} for text table generation...")
    # Construct prompt for LLM
    prompt = (
    "Extract plants and sun point information from the first 9 blocks of the game board. "
    "The provided image represents the game state of Plants vs. Zombies, where you select plants based on collected sun points. "
    "Follow these specific instructions to extract and format the data correctly:"
    "\n\n1. Identify the first 9 blocks of the game board. "
    "   - The first block displays the current number of sun points available. "
    "   - The remaining 8 blocks contain plant selection cards."
    "\n2. For each plant selection card, extract the following information: "
    "   - The plant type (e.g., sunflower, peashooter, snow peashooter, etc.). "
    "   - The sun point cost of the plant. "
    "   - The plant's availability: "
    "     - If the card is dark, it means the plant is on cooldown (CD = 0). "
    "     - If the card is bright, it is selectable (CD = 1)."
    "\n3. Format the output strictly as follows: "
    "   - Each block's data should be represented in the format: **id: (plant, cost, CD)** "
    "   - Maintain the board's original row layout when displaying the extracted data."
    "\n4. Example output format: "
    "   1:(sunflower,50,1) | 2:(peashooter,100,0) | 3:(wallnut,50,1) | ..."
    "\n5. Recognized plant types: sunflower, peashooter, snow peashooter, wallnut, cherry bomb, repeater pea, chomper, spikeweed."
    )

    
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
    
    # Process response and format as structured board output
    structured_board = response.strip()
    
    # Generate final text output
    final_output = "\nPlants VS Zombies Plants Status Representation:\n" + structured_board

    return final_output

def plants_board_read_worker(system_prompt, api_provider, model_name, image_path, modality="vision-text", thinking=False):
    base64_image = encode_image(image_path)
    print(f"Using {model_name} for text table generation...")
    # Construct prompt for LLM
    prompt = (
        "Extract the 2048 puzzel board layout from the provided image. "
        "Use the existing 4 * 4 grid to generate a text table to represent the game board. "
        "For each square block, recognize the value at center of this block. If it is empty just label it as empty "
        "Strictly format the output as: **value (row, column)**. "
        "Each row should reflect the board layout. "
        "Example format: \n2 (0, 0) | 4 (1, 0)| 16 (2, 0) | 8 (3, 0) \nempty (0,1) | 2 (1, 1)| empty (2, 1)... "
    )
    
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
    
    # Process response and format as structured board output
    structured_board = response.strip()
    
    # Generate final text output
    final_output = "\n2048 Puzzel Board Representation:\n" + structured_board

    return final_output

def plants_worker(system_prompt, api_provider, model_name, 
    prev_response="", 
    thinking=True, 
    modality="vision-text",
    ):
    """
    1) Captures a screenshot of the current game state.
    2) Calls an LLM to generate PyAutoGUI code for the next move.
    3) Logs latency and the generated code.
    """
    # Capture a screenshot of the current game state.
    # Save the screenshot directly in the cache directory.
    assert modality in ["text-only", "vision-text"], f"modality {modality} is not supported."

    thread1_dir = os.path.join(CACHE_DIR, "thread1")
    thread2_dir = os.path.join(CACHE_DIR, "thread2")

    os.makedirs(thread1_dir, exist_ok=True)
    os.makedirs(thread2_dir, exist_ok=True)
    
    screenshot_path = os.path.abspath(os.path.join(CACHE_DIR, "plants_screenshot.png"))
    # thread1_screenshot_path, position = take_screenshot("Plant VS Zombies Game" if platform.system() == "Windows" else "Terminal", screenshot_path)

    # thread 1 - cut off the board
    thread1_annotate_image_path, thread2_grid_annotation_path, thread1_annotate_cropped_image_path = get_annotate_img(screenshot_path, crop_left=30, crop_right=30, crop_top=0, crop_bottom=520, grid_rows=1, grid_cols=13, enable_digit_label=True, cache_dir=thread1_dir, line_thickness=3, black=True)
    # thread 2 - cut off the cards
    thread2_annotate_image_path, thread1_grid_annotation_path, thread2_annotate_cropped_image_path = get_annotate_img(screenshot_path, crop_left=30, crop_right=30, crop_top=85, crop_bottom=30, grid_rows=5, grid_cols=9, enable_digit_label=True, cache_dir=thread2_dir, line_thickness=3, black=True)
    table = plants_state_read_worker(system_prompt, api_provider, model_name, thread1_annotate_cropped_image_path, thinking=thinking, modality=modality)
    print(table)
    # print(table)
    # print(f"-------------- TABLE --------------\n{table}\n")
    # print(f"-------------- prev response --------------\n{prev_response}\n")

    # prompt = (
    # "## Previous Lessons Learned\n"
    # "- The 2048 board is structured as a 4x4 grid where each tile holds a power-of-two number.\n"
    # "- You can slide tiles in four directions (up, down, left, right), merging identical numbers when they collide.\n"
    # "- Your goal is to maximize the score and reach the highest possible tile, ideally 2048 or beyond.\n"
    # "- You are an expert AI agent specialized in solving 2048 optimally, utilizing advanced heuristic strategies such as the Monte Carlo Tree Search (MCTS) and Expectimax algorithm.\n"
    # "- Before making a move, evaluate all possible board states and consider which action maximizes the likelihood of long-term success.\n"
    # "- Prioritize maintaining an ordered grid structure to prevent the board from filling up prematurely.\n"
    # "- Always keep the highest-value tile in a stable corner to allow efficient merges and maintain control of the board.\n"
    # "- Minimize unnecessary movements that disrupt tile positioning and reduce future merge opportunities.\n"
    
    # "**IMPORTANT: You must always try a valid direction that leads to a merge or a tile move. If there are no available merges and no tile moves in the current direction, moving in that direction is invalid. In such cases, choose a new direction where at least two adjacent tiles can merge or where at least a tile can move. Every move should ensure the merging of two or more neighboring tiles to maintain board control and progress.**\n"

    # "## Potential Errors to Avoid:\n"
    # "1. Grid Disorder Error: Moving tiles in a way that disrupts the structured arrangement of numbers, leading to inefficient merges.\n"
    # "2. Edge Lock Error: Moving the highest tile out of a stable corner, reducing long-term strategic control.\n"
    # "3. Merge Delay Error: Failing to merge tiles early, causing a filled board with no valid moves.\n"
    # "4. Tile Isolation Error: Creating a situation where smaller tiles are blocked from merging due to inefficient movement.\n"
    # "5. Forced Move Error: Reaching a state where only one move is possible, reducing strategic flexibility.\n"

    # f"Here is your previous response: {prev_response}. Please evaluate your strategy and consider if any adjustments are necessary.\n"
    # "Here is the current state of the 2048 board:\n"
    # f"{table}\n\n"

    # "### Output Format:\n"
    # "move: up/down/left/right, thought: <brief reasoning>\n\n"
    # "Example output: move: left, thought: Maintaining the highest tile in the corner while creating merge opportunities."
    # )


    # base64_image = encode_image(annotate_cropped_image_path)
    # if "o3-mini" in model_name:
    #     base64_image = None
    # start_time = time.time()

    # print(f"Calling {model_name} API...")
    # # Call the LLM API based on the selected provider.
    # if api_provider == "anthropic" and modality=="text-only":
    #     response = anthropic_text_completion(system_prompt, model_name, prompt, thinking)
    # elif api_provider == "anthropic":
    #     response = anthropic_completion(system_prompt, model_name, base64_image, prompt, thinking)
    # elif api_provider == "openai" and "o3" in model_name and modality=="text-only":
    #     response = openai_text_reasoning_completion(system_prompt, model_name, prompt)
    # elif api_provider == "openai":
    #     response = openai_completion(system_prompt, model_name, base64_image, prompt)
    # elif api_provider == "gemini" and modality=="text-only":
    #     response = gemini_text_completion(system_prompt, model_name, prompt)
    # elif api_provider == "gemini":
    #     response = gemini_completion(system_prompt, model_name, base64_image, prompt)
    # elif api_provider == "deepseek":
    #     response = deepseek_text_reasoning_completion(system_prompt, model_name, prompt)
    # else:
    #     raise NotImplementedError(f"API provider: {api_provider} is not supported.")

    # latency = time.time() - start_time

    # pattern = r'move:\s*(\w+),\s*thought:\s*(.*)'
    # matches = re.findall(pattern, response, re.IGNORECASE)

    # move_thought_list = []
    # # Loop through every move in the order they appear
    # for move, thought in matches:
    #     move = move.strip().lower()
    #     thought = thought.strip()

    #     action_pair = {"move": move, "thought": thought}
    #     move_thought_list.append(action_pair)

    #     # Log move and thought
    #     log_output(
    #         "sokoban_worker",
    #         f"[INFO] Move executed: ({move}) | Thought: {thought} | Latency: {latency:.2f} sec",
    #         "sokoban",
    #         mode="a",
    #     )

    # # response
    # return move_thought_list