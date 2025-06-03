import time
import os
import pyautogui
import numpy as np
import datetime
import shutil

from tools.utils import encode_image, log_output, get_annotate_img
from tools.serving.api_manager import APIManager
import re
import json

CACHE_DIR = "cache/2048"

def get_session_dir(model_name, modality, datetime_str=None):
    """
    Creates and returns a session directory for the current API call.
    
    Args:
        model_name (str): Name of the model
        modality (str): Modality (vision_text or text_only)
        datetime_str (str, optional): Timestamp to use for the directory name
        
    Returns:
        str: Path to the session directory
    """
    # Clean model name for directory
    clean_model_name = model_name.lower().split('/')[-1] if '/' in model_name else model_name.lower()
    
    # Format modality to match API manager's format
    formatted_modality = modality.replace('-', '_')
    
    # Create timestamp if not provided
    if not datetime_str:
        datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create directory structure
    session_dir = os.path.join(CACHE_DIR, clean_model_name, formatted_modality, datetime_str)
    os.makedirs(session_dir, exist_ok=True)
    
    return session_dir

def log_move_and_thought(move, thought, latency, session_dir):
    """
    Logs the move and thought process into a log file inside the session directory.
    """
    log_file_path = os.path.join(session_dir, "moves.log")
    
    log_entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Move: {move}, Thought: {thought}, Latency: {latency:.2f} sec\n"
    
    try:
        with open(log_file_path, "a") as log_file:
            log_file.write(log_entry)
    except Exception as e:
        print(f"[ERROR] Failed to write log entry: {e}")

def game_2048_read_worker(system_prompt, api_provider, model_name, image_path, modality="vision_text", thinking=False, session_dir=None, datetime_str=None):
    """
    Extracts the 2048 board layout from an image using the API Manager.
    
    Args:
        system_prompt (str): System prompt for the API
        api_provider (str): Provider (anthropic, openai, gemini)
        model_name (str): Model name
        image_path (str): Path to the annotated image
        modality (str): Input modality (vision_text or text_only)
        thinking (bool): Whether to enable thinking mode
        session_dir (str): Directory to store outputs
        datetime_str (str): Datetime string for directory naming
        
    Returns:
        str: The extracted board layout as text
    """
    # If no session_dir is provided, create one
    if not session_dir:
        session_dir = get_session_dir(model_name, modality, datetime_str)
    
    # Save a copy of the image in the session directory
    board_image_path = os.path.join(session_dir, "board_image.png")
    shutil.copy(image_path, board_image_path)
    
    # Construct prompt for LLM
    prompt = (
        "Extract the 2048 puzzel board layout from the provided image. "
        "Use the existing 4 * 4 grid to generate a text table to represent the game board. "
        "For each square block, recognize the value at center of this block. If it is empty just label it as empty "
        "Strictly format the output as: **value (row, column)**. "
        "Each row should reflect the board layout. "
        "Example format: \n2 (0, 0) | 4 (1, 0)| 16 (2, 0) | 8 (3, 0) \nempty (0,1) | 2 (1, 1)| empty (2, 1)... "
    )
    
    # Initialize API Manager with info for directing output to the correct directory
    api_manager = APIManager(
        game_name="2048",
        base_cache_dir="cache",
        info={
            "model_name": model_name,
            "modality": modality,
            "datetime": datetime_str or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        }
    )
    
    # Call the appropriate API based on modality
    try:
        if modality == "text_only":
            completion, costs = api_manager.text_only_completion(
                model_name=model_name,
                system_prompt=system_prompt,
                prompt=prompt,
                thinking=thinking
            )
        else:  # vision_text
            completion, costs = api_manager.vision_text_completion(
                model_name=model_name,
                system_prompt=system_prompt,
                prompt=prompt,
                image_path=image_path,
                thinking=thinking
            )
            
        # Save costs to the session directory
        with open(os.path.join(session_dir, "board_extraction_costs.json"), "w") as f:
            json.dump({
                "prompt_tokens": costs.get("prompt_tokens", 0),
                "completion_tokens": costs.get("completion_tokens", 0),
                "prompt_cost": str(costs.get("prompt_cost", 0)),
                "completion_cost": str(costs.get("completion_cost", 0)),
                "image_tokens": costs.get("image_tokens", 0) if "image_tokens" in costs else 0
            }, f, indent=2)
        
        # Process response and format as structured board output
        structured_board = completion.strip()
        
        # Save the extracted board to a file
        with open(os.path.join(session_dir, "board_extraction.txt"), "w") as f:
            f.write(structured_board)
        
        # Generate final text output
        final_output = "\n2048 Puzzel Board Representation:\n" + structured_board
        
        return final_output
    
    except Exception as e:
        error_msg = f"Error in board extraction: {str(e)}"
        print(f"[ERROR] {error_msg}")
        
        # Log the error
        with open(os.path.join(session_dir, "errors.log"), "a") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {error_msg}\n")
        
        return "Error extracting board layout."


def game_2048_worker(system_prompt, api_provider, model_name, 
    prev_response="", 
    thinking=True, 
    modality="vision-text",
    datetime_str=None
    ):
    """
    1) Captures a screenshot of the current game state.
    2) Calls an LLM to generate the next move.
    3) Logs latency and the generated move.
    
    Args:
        system_prompt (str): System prompt for the API
        api_provider (str): Provider (anthropic, openai, gemini)
        model_name (str): Model name
        prev_response (str): Previous response from the model
        thinking (bool): Whether to enable thinking mode
        modality (str): Input modality (vision-text or text_only)
        datetime_str (str): Datetime string for directory naming
        
    Returns:
        list: List of move-thought pairs
    """
    # Convert modality to the format expected by the API Manager
    formatted_modality = modality.replace('-', '_')
    assert formatted_modality in ["text_only", "vision_text"], f"modality {modality} is not supported."

    # Create a timestamp if not provided
    if not datetime_str:
        datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize API Manager with the appropriate info
    api_manager = APIManager(
        game_name="2048",
        base_cache_dir="cache",
        info={
            "model_name": model_name,
            "modality": formatted_modality,
            "datetime": datetime_str
        }
    )
    
    # Get the session directory from API Manager
    session_dir = os.path.join(
        "cache", "2048", 
        model_name.lower().split('/')[-1] if '/' in model_name else model_name.lower(),
        formatted_modality, 
        datetime_str
    )
    
    # Create the directory if it doesn't exist
    os.makedirs(session_dir, exist_ok=True)
    
    # Capture and save screenshot
    temp_screenshot_path = os.path.abspath(os.path.join(session_dir, "temp_screenshot.png"))
    pyautogui.screenshot(temp_screenshot_path)
    
    # Now annotate the screenshot and save everything in the session directory
    annotate_image_path, grid_annotation_path, annotate_cropped_image_path = get_annotate_img(
        temp_screenshot_path, 
        crop_left=0, 
        crop_right=0, 
        crop_top=0, 
        crop_bottom=0, 
        grid_rows=4, 
        grid_cols=4,
        enable_digit_label=False, 
        cache_dir=session_dir, 
        black=True
    )
    
    # Extract the board layout using the first API call for text recognition
    print(f"Using {model_name} for text table generation...")
    
    # Construct prompt for LLM
    board_prompt = (
        "Extract the 2048 puzzel board layout from the provided image. "
        "Use the existing 4 * 4 grid to generate a text table to represent the game board. "
        "For each square block, recognize the value at center of this block. If it is empty just label it as empty "
        "Strictly format the output as: **value (row, column)**. "
        "Each row should reflect the board layout. "
        "Example format: \n2 (0, 0) | 4 (1, 0)| 16 (2, 0) | 8 (3, 0) \nempty (0,1) | 2 (1, 1)| empty (2, 1)... "
    )
    
    # First API call to extract the board
    start_time = time.time()
    try:
        if formatted_modality == "text_only":
            board_completion, board_costs = api_manager.text_only_completion(
                model_name=model_name,
                system_prompt=system_prompt,
                prompt=board_prompt,
                thinking=thinking
            )
        else:  # vision_text
            board_completion, board_costs = api_manager.vision_text_completion(
                model_name=model_name,
                system_prompt=system_prompt,
                prompt=board_prompt,
                image_path=annotate_cropped_image_path,
                thinking=thinking
            )
            
        # Process and save the board extraction
        structured_board = board_completion.strip()
        final_output = "\n2048 Puzzel Board Representation:\n" + structured_board
        
        # Save the extracted board to a file in the session directory
        with open(os.path.join(session_dir, "board_extraction.txt"), "w") as f:
            f.write(structured_board)
        
    except Exception as e:
        error_msg = f"Error in board extraction: {str(e)}"
        print(f"[ERROR] {error_msg}")
        
        # Create an error log file
        with open(os.path.join(session_dir, "errors.log"), "a") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {error_msg}\n")
            
        return []  # Return empty list on failure

    print(f"-------------- TABLE --------------\n{final_output}\n")
    print(f"-------------- prev response --------------\n{prev_response}\n")

    # Construct the prompt for the move decision
    move_prompt = (
    "## Previous Lessons Learned\n"
    "- The 2048 board is structured as a 4x4 grid where each tile holds a power-of-two number.\n"
    "- You can slide tiles in four directions (up, down, left, right), merging identical numbers when they collide.\n"
    "- Your goal is to maximize the score and reach the highest possible tile, ideally 2048 or beyond.\n"
    "- You are an expert AI agent specialized in solving 2048 optimally, utilizing advanced heuristic strategies such as the Monte Carlo Tree Search (MCTS) and Expectimax algorithm.\n"
    "- Before making a move, evaluate all possible board states and consider which action maximizes the likelihood of long-term success.\n"
    "- Prioritize maintaining an ordered grid structure to prevent the board from filling up prematurely.\n"
    "- Always keep the highest-value tile in a stable corner to allow efficient merges and maintain control of the board.\n"
    "- Minimize unnecessary movements that disrupt tile positioning and reduce future merge opportunities.\n"
    
    "**IMPORTANT: You must always try a valid direction that leads to a merge. If there are no available merges in the current direction, moving in that direction is invalid. In such cases, choose a new direction where at least two adjacent tiles can merge. Every move should ensure the merging of two or more neighboring tiles to maintain board control and progress.**\n"

    "## Potential Errors to Avoid:\n"
    "1. Grid Disorder Error: Moving tiles in a way that disrupts the structured arrangement of numbers, leading to inefficient merges.\n"
    "2. Edge Lock Error: Moving the highest tile out of a stable corner, reducing long-term strategic control.\n"
    "3. Merge Delay Error: Failing to merge tiles early, causing a filled board with no valid moves.\n"
    "4. Tile Isolation Error: Creating a situation where smaller tiles are blocked from merging due to inefficient movement.\n"
    "5. Forced Move Error: Reaching a state where only one move is possible, reducing strategic flexibility.\n"

    f"Here is your previous response: {prev_response}. Please evaluate your strategy and consider if any adjustments are necessary.\n"
    "Here is the current state of the 2048 board:\n"
    f"{final_output}\n\n"

    "### Output Format:\n"
    "move: up/down/left/right, thought: <brief reasoning>\n\n"
    "Example output: move: left, thought: Maintaining the highest tile in the corner while creating merge opportunities."
    )
    
    # Record the start time for latency calculation
    start_time = time.time()
    
    print(f"Calling {model_name} API for move decision...")
    
    # Make the API call based on the modality
    try:
        if formatted_modality == "text_only":
            completion, costs = api_manager.text_only_completion(
                model_name=model_name,
                system_prompt=system_prompt,
                prompt=move_prompt,
                thinking=thinking
            )
        else:  # vision_text
            # Skip image for o3-mini model
            if "o3-mini" in model_name:
                completion, costs = api_manager.text_only_completion(
                    model_name=model_name,
                    system_prompt=system_prompt,
                    prompt=move_prompt,
                    thinking=thinking
                )
            else:
                completion, costs = api_manager.vision_text_completion(
                    model_name=model_name,
                    system_prompt=system_prompt,
                    prompt=move_prompt,
                    image_path=annotate_cropped_image_path,
                    thinking=thinking
                )
        
        # Calculate latency
        latency = time.time() - start_time
        
        # Extract the move and thought from the response
        pattern = r'move:\s*(\w+),\s*thought:\s*(.*)'
        matches = re.findall(pattern, completion, re.IGNORECASE)
        
        # Save the full response
        with open(os.path.join(session_dir, "response.txt"), "w") as f:
            f.write(completion)
        
        move_thought_list = []
        # Process each move-thought pair
        for move, thought in matches:
            move = move.strip().lower()
            thought = thought.strip()
            
            action_pair = {"move": move, "thought": thought}
            move_thought_list.append(action_pair)
            
            # Log move and thought
            log_entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Move: {move}, Thought: {thought}, Latency: {latency:.2f} sec\n"
            
            # Write to moves.log in the session directory
            with open(os.path.join(session_dir, "moves.log"), "a") as log_file:
                log_file.write(log_entry)
            
            # Also log to the legacy format for backward compatibility
            log_output(
                "sokoban_worker",
                f"[INFO] Move executed: ({move}) | Thought: {thought} | Latency: {latency:.2f} sec",
                "sokoban",
                mode="a",
            )
        
        # Save the moves to a JSON file
        with open(os.path.join(session_dir, "moves.json"), "w") as f:
            json.dump(move_thought_list, f, indent=2)
        
        return move_thought_list
    
    except Exception as e:
        error_msg = f"Error in move decision: {str(e)}"
        print(f"[ERROR] {error_msg}")
        
        # Log the error
        with open(os.path.join(session_dir, "errors.log"), "a") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {error_msg}\n")
        
        # Return an empty list to indicate failure
        return []