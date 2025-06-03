import time
import os
import pyautogui
import numpy as np

import re
from PIL import Image
import json

import concurrent.futures

from tools.utils import encode_image, log_output, extract_python_code, read_log_to_string, extract_patch_table, extract_game_table, get_annotate_img, get_annotate_patched_img
from tools.serving.api_providers import anthropic_completion, openai_completion, gemini_completion, anthropic_text_completion, openai_text_completion, gemini_text_completion, openai_text_reasoning_completion

color_map = {
    0: "Empty",
    1: "Cyan",
    2: "Blue",
    3: "Orange",
    4: "Yellow",
    5: "Green",
    6: "Purple",
    7: "Red"
}

def state_to_text(json_file):
    with open(json_file, "r") as f:
        state = json.load(f)
    
    print("file loaded...")
    
    # Start building the output string
    output = f"Current Grid Status ({state['grid']['num_rows']} rows x {state['grid']['num_cols']} cols):\n"
    
    # Process each row of the grid
    for row in state["grid"]["cells"]:
        # Convert each cell number to its corresponding color name.
        row_str = " | ".join([color_map.get(cell, str(cell)) for cell in row])
        output += row_str + "\n"
    
    # Add details for the current block if present.
    if "current_block" in state:
        current = state["current_block"]
        output += f"\nCurrent Block ID: {current.get('id')}\nPositions: "
        positions = current.get("positions", [])
        output += ", ".join([f"({pos['row']}, {pos['column']})" for pos in positions])
        output += "\n"
    
    return output

def load_block_shapes(filename):
    # Read the JSON file and return the data
    with open(filename, 'r') as file:
        return json.load(file)

def create_block_shapes_prompt(data):
    prompt = f"Block shapes data:\n{json.dumps(data)}\n"
    return prompt

def game_table_to_matrix(game_table_text):
    """
    Convert a game table text (with rows like 'Row0:   0 1 0 1 ...') into a 2D list matrix.
    
    The expected format is:
    Column  0  1  2  3  4  5  6  7  8  9
    Row0:   0  1  0  1  0  0  1  0  1  0
    Row1:   1  0  1  0  1  1  0  1  0  0
    ...
    
    Returns:
        matrix (list of lists): Each inner list contains strings '0' or '1' for one row.
    """
    matrix = []
    # Find all row lines using regex; this captures the content after the row label.
    row_lines = re.findall(r"Row\d+:\s*(.*)", game_table_text)
    for line in row_lines:
        # Extract each digit (assumes digits are separated by whitespace)
        row = re.findall(r"([01])", line)
        matrix.append(row)
    return matrix

def matrix_to_text_table(matrix):
    """Convert a 2D list matrix into a structured text table."""
    header = "ID  | Item Type    | Position"
    line_separator = "-" * len(header)
    
    item_map = {
        '1': 'block',
        '0': 'Empty',
    }
    
    table_rows = [header, line_separator]
    item_id = 1
    
    for row_idx, row in enumerate(matrix):
        for col_idx, cell in enumerate(row):
            item_type = item_map.get(cell, 'Unknown')
            table_rows.append(f"{item_id:<3} | {item_type:<12} | ({col_idx}, {row_idx})")
            item_id += 1
    
    return "\n".join(table_rows)

def tetris_board_aggregator(system_prompt, api_provider, model_name, complete_annotate_cropped_image_path, patch_list):
    """
    Calls an LLM to merge multiple patch tables into one unified Tetris board.
    
    patch_list (list of tuples): Each element is (patch_num, patch_table).
        For example:
        [
            (0, 
                "Column 0 1 2\n
                Row0:   0 1 0\n
                Row1:   1 1 1\n
            ..."),
            (1, 
                "Column 3 4 5\n
                Row0:   0 0 1\n
                Row1:   0 0 1\n
            ..."),
            ...
        ]
    Returns:
        str: A single text block representing the merged board.
    """

    aggregator_prompt = (
        "You will also be provided a patch index number that counts for location of the provided 5x5 sub-grid  within the entire grid.\n"
        "Your task is to merge these patches into one unified board. "
        "Take care to align rows and columns appropriately.\n\n"
        "Here are the patch tables:\n\n"
    )

    for patch_num, patch_table in patch_list:
        aggregator_prompt += f"--- PATCH {patch_num} ---\n{patch_table}\n\n"

    # TODO (lanxiang): make patch granularity configurable
    aggregator_prompt += (
        "Patches are arranged as follows:\n"
        "[Patch 0][Patch 1]\n"
        "[Patch 2][Patch 3]\n"
        "...\n"
        "Now, please produce a final merged text table. "
        "Follow a similar format to the patches, labeling columns and rows clearly. "
        "Ensure that cells align properly across patches. "
        "No cells are overlapped across patches.\n\n"
        "Start both row and column indices with 0, and take patch offset (each is 5x5) into consideration.\n\n"
        "## Output Format:\n"
        "```game table\n"
        "Column  0  1  2  3  4  5  6  7  8  9\n"
        "Row0:   x  x  x  x  x  x  x  x  x  x\n"
        "Row1:   x  x  x  x  x  x  x  x  x  x\n"
        "Row2:   x  x  x  x  x  x  x  x  x  x\n"
        "...\n"
        "Row19:  x  x  x  x  x  x  x  x  x  x\n"
        "```\n" 
        "where each element 'x' takes either 0 or 1.\n\n"
    )

    base64_image = encode_image(complete_annotate_cropped_image_path)

    # Dispatch request to your chosen LLM
    if api_provider == "anthropic":
        # anthropic_completion(system_prompt, model_name, base64_image, aggregator_prompt)
        merged_response = anthropic_text_completion(system_prompt, model_name, aggregator_prompt)
    elif api_provider == "openai":
        merged_response = openai_text_completion(system_prompt, model_name, aggregator_prompt)
    elif api_provider == "gemini":
        merged_response = gemini_text_completion(system_prompt, model_name, aggregator_prompt)
    else:
        raise NotImplementedError(f"API provider: {api_provider} is not supported.")

    # Return the LLM's merged board text
    merged_table = extract_game_table(merged_response)
    return merged_table.strip()

def tetris_board_reader(system_prompt, api_provider, model_name, image_path, patch_num):
    """
    Reads part of the Tetris board from an image using a vision-language model (VLM).
    Returns a text-based representation of the board.
    
    Example:
    PATCH 0
      "Column   0  1  2  3  4  
      "Row0:    0  0  0  1  0\n"
      "Row1:    0  1  1  1  0\n"
      "Row2:    0  0  0  0  0\n"
      ...

    PATCH 1
      "Column   5  6  7  8  9
      "Row0:    1  1  1  0  0\n"
      "Row1:    1  1  1  0  0\n"
      "Row2:    1  1  1  0  0\n"
      ...
    """
    # TODO (lanxiang): make patch granularity configurable
    vlm_prompt = (
        "Extract the number grid layout from the provided image. "
        "Each block is represented by contiguous blocks with an unique background color.\n"
        "Only consider the following colors as color-filled:\n"
        "green = (47, 230, 23), "
	    "red = (232, 18, 18), "
        "orange = (226, 116, 17), "
	    "yellow = (237, 234, 4), "
	    "purple = (166, 0, 247), "
	    "cyan = (21, 204, 209), "
	    "light_blue = (59, 85, 162). "
        "## The following DO NOT count as color-filled:\n"
        "1. Dark (deep blue) color background.\n"
        "2. Green grid lines.\n"
        "DO NOT add annotated numbers to the grid, marking empty cells as 0 and color-filled cells by 1.\n\n"
        "## Output Format:\n"
        "```game patch table\n"
        "x x x x x\n"
        "x x x x x\n"
        "x x x x x\n"
        "x x x x x\n"
        "x x x x x\n"
        "```\n"
        "where each element 'x' takes either 0 or 1.\n"
        "Start both row and column indices with 0.\n\n"
        f"## Now start extracting the text table. The provided PATCH index number is: {patch_num}.\n"
    )

    base64_image = encode_image(image_path)
    
    if api_provider == "anthropic":
        response = anthropic_completion(system_prompt, model_name, base64_image, vlm_prompt)
    elif api_provider == "openai":
        response = openai_completion(system_prompt, model_name, base64_image, vlm_prompt)
    elif api_provider == "gemini":
        response = gemini_completion(system_prompt, model_name, base64_image, vlm_prompt)
    else:
        raise NotImplementedError(f"API provider: {api_provider} is not supported.")
    
    # The response should contain the textual board representation
    subboard_text = response.strip()
    patch_level_text_table = extract_patch_table(subboard_text)

    print(f"---- Tetris Sub-Board {patch_num} (small) ----")
    print(patch_level_text_table)
    print("--------------------------------")

    return patch_num, patch_level_text_table

def tetris_worker(
    thread_id,
    offset,
    system_prompt,
    board_reader_system_prompt,
    board_aggregator_system_prompt,
    api_provider,
    model_name,
    board_reader_api_provider,
    board_reader_model_name,
    modality,
    input_type="read-from-game-backend",
    plan_seconds=2.0,
    total_patch_num=8,
    crop_left=10,
    crop_right=182,
    crop_top=8,
    crop_bottom=2,
    grid_rows=20,
    grid_cols=10,
    cache_folder="/games/tetris/Python-Tetris-Game-Pygame/cache/tetris"
):
    """
    vision reasoning modality:
        A single Tetris worker that plans moves for 'plan_seconds'.
        1) Sleeps 'offset' seconds before starting (to stagger starts).
        2) Continuously:
            - Captures a screenshot
            - Calls the LLM with a Tetris prompt that includes 'plan_seconds'
            - Extracts the Python code from the LLM output
            - Executes the code with `exec()`
    vision-text reasoning modality:
        A single Tetris worker that plans moves for 'plan_seconds'.
        1) Sleeps 'offset' seconds before starting (to stagger starts).
        2) Continuously:
            - Captures a screenshot
            - Calls the VLM with a prompt to convert the game state into a text-table
            - Feed into a LLM , to generate the Python code
            - Extracts the Python code from the LLM output
            - Executes the code with `exec()`
    """
    assert modality in ["vision", "vision-text", "text-only"], f"{modality} modality is not supported."
    assert input_type in ["read-from-game-backend", "read-from-ui"], f"{input_type} input type is not supported."
    all_response_time = []

    time.sleep(offset)
    print(f"[Thread {thread_id}] Starting after {offset}s delay... (Plan: {plan_seconds} seconds)")

    tetris_prompt_template = """
Analyze the current Tetris board state and generate PyAutoGUI code to control the active Tetris piece for the next {plan_seconds} second(s).
An active Tetris piece appears from the top, and is not connected to the bottom of the screen is to be placed.

## Board state reference as a text table:
{board_text}

## General Tetris Controls (example keybinds).
- left: move the piece left by 1 grid unit.
- right: move the piece right by 1 grid unit.
- up: rotate the piece clockwise once by 90 degrees.
- down: accelerated drop (use ONLY IF you are very confident its control won't propagate to the next piece. DO NOT repeat more than 5 times).

## Tetris Geometry
- Tetris pieces in the game follow the following configurations:
{tetris_configurations}
- Every Tetris piece starts at state 0, every rotation will transit the piece to the next state modulo 4.
- Consider each Tetris piece occupies its nearest 3x3 grid (or 4x4 for I-shape). Each configurable specifies which grid unit are occupied by each rotation state.
- Place each piece such that the flat sides align with the sides or geometric structure at the bottom.

## Game Physics
- The game is played on a 10x20 grid.
- Blocks fall at a rate of approximately 1 grid unit every 3 seconds.
- Pressing the down key moves the block down by 1 grid unit.
- Rotations will be performed within the nearest 9x9 block, and shapes will be changed accordingly.

## Planning

### Principles
- Maximize Cleared Lines: prioritize moves that clear the most rows.
- Minimize Holes: avoid placements that create empty spaces that can only be filled by future pieces.
- Minimize Bumpiness: keep the playfield as flat as possible to avoid difficult-to-fill gaps.
- Minimize Aggregate Height: lower the total height of all columns to delay top-outs.
- Minimize Maximum Height: prevent any single column from growing too tall, which can lead to an early game over.

### Strategies
- Try clear the bottom-most line first.
- Imagine what shape the entire structure will form after the current active piece is placed. Avoid leaving any holes.
- Do not move a block piece back and forth. Plan a trajectory and generate the code.

### Code generation and latency
- In generated code, only consider the current block.
- At the time the code is executed, 3~5 seconds have elapsed.
- The entire sequence of key presses should be feasible within {plan_seconds} second(s).

### Lessons learned
{experience_summary}

## Output Format:
- Output ONLY the Python code for PyAutoGUI commands, e.g. `pyautogui.press("left")`.
- Include brief comments for each action.
- Do not print anything else besides these Python commands.
"""
    # TODO: make path configurable
    game_state_file_path = "/Users/lhu/workspace/game_arena_env/Python-Tetris-Game-Pygame/cache/tetris/state.json"

    block_shape_file_path = "games/tetris/data/block_shapes.json"
    block_shapes_info = load_block_shapes(block_shape_file_path)
    print("block shape file loaded.")
    block_shapes_prompt = create_block_shapes_prompt(block_shapes_info)
    
    iter_counter = 0
    try:
        while True:
            # Read information passed from the speculator cache
            try:
                # FIXME (lanxiang): make thread count configurable, currently planner is only in thread 0
                experience_summary = read_log_to_string(f"cache/tetris/thread_0/planner/experience_summary.log")
            except Exception as e:
                experience_summary = "- No lessons learned so far."
            
            print(f"-------------- experience summary --------------\n{experience_summary}\n------------------------------------\n")
            
            # Create a unique folder for this thread's cache
            screenshot_path = os.path.join(cache_folder, "screenshot.png")

            # Cache the screenshot content
            img = Image.open(screenshot_path)
            # Save the image to the new path.
            cache_path = f"cache/tetris/thread_{thread_id}/iter_{iter_counter}"
            os.makedirs(cache_path, exist_ok=True)
            cache_screenshot_path = os.path.join(cache_path, "screenshot.png")
            img.save(cache_screenshot_path)

            # Encode the screenshot
            print("starting a round of annotations...")
            _, _, annotate_cropped_image_paths = get_annotate_patched_img(screenshot_path, 
            crop_left=crop_left, crop_right=crop_right, 
            crop_top=crop_top, crop_bottom=crop_bottom, 
            grid_rows=grid_rows, grid_cols=grid_cols, 
            x_dim=5, y_dim=5, cache_dir=cache_folder)

            _, _, complete_annotate_cropped_image_path = get_annotate_img(screenshot_path, crop_left=crop_left, crop_right=crop_right, crop_top=crop_top, crop_bottom=crop_bottom, grid_rows=grid_rows, grid_cols=grid_cols, cache_dir=cache_folder)

            base64_image = encode_image(complete_annotate_cropped_image_path)

            print("finished a round of annotations.")

            patch_table_list = []
            if input_type == "read-from-game-backend":
                formatted_text_table = state_to_text(game_state_file_path)
            elif modality == "vision-text" or modality == "text-only":
                try:
                    threads = []
                    # read individual game sub-boards
                    with concurrent.futures.ThreadPoolExecutor(max_workers=total_patch_num) as executor:
                        for i in range(total_patch_num):
                            threads.append(
                                executor.submit(
                                    tetris_board_reader, board_reader_system_prompt, board_reader_api_provider, board_reader_model_name, annotate_cropped_image_paths[i], i
                                )
                            )
                        
                        for _ in concurrent.futures.as_completed(threads):
                            patch_table_list.append(_.result())
                    
                    print("patch table list generated.")

                    sorted_patch_table_list = sorted(patch_table_list, key=lambda x: x[0])
                    # aggreagte sub-boards to a bigger one
                    board_text = tetris_board_aggregator(board_aggregator_system_prompt, board_reader_api_provider, board_reader_model_name, complete_annotate_cropped_image_path, sorted_patch_table_list)

                    matrix = game_table_to_matrix(board_text)
                    formatted_text_table = matrix_to_text_table(matrix)
                    print("Formatted Text Table:")
                    print(formatted_text_table)

                except Exception as e:
                    print(f"Error extracting Tetris board text conversion: {e}")
                    formatted_text_table = "[NO CONVERTED BOARD TEXT]"
            elif modality == "vision":
                # In pure "vision" modality, we do not parse the board via text
                formatted_text_table = "[NO CONVERTED BOARD TEXT]"
            else:
                raise NotImplementedError(f"modality: {modality} is not supported.")

            print("---- Tetris Board (textual) ----")
            print(formatted_text_table)
            print("--------------------------------")
            
            tetris_prompt = tetris_prompt_template.format(
                board_text=formatted_text_table,
                tetris_configurations=block_shapes_prompt,
                plan_seconds=plan_seconds,
                experience_summary=experience_summary,
            )

            print(f"============ complete Tetris prompt ============\n{tetris_prompt}\n===========================\n")

            start_time = time.time()

            try:
                # HACK: o3-mini only support text-only modality for now
                if api_provider == "openai" and "o3" in model_name and modality=="text-only":
                    generated_code_str = openai_text_reasoning_completion(system_prompt, model_name, tetris_prompt)
                elif api_provider == "anthropic" and modality=="text-only":
                    print("calling text-only API...")
                    generated_code_str = anthropic_text_completion(system_prompt, model_name, tetris_prompt)
                elif api_provider == "anthropic":
                    print("calling vision API...")
                    generated_code_str = anthropic_completion(system_prompt, model_name, base64_image, tetris_prompt)
                elif api_provider == "openai":
                    generated_code_str = openai_completion(system_prompt, model_name, base64_image, tetris_prompt)
                elif api_provider == "gemini":
                    generated_code_str = gemini_completion(system_prompt, model_name, base64_image, tetris_prompt)
                else:
                    raise NotImplementedError(f"API provider: {api_provider} is not supported.")

            except Exception as e:
                print(f"[Thread {thread_id}] Error executing code: {e}")

            end_time = time.time()
            latency = end_time - start_time
            all_response_time.append(latency)

            print(f"[Thread {thread_id}] Request latency: {latency:.2f}s")
            avg_latency = np.mean(all_response_time)
            print(f"[Thread {thread_id}] Latencies: {all_response_time}")
            print(f"[Thread {thread_id}] Average latency: {avg_latency:.2f}s\n")

            print(f"[Thread {thread_id}] --- API output ---\n{generated_code_str}\n")

            # Extract Python code for execution
            clean_code = extract_python_code(generated_code_str)
            log_output(thread_id, f"[Thread {thread_id}] Python code to be executed:\n{clean_code}\n", "tetris", f"iter_{iter_counter}")
            print(f"[Thread {thread_id}] Python code to be executed:\n{clean_code}\n")

            try:
                exec(clean_code)
            except Exception as e:
                print(f"[Thread {thread_id}] Error executing code: {e}")
            
            iter_counter += 1

    except KeyboardInterrupt:
        print(f"[Thread {thread_id}] Interrupted by user. Exiting...")
