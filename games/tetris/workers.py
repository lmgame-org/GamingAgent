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

Tetris_Cache_Path = "/games/tetris/Python-Tetris-Game-Pygame/cache/tetris"

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

# Map grid cell values to text.
bit_map = {
    0: "0",
    1: "1",
    2: "1",
    3: "1",
    4: "1",
    5: "1",
    6: "1",
    7: "1"
}

# Predefined pivot (center) for each tetromino in its canonical (minimal) matrix.
pivot_map = {
    "O": (0, 0),
    "I": (0, 1),   # For a 1x4 representation, choose the second cell.
    "S": (1, 1),
    "Z": (1, 1),
    "T": (0, 1),   # For our canonical 2x3 T shape: top row is [1,1,1] and bottom row is [0,1,0].
    "J": (1, 1),
    "L": (1, 1)
}

def get_rotated_coordinates(block, block_pivot, target_pivot):
    """
    Given a rotated block (matrix) and its pivot, along with the target pivot in the grid,
    return the list of real grid coordinates for each '1' cell.
    """
    offset_r = target_pivot[0] - block_pivot[0]
    offset_c = target_pivot[1] - block_pivot[1]
    rotated_coords = []
    for i, row in enumerate(block):
        for j, cell in enumerate(row):
            if cell == 1:
                rotated_coords.append((offset_r + i, offset_c + j))
    return rotated_coords


def state_to_text_list(json_file):
    with open(json_file, "r") as f:
        state = json.load(f)
    
    num_rows = state['grid']['num_rows']
    num_cols = state['grid']['num_cols']
    
    # Build the static grid (board) from the state (without current block overlay).
    static_grid = [
        [bit_map.get(cell, str(cell)) for cell in row]
        for row in state["grid"]["cells"]
    ]
    
    # Build a base grid text (for reference).
    base_grid_text = f"Current Grid Status ({num_rows} rows x {num_cols} cols):\n"
    for row in static_grid:
        base_grid_text += " | ".join(row) + "\n"
    
    outputs = []
    
    if "current_block" in state:
        current = state["current_block"]
        positions = current.get("positions", [])
        
        # Mark the current block positions on the static grid (for the details output).
        marked_positions = [f"({pos['row']}, {pos['column']})" for pos in positions]
        block_details = f"Current Block ID: {current.get('id')}\nPositions: " + ", ".join(marked_positions) + "\n"
        
        # Detect the shape from positions.
        detected_shape = detect_shape(positions)
        
        # Compute the target pivot in the grid.
        # Normalize the current block positions.
        min_row = min(pos['row'] for pos in positions)
        min_col = min(pos['column'] for pos in positions)
        canonical_pivot = pivot_map.get(detected_shape, (0, 0))
        # The grid cell where the pivot is located.
        target_pivot = (min_row + canonical_pivot[0], min_col + canonical_pivot[1])
        
        # Generate rotation states (each is a tuple: (matrix, pivot) in that matrix).
        rotations = generate_rotations(detected_shape)
        
        # For each rotation state, overlay the rotated block onto a copy of the static grid.
        for i, (rot_matrix, rot_pivot) in enumerate(rotations):
            overlay_grid = overlay_block_on_grid_with_pivot(static_grid, rot_matrix, rot_pivot, target_pivot)
            overlay_text = table_to_text(overlay_grid)
            # full_output = base_grid_text + "\n" + block_details + \
            #               f"\nRotation {i+1} for shape {detected_shape} (pivot remains at {target_pivot}):\n" + overlay_text + "\n"
            rotated_coords = get_rotated_coordinates(rot_matrix, rot_pivot, target_pivot)
            coords_text = "Rotated Coordinates: " + ", ".join(str(coord) for coord in rotated_coords)
            full_output = (
                f"\nRotation {i+1} for shape {detected_shape} (pivot remains at {target_pivot}):\n"
                f"\nThe input block shape: {detected_shape}.\n"
                f"{overlay_text}\n{coords_text}\n"
            )
            outputs.append(full_output)
    else:
        outputs.append(base_grid_text)
    
    return outputs

def rotate_shape_about_pivot(matrix, pivot):
    """
    Rotate the shape (matrix) 90° clockwise about the given pivot.
    The pivot is given as a coordinate (row, col) relative to the matrix.
    Returns a tuple: (new_matrix, new_pivot) where new_matrix is the rotated shape
    in a minimal bounding box and new_pivot is the pivot's coordinate in that new matrix.
    """
    coords = []
    for i, row in enumerate(matrix):
        for j, cell in enumerate(row):
            if cell:
                coords.append((i, j))
    new_coords = []
    pr, pc = pivot
    for (i, j) in coords:
        # Compute offset from pivot.
        off_i = i - pr
        off_j = j - pc
        # Rotate offset: (off_i, off_j) -> (off_j, -off_i)
        new_i = pr + off_j
        new_j = pc - off_i
        new_coords.append((new_i, new_j))
    # Re-normalize: shift so that the minimal coordinate becomes (0,0).
    min_i = min(i for i, j in new_coords)
    min_j = min(j for i, j in new_coords)
    normalized = {(i - min_i, j - min_j) for (i, j) in new_coords}
    # The new pivot in the normalized coordinates.
    new_pivot = (pivot[0] - min_i, pivot[1] - min_j)
    # Build the new matrix.
    max_i = max(i for i, j in normalized)
    max_j = max(j for i, j in normalized)
    new_matrix = [[0]*(max_j+1) for _ in range(max_i+1)]
    for (i, j) in normalized:
        new_matrix[i][j] = 1
    return new_matrix, new_pivot

def generate_rotations(shape):
    """
    Return a list of rotation states for the given tetromino shape.
    Each state is a tuple: (matrix, pivot), where matrix is a 2D list
    representing the shape (with 1's for blocks) and pivot is the center coordinate
    that should remain fixed.
    Number of states:
      O: 1, I/S/Z: 2, T/J/L: 4.
    """
    shapes = {
        "O": [[1, 1],
              [1, 1]],
        "I": [[1, 1, 1, 1]],
        "S": [[0, 1, 1],
              [1, 1, 0]],
        "Z": [[1, 1, 0],
              [0, 1, 1]],
        "T": [[1, 1, 1],
              [0, 1, 0]],
        "J": [[1, 0, 0],
              [1, 1, 1]],
        "L": [[0, 0, 1],
              [1, 1, 1]],
    }
    base = shapes.get(shape)
    if base is None:
        return []
    pivot = pivot_map.get(shape, (0,0))
    
    rotations = []
    if shape == "O":
        rotations.append((base, pivot))
    elif shape in ["I", "S", "Z"]:
        # Two states: base and one rotation.
        rotations.append((base, pivot))
        rot_matrix, rot_pivot = rotate_shape_about_pivot(base, pivot)
        rotations.append((rot_matrix, rot_pivot))
    elif shape in ["T", "J", "L"]:
        # Four rotations.
        current_matrix = base
        current_pivot = pivot
        for _ in range(4):
            rotations.append((current_matrix, current_pivot))
            current_matrix, current_pivot = rotate_shape_about_pivot(current_matrix, current_pivot)
    else:
        rotations.append((base, pivot))
    return rotations

def overlay_block_on_grid_with_pivot(base_grid, block, block_pivot, target_pivot):
    """
    Overlay the block (2D list) onto a copy of the base_grid such that the block's pivot
    (block_pivot) is mapped to target_pivot in the grid.
    """
    new_grid = [row.copy() for row in base_grid]
    offset_r = target_pivot[0] - block_pivot[0]
    offset_c = target_pivot[1] - block_pivot[1]
    for i, row in enumerate(block):
        for j, cell in enumerate(row):
            if cell == 1:
                new_r = offset_r + i
                new_c = offset_c + j
                if 0 <= new_r < len(new_grid) and 0 <= new_c < len(new_grid[0]):
                    new_grid[new_r][new_c] = "1"
    return new_grid

def table_to_text(table):
    """Convert a 2D list (table) to a text table string with cells separated by ' | '."""
    lines = []
    for i, row in enumerate(table):
        line = " | ".join(str(x) for x in row)
        line = f"Row {i:02d}: " + line
        lines.append(line)
    return "\n".join(lines)

def detect_shape(positions):
    """
    Detect the tetromino shape from the positions of the current block.
    Returns one of: O, I, T, S, Z, J, L, or "?" if not detected.
    """
    if not positions or len(positions) != 4:
        return "?"
    coords = [(pos['row'], pos['column']) for pos in positions]
    min_r = min(r for r, c in coords)
    min_c = min(c for r, c in coords)
    norm = {(r - min_r, c - min_c) for r, c in coords}
    canonical = {
        "O": {(0,0), (0,1), (1,0), (1,1)},
        "I": {(0,0), (0,1), (0,2), (0,3)},
        "T": {(0,0), (0,1), (0,2), (1,1)},
        "S": {(0,1), (0,2), (1,0), (1,1)},
        "Z": {(0,0), (0,1), (1,1), (1,2)},
        "J": {(0,0), (1,0), (1,1), (1,2)},
        "L": {(0,2), (1,0), (1,1), (1,2)},
    }
    for shape, pattern in canonical.items():
        candidate = norm
        for _ in range(4):
            if candidate == pattern:
                return shape
            candidate = rotate_coords(candidate)
    return "?"

def rotate_coords(coords):
    """Rotate a set of normalized coordinates 90° clockwise and re-normalize."""
    max_r = max(r for r, c in coords)
    rotated = {(c, max_r - r) for r, c in coords}
    min_r = min(r for r, c in rotated)
    min_c = min(c for r, c in rotated)
    normalized = {(r - min_r, c - min_c) for r, c in rotated}
    return normalized
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

import os

def vision_worker(game_state_file_path="games/tetris/Python-Tetris-Game-Pygame/cache/tetris/state.json"):
    """
    Interpret game state if the file exists; otherwise, return an error message.
    """
    if not os.path.exists(game_state_file_path):
        return f"Error: File '{game_state_file_path}' does not exist."
    
    return state_to_text_list(game_state_file_path)

def plan_worker(system_prompt, api_provider, model_name, input_text_table, base64_image=None, modality="text-only"):

    tetris_prompt = f"""
You are an expert Tetris move planner. Analyze the current board (provided below) and determine the best move based on these priorities:
1. **Clear Lines:** Aim to form or clear as many lines as possible.(First Priority)
2. **Avoid Bubbles:** Do not create any empty spaces completely enclosed by blocks.(Second Priority)
3. **Maintain Geometry:** Keep the falling piece’s original shape intact.
4. Even Distribution: Strive for a balanced layout by spreading elements evenly. Avoid creating tall, narrow blocks that are difficult to complete or fill, ensuring each block remains manageable in size and shape.

The input is a game state of a Tetris game. Block areas are 1 and empty areas are 0. The input block is in the middle of the table and the input block is falling down. Don't need to rotate the block.

Current Board:
{input_text_table}

Please find the best move.

Output your answer in the following JSON-like format:
{{
    move: (a string describing the horizontal moves needed, e.g., "right right" if the block must move two cells to the right),
    result_table: (a text table showing the board state immediately after the falling block locks in place, before any automatic line clearance),
    result_summary: (a brief summary describing how many bubbles were created and how many full lines were formed by this move)
}}

Generate the best move and the immediate resulting board state after the falling block.
"""
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
    
    return generated_code_str



def tetris_worker(
    system_prompt,
    api_provider,
    model_name,
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
    # assert modality in ["vision", "vision-text", "text-only"], f"{modality} modality is not supported."
    # assert input_type in ["read-from-game-backend", "read-from-ui"], f"{input_type} input type is not supported."
    # all_response_time = []

    # time.sleep(offset)
    # print(f"[Thread {thread_id}] Starting after {offset}s delay... (Plan: {plan_seconds} seconds)")

#     tetris_prompt_template = """
# Analyze the current Tetris board state and generate PyAutoGUI code to control the active Tetris piece for the next {plan_seconds} second(s).
# An active Tetris piece appears from the top, and is not connected to the bottom of the screen is to be placed.

# ## Board state reference as a text table:
# {board_text}

# ## General Tetris Controls (example keybinds).
# - left: move the piece left by 1 grid unit.
# - right: move the piece right by 1 grid unit.
# - up: rotate the piece clockwise once by 90 degrees.
# - down: accelerated drop (use ONLY IF you are very confident its control won't propagate to the next piece. DO NOT repeat more than 5 times).

# ## Tetris Geometry
# - Tetris pieces in the game follow the following configurations:
# {tetris_configurations}
# - Every Tetris piece starts at state 0, every rotation will transit the piece to the next state modulo 4.
# - Consider each Tetris piece occupies its nearest 3x3 grid (or 4x4 for I-shape). Each configurable specifies which grid unit are occupied by each rotation state.
# - Place each piece such that the flat sides align with the sides or geometric structure at the bottom.

# ## Game Physics
# - The game is played on a 10x20 grid.
# - Blocks fall at a rate of approximately 1 grid unit every 3 seconds.
# - Pressing the down key moves the block down by 1 grid unit.
# - Rotations will be performed within the nearest 9x9 block, and shapes will be changed accordingly.

# ## Planning

# ### Principles
# - Maximize Cleared Lines: prioritize moves that clear the most rows.
# - Minimize Holes: avoid placements that create empty spaces that can only be filled by future pieces.
# - Minimize Bumpiness: keep the playfield as flat as possible to avoid difficult-to-fill gaps.
# - Minimize Aggregate Height: lower the total height of all columns to delay top-outs.
# - Minimize Maximum Height: prevent any single column from growing too tall, which can lead to an early game over.

# ### Strategies
# - Try clear the bottom-most line first.
# - Imagine what shape the entire structure will form after the current active piece is placed. Avoid leaving any holes.
# - Do not move a block piece back and forth. Plan a trajectory and generate the code.

# ### Code generation and latency
# - In generated code, only consider the current block.
# - At the time the code is executed, 3~5 seconds have elapsed.
# - The entire sequence of key presses should be feasible within {plan_seconds} second(s).

# ### Lessons learned
# {experience_summary}

# ## Output Format:
# - Output ONLY the Python code for PyAutoGUI commands, e.g. `pyautogui.press("left")`.
# - Include brief comments for each action.
# - Do not print anything else besides these Python commands.
# """
    # TODO: make path configurable
    text_tables = vision_worker()
    for text_table in text_tables:
        print(text_table)

    
    response = plan_worker(system_prompt, api_provider, model_name, text_table, base64_image=None, modality="text-only")
    print(response)
    # block_shape_file_path = "games/tetris/data/block_shapes.json"
    # block_shapes_info = load_block_shapes(block_shape_file_path)
    # block_shapes_prompt = create_block_shapes_prompt(block_shapes_info)
    # print(f"block_shape_prompt: {block_shapes_prompt}")
    
    # iter_counter = 0
    # try:
    #     while True:
    #         # Read information passed from the speculator cache
    #         # try:
    #         #     # FIXME (lanxiang): make thread count configurable, currently planner is only in thread 0
    #         #     experience_summary = read_log_to_string(f"cache/tetris/thread_0/planner/experience_summary.log")
    #         # except Exception as e:
    #         #     experience_summary = "- No lessons learned so far."
            
    #         # print(f"-------------- experience summary --------------\n{experience_summary}\n------------------------------------\n")
            
    #         # # Create a unique folder for this thread's cache
    #         # screenshot_path = os.path.join(cache_folder, "screenshot.png")

    #         # # Cache the screenshot content
    #         # img = Image.open(screenshot_path)
    #         # # Save the image to the new path.
    #         # cache_path = f"cache/tetris/thread_{thread_id}/iter_{iter_counter}"
    #         # os.makedirs(cache_path, exist_ok=True)
    #         # cache_screenshot_path = os.path.join(cache_path, "screenshot.png")
    #         # img.save(cache_screenshot_path)

    #         # # Encode the screenshot
    #         # print("starting a round of annotations...")
    #         # _, _, annotate_cropped_image_paths = get_annotate_patched_img(screenshot_path, 
    #         # crop_left=crop_left, crop_right=crop_right, 
    #         # crop_top=crop_top, crop_bottom=crop_bottom, 
    #         # grid_rows=grid_rows, grid_cols=grid_cols, 
    #         # x_dim=5, y_dim=5, cache_dir=cache_folder)

    #         # _, _, complete_annotate_cropped_image_path = get_annotate_img(screenshot_path, crop_left=crop_left, crop_right=crop_right, crop_top=crop_top, crop_bottom=crop_bottom, grid_rows=grid_rows, grid_cols=grid_cols, cache_dir=cache_folder)

    #         # base64_image = encode_image(complete_annotate_cropped_image_path)

    #         # print("finished a round of annotations.")

    #         # patch_table_list = []
    #         # if input_type == "read-from-game-backend":
    #             formatted_text_table = state_to_text(game_state_file_path)
    #         # elif modality == "vision-text" or modality == "text-only":
    #         #     try:
    #         #         threads = []
    #         #         # read individual game sub-boards
    #         #         with concurrent.futures.ThreadPoolExecutor(max_workers=total_patch_num) as executor:
    #         #             for i in range(total_patch_num):
    #         #                 threads.append(
    #         #                     executor.submit(
    #         #                         tetris_board_reader, board_reader_system_prompt, board_reader_api_provider, board_reader_model_name, annotate_cropped_image_paths[i], i
    #         #                     )
    #         #                 )
                        
    #         #             for _ in concurrent.futures.as_completed(threads):
    #         #                 patch_table_list.append(_.result())
                    
    #         #         print("patch table list generated.")

    #         #         sorted_patch_table_list = sorted(patch_table_list, key=lambda x: x[0])
    #         #         # aggreagte sub-boards to a bigger one
    #         #         board_text = tetris_board_aggregator(board_aggregator_system_prompt, board_reader_api_provider, board_reader_model_name, complete_annotate_cropped_image_path, sorted_patch_table_list)

    #         #         matrix = game_table_to_matrix(board_text)
    #         #         formatted_text_table = matrix_to_text_table(matrix)
    #         #         print("Formatted Text Table:")
    #         #         print(formatted_text_table)

    #         #     except Exception as e:
    #         #         print(f"Error extracting Tetris board text conversion: {e}")
    #         #         formatted_text_table = "[NO CONVERTED BOARD TEXT]"
    #         # elif modality == "vision":
    #         #     # In pure "vision" modality, we do not parse the board via text
    #         #     formatted_text_table = "[NO CONVERTED BOARD TEXT]"
    #         # else:
    #         #     raise NotImplementedError(f"modality: {modality} is not supported.")

    #         print("---- Tetris Board (textual) ----")
    #         print(formatted_text_table)
    #         print("--------------------------------")
            
    #         tetris_prompt = tetris_prompt_template.format(
    #             board_text=formatted_text_table,
    #             tetris_configurations=block_shapes_prompt,
    #             plan_seconds=plan_seconds,
    #             experience_summary=experience_summary,
    #         )

    #         print(f"============ complete Tetris prompt ============\n{tetris_prompt}\n===========================\n")

    #         start_time = time.time()

    #         try:
    #             # HACK: o3-mini only support text-only modality for now
    #             if api_provider == "openai" and "o3" in model_name and modality=="text-only":
    #                 generated_code_str = openai_text_reasoning_completion(system_prompt, model_name, tetris_prompt)
    #             elif api_provider == "anthropic" and modality=="text-only":
    #                 print("calling text-only API...")
    #                 generated_code_str = anthropic_text_completion(system_prompt, model_name, tetris_prompt)
    #             elif api_provider == "anthropic":
    #                 print("calling vision API...")
    #                 generated_code_str = anthropic_completion(system_prompt, model_name, base64_image, tetris_prompt)
    #             elif api_provider == "openai":
    #                 generated_code_str = openai_completion(system_prompt, model_name, base64_image, tetris_prompt)
    #             elif api_provider == "gemini":
    #                 generated_code_str = gemini_completion(system_prompt, model_name, base64_image, tetris_prompt)
    #             else:
    #                 raise NotImplementedError(f"API provider: {api_provider} is not supported.")

    #         except Exception as e:
    #             print(f"[Thread {thread_id}] Error executing code: {e}")

    #         end_time = time.time()
    #         latency = end_time - start_time
    #         all_response_time.append(latency)

    #         print(f"[Thread {thread_id}] Request latency: {latency:.2f}s")
    #         avg_latency = np.mean(all_response_time)
    #         print(f"[Thread {thread_id}] Latencies: {all_response_time}")
    #         print(f"[Thread {thread_id}] Average latency: {avg_latency:.2f}s\n")

    #         print(f"[Thread {thread_id}] --- API output ---\n{generated_code_str}\n")

    #         # Extract Python code for execution
    #         clean_code = extract_python_code(generated_code_str)
    #         log_output(thread_id, f"[Thread {thread_id}] Python code to be executed:\n{clean_code}\n", "tetris", f"iter_{iter_counter}")
    #         print(f"[Thread {thread_id}] Python code to be executed:\n{clean_code}\n")

    #         try:
    #             exec(clean_code)
    #         except Exception as e:
    #             print(f"[Thread {thread_id}] Error executing code: {e}")
            
    #         iter_counter += 1

    # except KeyboardInterrupt:
    #     print(f"[Thread {thread_id}] Interrupted by user. Exiting...")
