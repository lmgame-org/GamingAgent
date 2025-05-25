import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
# import imageio # No longer used for video writing
import tempfile
import shutil
import statistics # For median calculation
import subprocess # For calling ffmpeg
from typing import Optional, Dict, List, Tuple, Union # Added Dict, List, Tuple, Union
import ast # Added import
import re # Added for Candy Crush textual parsing

# Import for Candy Crush image creation
# from gamingagent.envs.custom_03_candy_crush.candy_crush_env import create_board_image_candy_crush, COLOR_MAP as CANDY_CRUSH_ENV_COLOR_MAP


# Default seconds per frame for videos (1 FPS)
DEFAULT_SECONDS_PER_FRAME = 1.0

# --- Sokoban Specific Constants (adapted from sokobanEnv.py) ---
SOKOBAN_ASSET_DIR = os.path.join(os.path.dirname(__file__), "..", "gamingagent", "envs", "custom_02_sokoban", "assets", "images")

# Mapping from room_state numerical values to characters
SOKOBAN_ROOM_STATE_TO_CHAR = {
    0: '#', 1: ' ', 2: '?', 3: '*', 4: '$', 5: '@', 6: '+' # Changed 2: '.' to 2: '?'
}
# Mapping from characters in textual_representation to numerical room_state values
SOKOBAN_CHAR_TO_ROOM_STATE = {v: k for k, v in SOKOBAN_ROOM_STATE_TO_CHAR.items()}
# The explicit SOKOBAN_CHAR_TO_ROOM_STATE['?'] = 2 is no longer needed due to the change above


# --- Candy Crush Specific Constants & Utilities (Copied and adapted) ---
CANDY_CRUSH_REPLAY_COLOR_MAP = { # Renamed from COLOR_MAP
    0: " ",  # Empty or background
    1: "G",  # Green
    2: "C",  # Cyan
    3: "P",  # Purple
    4: "R",  # Red
    5: "Y",  # Yellow 
    6: "B",  # Blue 
}
# Update CANDY_CRUSH_CHAR_TO_IDX to use the new local map
CANDY_CRUSH_CHAR_TO_IDX = {v: k for k, v in CANDY_CRUSH_REPLAY_COLOR_MAP.items() if v.strip()}


# --- Font Loading Utilities (can be shared or adapted) ---
POTENTIAL_FONTS = [
    "arial.ttf", "Arial.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    # Add other common system font paths if necessary
]

def _load_font(font_name_list: List[str], size: int, default_message: Optional[str] = None) -> ImageFont.FreeTypeFont:
    font = None
    for font_name in font_name_list:
        try:
            font = ImageFont.truetype(font_name, size)
            break
        except (OSError, IOError):
            continue
    if font is None:
        try:
            font = ImageFont.load_default(size=size)
        except TypeError: 
             font = ImageFont.load_default() # type: ignore
        if default_message:
            print(f"[ReplayUtils] {default_message}")
    return font

# --- Sokoban Asset Loading Utility ---
def load_sokoban_asset_image_for_replay(path: str, size: Tuple[int, int]) -> Optional[Image.Image]:
    """Loads and resizes a Sokoban asset image."""
    asset_full_path = os.path.join(SOKOBAN_ASSET_DIR, path)
    if not os.path.exists(asset_full_path):
        print(f"[ReplayUtils] Warning: Sokoban asset not found at {asset_full_path}")
        return None
    try:
        img = Image.open(asset_full_path).convert("RGBA")
        return img.resize(size, Image.Resampling.LANCZOS)
    except Exception as e:
        print(f"[ReplayUtils] Error loading Sokoban asset {asset_full_path}: {e}")
        return None

# --- Sokoban Textual Board Parser ---
def parse_sokoban_textual_board(text_board_str: str, char_to_state_map: Dict[str, int]) -> Optional[np.ndarray]:
    """Parses a textual representation of the Sokoban board into a numerical numpy array."""
    if not text_board_str:
        return None
    
    try:
        # Safely evaluate the string representation of the list of lists
        board_list_of_lists = ast.literal_eval(text_board_str)
        if not isinstance(board_list_of_lists, list) or \
           not all(isinstance(row, list) for row in board_list_of_lists) or \
           not board_list_of_lists: # check for empty list of lists
            print(f"[ReplayUtils] Warning: Parsed textual board is not a non-empty list of lists. Input: {text_board_str[:100]}...")
            return None
        if not all(all(isinstance(char, str) for char in row) for row in board_list_of_lists): # Check all elements are strings
            print(f"[ReplayUtils] Warning: Parsed textual board contains non-string elements. Input: {text_board_str[:100]}...")
            return None
        # Convert list of lists of chars to list of strings (one string per row)
        lines = ["".join(row) for row in board_list_of_lists]
    except (ValueError, SyntaxError) as e:
        print(f"[ReplayUtils] Warning: Could not parse textual board string '{text_board_str[:100]}...' as list of lists: {e}")
        return None
    
    if not lines:
        return None
    
    rows = len(lines)
    cols = len(lines[0])
    
    # Validate consistent line lengths
    if not all(len(line) == cols for line in lines):
        print("[ReplayUtils] Warning: Inconsistent line lengths in Sokoban textual board. Cannot parse.")
        return None
        
    board_numerical = np.ones((rows, cols), dtype=np.uint8) # Default to floor (1)

    for r, line in enumerate(lines):
        for c, char_val in enumerate(line):
            # The _parse_level_data in sokobanEnv uses a two-pass approach.
            # For replay, we directly map the character representing the final state of the tile.
            # We need to handle cases where a character might imply an underlying feature (e.g. player on target)
            # For simplicity here, we'll use direct mapping. More complex rendering could infer fixed vs state.
            
            # Base state from char
            state_val = char_to_state_map.get(char_val)
            
            if state_val is not None:
                board_numerical[r, c] = state_val
            else:
                # Fallback for unknown characters - could be floor or an error
                board_numerical[r, c] = 1 # Default to floor
                print(f"[ReplayUtils] Warning: Unknown character '{char_val}' in Sokoban textual board at ({r},{c}). Treating as floor.")
    return board_numerical

# --- Sokoban Board Image Creation ---
def create_board_image_sokoban_for_replay(
    board_state_numerical: np.ndarray,
    save_path: str,
    tile_size: int = 32,
    perf_score: Optional[float] = None,
    action_taken_str: Optional[str] = None
):
    """Creates a visualization of the Sokoban board for replay frames, including score and action text."""
    if board_state_numerical is None:
        img = Image.new('RGB', (tile_size * 5, tile_size * 5), (128, 128, 128))
        draw = ImageDraw.Draw(img)
        error_font = _load_font(POTENTIAL_FONTS, tile_size // 2, "Error font not found, using default.")
        draw.text((10,10), "Error: No board state", fill=(255,0,0), font=error_font)
        if save_path:
             os.makedirs(os.path.dirname(save_path), exist_ok=True)
             img.save(save_path)
        return

    rows, cols = board_state_numerical.shape
    
    # Determine font size for info text
    info_font_size = max(10, tile_size // 3) # Adjusted for potentially smaller tiles
    info_font = _load_font(POTENTIAL_FONTS, info_font_size, "Info font for Sokoban replay not found.")
    text_color = (0, 0, 0) # Black text
    text_padding = 5 # Padding from edges

    # Calculate potential additional height needed for the text strip at the top
    # This is a rough estimate; more precise calculation would involve font metrics
    num_info_lines = 0
    if perf_score is not None: num_info_lines +=1
    if action_taken_str is not None: num_info_lines +=1
    
    # Estimate line height based on font size. Add a small buffer.
    estimated_line_height = info_font_size + 4 
    info_strip_height = 0
    if num_info_lines > 0:
        info_strip_height = (num_info_lines * estimated_line_height) + (text_padding * 2)

    img_width = cols * tile_size
    img_height_board = rows * tile_size
    total_img_height = img_height_board + info_strip_height

    img = Image.new('RGB', (img_width, total_img_height), (200, 200, 200)) # Background
    draw = ImageDraw.Draw(img)

    # Draw the info text first (if any) in the top strip
    current_y_text = text_padding
    if info_strip_height > 0:
        draw.rectangle([0, 0, img_width, info_strip_height], fill=(180, 180, 180)) # Light gray for text bg
        if perf_score is not None:
            score_text = f"Perf: {perf_score:.2f}"
            draw.text((text_padding, current_y_text), score_text, fill=text_color, font=info_font)
            current_y_text += estimated_line_height
        if action_taken_str is not None:
            action_text = f"Action: {action_taken_str}"
            # Truncate action_text if too long to fit (simple truncation)
            max_action_len = (img_width - 2 * text_padding) // (info_font_size // 2) # Rough estimate
            if len(action_text) > max_action_len and max_action_len > 3:
                action_text = action_text[:max_action_len-3] + "..."
            draw.text((text_padding, current_y_text), action_text, fill=text_color, font=info_font)

    # Asset loading (remains the same)
    asset_files = { 
        "wall": "wall.png", "floor": "floor.png", "box": "box.png",
        "box_on_target": "box_docked.png", "player": "worker.png",
        "player_on_target": "worker_dock.png", "target": "dock.png",
    }
    assets = {k: load_sokoban_asset_image_for_replay(p, (tile_size, tile_size)) for k, p in asset_files.items()}

    # Draw the board, offset by info_strip_height
    board_offset_y = info_strip_height
    for r in range(rows):
        for c in range(cols):
            x0, y0 = c * tile_size, r * tile_size + board_offset_y # Apply offset here
            tile_val = board_state_numerical[r, c]

            if assets["floor"]:
                img.paste(assets["floor"], (x0, y0), assets["floor"] if assets["floor"].mode == 'RGBA' else None)
            
            asset_to_draw = None
            if tile_val == 0: asset_to_draw = assets["wall"]
            elif tile_val == 2: asset_to_draw = assets["target"]
            elif tile_val == 3:
                if assets["target"]: img.paste(assets["target"], (x0, y0), assets["target"] if assets["target"].mode == 'RGBA' else None)
                asset_to_draw = assets["box_on_target"]
            elif tile_val == 4: asset_to_draw = assets["box"]
            elif tile_val == 5: asset_to_draw = assets["player"]
            elif tile_val == 6:
                if assets["target"]: img.paste(assets["target"], (x0, y0), assets["target"] if assets["target"].mode == 'RGBA' else None)
                asset_to_draw = assets["player_on_target"]
            
            if asset_to_draw:
                img.paste(asset_to_draw, (x0, y0), asset_to_draw if asset_to_draw.mode == 'RGBA' else None)
            elif tile_val == 1: pass # Floor already drawn
            else: 
                draw.rectangle([x0, y0, x0 + tile_size, y0 + tile_size], fill=(100, 100, 100))
                char_for_val = SOKOBAN_ROOM_STATE_TO_CHAR.get(tile_val, "?")
                fallback_font = _load_font(POTENTIAL_FONTS, tile_size // 2, "Fallback font for unknown tile not found.")
                # Simple text centering for fallback char
                text_w_fb, text_h_fb = tile_size // 4, tile_size // 2 # Rough estimate
                if hasattr(fallback_font, 'getbbox'):
                    bbox = fallback_font.getbbox(char_for_val)
                    text_w_fb = bbox[2] - bbox[0]
                    text_h_fb = bbox[3] - bbox[1]
                elif hasattr(fallback_font, 'getsize'):
                     text_w_fb, text_h_fb = fallback_font.getsize(char_for_val)

                draw.text((x0 + (tile_size - text_w_fb) // 2, y0 + (tile_size - text_h_fb) // 2), char_for_val, fill=(255,255,255), font=fallback_font)

    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir: os.makedirs(save_dir, exist_ok=True)
        try:
            img.save(save_path)
        except Exception as e:
            print(f"[ReplayUtils] Error saving Sokoban board image to {save_path}: {e}")

# --- Candy Crush Textual Board Parser ---
def parse_candy_crush_textual_board(text_board_str: str, char_to_idx_map: Dict[str, int]) -> Tuple[Optional[np.ndarray], Optional[int], Optional[int]]:
    """
    Parses a textual representation of the Candy Crush board (and score/moves) 
    into a numerical numpy array for the board, and extracts score and moves left.
    Example textual_representation format:
    Board:
    0| R C G C P P C R
    1| P C R R C G P P
    ...
    Score: 120
    Moves Left: 35
    Last Action: ((1,2),(1,3))
    """
    if not text_board_str:
        return None, None, None

    board_lines = []
    current_score = None
    moves_left = None
    
    in_board_section = False
    for line in text_board_str.split('\\\\n'): # Using escaped newline
        line_strip = line.strip()
        if line_strip.startswith("Board:"):
            in_board_section = True
            continue
        
        if in_board_section:
            if re.match(r"^\d+\|", line_strip): # Line like "0| R C G..."
                board_content = line_strip.split("|", 1)[-1].strip()
                board_lines.append(board_content)
            else: # End of board section (e.g., encountering "Score:")
                in_board_section = False
        
        # Check for Score and Moves Left even if not in board section anymore
        if line_strip.startswith("Score:"):
            try:
                current_score = int(line_strip.split(":", 1)[-1].strip())
            except ValueError:
                print(f"[ReplayUtils] Warning: Could not parse score from line: {line_strip}")
        elif line_strip.startswith("Moves Left:"):
            try:
                moves_left = int(line_strip.split(":", 1)[-1].strip())
            except ValueError:
                print(f"[ReplayUtils] Warning: Could not parse moves left from line: {line_strip}")

    if not board_lines:
        print("[ReplayUtils] Warning: No board lines found in Candy Crush textual representation.")
        return None, current_score, moves_left

    try:
        rows = len(board_lines)
        # Assume all board lines have space-separated characters
        cols = len(board_lines[0].split()) 
        
        board_numerical = np.zeros((rows, cols), dtype=np.int32) # Default to 0 (empty or background)

        for r, line_content in enumerate(board_lines):
            chars = line_content.split()
            if len(chars) != cols:
                print(f"[ReplayUtils] Warning: Inconsistent column count in Candy Crush textual board at row {r}. Expected {cols}, got {len(chars)}. Line: '{line_content}'")
                return None, current_score, moves_left # Or handle more gracefully
            for c, char_val in enumerate(chars):
                idx_val = char_to_idx_map.get(char_val)
                if idx_val is not None:
                    board_numerical[r, c] = idx_val
                else:
                    board_numerical[r, c] = 0 # Default for unknown char
                    print(f"[ReplayUtils] Warning: Unknown character '{char_val}' in Candy Crush textual board at ({r},{c}). Treating as empty.")
        return board_numerical, current_score, moves_left
    except Exception as e:
        print(f"[ReplayUtils] Error parsing Candy Crush board_lines: {e}. Lines: {board_lines}")
        return None, current_score, moves_left

def overlay_text_on_image(
    source_image_path: str,
    output_save_path: str,
    perf_score: Optional[float] = None,
    action_taken_str: Optional[str] = None,
    reference_size_for_font: int = 32 # Similar to tile_size for consistency
):
    """Opens an existing image, overlays performance score and action text, and saves it."""
    if not os.path.exists(source_image_path):
        print(f"[ReplayUtils] Source image for overlay not found: {source_image_path}")
        error_img_width = reference_size_for_font * 10 
        error_img_height = reference_size_for_font * 8
        img = Image.new('RGB', (error_img_width, error_img_height), (50, 50, 50))
        draw = ImageDraw.Draw(img)
        error_font = _load_font(POTENTIAL_FONTS, reference_size_for_font // 2, "Error font for overlay fallback.")
        draw.text((10,10), f"Source Missing:\\n{os.path.basename(source_image_path)}", fill=(255,0,0), font=error_font)
        if output_save_path:
            os.makedirs(os.path.dirname(output_save_path), exist_ok=True)
            img.save(output_save_path)
        return

    try:
        # img = Image.open(source_image_path).convert("RGB") # No longer needed if just copying
        # draw = ImageDraw.Draw(img) # No longer needed
        # img_width, img_height = img.size # No longer needed

        # Font and text drawing logic -- ENTIRE BLOCK TO BE REMOVED
        # info_font_size = max(10, reference_size_for_font // 2 - 2)
        # if img_height < 100 : 
        #     info_font_size = max(8, img_height // 10)
        # common_font = _load_font(POTENTIAL_FONTS, info_font_size, "Info font for overlay not found.")
        # shadow_offset = 1
        # text_margin = 5
        # if perf_score is not None:
        #     score_text_content = f"Total Score: {perf_score:.1f}" 
        #     draw.text((text_margin, text_margin), score_text_content, fill=(0,0,0), font=common_font)
        # if action_taken_str is not None:
        #     action_text_content = f"Action: {action_taken_str}"
        #     text_h = info_font_size
        #     if hasattr(common_font, 'getbbox'):
        #         bbox = common_font.getbbox(action_text_content)
        #         text_h = bbox[3] - bbox[1]
        #     elif hasattr(common_font, 'getsize'): 
        #         _, text_h = common_font.getsize(action_text_content)
        #     text_x_action = text_margin
        #     text_y_action = img_height - text_h - text_margin
        #     draw.text((text_x_action + shadow_offset, text_y_action + shadow_offset), action_text_content, fill=(0,0,0), font=common_font)
        #     draw.text((text_x_action, text_y_action), action_text_content, fill=(255,255,255), font=common_font)

        # Save the modified image - Now just copy
        save_dir = os.path.dirname(output_save_path)
        if save_dir: os.makedirs(save_dir, exist_ok=True)
        # img.save(output_save_path)
        shutil.copy(source_image_path, output_save_path)

    except Exception as e:
        print(f"[ReplayUtils] Error processing image {source_image_path} for copy/overlay: {e}")

# --- 2048 Specific Functions (existing code) ---
def create_board_image_2048(board_powers: np.ndarray, save_path: str, size: int = 400, perf_score: Optional[float] = None) -> None:
    """Create a visualization of the 2048 board, incorporating new styling and perf_score display."""
    cell_size = size // 4
    padding = cell_size // 10

    img = Image.new('RGB', (size, size), (250, 248, 239)) # Overall image background (can be overridden by grid bg)
    draw = ImageDraw.Draw(img)

    # Color mapping for different tile values
    colors = {
        0: (205, 193, 180),      # Empty cell
        2: (238, 228, 218),      # 2
        4: (237, 224, 200),      # 4
        8: (242, 177, 121),      # 8
        16: (245, 149, 99),      # 16
        32: (246, 124, 95),      # 32
        64: (246, 94, 59),       # 64
        128: (237, 207, 114),    # 128
        256: (237, 204, 97),     # 256
        512: (237, 200, 80),     # 512
        1024: (237, 197, 63),    # 1024
        2048: (237, 194, 46),    # 2048
        4096: (60, 58, 50),      # 4096
        8192: (60, 58, 50)       # 8192
    }
    
    dark_text_color = (119, 110, 101)
    light_text_color = (249, 246, 242)

    font = _load_font(POTENTIAL_FONTS, cell_size // 3, "Main font for 2048 not found, using PIL default.")
    perf_score_display_font = _load_font(POTENTIAL_FONTS, max(15, size // 25), "Perf score font for 2048 not found, using PIL default.")

    draw.rectangle([0, 0, size, size], fill=(187, 173, 160))

    for r_idx in range(4):
        for c_idx in range(4):
            power = int(board_powers[r_idx, c_idx])
            value = 0 if power == 0 else 2**power
            
            x0 = c_idx * cell_size + padding
            y0 = r_idx * cell_size + padding
            x1 = (c_idx + 1) * cell_size - padding
            y1 = (r_idx + 1) * cell_size - padding
            
            cell_color = colors.get(value, (60, 58, 50)) 
            draw.rectangle([x0, y0, x1, y1], fill=cell_color)
            
            if value == 0:
                continue
            
            text_content = str(value)
            current_text_color = light_text_color if value > 4 else dark_text_color
            
            # Dynamic font sizing for 2048 tile numbers
            current_tile_font_size = cell_size // 3
            if len(text_content) == 3: current_tile_font_size = int(cell_size * 0.8)
            elif len(text_content) >= 4: current_tile_font_size = int(cell_size * 0.65)
            
            final_font_for_tile = font
            if current_tile_font_size != cell_size // 3:
                 final_font_for_tile = _load_font(POTENTIAL_FONTS, current_tile_font_size) # Use default message from _load_font
            
            text_width, text_height = 0, 0
            try:
                if hasattr(final_font_for_tile, 'getbbox'):
                    bbox = final_font_for_tile.getbbox(text_content)
                    text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
                elif hasattr(final_font_for_tile, 'getsize'): # Legacy
                    text_width, text_height = final_font_for_tile.getsize(text_content)
                else: # Basic fallback
                    text_width = len(text_content) * current_tile_font_size // 2 
                    text_height = current_tile_font_size
            except Exception as e:
                 print(f"[ReplayUtils] Error getting text size for 2048 tile: {e}. Using fallback.")
                 text_width = len(text_content) * current_tile_font_size // 2
                 text_height = current_tile_font_size
            
            cell_center_x = (x0 + x1) // 2
            cell_center_y = (y0 + y1) // 2
            text_x = cell_center_x - text_width // 2
            text_y = cell_center_y - text_height // 2 - (cell_size // 20) # Small adjustment for better centering
            
            draw.text((text_x, text_y), text_content, fill=current_text_color, font=final_font_for_tile)
            if value >= 8: # Bold effect for larger numbers
                draw.text((text_x + 1, text_y), text_content, fill=current_text_color, font=final_font_for_tile)

    if perf_score is not None:
        score_text_content = f"Total Score: {perf_score:.0f}" # For 2048, usually integer total game score
        score_display_text_color = (10, 10, 10)
        score_pos_x = padding 
        score_pos_y = padding // 2 
        try:
            draw.text((score_pos_x, score_pos_y), score_text_content, fill=score_display_text_color, font=perf_score_display_font)
        except Exception as e:
            print(f"[ReplayUtils] Error drawing 2048 perf_score on image: {e}")

    try:
        save_dir = os.path.dirname(save_path)
        if save_dir: os.makedirs(save_dir, exist_ok=True)
        img.save(save_path)
    except Exception as e:
        print(f"[ReplayUtils] Error saving 2048 board image to {save_path}: {e}")

# --- Main Replay Generation Functions ---
def generate_2048_median_score_replay(
    game_perf_json_path: str, 
    model_name_prefix: str, 
    game_display_name: str, # e.g. "2048" or "twenty_forty_eight" as key in game_perf.json
    harness_status_key: str, # e.g. "harness_false"
    video_output_base_dir: str,
    seconds_per_frame: float = DEFAULT_SECONDS_PER_FRAME
):
    """
    Generates an MP4 video replay for the median-scoring episode of a given model 
    in the 2048 game, using data from game_perf.json.
    Frames are generated using step_infos (board and step_score).
    Video is created using ffmpeg.
    """
    print(f"Attempting to generate median score MP4 replay for {model_name_prefix} in game '{game_display_name}' ({harness_status_key}).")

    # Check for ffmpeg first
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        print("[ReplayUtils] ffmpeg found.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[ReplayUtils] Error: ffmpeg is not installed or not found in PATH. Video generation aborted.")
        print("[ReplayUtils] Please install ffmpeg: sudo apt-get install ffmpeg (on Debian/Ubuntu) or brew install ffmpeg (on macOS)")
        return

    try:
        with open(game_perf_json_path, 'r') as f:
            all_game_perf_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: game_perf.json not found at {game_perf_json_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {game_perf_json_path}")
        return

    try:
        # Use game_display_name to access the game's data
        model_data = all_game_perf_data.get(game_display_name, {}).get(model_name_prefix, {}).get(harness_status_key, {})
        if not model_data or "episodes_data" not in model_data:
            print(f"No data found for {model_name_prefix} - game '{game_display_name}' - {harness_status_key} in {game_perf_json_path}")
            return

        episodes_data = model_data["episodes_data"]
        if not episodes_data:
            print(f"No episodes found for {model_name_prefix} - game '{game_display_name}' - {harness_status_key}.")
            return

        # Sort episodes by total_episode_perf_score to find the median
        # If even number of episodes, statistics.median will average the two middle scores.
        # We need to pick one actual episode.
        scores = [ep["total_episode_perf_score"] for ep in episodes_data]
        if not scores:
            print("No scores available to find median.")
            return
            
        median_score_value = statistics.median(scores)
        
        # Find all episodes that match the median score
        median_episodes = [ep for ep in episodes_data if ep["total_episode_perf_score"] == median_score_value]
        
        if not median_episodes: # Should not happen if scores list was not empty
             # Fallback: if median is an average, pick the one closest (lower bias)
            episodes_data.sort(key=lambda ep: ep["total_episode_perf_score"])
            if len(episodes_data) % 2 == 0 and len(episodes_data) > 0 : # Even number
                median_episode_idx = len(episodes_data) // 2 -1 # pick the lower of the two middle ones
            elif len(episodes_data) > 0: # Odd number
                median_episode_idx = len(episodes_data) // 2
            else: # No episodes
                print(f"No episodes to select for median replay generation for {model_name_prefix}, {harness_status_key}.")
                return
            median_episode = episodes_data[median_episode_idx]
            print(f"Could not find exact median score match. Using episode with score: {median_episode['total_episode_perf_score']}")
        else:
            median_episode = median_episodes[0] # Pick the first one if multiple have the exact median score
            print(f"Selected median episode with score: {median_episode['total_episode_perf_score']}")


        episode_id_str = median_episode.get("episode_id", "unknown_ep")
        actual_median_score = median_episode["total_episode_perf_score"]
        
        step_infos = median_episode.get("step_infos")
        if not step_infos:
            print(f"No step_infos found for median episode {episode_id_str} of {model_name_prefix}.")
            return

        temp_dir = tempfile.mkdtemp()
        frame_files = []
        print(f"Generating frames in temporary directory: {temp_dir}")

        cumulative_step_score = 0.0 # Initialize cumulative score

        for idx, step_info in enumerate(step_infos):
            board_raw = step_info.get("board")
            current_step_score = step_info.get("step_score", 0.0) # Get current step's score
            cumulative_step_score += current_step_score # Add to cumulative

            if board_raw is None:
                print(f"Warning: 'board' key not found in step_info {idx} for episode {episode_id_str}. Skipping frame.")
                continue
            try:
                board_powers = np.array(board_raw, dtype=int)
                if board_powers.shape != (4, 4):
                    print(f"Warning: Board in step_info {idx} (ep {episode_id_str}) does not have shape (4,4). Actual: {board_powers.shape}. Skipping frame.")
                    continue
            except Exception as e:
                print(f"Warning: Could not convert board in step_info {idx} (ep {episode_id_str}) to numpy array: {e}. Skipping frame.")
                continue
            
            frame_path = os.path.join(temp_dir, f"frame_{idx:04d}.png")
            # Pass the cumulative_step_score to be displayed on the image
            create_board_image_2048(board_powers, frame_path, perf_score=cumulative_step_score)
            frame_files.append(frame_path)

        if not frame_files:
            print(f"No frames generated for median episode {episode_id_str} of {model_name_prefix}. Video not created.")
            shutil.rmtree(temp_dir)
            return

        harness_short = "hT" if "true" in harness_status_key else "hF"
        safe_model_name = "".join(c if c.isalnum() or c in ['-', '_'] else '_' for c in model_name_prefix)
        safe_model_name = safe_model_name[:15]

        
        video_name = f"{game_display_name.replace('_', '').replace(' ', '')}_{safe_model_name}_{harness_short}_median_ep{episode_id_str}_score{actual_median_score:.0f}.mp4"
        
        model_video_dir = os.path.join(video_output_base_dir, safe_model_name)
        os.makedirs(model_video_dir, exist_ok=True)
        output_video_path = os.path.join(model_video_dir, video_name)

        print(f"Compiling {len(frame_files)} frames into MP4 video: {output_video_path} at {1/seconds_per_frame:.2f} FPS.")
        
        # ffmpeg command
        # -r: framerate (input/output based on context)
        # -framerate: specific input framerate
        # -i: input pattern
        # -c:v: video codec
        # -pix_fmt: pixel format, yuv420p is widely compatible for mp4
        # -y: overwrite output without asking
        framerate = 1.0 / seconds_per_frame
        ffmpeg_cmd = [
            "ffmpeg",
            "-framerate", str(framerate),
            "-i", os.path.join(temp_dir, "frame_%04d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-y", # Overwrite output
            output_video_path
        ]

        try:
            print(f"[ReplayUtils] Executing ffmpeg command: {' '.join(ffmpeg_cmd)}")
            process = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)
            print("[ReplayUtils] ffmpeg stdout:")
            print(process.stdout)
            print(f"Median score replay MP4 video saved to {output_video_path}")
        except subprocess.CalledProcessError as e:
            print(f"[ReplayUtils] Error creating MP4 video with ffmpeg:")
            print(f"[ReplayUtils] Command: {' '.join(e.cmd)}")
            print(f"[ReplayUtils] Return code: {e.returncode}")
            print(f"[ReplayUtils] stdout:{e.stdout}")
            print(f"[ReplayUtils] stderr:{e.stderr}")
            print(f"Frames are available in {temp_dir}")
            return # Keep temp_dir for debugging if ffmpeg fails
        except FileNotFoundError: # Should have been caught by the initial check, but as a safeguard
            print("[ReplayUtils] Error: ffmpeg not found during video compilation. This should not happen after initial check.")
            print(f"Frames are available in {temp_dir}")
            return
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                print(f"Cleaned up temporary directory: {temp_dir}")

    except Exception as e:
        print(f"An unexpected error occurred in generate_2048_median_score_replay for {model_name_prefix}, {harness_status_key}: {e}")
        # Clean up temp dir if it exists and an error occurs mid-process
        if 'temp_dir' in locals() and os.path.exists(temp_dir) and os.path.isdir(temp_dir): # Ensure it's a directory
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory due to error: {temp_dir}")
 
def generate_sokoban_median_score_replay(
    game_perf_json_path: str, 
    model_name_prefix: str, 
    game_display_name: str, # e.g., "sokoban"
    harness_status_key: str, # e.g., "harness_true"
    video_output_base_dir: str,
    seconds_per_frame: float = DEFAULT_SECONDS_PER_FRAME,
    default_tile_size: int = 32 # Default tile size for Sokoban rendering
):
    """
    Generates an MP4 video replay for the median-scoring episode of a given model 
    in the Sokoban game, using data from game_perf.json.
    Frames are generated from textual_representation in step_infos.
    Video is created using ffmpeg.
    """
    print(f"Attempting to generate median score MP4 replay for {model_name_prefix} in game '{game_display_name}' ({harness_status_key}) for Sokoban.")

    # Check for ffmpeg (same as 2048)
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        print("[ReplayUtils] ffmpeg found.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[ReplayUtils] Error: ffmpeg is not installed or not found in PATH. Video generation aborted.")
        return

    try:
        with open(game_perf_json_path, 'r') as f:
            all_game_perf_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: game_perf.json not found at {game_perf_json_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {game_perf_json_path}")
        return

    try:
        model_data = all_game_perf_data.get(game_display_name, {}).get(model_name_prefix, {}).get(harness_status_key, {})
        if not model_data or "episodes_data" not in model_data:
            print(f"No data found for {model_name_prefix} - game '{game_display_name}' - {harness_status_key} in {game_perf_json_path}")
            return

        episodes_data = model_data["episodes_data"]
        if not episodes_data:
            print(f"No episodes found for {model_name_prefix} - game '{game_display_name}' - {harness_status_key}.")
            return

        scores = [ep.get("total_episode_perf_score", 0) for ep in episodes_data] # Use .get for safety
        if not scores:
            print("No scores available to find median for Sokoban.")
            return
            
        median_score_value = statistics.median(scores)
        
        median_episodes = [ep for ep in episodes_data if ep.get("total_episode_perf_score", 0) == median_score_value]
        
        if not median_episodes:
            episodes_data.sort(key=lambda ep: ep.get("total_episode_perf_score", 0))
            if len(episodes_data) % 2 == 0 and len(episodes_data) > 0 :
                median_episode_idx = len(episodes_data) // 2 -1 
            elif len(episodes_data) > 0:
                median_episode_idx = len(episodes_data) // 2
            else:
                print(f"No episodes to select for Sokoban median replay for {model_name_prefix}, {harness_status_key}.")
                return
            median_episode = episodes_data[median_episode_idx]
            print(f"Could not find exact median score match for Sokoban. Using episode with score: {median_episode.get('total_episode_perf_score', 'N/A')}")
        else:
            median_episode = median_episodes[0]
            print(f"Selected Sokoban median episode with score: {median_episode.get('total_episode_perf_score', 'N/A')}")

        episode_id_str = median_episode.get("episode_id", "unknown_ep")
        actual_median_score = median_episode.get("total_episode_perf_score", 0.0)
        
        # Get the replayable_steps list
        replayable_step_data = median_episode.get("replayable_steps")
        
        if not replayable_step_data:
            print(f"No 'replayable_steps' data found for Sokoban median episode {episode_id_str} of {model_name_prefix}.")
            return

        temp_dir = tempfile.mkdtemp()
        frame_files = []
        print(f"Generating Sokoban frames in temporary directory: {temp_dir}")

        for idx, step_data in enumerate(replayable_step_data):
            textual_board = step_data.get("textual_representation")
            action_taken = step_data.get("agent_action", "N/A")
            perf_score_for_frame = step_data.get("perf_score") 
            img_path_from_log = step_data.get("img_path") # Get the img_path from replayable_steps

            frame_path = os.path.join(temp_dir, f"frame_{idx:04d}.png")

            if textual_board:
                board_numerical = parse_sokoban_textual_board(textual_board, SOKOBAN_CHAR_TO_ROOM_STATE)
                if board_numerical is None:
                    print(f"Warning: Failed to parse textual board in replayable_step {idx} (ep {episode_id_str}). Skipping frame.")
                    # Potentially create an error frame or use img_path_from_log with overlay if available
                    if img_path_from_log and os.path.exists(img_path_from_log):
                        overlay_text_on_image(
                            source_image_path=img_path_from_log, 
                            output_save_path=frame_path, 
                            perf_score=perf_score_for_frame, 
                            action_taken_str=action_taken, 
                            reference_size_for_font=default_tile_size
                        )
                        frame_files.append(frame_path)
                    else:
                         # Create a distinct error placeholder if both textual and image log are bad/missing
                        error_img = Image.new('RGB', (default_tile_size * 10, default_tile_size * 8), (70, 20, 20))
                        draw = ImageDraw.Draw(error_img)
                        err_font = _load_font(POTENTIAL_FONTS, default_tile_size // 2, "Error font for parse failure.")
                        draw.text((10,10), f"Frame {idx}: Parse Fail\nNo Log Img", fill=(255,100,100), font=err_font)
                        error_img.save(frame_path)
                        frame_files.append(frame_path)
                    continue # Move to next step_data
                
                create_board_image_sokoban_for_replay(
                    board_numerical, 
                    frame_path, 
                    tile_size=default_tile_size,
                    perf_score=perf_score_for_frame, 
                    action_taken_str=action_taken
                )
                frame_files.append(frame_path)
            elif img_path_from_log and os.path.exists(img_path_from_log):
                overlay_text_on_image(
                    source_image_path=img_path_from_log, 
                    output_save_path=frame_path, 
                    perf_score=perf_score_for_frame, 
                    action_taken_str=action_taken, 
                    reference_size_for_font=default_tile_size
                )
                frame_files.append(frame_path)
            else:
                print(f"Warning: Both textual_representation and valid img_path are missing for replayable_step {idx} (ep {episode_id_str}). Skipping frame.")
                # Create a distinct error placeholder if both textual and image log are bad/missing
                error_img = Image.new('RGB', (default_tile_size * 10, default_tile_size * 8), (20, 20, 70))
                draw = ImageDraw.Draw(error_img)
                err_font = _load_font(POTENTIAL_FONTS, default_tile_size // 2, "Error font for missing data.")
                draw.text((10,10), f"Frame {idx}: No Data", fill=(100,100,255), font=err_font)
                error_img.save(frame_path)
                frame_files.append(frame_path)
                continue

        if not frame_files:
            print(f"No Sokoban frames generated for median episode {episode_id_str} of {model_name_prefix}. Video not created.")
            shutil.rmtree(temp_dir)
            return

        harness_short = "hT" if "true" in harness_status_key.lower() else "hF" # Make comparison case-insensitive
        safe_model_name = "".join(c if c.isalnum() or c in ['-', '_'] else '_' for c in model_name_prefix)[:15]
        
        video_name = f"sokoban_{safe_model_name}_{harness_short}_median_ep{episode_id_str}_score{actual_median_score:.1f}.mp4"
        
        model_video_dir = os.path.join(video_output_base_dir, safe_model_name)
        os.makedirs(model_video_dir, exist_ok=True)
        output_video_path = os.path.join(model_video_dir, video_name)

        print(f"Compiling {len(frame_files)} Sokoban frames into MP4 video: {output_video_path} at {1/seconds_per_frame:.2f} FPS.")
        
        framerate = 1.0 / seconds_per_frame
        ffmpeg_cmd = [
            "ffmpeg", "-framerate", str(framerate),
            "-i", os.path.join(temp_dir, "frame_%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-y", output_video_path
        ]

        try:
            print(f"[ReplayUtils] Executing ffmpeg command: {' '.join(ffmpeg_cmd)}")
            process = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)
            print("[ReplayUtils] ffmpeg stdout (Sokoban):")
            print(process.stdout) # Print less verbose output if successful, or more on error
            print(f"Sokoban median score replay MP4 video saved to {output_video_path}")
        except subprocess.CalledProcessError as e:
            print(f"[ReplayUtils] Error creating Sokoban MP4 video with ffmpeg:")
            print(f"[ReplayUtils] Command: {' '.join(e.cmd)}")
            print(f"[ReplayUtils] Return code: {e.returncode}")
            print(f"[ReplayUtils] stdout:{e.stdout}")
            print(f"[ReplayUtils] stderr:{e.stderr}")
            print(f"Sokoban frames are available in {temp_dir}")
            return 
        finally:
            if os.path.exists(temp_dir) and os.path.isdir(temp_dir):
                shutil.rmtree(temp_dir)
                print(f"Cleaned up temporary directory for Sokoban: {temp_dir}")
    
    except Exception as e:
        print(f"An unexpected error occurred in generate_sokoban_median_score_replay for {model_name_prefix}, {harness_status_key}: {e}")
        if 'temp_dir' in locals() and os.path.exists(temp_dir) and os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory due to error: {temp_dir}")

def generate_candy_crush_median_score_replay(
    game_perf_json_path: str,
    model_name_prefix: str,
    game_display_name: str, 
    harness_status_key: str, 
    video_output_base_dir: str,
    seconds_per_frame: float = DEFAULT_SECONDS_PER_FRAME,
    default_tile_size: int = 32 
):
    print(f"Attempting to generate median score MP4 replay for {model_name_prefix} in game '{game_display_name}' ({harness_status_key}) for Candy Crush.")

    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[ReplayUtils] Error: ffmpeg is not installed or not found in PATH. Video generation aborted.")
        return

    try:
        with open(game_perf_json_path, 'r') as f: all_game_perf_data = json.load(f)
    except FileNotFoundError: print(f"Error: game_perf.json not found at {game_perf_json_path}"); return
    except json.JSONDecodeError: print(f"Error: Could not decode JSON from {game_perf_json_path}"); return

    try:
        model_data = all_game_perf_data.get(game_display_name, {}).get(model_name_prefix, {}).get(harness_status_key, {})
        if not model_data or "episodes_data" not in model_data:
            print(f"No data for {model_name_prefix} - game '{game_display_name}' - {harness_status_key}"); return
        episodes_data = model_data["episodes_data"]
        if not episodes_data: print(f"No episodes for {model_name_prefix} - game '{game_display_name}' - {harness_status_key}."); return

        scores = [ep.get("total_episode_perf_score", 0) for ep in episodes_data]
        if not scores: print("No scores for Candy Crush median."); return
            
        median_score_value = statistics.median(scores)
        median_episodes = [ep for ep in episodes_data if ep.get("total_episode_perf_score", 0) == median_score_value]
        
        if not median_episodes:
            episodes_data.sort(key=lambda ep: ep.get("total_episode_perf_score", 0))
            median_episode_idx = (len(episodes_data) - 1) // 2 
            if not episodes_data: print(f"No episodes to select for CC median replay for {model_name_prefix}, {harness_status_key}."); return
            median_episode = episodes_data[median_episode_idx]
            print(f"Exact median score not found for CC. Using ep score: {median_episode.get('total_episode_perf_score', 'N/A')}")
        else:
            median_episode = median_episodes[0]
            print(f"Selected CC median episode score: {median_episode.get('total_episode_perf_score', 'N/A')}")

        episode_id_str = median_episode.get("episode_id", "unknown_ep")
        actual_median_score = median_episode.get("total_episode_perf_score", 0.0)
        replayable_step_data = median_episode.get("replayable_steps")
        
        if not replayable_step_data: print(f"No 'replayable_steps' for CC median ep {episode_id_str} of {model_name_prefix}."); return

        temp_dir = tempfile.mkdtemp()
        frame_files = []
        print(f"Generating CC frames in temp dir: {temp_dir}")

        for idx, step_data in enumerate(replayable_step_data):
            textual_rep = step_data.get("textual_representation")
            action_taken = step_data.get("agent_action", "N/A")
            img_path_from_log = step_data.get("img_path")
            raw_env_info_for_step = step_data.get("raw_env_info", {})
            score_to_display_on_frame = raw_env_info_for_step.get("total_score") 
            moves_left_for_display = raw_env_info_for_step.get("num_moves_left")
            frame_path = os.path.join(temp_dir, f"frame_{idx:04d}.png")

            if img_path_from_log and os.path.exists(img_path_from_log):
                # print(f"  Frame {idx}: Using logged image: {img_path_from_log}") # Removed
                shutil.copy(img_path_from_log, frame_path)
                frame_files.append(frame_path)
            elif textual_rep:
                # print(f"  Frame {idx}: Reconstructing from textual representation.") # Removed
                board_numerical, score_from_text, moves_from_text = parse_candy_crush_textual_board(textual_rep, CANDY_CRUSH_CHAR_TO_IDX) 
                
                final_score_for_render = score_to_display_on_frame if score_to_display_on_frame is not None else score_from_text
                final_moves_for_render = moves_left_for_display if moves_left_for_display is not None else moves_from_text

                if board_numerical is not None:
                    _create_board_image_candy_crush_for_replay( 
                        board_numerical,
                        frame_path,
                        tile_size=default_tile_size,
                        perf_score=final_score_for_render, 
                        action_taken_str=action_taken, 
                        moves_left=final_moves_for_render
                    )
                    frame_files.append(frame_path)
                else:
                    # print(f"  Warning: Failed to parse textual board for Candy Crush step {idx}. Creating error frame.") # Optional: keep for debugging
                    error_img = Image.new('RGB', (default_tile_size * 8, default_tile_size * 6), (70, 20, 20))
                    draw = ImageDraw.Draw(error_img); err_font = _load_font(POTENTIAL_FONTS, default_tile_size // 2)
                    draw.text((10,10), f"Frame {idx}: Text Parse Fail", fill=(255,100,100), font=err_font)
                    error_img.save(frame_path); frame_files.append(frame_path)
            else: 
                # print(f"  Warning: Both img_path and textual_representation missing for Candy Crush step {idx}. Creating error frame.") # Optional
                error_img = Image.new('RGB', (default_tile_size * 8, default_tile_size * 6), (20, 20, 70))
                draw = ImageDraw.Draw(error_img); err_font = _load_font(POTENTIAL_FONTS, default_tile_size // 2)
                draw.text((10,10), f"Frame {idx}: No Data", fill=(100,100,255), font=err_font)
                error_img.save(frame_path); frame_files.append(frame_path)

        if not frame_files: print(f"No CC frames for median ep {episode_id_str} of {model_name_prefix}."); shutil.rmtree(temp_dir); return

        harness_short = "hT" if "true" in harness_status_key.lower() else "hF"
        safe_model_name = "".join(c if c.isalnum() or c in ['-', '_'] else '_' for c in model_name_prefix)[:15]
        video_name = f"candycrush_{safe_model_name}_{harness_short}_median_ep{episode_id_str}_score{actual_median_score:.0f}.mp4"
        
        model_video_dir = os.path.join(video_output_base_dir, safe_model_name)
        os.makedirs(model_video_dir, exist_ok=True)
        output_video_path = os.path.join(model_video_dir, video_name)

        print(f"Compiling {len(frame_files)} CC frames to {output_video_path} ({1/seconds_per_frame:.2f} FPS).")
        framerate = 1.0 / seconds_per_frame
        ffmpeg_cmd = ["ffmpeg", "-framerate", str(framerate), "-i", os.path.join(temp_dir, "frame_%04d.png"),
                      "-c:v", "libx264", "-pix_fmt", "yuv420p", "-y", output_video_path]
        try:
            process = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)
            # Optionally print stdout/stderr if needed for debugging ffmpeg
            # if process.stdout: print(f"[ReplayUtils] ffmpeg stdout (CC):\n{process.stdout}")
            # if process.stderr: print(f"[ReplayUtils] ffmpeg stderr (CC):\n{process.stderr}")
            print(f"CC median score replay MP4 saved to {output_video_path}")
        except subprocess.CalledProcessError as e:
            print(f"[ReplayUtils] Error creating CC MP4 with ffmpeg:\nCmd: {' '.join(e.cmd)}\nReturn: {e.returncode}\nstdout:\n{e.stdout}\nstderr:\n{e.stderr}")
            print(f"CC frames are in {temp_dir}")
            return 
        finally:
            if os.path.exists(temp_dir) and os.path.isdir(temp_dir): shutil.rmtree(temp_dir)
    
    except Exception as e:
        print(f"Unexpected error in generate_candy_crush_median_score_replay for {model_name_prefix}, {harness_status_key}: {e}")
        if 'temp_dir' in locals() and os.path.exists(temp_dir) and os.path.isdir(temp_dir): shutil.rmtree(temp_dir)

def generate_super_mario_bros_median_replay(
    game_perf_json_path: str,
    model_name_prefix: str,
    game_display_name: str, # e.g., "super_mario_bros" or "Super Mario Bros"
    harness_status_key: str, # e.g., "harness_false"
    video_output_base_dir: str,
    # seconds_per_frame is not directly used by playback_movie, but kept for signature consistency if needed later
    seconds_per_frame: float = DEFAULT_SECONDS_PER_FRAME 
):
    """
    Generates an MP4 video replay from a .bk2 file for the median-scoring Super Mario Bros episode.
    Uses `python3 -m retro.scripts.playback_movie`.
    """
    print(f"Attempting to generate Super Mario Bros median score MP4 replay for {model_name_prefix} (game: '{game_display_name}', harness: {harness_status_key}).")

    # Check for ffmpeg (required by retro.scripts.playback_movie for .mp4 output)
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True, text=True)
        print("[ReplayUtils] ffmpeg found.")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print("[ReplayUtils] Error: ffmpeg is not installed or not found in PATH. Video generation aborted for Super Mario Bros.")
        if isinstance(e, subprocess.CalledProcessError):
            print(f"ffmpeg stdout: {e.stdout}")
            print(f"ffmpeg stderr: {e.stderr}")
        return

    try:
        with open(game_perf_json_path, 'r') as f:
            all_game_perf_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: game_perf.json not found at {game_perf_json_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {game_perf_json_path}")
        return

    try:
        model_data = all_game_perf_data.get(game_display_name, {}).get(model_name_prefix, {}).get(harness_status_key, {})
        if not model_data or "episodes_data" not in model_data:
            print(f"No data found for {model_name_prefix} - game '{game_display_name}' - {harness_status_key} in {game_perf_json_path}")
            return

        episodes_data = model_data["episodes_data"]
        if not episodes_data:
            print(f"No episodes found for {model_name_prefix} - game '{game_display_name}' - {harness_status_key}.")
            return

        scores = [ep.get("total_episode_perf_score", 0) for ep in episodes_data]
        if not scores:
            print("No scores available to find median for Super Mario Bros.")
            return
            
        median_score_value = statistics.median(scores)
        median_episodes = [ep for ep in episodes_data if ep.get("total_episode_perf_score", 0) == median_score_value]
        
        if not median_episodes:
            episodes_data.sort(key=lambda ep: ep.get("total_episode_perf_score", 0))
            if not episodes_data: # Should be caught by `if not scores` earlier
                 print(f"No episodes to select for Super Mario Bros median replay for {model_name_prefix}, {harness_status_key}.")
                 return
            median_episode_idx = (len(episodes_data) -1) // 2 # Fallback: pick middle or lower-middle
            median_episode = episodes_data[median_episode_idx]
            print(f"Could not find exact median score match for Super Mario Bros. Using episode with score: {median_episode.get('total_episode_perf_score', 'N/A')}")
        else:
            median_episode = median_episodes[0] # Pick the first one if multiple have the exact median score
            print(f"Selected Super Mario Bros median episode with score: {median_episode.get('total_episode_perf_score', 'N/A')}")

        episode_id_str = median_episode.get("episode_id", "unknown_ep") # e.g., "001", "002"
        actual_median_score = median_episode.get("total_episode_perf_score", 0.0)
        session_log_dir = median_episode.get("log_dir_path") # Path to session cache dir

        if not session_log_dir:
            print(f"Error: 'log_dir_path' not found for median episode {episode_id_str} of {model_name_prefix}. Cannot locate .bk2 file.")
            return

        bk2_recordings_dir = os.path.join(session_log_dir, "bk2_recordings")
        if not os.path.isdir(bk2_recordings_dir):
            print(f"Error: bk2_recordings directory not found at {bk2_recordings_dir} for episode {episode_id_str}.")
            return

        # Convert 1-indexed episode_id_str (e.g., "001") to 0-indexed for Retro's -epXXXXXX- pattern
        try:
            retro_ep_idx = int(episode_id_str) - 1
            if retro_ep_idx < 0: raise ValueError("Episode ID must be positive.")
        except ValueError:
            print(f"Error: Invalid episode_id_str '{episode_id_str}'. Cannot determine .bk2 file.")
            return
        
        # The file name typically ends with -<run_number>.bk2, where run_number is 0-indexed episode
        target_bk2_file_suffix = f"-{retro_ep_idx:06d}.bk2" # NEW LOGIC
        found_bk2_file = None

        print(f"Searching for .bk2 file in {bk2_recordings_dir} ending with: '{target_bk2_file_suffix}'")
        for filename in os.listdir(bk2_recordings_dir):
            # We also need to make sure it's for the correct game, though bk2_recordings_dir should be specific enough.
            # A more robust check could involve parsing the game name from the .bk2 file if needed,
            # but for now, ending with the correct run number should suffice within the specific session's bk2_recordings dir.
            if filename.endswith(target_bk2_file_suffix): # NEW LOGIC
                # Example: SuperMarioBros-Nes-Level1-1-000000.bk2
                # We expect self.env_id (e.g., "SuperMarioBros-Nes") to be part of the filename.
                # This check might be overly strict if the env_id in the config isn't exactly matching the prefix
                # For now, let's be a bit more lenient and just check the suffix, as the directory is already specific.
                # A better check would be `filename.startswith(self.env_id)` if we had env_id here.
                # Since env_id isn't directly passed, we rely on the specific log_dir_path.
                found_bk2_file = os.path.join(bk2_recordings_dir, filename)
                print(f"Found .bk2 file: {found_bk2_file}")
                break 
        
        if not found_bk2_file:
            print(f"Error: No .bk2 file found for episode {episode_id_str} (pattern part '{target_bk2_file_suffix}') in {bk2_recordings_dir}.")
            print(f"Files in directory: {os.listdir(bk2_recordings_dir)}")
            return

        harness_short = "hT" if "true" in harness_status_key.lower() else "hF"
        safe_model_name = "".join(c if c.isalnum() or c in ['-', '_'] else '_' for c in model_name_prefix)[:15]
        
        # Use a distinct prefix for SMB, e.g., "smb"
        video_name = f"smb_{safe_model_name}_{harness_short}_median_ep{episode_id_str}_score{actual_median_score:.0f}.mp4"
        
        model_video_dir = os.path.join(video_output_base_dir, safe_model_name)
        os.makedirs(model_video_dir, exist_ok=True)
        output_video_path = os.path.join(model_video_dir, video_name)

        print(f"Preparing to generate MP4 video: {output_video_path} from {found_bk2_file}")
        
        # Command for retro.scripts.playback_movie
        # Original attempt:
        # python3 -m retro.scripts.playback_movie <movie_file> --output-video <output_mp4_path>
        # New approach: Generate in place, then move.
        
        # Predict the output filename that playback_movie will create in found_bk2_file's directory
        bk2_basename_no_ext = os.path.splitext(os.path.basename(found_bk2_file))[0]
        intermediate_video_filename = f"{bk2_basename_no_ext}.mp4"
        intermediate_video_path = os.path.join(bk2_recordings_dir, intermediate_video_filename)

        playback_cmd = [
            "python3", 
            "-m", "retro.scripts.playback_movie",
            found_bk2_file,
            "--lossless", "mp4"  # Use mp4 for better color compatibility
        ]

        try:
            print(f"[ReplayUtils] Executing playback command to generate intermediate video: {' '.join(playback_cmd)}")
            if os.path.exists(intermediate_video_path):
                print(f"[ReplayUtils] Warning: Intermediate video path {intermediate_video_path} already exists. Removing before generation.")
                os.remove(intermediate_video_path)

            process = subprocess.run(playback_cmd, capture_output=True, text=True, check=True)
            if process.stdout: print(f"[ReplayUtils] playback_movie stdout:\n{process.stdout}")
            if process.stderr: print(f"[ReplayUtils] playback_movie stderr:\n{process.stderr}")

            if not os.path.exists(intermediate_video_path):
                print(f"[ReplayUtils] Error: Intermediate video file {intermediate_video_path} was not created by playback_movie.py.")
                print(f"[ReplayUtils] Check stderr above for clues. It might be using a different output name or failed silently.")
                return # Give up if the file isn't created

            print(f"[ReplayUtils] Intermediate video successfully created at: {intermediate_video_path}")
            print(f"[ReplayUtils] Moving intermediate video to final path: {output_video_path}")
            shutil.move(intermediate_video_path, output_video_path)
            print(f"Super Mario Bros median score replay MP4 video saved to {output_video_path}")

        except subprocess.CalledProcessError as e:
            print(f"[ReplayUtils] Error running retro.scripts.playback_movie for Super Mario Bros:")
            print(f"[ReplayUtils] Command: {' '.join(e.cmd)}")
            print(f"[ReplayUtils] Return code: {e.returncode}")
            print(f"[ReplayUtils] stdout:\n{e.stdout}")
            print(f"[ReplayUtils] stderr:\n{e.stderr}")
            # Do not clean up anything here, user might want to inspect video_output_base_dir
            return
        except FileNotFoundError: # e.g. python3 not found
            print("[ReplayUtils] Error: python3 (or the interpreter) not found. Cannot run playback_movie script.")
            return
    
    except Exception as e:
        print(f"An unexpected error occurred in generate_super_mario_bros_median_replay for {model_name_prefix}, game '{game_display_name}', harness {harness_status_key}: {type(e).__name__} - {e}")
        import traceback
        traceback.print_exc()

# Ensure the script can be called, e.g. if __name__ == "__main__": block
# For now, just defining the functions.
# Example usage would be added in a main block or a separate script. 