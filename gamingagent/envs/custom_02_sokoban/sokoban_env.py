import gymnasium as gym
import numpy as np
import os
import sys # Import sys for sys.path manipulation
from PIL import Image, ImageDraw, ImageFont # We'll need these for rendering
from typing import Any, Dict, Tuple, Callable, Optional, List
import json
import hashlib
import imageio
import tempfile
import shutil
import re # For parsing levels.txt

from gamingagent.envs.base_env import BaseGameEnv
from gamingagent.modules.core_module import Observation
from gymnasium.spaces import Discrete, Box # For defining action and observation space

# --- Try to import generate_room from the pip-installed gym-sokoban ---
generate_room = None
try:
    from gym_sokoban.envs.room_utils import generate_room
    print("[SokobanEnv] Successfully imported generate_room from pip-installed gym_sokoban.envs.room_utils.")
except ImportError as e:
    print(f"[SokobanEnv] WARNING: Failed to import generate_room from pip-installed gym_sokoban. Error: {e}")
    print("[SokobanEnv] Please ensure 'gym-sokoban' is installed via pip (e.g., 'pip install gym-sokoban').")
    print("[SokobanEnv] Level generation will not work if generate_room is unavailable.")
except Exception as e_gen:
    print(f"[SokobanEnv] WARNING: An unexpected error occurred during generate_room import from pip-installed package: {e_gen}")
# --- End utility import ---


# Constants for game logic
ACTION_LOOKUP = {
    0: 'no operation',
    1: 'push up',
    2: 'push down',
    3: 'push left',
    4: 'push right',
    5: 'move up',
    6: 'move down',
    7: 'move left',
    8: 'move right',
}

CHANGE_COORDINATES = {
    0: (-1, 0),  # Up
    1: (1, 0),   # Down
    2: (0, -1),  # Left
    3: (0, 1)    # Right
}

ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets", "images")
LEVELS_FILE_PATH = os.path.join(os.path.dirname(__file__), "assets", "levels.txt") # Path to levels.txt

WALL_IMG_PATH = os.path.join(ASSET_DIR, "wall.png")
FLOOR_IMG_PATH = os.path.join(ASSET_DIR, "floor.png")
BOX_IMG_PATH = os.path.join(ASSET_DIR, "box.png")
BOX_ON_TARGET_IMG_PATH = os.path.join(ASSET_DIR, "box_docked.png")
PLAYER_IMG_PATH = os.path.join(ASSET_DIR, "worker.png")
PLAYER_ON_TARGET_IMG_PATH = os.path.join(ASSET_DIR, "worker_dock.png")
TARGET_IMG_PATH = os.path.join(ASSET_DIR, "dock.png")

ROOM_STATE_TO_CHAR = {
    0: '#',  # Wall
    1: ' ',  # Floor
    2: '.',  # Target (empty)
    3: '*',  # Box on Target
    4: '$',  # Box (off Target)
    5: '@',  # Player (on Floor)
    6: '+',  # Player on Target
}

# Mapping from characters in levels.txt to internal numerical representation
# This will be used when parsing custom levels
LEVEL_CHAR_TO_NUM = {
    '#': 0,  # Wall
    ' ': 1,  # Floor
    '?': 2,  # Target
    '*': 3,  # Box on Target
    '$': 4,  # Box
    '@': 5,  # Player
    # '+' for Player on Target will be handled by checking underlying fixed char
}


ROOM_STATE_TO_ASSET_KEY = {
    0: "wall",
    1: "floor",
    2: "target",
    3: "box_on_target",
    4: "box",
    5: "player",
    6: "player_on_target"
}


def load_sokoban_asset(path, size):
    """Helper to load and resize an asset if it exists."""
    if not os.path.exists(path):
        print(f"[SokobanEnv] Warning: Asset not found at {path}")
        return None
    try:
        img = Image.open(path).convert("RGBA")
        return img.resize(size, Image.Resampling.LANCZOS)
    except Exception as e:
        print(f"[SokobanEnv] Error loading asset {path}: {e}")
        return None

def create_board_image_sokoban(board_state: np.ndarray, save_path: str, tile_size: int = 32, perf_score: Optional[float] = None, action_taken_str: Optional[str] = None) -> None:
    """
    Create a visualization of the Sokoban board.
    board_state: A 2D numpy array representing the Sokoban level.
    save_path: Path to save the image.
    tile_size: Size of each tile in pixels.
    perf_score: Optional performance score to display.
    action_taken_str: Optional string of the action taken to display.
    """
    if board_state is None:
        print("[SokobanEnv] Error: board_state is None in create_board_image_sokoban.")
        img = Image.new('RGB', (tile_size * 4, tile_size * 4), (128, 128, 128))
        draw = ImageDraw.Draw(img)
        draw.text((10,10), "Error: No board state", fill=(255,0,0))
        if save_path:
             os.makedirs(os.path.dirname(save_path), exist_ok=True)
             img.save(save_path)
        return

    rows, cols = board_state.shape
    img_width = cols * tile_size
    img_height = rows * tile_size

    img = Image.new('RGB', (img_width, img_height), (200, 200, 200))

    asset_size = (tile_size, tile_size)
    assets = {
        "wall": load_sokoban_asset(WALL_IMG_PATH, asset_size),
        "floor": load_sokoban_asset(FLOOR_IMG_PATH, asset_size),
        "box": load_sokoban_asset(BOX_IMG_PATH, asset_size),
        "box_on_target": load_sokoban_asset(BOX_ON_TARGET_IMG_PATH, asset_size),
        "player": load_sokoban_asset(PLAYER_IMG_PATH, asset_size),
        "player_on_target": load_sokoban_asset(PLAYER_ON_TARGET_IMG_PATH, asset_size),
        "target": load_sokoban_asset(TARGET_IMG_PATH, asset_size),
    }

    for r in range(rows):
        for c in range(cols):
            x0, y0 = c * tile_size, r * tile_size
            tile_val = board_state[r, c]

            if assets["floor"]:
                img.paste(assets["floor"], (x0, y0), assets["floor"] if assets["floor"].mode == 'RGBA' else None)

            specific_asset = None
            if tile_val == 0: specific_asset = assets.get("wall")
            elif tile_val == 1: pass
            elif tile_val == 2: specific_asset = assets.get("target")
            elif tile_val == 3:
                if assets.get("target"):
                    img.paste(assets["target"], (x0, y0), assets["target"] if assets["target"].mode == 'RGBA' else None)
                specific_asset = assets.get("box_on_target")
            elif tile_val == 4: specific_asset = assets.get("box")
            elif tile_val == 5: specific_asset = assets.get("player")
            elif tile_val == 6:
                if assets.get("target"):
                    img.paste(assets["target"], (x0, y0), assets["target"] if assets["target"].mode == 'RGBA' else None)
                specific_asset = assets.get("player_on_target") or assets.get("player")
            
            if specific_asset:
                img.paste(specific_asset, (x0, y0), specific_asset if specific_asset.mode == 'RGBA' else None)
            elif tile_val not in [0, 1]:
                draw_obj_for_fallback = ImageDraw.Draw(img)
                fallback_colors = { 2: (255,255,0), 3: (0,128,0), 4: (139,69,19), 5: (0,0,255), 6: (0,255,255) }
                color = fallback_colors.get(tile_val, (128,128,128))
                draw_obj_for_fallback.rectangle([x0, y0, x0+tile_size, y0+tile_size], fill=color)
                try:
                    font_size_fallback = max(8, tile_size // 3)
                    try: font_fallback = ImageFont.truetype("arial.ttf", font_size_fallback)
                    except IOError: font_fallback = ImageFont.load_default(size=font_size_fallback)
                    if hasattr(font_fallback, 'getbbox'):
                        bbox = font_fallback.getbbox(str(tile_val)); text_w, text_h = bbox[2]-bbox[0], bbox[3]-bbox[1]
                    else: text_w, text_h = font_fallback.getsize(str(tile_val))
                    draw_obj_for_fallback.text((x0+(tile_size-text_w)//2, y0+(tile_size-text_h)//2), str(tile_val), fill=(0,0,0), font=font_fallback)
                except Exception as e_font: print(f"[SokobanEnv] Fallback text drawing error: {e_font}")

    if perf_score is not None:
        draw_perf = ImageDraw.Draw(img)
        try:
            font_size_perf = max(12, tile_size // 2)
            font_for_text = ImageFont.load_default(size=font_size_perf)
            try: font_for_text = ImageFont.truetype("arial.ttf", font_size_perf)
            except IOError: pass
            text_content = f"Perf: {perf_score:.2f}"
            shadow_offset = max(1, tile_size // 32)
            draw_perf.text((5 + shadow_offset, 5 + shadow_offset), text_content, fill=(0,0,0), font=font_for_text)
            draw_perf.text((5, 5), text_content, fill=(255, 255, 255), font=font_for_text)
        except Exception as e: print(f"[SokobanEnv] Error drawing perf_score on image: {e}")

    if action_taken_str is not None:
        draw_action = ImageDraw.Draw(img)
        try:
            font_size_action = max(10, tile_size // 2 - 2)
            font_for_action = ImageFont.load_default(size=font_size_action)
            try: font_for_action = ImageFont.truetype("arial.ttf", font_size_action)
            except IOError: pass
            action_text_content = f"Action: {action_taken_str}"
            shadow_offset = max(1, tile_size // 32)
            text_x = 5
            text_y = img_height - font_size_action - 5
            if hasattr(font_for_action, 'getbbox'):
                bbox = font_for_action.getbbox(action_text_content); text_h = bbox[3] - bbox[1]
                text_y = img_height - text_h - 5
            draw_action.text((text_x + shadow_offset, text_y + shadow_offset), action_text_content, fill=(0,0,0), font=font_for_action)
            draw_action.text((text_x, text_y), action_text_content, fill=(255,255,255), font=font_for_action)
        except Exception as e: print(f"[SokobanEnv] Error drawing action_taken_str on image: {e}")
            
    try:
        if save_path:
            save_dir = os.path.dirname(save_path)
            if save_dir: os.makedirs(save_dir, exist_ok=True)
            img.save(save_path)
    except Exception as e: print(f"[SokobanEnv] Error saving Sokoban board image to {save_path}: {e}")

class SokobanEnv(BaseGameEnv):
    """Sokoban game environment implementing game logic directly."""

    def __init__(self,
                 game_name: str,
                 observation_mode: str,
                 agent_observations_base_dir: str,
                 env_type: str,
                 config_root_dir: str = "configs",
                 log_root_dir: str = "runs_output"):
        
        super().__init__(game_name,
                         observation_mode,
                         agent_observations_base_dir,
                         env_type,
                         config_root_dir,
                         log_root_dir,
                         skip_env_init=True)

        self.env_init_kwargs = {}
        self.tile_size_for_render = 32
        self._max_unchanged_steps = 20
        self.level_to_load: Optional[int] = None # For loading specific level from levels.txt
        self.predefined_levels: Dict[int, List[str]] = {} # To store parsed levels from levels.txt

        current_dir = os.path.dirname(os.path.abspath(__file__))
        game_specific_config_path = os.path.join(current_dir, "game_env_config.json")
        
        if os.path.exists(game_specific_config_path):
            try:
                with open(game_specific_config_path, 'r') as f: game_specific_config = json.load(f)
                self.env_id = game_specific_config.get("env_id", "SokobanCustom-v0")
                self.env_init_kwargs = game_specific_config.get("env_init_kwargs", {})
                self.level_to_load = game_specific_config.get("level_to_load") # Load this new config
                
                override_action_map_config = game_specific_config.get("action_mapping")
                if isinstance(override_action_map_config, dict):
                    try:
                        self.move_to_action_idx = {str(k).lower(): int(v) for k,v in override_action_map_config.items()}
                        self.action_idx_to_move = {int(v): str(k).lower() for k,v in override_action_map_config.items()}
                        self.action_mapping_config = override_action_map_config
                        print(f"[SokobanEnv] Successfully applied action_mapping override from {game_specific_config_path}")
                    except ValueError: print(f"[SokobanEnv] Warning: Values in action_mapping from {game_specific_config_path} must be integers.")
                else:
                    self.move_to_action_idx = {name: idx for idx, name in ACTION_LOOKUP.items()}
                    self.action_idx_to_move = {idx: name for idx, name in ACTION_LOOKUP.items()}
                    self.action_mapping_config = ACTION_LOOKUP
                    print(f"[SokobanEnv] Using default ACTION_LOOKUP for action mapping.")

                self.tile_size_for_render = game_specific_config.get("tile_size_for_render", self.tile_size_for_render)
                loaded_max_unchanged = game_specific_config.get("max_unchanged_steps_for_termination")
                if isinstance(loaded_max_unchanged, int) and loaded_max_unchanged > 0: self._max_unchanged_steps = loaded_max_unchanged
                print(f"[SokobanEnv] Successfully loaded game-specific parameters from {game_specific_config_path}")
            except Exception as e:
                print(f"[SokobanEnv] Error loading game-specific config from {game_specific_config_path}: {e}. Using defaults.")
                self.move_to_action_idx = {name: idx for idx, name in ACTION_LOOKUP.items()}
                self.action_idx_to_move = {idx: name for idx, name in ACTION_LOOKUP.items()}
                self.action_mapping_config = ACTION_LOOKUP
        else:
            print(f"[SokobanEnv] Game-specific config {game_specific_config_path} not found. Using defaults.")
            self.env_id = "SokobanCustom-v0"
            self.move_to_action_idx = {name: idx for idx, name in ACTION_LOOKUP.items()}
            self.action_idx_to_move = {idx: name for idx, name in ACTION_LOOKUP.items()}
            self.action_mapping_config = ACTION_LOOKUP

        self.dim_room = tuple(self.env_init_kwargs.get("dim_room", [10, 10]))
        self.num_boxes_total = self.env_init_kwargs.get("num_boxes", 3)
        self.max_steps_episode = self.env_init_kwargs.get("max_steps", 200)
        
        # num_gen_steps is only relevant if using generate_room
        self.num_gen_steps = self.env_init_kwargs.get("num_gen_steps")
        if self.num_gen_steps is None and self.dim_room and (not self.level_to_load or self.level_to_load == 0) : 
            self.num_gen_steps = int(1.7 * (self.dim_room[0] + self.dim_room[1]))

        # Load predefined levels if level_to_load is specified
        if self.level_to_load and self.level_to_load > 0:
            self._load_predefined_levels()
            if self.level_to_load not in self.predefined_levels:
                print(f"[SokobanEnv] WARNING: Level {self.level_to_load} not found in {LEVELS_FILE_PATH}. Will attempt to use generate_room if available.")
                self.level_to_load = None # Fallback to random generation
            else:
                # If level loaded, dim_room might be determined by the level itself later in reset
                print(f"[SokobanEnv] Will load predefined level {self.level_to_load} from {LEVELS_FILE_PATH}.")
        else:
            print("[SokobanEnv] No specific level_to_load or set to 0. Will use generate_room if available.")

        self.room_fixed: Optional[np.ndarray] = None
        self.room_state: Optional[np.ndarray] = None
        self.box_mapping: Optional[Dict[Tuple[int, int], Tuple[int, int]]] = None
        self.player_position: Optional[np.ndarray] = None
        self.num_env_steps: int = 0
        self.boxes_on_target: int = 0
        
        self.penalty_for_step = -0.1
        self.penalty_box_off_target = -1.0
        self.reward_box_on_target = 1.0
        self.reward_finished = 10.0

        self._action_space = Discrete(len(ACTION_LOOKUP))
        obs_shape = (self.dim_room[0] * self.tile_size_for_render, self.dim_room[1] * self.tile_size_for_render, 3)
        self._observation_space = Box(0, 255, shape=obs_shape, dtype=np.uint8)
        
        print(f"[SokobanEnv] Initialized. Action space: {self.action_space}. Assumed Obs Space Shape: {self.observation_space.shape}")

        self._last_observation_hash: Optional[str] = None
        self._unchanged_obs_count: int = 0

    def _load_predefined_levels(self):
        """Loads levels from the levels.txt file."""
        if not os.path.exists(LEVELS_FILE_PATH):
            print(f"[SokobanEnv] ERROR: Levels file not found at {LEVELS_FILE_PATH}")
            return

        print(f"[SokobanEnv] Loading predefined levels from: {LEVELS_FILE_PATH}")
        with open(LEVELS_FILE_PATH, 'r') as f:
            content = f.read()
        
        # Split levels by "Level X" header, then process each block
        level_blocks = re.split(r"Level\s+\d+", content)
        current_level_num = 0
        for block in level_blocks:
            if not block.strip():
                continue
            current_level_num +=1 # Assuming Level headers are sequential starting from 1

            level_lines = [line.rstrip('\n') for line in block.strip().split('\n') if line.strip()] #rstrip to remove trailing newlines from split
            
            if level_lines:
                # Validate level lines for consistent width
                first_line_width = len(level_lines[0])
                if not all(len(line) == first_line_width for line in level_lines):
                    print(f"[SokobanEnv] WARNING: Level {current_level_num} in {LEVELS_FILE_PATH} has inconsistent line widths. Skipping this level.")
                    continue
                self.predefined_levels[current_level_num] = level_lines
                print(f"[SokobanEnv] Loaded Level {current_level_num} with {len(level_lines)} rows and {first_line_width} columns.")
            else:
                print(f"[SokobanEnv] WARNING: Level block for potential level {current_level_num} in {LEVELS_FILE_PATH} is empty or invalid.")
        
        if not self.predefined_levels:
            print(f"[SokobanEnv] WARNING: No valid levels loaded from {LEVELS_FILE_PATH}.")

    def _parse_and_set_custom_level(self, level_str_lines: List[str]):
        """Parses a list of strings representing a level and sets up the board."""
        rows = len(level_str_lines)
        if rows == 0:
            print("[SokobanEnv] CRITICAL: Custom level has 0 rows.")
            return False
        cols = len(level_str_lines[0])
        if cols == 0:
            print("[SokobanEnv] CRITICAL: Custom level has 0 columns.")
            return False

        self.dim_room = (rows, cols)
        self.room_fixed = np.zeros(self.dim_room, dtype=np.uint8)
        self.room_state = np.zeros(self.dim_room, dtype=np.uint8)
        self.num_boxes_total = 0
        found_player = False

        for r in range(rows):
            for c in range(cols):
                char = level_str_lines[r][c]
                
                # Default to floor
                self.room_fixed[r, c] = 1 
                self.room_state[r, c] = 1

                if char == '#': # Wall
                    self.room_fixed[r, c] = 0
                    self.room_state[r, c] = 0
                elif char == '?': # Target
                    self.room_fixed[r, c] = 2
                    self.room_state[r, c] = 2
                elif char == '$': # Box
                    self.room_state[r, c] = 4
                    self.num_boxes_total += 1
                elif char == '*': # Box on Target
                    self.room_fixed[r, c] = 2 # Underlying is a target
                    self.room_state[r, c] = 3
                    self.num_boxes_total += 1
                elif char == '@': # Player
                    if found_player:
                        print("[SokobanEnv] WARNING: Multiple players found in custom level. Using the first one.")
                    else:
                        self.player_position = np.array([r, c])
                        # If player is on a target char '?' in levels.txt, it's player_on_target (6)
                        # but levels.txt uses '@' for player on target square too.
                        # We check the original char from levels.txt.
                        # No, we should check room_fixed. If room_fixed is target (2), then player is on target.
                        # The issue is that `level_str_lines[r][c]` is already '@'.
                        # The logic for player on target needs to be: if fixed is 2, and current is player, it is 6.
                        # For now, let's assume if it's '@', it could be on floor or on target.
                        # The room_fixed for '@' should be 1 (floor) unless it was also '?'
                        # This parsing needs to be careful. Let's refine.

                        # If player is on a spot that was defined as target '?' earlier in parsing or by fixed rule
                        # The problem: player char '@' overrides target char '?' in level_str_lines
                        # Simpler: If player is at (r,c), check if underlying fixed tile is target
                        # This can only be done *after* all fixed elements (walls, floors, targets) are set.
                        # So, process fixed elements first, then state elements.

                        # Let's try another way:
                        # 1. Fill room_fixed and room_state with floors (1)
                        # 2. Place Walls (0) in both
                        # 3. Place Targets (2) in both room_fixed and room_state
                        # 4. Then process dynamic elements for room_state:
                        #    - Box ($) -> 4
                        #    - Box on Target (*) -> 3 (fixed remains 2)
                        #    - Player (@) -> 5. If fixed is 2, then 6.

                        # Revised parsing logic in the loop:
                        # Initial pass for fixed elements:
                        if char == '#': self.room_fixed[r,c] = 0; self.room_state[r,c] = 0
                        elif char == '?': self.room_fixed[r,c] = 2; self.room_state[r,c] = 2
                        elif char == '*': self.room_fixed[r,c] = 2; self.room_state[r,c] = 1 # Underlying fixed is target, state will be box_on_target
                        elif char == ' ': self.room_fixed[r,c] = 1; self.room_state[r,c] = 1
                        elif char == '$': self.room_fixed[r,c] = 1; self.room_state[r,c] = 1 # Box is on floor
                        elif char == '@': self.room_fixed[r,c] = 1; self.room_state[r,c] = 1 # Player is on floor
                        # Any other char -> treat as floor for now
                        else: self.room_fixed[r,c] = 1; self.room_state[r,c] = 1
        
        # Second pass for state elements (player, boxes)
        for r_s in range(rows):
            for c_s in range(cols):
                char_s = level_str_lines[r_s][c_s]
                if char_s == '$': # Box
                    self.room_state[r_s, c_s] = 4 # Box on floor/target (if target, fixed is 2, state becomes 3 later)
                    if self.room_fixed[r_s, c_s] == 2: # Box starts on a target
                        self.room_state[r_s,c_s] = 3
                    self.num_boxes_total += 1
                elif char_s == '*': # Box on Target
                    # self.room_fixed[r_s,c_s] was already set to 2
                    self.room_state[r_s, c_s] = 3
                    self.num_boxes_total += 1
                elif char_s == '@': # Player
                    if found_player:
                        print("[SokobanEnv] WARNING: Multiple players defined. Using first one.")
                    else:
                        self.player_position = np.array([r_s, c_s])
                        if self.room_fixed[r_s, c_s] == 2: # Player on Target
                            self.room_state[r_s, c_s] = 6
                        else: # Player on Floor
                            self.room_state[r_s, c_s] = 5
                        found_player = True
        
        if not found_player:
            print("[SokobanEnv] CRITICAL: No player '@' found in the custom level. Placing at first available floor.")
            floor_locs = np.argwhere(self.room_state == 1)
            if floor_locs.size > 0:
                self.player_position = floor_locs[0]
                r_p, c_p = self.player_position
                self.room_state[r_p,c_p] = 6 if self.room_fixed[r_p,c_p] == 2 else 5
            else: # Absolute fallback
                self.player_position = np.array([0,0]) 
                if 0 <= self.player_position[0] < self.dim_room[0] and 0 <= self.player_position[1] < self.dim_room[1]:
                     self.room_state[self.player_position[0], self.player_position[1]] = 5 # Assume (0,0) is not a wall
                print("[SokobanEnv] CRITICAL: No floor found for player fallback. Player at (0,0).")

        # Update observation space based on loaded level dimensions
        obs_shape = (self.dim_room[0] * self.tile_size_for_render, self.dim_room[1] * self.tile_size_for_render, 3)
        self._observation_space = Box(0, 255, shape=obs_shape, dtype=np.uint8)
        print(f"[SokobanEnv] Custom level loaded. New dim_room: {self.dim_room}. Player: {self.player_position}. Boxes: {self.num_boxes_total}. New obs_shape: {obs_shape}")
        return True

    def _perform_environment_reset(self, seed: Optional[int] = None, episode_id: int = 1) -> Tuple[Any, Dict[str, Any]]:
        self._last_observation_hash = None
        self._unchanged_obs_count = 0
        self.num_env_steps = 0
        self.boxes_on_target = 0

        level_loaded_from_file = False
        if self.level_to_load and self.level_to_load in self.predefined_levels:
            level_data = self.predefined_levels[self.level_to_load]
            if self._parse_and_set_custom_level(level_data):
                self.boxes_on_target = np.count_nonzero(self.room_state == 3)
                print(f"[SokobanEnv] Successfully reset to predefined level {self.level_to_load}. Player at {self.player_position}. Boxes on target: {self.boxes_on_target}/{self.num_boxes_total}")
                level_loaded_from_file = True
            else:
                print(f"[SokobanEnv] CRITICAL: Failed to parse/set predefined level {self.level_to_load}. Falling back to generate_room.")
        
        if not level_loaded_from_file:
            if generate_room is None:
                print("[SokobanEnv] CRITICAL: generate_room is not available AND failed to load/parse custom level. Cannot reset/create level.")
                # Create a minimal empty room as a last resort
                self.dim_room = tuple(self.env_init_kwargs.get("dim_room", [5,5])) # Use default or smaller
                self.room_fixed = np.zeros(self.dim_room, dtype=np.uint8) # All walls
                self.room_state = np.zeros(self.dim_room, dtype=np.uint8) # All walls
                if self.dim_room[0] > 2 and self.dim_room[1] > 2: # Try to make a tiny clear space
                    self.room_fixed[1:self.dim_room[0]-1, 1:self.dim_room[1]-1] = 1
                    self.room_state[1:self.dim_room[0]-1, 1:self.dim_room[1]-1] = 1
                    self.player_position = np.array([self.dim_room[0]//2, self.dim_room[1]//2])
                    self.room_state[self.player_position[0], self.player_position[1]] = 5
                else:
                    self.player_position = np.array([0,0])

                self.num_boxes_total = 0
                self.box_mapping = {} # Not used with custom levels this way
                print("[SokobanEnv] Created a minimal fallback empty room due to multiple failures.")
            else:
                try:
                    # Use dim_room from env_init_kwargs if not overridden by a loaded level
                    current_dim_room = tuple(self.env_init_kwargs.get("dim_room", [10,10]))
                    current_num_boxes = self.env_init_kwargs.get("num_boxes", 3)
                    current_num_gen_steps = self.num_gen_steps # Use the one calculated in init
                    if current_num_gen_steps is None: # Fallback if still none
                         current_num_gen_steps = int(1.7 * (current_dim_room[0] + current_dim_room[1]))


                    self.room_fixed, self.room_state, self.box_mapping = generate_room(
                        dim=current_dim_room, num_steps=current_num_gen_steps, num_boxes=current_num_boxes, second_player=False)
                    
                    self.dim_room = current_dim_room # Ensure self.dim_room reflects generated level
                    self.num_boxes_total = current_num_boxes

                    player_loc = np.argwhere(self.room_state == 5) # Player is 5
                    if player_loc.size > 0: self.player_position = player_loc[0]
                    else:
                        print("[SokobanEnv] CRITICAL: Player not found in generated room. Placing at first floor or (0,0).")
                        self.player_position = np.array([0,0])
                        if self.room_state.size > 0 and self.room_state[0,0] == 0: # if (0,0) is wall
                            floor_locs = np.argwhere(self.room_state == 1) # find floor
                            if floor_locs.size > 0: self.player_position = floor_locs[0]
                        # Ensure player position is valid before assignment to room_state
                        px, py = self.player_position[0], self.player_position[1]
                        if 0 <= px < self.dim_room[0] and 0 <= py < self.dim_room[1]:
                             # Check if player is on a target based on room_fixed
                            self.room_state[px, py] = 6 if self.room_fixed[px,py] == 2 else 5
                        else: # Fallback if position is still invalid
                            self.player_position = np.array([min(px, self.dim_room[0]-1), min(py, self.dim_room[1]-1)])
                            corrected_px, corrected_py = self.player_position
                            self.room_state[corrected_px, corrected_py] = 6 if self.room_fixed[corrected_px, corrected_py] == 2 else 5


                    self.boxes_on_target = np.count_nonzero(self.room_state == 3) # Box on target is 3
                    # Update observation space if using generate_room and dims might change from default
                    obs_shape = (self.dim_room[0] * self.tile_size_for_render, self.dim_room[1] * self.tile_size_for_render, 3)
                    self._observation_space = Box(0, 255, shape=obs_shape, dtype=np.uint8)
                    print(f"[SokobanEnv] Level reset using generate_room. Dim: {self.dim_room}. Player at {self.player_position}. Boxes on target: {self.boxes_on_target}/{self.num_boxes_total}. Obs shape: {obs_shape}")
                except Exception as e:
                    print(f"[SokobanEnv] CRITICAL: Error during level generation with generate_room: {e}. Creating fallback room.")
                    # Fallback to minimal room if generate_room fails
                    self.dim_room = tuple(self.env_init_kwargs.get("dim_room", [5,5]))
                    self.room_fixed = np.ones(self.dim_room, dtype=np.uint8) 
                    if self.dim_room[0]>0 and self.dim_room[1]>0: self.room_fixed[0,:]=0; self.room_fixed[-1,:]=0; self.room_fixed[:,0]=0; self.room_fixed[:,-1]=0
                    self.room_state = self.room_fixed.copy()
                    self.player_position = np.array([self.dim_room[0]//2, self.dim_room[1]//2])
                    if 0 <= self.player_position[0] < self.dim_room[0] and 0 <= self.player_position[1] < self.dim_room[1]:
                         self.room_state[self.player_position[0], self.player_position[1]] = 5
                    self.num_boxes_total = 0; self.box_mapping = {}
                    self.boxes_on_target = 0
                    obs_shape = (self.dim_room[0] * self.tile_size_for_render, self.dim_room[1] * self.tile_size_for_render, 3)
                    self._observation_space = Box(0, 255, shape=obs_shape, dtype=np.uint8)

        return self.room_state, self._get_info()

    def _perform_environment_step(self, action: int) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        if not (0 <= action < len(ACTION_LOOKUP)):
            print(f"[SokobanEnv] WARNING: Invalid action {action}. Taking no_op (0).")
            action = 0

        self.num_env_steps += 1
        reward = self.penalty_for_step
        moved_player, moved_box = False, False

        if action == 0: pass
        elif 1 <= action <= 4: moved_player, moved_box = self._push(action - 1)
        elif 5 <= action <= 8: moved_player = self._move(action - 5)
        
        new_boxes_on_target = np.count_nonzero(self.room_state == 3)
        if new_boxes_on_target > self.boxes_on_target: reward += self.reward_box_on_target * (new_boxes_on_target - self.boxes_on_target)
        elif new_boxes_on_target < self.boxes_on_target: reward += self.penalty_box_off_target * (self.boxes_on_target - new_boxes_on_target)
        self.boxes_on_target = new_boxes_on_target

        all_boxes_on_target = self._check_if_all_boxes_on_target()
        if all_boxes_on_target: reward += self.reward_finished
        
        terminated = all_boxes_on_target
        truncated = self.num_env_steps >= self.max_steps_episode
        
        info_to_return = self._get_info()
        info_to_return["action.name"] = ACTION_LOOKUP.get(action, "unknown")
        info_to_return["action.moved_player"] = moved_player
        info_to_return["action.moved_box"] = moved_box
        if terminated or truncated:
            info_to_return["maxsteps_used"] = self.num_env_steps >= self.max_steps_episode
            info_to_return["all_boxes_on_target"] = all_boxes_on_target
            
        return self.room_state, float(reward), terminated, truncated, info_to_return

    def _push(self, move_direction_idx: int) -> Tuple[bool, bool]:
        change = CHANGE_COORDINATES[move_direction_idx]
        player_r, player_c = self.player_position
        next_r, next_c = player_r + change[0], player_c + change[1]
        box_next_r, box_next_c = next_r + change[0], next_c + change[1]

        if not (0 <= next_r < self.dim_room[0] and 
                0 <= next_c < self.dim_room[1] and 
                0 <= box_next_r < self.dim_room[0] and 
                0 <= box_next_c < self.dim_room[1]):
            return False, False

        is_box_at_next = self.room_state[next_r, next_c] in [4, 3]
        is_space_clear_beyond_box = self.room_state[box_next_r, box_next_c] in [1, 2]

        if is_box_at_next and is_space_clear_beyond_box:
            self.room_state[box_next_r, box_next_c] = 3 if self.room_fixed[box_next_r, box_next_c] == 2 else 4
            self.room_state[next_r, next_c] = 5 if self.room_fixed[next_r, next_c] != 2 else 6
            self.player_position = np.array([next_r, next_c])
            self.room_state[player_r, player_c] = self.room_fixed[player_r, player_c]
            return True, True
        else:
            # Cannot push, try to move instead
            moved_player_instead = self._move(move_direction_idx)
            return moved_player_instead, False

    def _move(self, move_direction_idx: int) -> bool:
        change = CHANGE_COORDINATES[move_direction_idx]
        player_r, player_c = self.player_position
        next_r, next_c = player_r + change[0], player_c + change[1]

        if not (0 <= next_r < self.dim_room[0] and 0 <= next_c < self.dim_room[1]): return False

        if self.room_state[next_r, next_c] in [1, 2]:
            self.room_state[next_r, next_c] = 5 if self.room_fixed[next_r, next_c] != 2 else 6
            self.player_position = np.array([next_r, next_c])
            self.room_state[player_r, player_c] = self.room_fixed[player_r, player_c]
            return True
        return False
        
    def _check_if_all_boxes_on_target(self) -> bool:
        if self.room_state is None: return False
        if self.num_boxes_total == 0: return True
        num_boxes_off_target = np.count_nonzero(self.room_state == 4)
        return num_boxes_off_target == 0 and self.boxes_on_target == self.num_boxes_total

    def _get_info(self) -> Dict[str, Any]:
        return {
            "num_env_steps": self.num_env_steps,
            "player_position": self.player_position.tolist() if self.player_position is not None else None,
            "boxes_on_target": self.boxes_on_target,
            "num_boxes": self.num_boxes_total,
            "all_boxes_on_target": self._check_if_all_boxes_on_target(),
        }

    def get_board_state(self, raw_observation: Any, info: Dict[str, Any]) -> Optional[np.ndarray]:
        if self.room_state is not None: return np.array(self.room_state, dtype=int)
        else:
            print("[SokobanEnv] CRITICAL: get_board_state called but self.room_state is None.")
            return np.zeros(self.dim_room, dtype=int)

    def extract_observation(self, raw_observation: Any, info: Dict[str, Any]) -> Observation:
        board_state_numerical = raw_observation if isinstance(raw_observation, np.ndarray) else self.get_board_state(None, None)
        img_path_for_agent, text_representation_for_agent = None, None
        current_ep, current_st = self.get_current_episode_step_num()

        if board_state_numerical is None:
            print("[SokobanEnv] Error: board_state_numerical is None. Creating dummy observation.")
            return Observation(img_path=None, symbolic_representation="Error: Could not extract board state.")

        if self.observation_mode in ["vision", "both"]:
            img_path_for_agent = self._create_agent_observation_path(current_ep, current_st)
            create_board_image_sokoban(board_state_numerical, img_path_for_agent, tile_size=self.tile_size_for_render)
        
        if self.observation_mode in ["text", "both"]:
            text_representation_for_agent = "\n".join(["".join([ROOM_STATE_TO_CHAR.get(tile, '?') for tile in row]) for row in board_state_numerical.tolist()])
            
        return Observation(img_path=img_path_for_agent, symbolic_representation=text_representation_for_agent)

    def verify_termination(self, observation: Observation, current_terminated: bool, current_truncated: bool) -> Tuple[bool, bool]:
        if current_terminated or current_truncated: return current_terminated, current_truncated
        current_obs_hash: Optional[str] = None
        if observation.symbolic_representation: current_obs_hash = hashlib.md5(observation.symbolic_representation.encode()).hexdigest()
        elif observation.img_path and os.path.exists(observation.img_path):
            try:
                with Image.open(observation.img_path) as img: current_obs_hash = hashlib.md5(img.tobytes()).hexdigest()
            except Exception as e: print(f"[SokobanEnv] Warning: Could not hash image {observation.img_path}: {e}"); return current_terminated, current_truncated
        else: return current_terminated, current_truncated

        if self._last_observation_hash == current_obs_hash: self._unchanged_obs_count += 1
        else: self._unchanged_obs_count = 0
        self._last_observation_hash = current_obs_hash

        if self._unchanged_obs_count >= self._max_unchanged_steps:
            print(f"[SokobanEnv] Terminating: unchanged observation for {self._max_unchanged_steps} steps.")
            return True, current_truncated
        return current_terminated, current_truncated

    def game_replay(self, trajectory_data: List[Dict[str, Any]], perf_score_list: List[float],
                    output_video_path: str = "sokoban_replay.gif", frame_duration: float = 0.3) -> None:
        if not trajectory_data: print("[SokobanEnv] No trajectory data for replay."); return
        display_scores = len(trajectory_data) == len(perf_score_list)
        if not display_scores: print(f"[SokobanEnv] Warning: Trajectory data ({len(trajectory_data)}) and score list ({len(perf_score_list)}) length mismatch. Scores not shown.")

        temp_dir = tempfile.mkdtemp(); frame_files = []
        print(f"[SokobanEnv] Generating frames for replay in {temp_dir}...")

        for idx, step_data in enumerate(trajectory_data):
            board_raw = step_data.get("raw_env_observation")
            agent_action = step_data.get("agent_action", "N/A")
            
            if board_raw is None: # Fallback attempt (less ideal)
                obs_obj_str = step_data.get("agent_observation") # Check "agent_observation"
                if isinstance(obs_obj_str, str):
                    # This is a very brittle fallback. The symbolic representation is char-based.
                    # The create_board_image_sokoban expects numerical. This fallback is unlikely to work
                    # without a proper parser from symbolic char to numerical, which is not implemented.
                    # For now, just log that raw_env_observation should be used.
                    print(f"[SokobanEnv] Replay step {idx}: raw_env_observation missing. Fallback to agent_observation.symbolic not robust for image creation.")
                    # If a future parser from ROOM_STATE_TO_CHAR back to numerical existed, it would go here.
                if board_raw is None: # If still None after trying fallback
                    print(f"[SokobanEnv] Warning: Board (raw_env_observation) missing for step {idx}. Skipping frame.")
                    continue
            try:
                board_numerical = np.array(board_raw, dtype=int)
                if board_numerical.ndim != 2 or board_numerical.shape[0] == 0 or board_numerical.shape[1] == 0:
                    print(f"[SokobanEnv] Warning: Board step {idx} invalid 2D array. Shape: {board_numerical.shape}. Skip.")
                    continue
            except Exception as e: print(f"[SokobanEnv] Warning: Board step {idx} to numpy array conversion failed: {e}. Skip."); continue

            frame_path = os.path.join(temp_dir, f"frame_{idx:04d}.png")
            current_perf_score = perf_score_list[idx] if display_scores and idx < len(perf_score_list) else None
            try:
                create_board_image_sokoban(board_numerical, frame_path, tile_size=self.tile_size_for_render, perf_score=current_perf_score, action_taken_str=agent_action)
                frame_files.append(frame_path)
            except Exception as e: print(f"[SokobanEnv] Error creating board image step {idx}: {e}. Skip.")

        if not frame_files: print("[SokobanEnv] No frames generated for replay.")
        else:
            print(f"[SokobanEnv] Compiling {len(frame_files)} frames to {output_video_path}")
            try:
                images_data = [imageio.imread(f) for f in frame_files]
                imageio.mimsave(output_video_path, images_data, format='GIF', duration=frame_duration, subrectangles=True)
                print(f"[SokobanEnv] Replay video saved to {output_video_path}")
            except Exception:
                try: imageio.mimsave(output_video_path, images_data, duration=frame_duration); print(f"[SokobanEnv] Replay video saved (default GIF writer).")
                except Exception as e2: print(f"[SokobanEnv] Error creating video (default writer failed): {e2}")
        try: shutil.rmtree(temp_dir)
        except Exception as e: print(f"[SokobanEnv] Error cleaning temp dir {temp_dir}: {e}")

    def perf_score(self, reward: float, info: Dict[str, Any]) -> float:
        num_boxes_on_target = info.get("boxes_on_target", 0)
        all_on_target = info.get("all_boxes_on_target", False)
        total_boxes = info.get("num_boxes", 1)
        if total_boxes <= 0: return 100.0 + self.reward_finished + float(reward) if self.num_boxes_total == 0 else float(reward)
        progress = (num_boxes_on_target / total_boxes) * 100.0
        solve_bonus = self.reward_finished if all_on_target else 0.0
        return float(progress + solve_bonus + float(reward))

    def close(self):
        super().close()
        print("[SokobanEnv] Closed.")

    @property
    def action_space(self) -> gym.Space: return self._action_space
    @property
    def observation_space(self) -> gym.Space: return self._observation_space
