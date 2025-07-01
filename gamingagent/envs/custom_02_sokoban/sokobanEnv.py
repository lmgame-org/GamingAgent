import gymnasium as gym
import numpy as np
import os
import pygame # For rendering
from PIL import Image, ImageDraw, ImageFont 
from typing import Any, Dict, Tuple, Optional, List, Union
import json
import re # For parsing levels.txt

from gamingagent.envs.gym_env_adapter import GymEnvAdapter
from gamingagent.modules.core_module import Observation
from gymnasium.spaces import Discrete, Box 
from gymnasium.core import RenderFrame

# It's better to move create_board_image_sokoban to env_utils if it's generic enough,
# but for now, let's assume it's defined here or imported from a local helper.
# For this refactor, we will redefine it here, taking inspiration from sokoban_env.py.

# --- Constants and Asset Paths (from sokoban_env.py) ---
ACTION_LOOKUP = { # Simplified, adapter will handle mapping from string like "up"
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

# Raw actions for the environment itself (0-3 for up, down, left, right)
# These are the actions the _push and _move would expect if we simplify.
# The GymEnvAdapter will map agent string actions ("up", "right", "push up") to these.
# Let's define the core env actions based on sokoban_env_old.py for _move and _push.
# For internal logic, we can map:
# - 0: Up (for move/push)
# - 1: Down (for move/push)
# - 2: Left (for move/push)
# - 3: Right (for move/push)
# The adapter's action_mapping in game_env_config.json will map strings like "move up" to 5, "push up" to 1, etc.
# And then the SokobanEnv can internally map these to the 0-3 directional indices.
# For simplicity in this refactor, let's assume the direct action indices for _push and _move
# will be 0:up, 1:down, 2:left, 3:right.
# The ACTION_LOOKUP above is more for info. The real mapping will be in game_env_config.json

CHANGE_COORDINATES = { # (row_change, col_change)
    0: (-1, 0),  # Up
    1: (1, 0),   # Down
    2: (0, -1),  # Left
    3: (0, 1)    # Right
}

ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets", "images")
LEVELS_FILE_PATH = os.path.join(os.path.dirname(__file__), "assets", "levels.txt")

ROOM_STATE_TO_CHAR = {
    0: '#', 1: ' ', 2: '?', 3: '*', 4: '$', 5: '@', 6: '+'
}
CHAR_TO_ROOM_STATE = {v: k for k, v in ROOM_STATE_TO_CHAR.items()}


def load_sokoban_asset_image(path, size):
    if not os.path.exists(path): return None
    try:
        img = Image.open(path).convert("RGBA")
        return img.resize(size, Image.Resampling.LANCZOS)
    except Exception: return None

def create_board_image_sokoban(board_state: np.ndarray, save_path: str, tile_size: int = 32, perf_score: Optional[float] = None, action_taken_str: Optional[str] = None):
    if board_state is None:
        img = Image.new('RGB', (tile_size * 5, tile_size * 5), (128, 128, 128)) # Default small error image
        draw = ImageDraw.Draw(img)
        draw.text((10,10), "Error: No board state", fill=(255,0,0))
        if save_path:
             os.makedirs(os.path.dirname(save_path), exist_ok=True)
             img.save(save_path)
        return

    rows, cols = board_state.shape
    img_width = cols * tile_size
    img_height = rows * tile_size
    img = Image.new('RGB', (img_width, img_height), (200, 200, 200)) # Background
    draw = ImageDraw.Draw(img)

    asset_paths = {
        "wall": os.path.join(ASSET_DIR, "wall.png"),
        "floor": os.path.join(ASSET_DIR, "floor.png"),
        "box": os.path.join(ASSET_DIR, "box.png"),
        "box_on_target": os.path.join(ASSET_DIR, "box_docked.png"),
        "player": os.path.join(ASSET_DIR, "worker.png"),
        "player_on_target": os.path.join(ASSET_DIR, "worker_dock.png"),
        "target": os.path.join(ASSET_DIR, "dock.png"),
    }
    assets = {k: load_sokoban_asset_image(p, (tile_size, tile_size)) for k, p in asset_paths.items()}

    for r in range(rows):
        for c in range(cols):
            x0, y0 = c * tile_size, r * tile_size
            tile_val = board_state[r, c] # Numerical value from room_state

            # Draw floor first
            if assets["floor"]:
                img.paste(assets["floor"], (x0, y0), assets["floor"] if assets["floor"].mode == 'RGBA' else None)
            
            asset_to_draw = None
            if tile_val == 0: asset_to_draw = assets["wall"]      # Wall
            elif tile_val == 2: asset_to_draw = assets["target"]   # Target
            elif tile_val == 3: # Box on Target
                if assets["target"]: # Draw target beneath box_on_target
                    img.paste(assets["target"], (x0, y0), assets["target"] if assets["target"].mode == 'RGBA' else None)
                asset_to_draw = assets["box_on_target"]
            elif tile_val == 4: asset_to_draw = assets["box"]      # Box
            elif tile_val == 5: asset_to_draw = assets["player"]   # Player
            elif tile_val == 6: # Player on Target
                if assets["target"]: # Draw target beneath player
                    img.paste(assets["target"], (x0, y0), assets["target"] if assets["target"].mode == 'RGBA' else None)
                asset_to_draw = assets["player_on_target"]
            
            if asset_to_draw:
                img.paste(asset_to_draw, (x0, y0), asset_to_draw if asset_to_draw.mode == 'RGBA' else None)
            elif tile_val == 1: # Floor, already drawn
                pass
            else: # Fallback for unknown tile values
                draw.rectangle([x0, y0, x0 + tile_size, y0 + tile_size], fill=(100, 100, 100))
                draw.text((x0 + 5, y0 + 5), ROOM_STATE_TO_CHAR.get(tile_val, "?"), fill=(255,255,255))

    common_font = None
    try:
        font_size = max(10, tile_size // 2 - 2)
        common_font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        common_font = ImageFont.load_default(size=font_size if 'font_size' in locals() else 12)

    if perf_score is not None:
        text_content = f"Perf: {perf_score:.2f}"
        shadow_offset = 1
        draw.text((5 + shadow_offset, 5 + shadow_offset), text_content, fill=(0,0,0), font=common_font)
        draw.text((5, 5), text_content, fill=(255, 255, 255), font=common_font)

    if action_taken_str is not None:
        action_text_content = f"Action: {action_taken_str}"
        shadow_offset = 1
        # Calculate text size to position at bottom
        if hasattr(common_font, 'getbbox'):
            bbox = common_font.getbbox(action_text_content)
            text_h = bbox[3] - bbox[1]
        elif hasattr(common_font, 'getsize'): # Legacy
            _, text_h = common_font.getsize(action_text_content)
        else: # Fallback
            text_h = font_size if 'font_size' in locals() else 12

        text_x = 5
        text_y = img_height - text_h - 5
        draw.text((text_x + shadow_offset, text_y + shadow_offset), action_text_content, fill=(0,0,0), font=common_font)
        draw.text((text_x, text_y), action_text_content, fill=(255,255,255), font=common_font)
            
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir: os.makedirs(save_dir, exist_ok=True)
        img.save(save_path)


class SokobanEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array', 'raw'], 'render_fps': 4}

    def __init__(self,
                 render_mode: Optional[str] = None,
                 # Sokoban specific params from game_env_config.json
                 dim_room: Tuple[int, int] = (10, 10),
                 max_steps_episode: int = 200,
                 num_boxes: int = 3,
                 num_gen_steps: Optional[int] = None, # For procedural generation
                 level_to_load: Optional[int] = None, # To load a specific level
                 tile_size_for_render: int = 32,
                 # Adapter related parameters (will be passed by the runner)
                 game_name_for_adapter: str = "sokoban",
                 observation_mode_for_adapter: str = "vision",
                 agent_cache_dir_for_adapter: str = "cache/sokoban/default_run",
                 game_specific_config_path_for_adapter: str = "gamingagent/envs/custom_02_sokoban/game_env_config.json",
                 max_stuck_steps_for_adapter: Optional[int] = 20
                 ):
        
        self.dim_room = dim_room
        self.max_steps_episode = max_steps_episode
        self.num_boxes_initial = num_boxes # Store initial config
        self.level_to_load = level_to_load
        self.tile_size_for_render = tile_size_for_render
        self.current_level = level_to_load if level_to_load is not None else 1
        self.max_level = 6  # Maximum level number in levels.txt
        
        if num_gen_steps is None and not self.level_to_load : 
            self.num_gen_steps = int(1.7 * (self.dim_room[0] + self.dim_room[1]))
        else:
            self.num_gen_steps = num_gen_steps if num_gen_steps is not None else 0 # default if level_to_load

        # Penalties and Rewards from sokoban_env_old.py
        self.penalty_for_step = -0.1
        self.penalty_box_off_target = -1.0 
        self.reward_box_on_target = 1.0
        self.reward_finished = 10.0
        
        self.action_space = Discrete(9) # 0: no_op, 1-4: push, 5-8: move (corresponds to ACTION_LOOKUP length)
        # Observation space will be dynamically set in reset based on room_fixed and tile_size
        # For now, set a placeholder. The adapter gets Observation objects.
        # The gym observation space for this env will be the raw board state (numerical)
        self.observation_space = Box(low=0, high=6, shape=self.dim_room, dtype=np.uint8) 

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None # For pygame rendering
        self.clock = None  # For pygame rendering

        # Initialize Adapter
        self.adapter = GymEnvAdapter(
            game_name=game_name_for_adapter,
            observation_mode=observation_mode_for_adapter,
            agent_cache_dir=agent_cache_dir_for_adapter,
            game_specific_config_path=game_specific_config_path_for_adapter, # Adapter loads its own action mapping
            max_steps_for_stuck=max_stuck_steps_for_adapter
        )

        # Sokoban-specific performance score tracking
        self.previous_boxes_on_target_for_perf: int = 0
        self.current_episode_cumulative_perf_score: float = 0.0

        self.room_fixed: Optional[np.ndarray] = None
        self.room_state: Optional[np.ndarray] = None
        self.player_position: Optional[np.ndarray] = None
        self.num_boxes_current: int = 0 # Number of boxes in the current level
        self.boxes_on_target: int = 0
        self.num_env_steps: int = 0
        self.current_reward_last_step: float = 0.0
        
        self.predefined_levels: Dict[int, List[str]] = {}
        self._load_predefined_levels() # Load levels from levels.txt
        
        # For gym_sokoban compatibility if used
        self.box_mapping: Optional[Dict[Tuple[int, int], Tuple[int, int]]] = None


    def _load_predefined_levels(self):
        if not os.path.exists(LEVELS_FILE_PATH): return
        with open(LEVELS_FILE_PATH, 'r') as f: content = f.read()
        
        level_blocks = re.split(r"Level\s+\d+", content)
        current_level_num = 0
        for block in level_blocks:
            if not block.strip(): continue
            current_level_num +=1
            # Corrected line splitting and stripping
            level_lines_raw = block.strip().splitlines()
            level_lines = [line.strip() for line in level_lines_raw if line.strip()]
            
            if level_lines:
                first_line_width = len(level_lines[0])
                if not all(len(line) == first_line_width for line in level_lines):
                    print(f"[SokobanEnv] Warning: Level {current_level_num} has inconsistent line widths. Skipping.")
                    continue
                self.predefined_levels[current_level_num] = level_lines

    def _parse_level_data(self, level_str_lines: List[str]) -> bool:
        rows = len(level_str_lines)
        if rows == 0: return False
        cols = len(level_str_lines[0])
        if cols == 0: return False

        self.dim_room = (rows, cols)
        _room_fixed = np.ones(self.dim_room, dtype=np.uint8) # Default to floor (1)
        _player_pos_list = []
        _num_boxes = 0

        # First pass: Set up _room_fixed (walls, targets, default floor for dynamic items)
        for r in range(rows):
            for c in range(cols):
                char = level_str_lines[r][c]
                if char == '#': # Wall
                    _room_fixed[r, c] = 0
                elif char == '?': # Target
                    _room_fixed[r, c] = 2
                # For '$', '*', '@', the underlying _room_fixed is floor (1), which is the default.
                # If '*' implies a target, it's handled in the second pass by explicitly setting _room_fixed.

        _room_state = _room_fixed.copy() # Initialize _room_state based on fixed layout

        # Second pass: Place dynamic entities (player, boxes) onto _room_state
        for r in range(rows):
            for c in range(cols):
                char = level_str_lines[r][c]
                if char == '$': # Box
                    if _room_fixed[r, c] == 2: # Box placed on a target square (e.g. '$' on a '?')
                        _room_state[r, c] = 3 # Box on Target
                    else: # Box on Floor
                        _room_state[r, c] = 4 # Box
                    _num_boxes += 1
                elif char == '*': # Box explicitly on Target
                    _room_fixed[r, c] = 2 # Ensure underlying fixed map shows a target
                    _room_state[r, c] = 3 # Box on Target
                    _num_boxes += 1
                elif char == '@': # Player
                    if _room_fixed[r, c] == 2: # Player on a Target square (e.g. '@' on a '?')
                        _room_state[r, c] = 6 # Player on Target
                    else: # Player on Floor
                        _room_state[r, c] = 5 # Player on Floor
                    _player_pos_list.append(np.array([r, c]))
                # The character '+' for player on target is not used by the provided levels.txt
        
        self.room_fixed = _room_fixed
        self.room_state = _room_state
        self.num_boxes_current = _num_boxes

        if not _player_pos_list: # No player '@' found in level string
            print("[SokobanEnv] Warning: No player '@' found in level. Attempting to place player on an available floor or target square.")
            # Try to find a floor or target square to place the player
            available_squares = np.argwhere((self.room_state == 1) | (self.room_state == 2)) # Floor or Empty Target
            if available_squares.size > 0:
                # Place player randomly on one of the available squares
                self.player_position = available_squares[np.random.choice(len(available_squares))]
            else: # Absolute fallback: try center; this might be a wall.
                print("[SokobanEnv] Warning: No floor or target squares available for fallback player placement. Placing at center.")
                self.player_position = np.array([self.dim_room[0] // 2, self.dim_room[1] // 2])

            r_p, c_p = self.player_position
            # Check if chosen position is valid and not a wall before placing player state
            if self.room_fixed[r_p, c_p] != 0: # If not a wall
                 self.room_state[r_p, c_p] = 6 if self.room_fixed[r_p, c_p] == 2 else 5
            else: # Chosen fallback is a wall, this is bad. Player is effectively stuck or invalid.
                 print(f"[SokobanEnv] CRITICAL: Fallback player position ({r_p},{c_p}) is a wall. Level may be unplayable.")
                 # Keep player_position, but room_state at wall remains wall. Agent might get stuck immediately.
        elif len(_player_pos_list) > 1:
            print(f"[SokobanEnv] Warning: Multiple players ({len(_player_pos_list)}) found in level. Using the first one.")
            self.player_position = _player_pos_list[0]
        else: # Exactly one player found
            self.player_position = _player_pos_list[0]

        # Update observation space based on actual loaded level dimensions
        self.observation_space = Box(low=0, high=6, shape=self.dim_room, dtype=np.uint8)
        return True

    def _generate_procedural_level(self):
        # This uses the generate_room from gym_sokoban if available
        # Adapted from sokoban_env_old.py
        try:
            from gym_sokoban.envs.room_utils import generate_room as gr_func
            self.room_fixed, self.room_state, self.box_mapping = gr_func(
                dim=self.dim_room,
                num_steps=self.num_gen_steps if self.num_gen_steps else int(1.7 * (self.dim_room[0]+self.dim_room[1])),
                num_boxes=self.num_boxes_initial,
                second_player=False # Not supporting second player
            )
            self.player_position = np.argwhere(self.room_state == 5)[0] # Player is 5
            self.num_boxes_current = self.num_boxes_initial
            self.observation_space = Box(low=0, high=6, shape=self.dim_room, dtype=np.uint8)
            return True
        except Exception as e:
            print(f"[SokobanEnv] Failed to generate procedural level: {e}. Fallback to empty level.")
            # Fallback to a simple empty level if generation fails
            self.dim_room = (5,5) # Smaller default
            self.room_fixed = np.ones(self.dim_room, dtype=np.uint8)
            self.room_state = np.ones(self.dim_room, dtype=np.uint8)
            self.room_fixed[0,:]=0; self.room_fixed[-1,:]=0; self.room_fixed[:,0]=0; self.room_fixed[:,-1]=0 # Walls
            self.room_state = self.room_fixed.copy()
            self.player_position = np.array([self.dim_room[0]//2, self.dim_room[1]//2])
            self.room_state[self.player_position[0], self.player_position[1]] = 5
            self.num_boxes_current = 0
            self.observation_space = Box(low=0, high=6, shape=self.dim_room, dtype=np.uint8)
            return False


    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None, max_memory: Optional[int] = 10, episode_id: int = 1) -> Tuple[Observation, Dict[str, Any]]:
        super().reset(seed=seed)
        self.num_env_steps = 0
        self.current_reward_last_step = 0.0
        
        # Reset Sokoban-specific performance score trackers
        self.previous_boxes_on_target_for_perf = 0
        self.current_episode_cumulative_perf_score = 0.0

        level_loaded_ok = False
        if self.level_to_load and self.level_to_load in self.predefined_levels:
            level_data_str = self.predefined_levels[self.level_to_load]
            if self._parse_level_data(level_data_str):
                level_loaded_ok = True
        
        if not level_loaded_ok: # Fallback to procedural generation or empty
            self._generate_procedural_level()

        self.boxes_on_target = np.count_nonzero(self.room_state == 3) # Box on target is 3
        
        self.adapter.reset_episode(episode_id)
        raw_board_obs = self._get_raw_board_obs()
        info_dict = self._get_info()
        
        # Calculate initial perf score using the overridden method
        initial_perf_score = self.calculate_perf_score(0, info_dict) 
    
        img_path_for_adapter = None
        text_representation_for_adapter = None
        if self.adapter.observation_mode in ["vision", "both"]:
            img_path_for_adapter = self.adapter._create_agent_observation_path(self.adapter.current_episode_id, self.adapter.current_step_num)
            create_board_image_sokoban(raw_board_obs, img_path_for_adapter, tile_size=self.tile_size_for_render, perf_score=initial_perf_score)
        
        if self.adapter.observation_mode in ["text", "both"]:
            char_board_2d_list = [[ROOM_STATE_TO_CHAR.get(tile, '?') for tile in row] for row in raw_board_obs.tolist()]
            # text_representation_for_adapter = str(char_board_2d_list)
            text_representation_for_adapter = self.matrix_to_text_table(char_board_2d_list)

        agent_observation = self.adapter.create_agent_observation(
            img_path=img_path_for_adapter,
            text_representation=text_representation_for_adapter,
            max_memory=max_memory
        )

        if self.render_mode == "human": self._render_frame()
        return agent_observation, info_dict

    def _get_raw_board_obs(self) -> np.ndarray:
        return self.room_state.copy() if self.room_state is not None else np.zeros(self.dim_room, dtype=np.uint8)

    def _get_info(self) -> Dict[str, Any]:
        return {
            "num_env_steps": self.num_env_steps,
            "player_position": self.player_position.tolist() if self.player_position is not None else None,
            "boxes_on_target": self.boxes_on_target,
            "num_boxes": self.num_boxes_current,
            "all_boxes_on_target": self._check_if_all_boxes_on_target(),
            "reward_last_step": self.current_reward_last_step
        }

    def _check_if_all_boxes_on_target(self) -> bool:
        if self.room_state is None or self.num_boxes_current == 0: return True # No boxes means solved
        num_boxes_off_target = np.count_nonzero(self.room_state == 4) # Count boxes not on target (state 4)
        return num_boxes_off_target == 0 and self.boxes_on_target == self.num_boxes_current
        
    def _check_if_maxsteps(self) -> bool:
        return self.num_env_steps >= self.max_steps_episode

    def _calc_reward(self) -> float:
        # Adapted from sokoban_env_old.py
        reward = self.penalty_for_step
        
        current_boxes_on_target = np.count_nonzero(self.room_state == 3) # state 3 is box_on_target
        
        if current_boxes_on_target > self.boxes_on_target:
            reward += self.reward_box_on_target * (current_boxes_on_target - self.boxes_on_target)
        elif current_boxes_on_target < self.boxes_on_target:
            # This implies a box was moved OFF a target.
            reward += self.penalty_box_off_target * (self.boxes_on_target - current_boxes_on_target)
        
        self.boxes_on_target = current_boxes_on_target
        
        if self._check_if_all_boxes_on_target():
            reward += self.reward_finished
        
        return float(reward)

    def _internal_push_or_move(self, action_mapped_idx: int) -> Tuple[bool, bool]:
        # action_mapped_idx: 0-no_op, 1-push_up, ..., 8-move_right
        # We need to map this to simple directions 0-up, 1-down, 2-left, 3-right
        # and determine if it's a push or move.
        
        moved_player, moved_box = False, False
        if action_mapped_idx == 0: return False, False # no_op

        is_push_action = 1 <= action_mapped_idx <= 4
        is_move_action = 5 <= action_mapped_idx <= 8
        
        direction_idx = -1
        if is_push_action: direction_idx = action_mapped_idx - 1 # 1-4 -> 0-3
        elif is_move_action: direction_idx = action_mapped_idx - 5 # 5-8 -> 0-3
        else: return False, False # Should not happen if action mapping is correct

        change = CHANGE_COORDINATES[direction_idx]
        player_r, player_c = self.player_position
        next_player_r, next_player_c = player_r + change[0], player_c + change[1]

        # Boundary checks for next_player_pos
        if not (0 <= next_player_r < self.dim_room[0] and 0 <= next_player_c < self.dim_room[1]):
            return False, False # Player would move out of bounds

        if is_push_action:
            box_target_r, box_target_c = next_player_r + change[0], next_player_c + change[1]
            # Boundary checks for box_target_pos
            if not (0 <= box_target_r < self.dim_room[0] and 0 <= box_target_c < self.dim_room[1]):
                return False, False # Box would be pushed out of bounds

            is_box_at_next = self.room_state[next_player_r, next_player_c] in [3, 4] # Box or BoxOnTarget
            can_box_move_to_target = self.room_state[box_target_r, box_target_c] in [1, 2] # Floor or Target

            if is_box_at_next and can_box_move_to_target:
                # Move Box
                self.room_state[box_target_r, box_target_c] = 3 if self.room_fixed[box_target_r, box_target_c] == 2 else 4
                moved_box = True
                # Move Player (follows box)
                self.room_state[next_player_r, next_player_c] = 6 if self.room_fixed[next_player_r, next_player_c] == 2 else 5
                self.room_state[player_r, player_c] = self.room_fixed[player_r, player_c] # Restore original player tile
                self.player_position = np.array([next_player_r, next_player_c])
                moved_player = True
            # else: # Cannot push, try to move if agent intended push but couldn't
            # This behavior (falling back to move if push fails) can be complex.
            # Let's make it explicit: if a push action is chosen, it either pushes or does nothing.
            # If agent wants to "try push then move", it should send a "move" action if push is known to fail.
            # For now, a failed push is just a failed push.
            
        elif is_move_action:
            if self.room_state[next_player_r, next_player_c] in [1, 2]: # Floor or Target
                self.room_state[next_player_r, next_player_c] = 6 if self.room_fixed[next_player_r, next_player_c] == 2 else 5
                self.room_state[player_r, player_c] = self.room_fixed[player_r, player_c] # Restore original player tile
                self.player_position = np.array([next_player_r, next_player_c])
                moved_player = True
        
        return moved_player, moved_box

    def matrix_to_text_table(self, matrix: List[List[str]]) -> str:
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

    def _progress_to_next_level(self) -> bool:
        """Progress to the next level if available. Returns True if progressed, False if at max level."""
        if self.current_level < self.max_level:
            self.current_level += 1
            self.level_to_load = self.current_level
            return True
        return False

    def step(self, agent_action_str: Optional[str], thought_process: str = "", time_taken_s: float = 0.0) -> Tuple[Observation, float, bool, bool, Dict[str, Any], float]:
        self.adapter.increment_step()
        
        # Map agent string action to environment action index using adapter
        env_action_idx = self.adapter.map_agent_action_to_env_action(agent_action_str)
        
        reward = 0.0
        terminated = False
        truncated = False
        
        if env_action_idx is not None and self.action_space.contains(env_action_idx):
            self._internal_push_or_move(env_action_idx)
            reward = self._calc_reward()
            terminated = self._check_if_all_boxes_on_target()
            
            # If level is completed, try to progress to next level
            if terminated and self._progress_to_next_level():
                # Reset the environment for the new level
                self.reset()
                # Return the new observation and info
                raw_board_obs = self._get_raw_board_obs()
                info_dict = self._get_info()
                current_perf_score = self.calculate_perf_score(reward, info_dict)
                
                img_path_for_adapter = None
                text_representation_for_adapter = None
                if self.adapter.observation_mode in ["vision", "both"]:
                    img_path_for_adapter = self.adapter._create_agent_observation_path(self.adapter.current_episode_id, self.adapter.current_step_num)
                    create_board_image_sokoban(raw_board_obs, img_path_for_adapter, tile_size=self.tile_size_for_render, perf_score=current_perf_score, action_taken_str=agent_action_str)
                
                if self.adapter.observation_mode in ["text", "both"]:
                    char_board_2d_list = [[ROOM_STATE_TO_CHAR.get(tile, '?') for tile in row] for row in raw_board_obs.tolist()]
                    text_representation_for_adapter = self.matrix_to_text_table(char_board_2d_list)

                agent_observation = self.adapter.create_agent_observation(
                    img_path=img_path_for_adapter,
                    text_representation=text_representation_for_adapter
                )
                
                return agent_observation, reward, False, False, info_dict, current_perf_score
        else: # Invalid or no action from agent
            print(f"[SokobanEnv] Action '{agent_action_str}' (mapped to {env_action_idx}) is skip/invalid. Env not stepped.")
            reward = self.penalty_for_step
            terminated = self._check_if_all_boxes_on_target()

        self.num_env_steps += 1
        truncated = self._check_if_maxsteps()
        self.current_reward_last_step = reward

        raw_board_obs = self._get_raw_board_obs()
        info_dict = self._get_info()
        current_perf_score = self.calculate_perf_score(reward, info_dict)
        
        img_path_for_adapter = None
        text_representation_for_adapter = None
        if self.adapter.observation_mode in ["vision", "both"]:
            img_path_for_adapter = self.adapter._create_agent_observation_path(self.adapter.current_episode_id, self.adapter.current_step_num)
            create_board_image_sokoban(raw_board_obs, img_path_for_adapter, tile_size=self.tile_size_for_render, perf_score=current_perf_score, action_taken_str=agent_action_str)
        
        if self.adapter.observation_mode in ["text", "both"]:
            char_board_2d_list = [[ROOM_STATE_TO_CHAR.get(tile, '?') for tile in row] for row in raw_board_obs.tolist()]
            text_representation_for_adapter = self.matrix_to_text_table(char_board_2d_list)

        agent_observation = self.adapter.create_agent_observation(
            img_path=img_path_for_adapter,
            text_representation=text_representation_for_adapter
        )
        
        final_terminated, final_truncated = self.adapter.verify_termination(agent_observation, terminated, truncated)

        self.adapter.log_step_data(
            agent_action_str=agent_action_str,
            thought_process=thought_process,
            reward=reward,
            info=info_dict,
            terminated=final_terminated,
            truncated=final_truncated,
            time_taken_s=time_taken_s,
            perf_score=current_perf_score,
            agent_observation=agent_observation
        )

        if self.render_mode == "human": self._render_frame()
        return agent_observation, reward, final_terminated, final_truncated, info_dict, current_perf_score

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]: # Type hint from gym.Env
        if self.render_mode == "rgb_array":
            return self._render_frame_rgb()
        elif self.render_mode == "human":
            self._render_frame() # Handles its own display
            return None 
        elif self.render_mode == "raw": # For debugging or specific agent needs
            return self.room_fixed, self.room_state, self.player_position, self.boxes_on_target
        return None # Default for other modes or if no rendering

    def _render_frame_rgb(self) -> Optional[np.ndarray]:
        if self.room_state is None: return None
        
        # Create a temporary path to save the image, then load it as numpy array
        # This is a bit indirect but reuses create_board_image_sokoban
        temp_img_path = os.path.join(self.adapter.agent_cache_dir, "_temp_render.png")
        create_board_image_sokoban(self.room_state, temp_img_path, self.tile_size_for_render)
        if os.path.exists(temp_img_path):
            img = Image.open(temp_img_path).convert('RGB')
            rgb_array = np.array(img)
            os.remove(temp_img_path)
            return rgb_array
        return None

    def _render_frame(self): # For human mode
        if self.room_state is None: return

        img_rgb_array = self._render_frame_rgb()
        if img_rgb_array is None: return

        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((img_rgb_array.shape[1], img_rgb_array.shape[0])) # W, H
            pygame.display.set_caption(f"Sokoban - Level {self.current_level}")
        else:
            # Check if window size needs to be updated for new level
            current_size = self.window.get_size()
            new_size = (img_rgb_array.shape[1], img_rgb_array.shape[0])
            if current_size != new_size:
                self.window = pygame.display.set_mode(new_size)
            pygame.display.set_caption(f"Sokoban - Level {self.current_level}")

        if self.clock is None:
            self.clock = pygame.time.Clock()

        surf = pygame.surfarray.make_surface(img_rgb_array.swapaxes(0, 1)) # Pygame needs WxH
        self.window.blit(surf, (0, 0))
        pygame.event.pump()
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
        self.adapter.close_log_file()
        print("[SokobanEnv] Closed.")

    def calculate_perf_score(self, reward: float, info: Dict[str, Any]) -> float:
        """
        Calculates a performance score for the current step for Sokoban.
        This is based on the cumulative count of newly placed boxes on targets 
        within the current episode.

        Args:
            reward (float): The reward received for the step (not directly used here).
            info (Dict[str, Any]): Additional information from the environment, expected
                                   to contain "boxes_on_target".

        Returns:
            float: The cumulative performance score for the episode up to this step.
        """
        current_boxes_on_target = info.get("boxes_on_target", 0)
        
        delta_boxes = 0
        if current_boxes_on_target > self.previous_boxes_on_target_for_perf:
            delta_boxes = current_boxes_on_target - self.previous_boxes_on_target_for_perf
        
        # Only add positive delta to the cumulative score
        if delta_boxes > 0:
            self.current_episode_cumulative_perf_score += float(delta_boxes)
            
        self.previous_boxes_on_target_for_perf = current_boxes_on_target
        
        return self.current_episode_cumulative_perf_score
