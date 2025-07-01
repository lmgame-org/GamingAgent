import gymnasium as gym
import numpy as np
import os
import json
import re
from typing import Any, Dict, Tuple, Optional, List, Union
from PIL import Image, ImageDraw, ImageFont
import hashlib
# import imageio # Commented out for now
# import tempfile # Commented out for now
# import shutil # Commented out for now
from collections import OrderedDict

from gamingagent.modules.core_module import Observation
from gamingagent.envs.gym_env_adapter import GymEnvAdapter # Added
from gymnasium.spaces import Discrete, Box

# Imports from TileMatchEnv
from tile_match_gym.board import Board
from tile_match_gym.board import is_move_effective
from tile_match_gym.renderer import Renderer

# Define constants for Candy Crush elements (example)
# These should match what TileMatchEnv uses or how you want to represent them textually
COLOR_MAP = {
    0: " ",  # Empty or background
    1: "G",  # Green
    2: "C",  # Cyan
    3: "P",  # Purple
    4: "R",  # Red
    5: "Y",  # Yellow (if used)
    6: "B",  # Blue (if used)
    # Add more colors/specials as needed by your TileMatchEnv config
}

def create_board_image_candy_crush(board_state: np.ndarray, save_path: str, tile_size: int = 32, perf_score: Optional[float] = None, action_taken_str: Optional[str] = None, moves_left: Optional[int] = None) -> None:
    """
    Create a visualization of the Candy Crush board.
    board_state: A 2D numpy array representing the Candy Crush board (color indices).
    """
    if board_state is None or board_state.size == 0:
        # Create a dummy image indicating an error
        img = Image.new('RGB', (tile_size * 5, tile_size * 2), (128, 128, 128))
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), "Error: No board state", fill=(255, 0, 0))
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            img.save(save_path)
        return

    rows, cols = board_state.shape
    img_width = cols * tile_size
    img_height = rows * tile_size # Height of the board itself

    # Determine font size for info text (used for both calculation and drawing)
    font_size_info = max(12, tile_size // 2)

    # Determine number of lines for info text and calculate required height
    num_info_lines = 0
    if moves_left is not None: num_info_lines += 1
    if perf_score is not None: num_info_lines += 1

    info_strip_alloc_height = 0
    top_text_padding = 3 # Desired padding above the text block (reduced from implied 5)
    
    if num_info_lines > 0:
        # Estimate character pixel height based on font point size
        # Multiplying by 1.25 as a heuristic for typical font height vs. point size
        estimated_char_pixel_height = int(font_size_info * 1.25) 
        bottom_text_padding = 3 # Desired padding below the text block
        inter_line_padding = 2 # Pixels between lines of text, based on current_y update logic

        # Calculate the height needed for the text block itself
        text_block_actual_height = (num_info_lines * estimated_char_pixel_height) + \
                                   max(0, num_info_lines - 1) * inter_line_padding
        
        # Total height for the info strip including padding
        calculated_total_info_strip_height = top_text_padding + text_block_actual_height + bottom_text_padding
        
        # Ensure info strip is at least tile_size (original behavior if that's larger) or calculated height
        info_strip_alloc_height = int(max(tile_size, calculated_total_info_strip_height))
    
    # Create image: board part + info strip part
    img = Image.new('RGB', (img_width, img_height + info_strip_alloc_height), (200, 200, 200))
    draw = ImageDraw.Draw(img)

    # Define some basic colors for tiles - extend as needed
    tile_colors_rgb = {
        0: (200, 200, 200), # Empty
        1: (0, 255, 0),     # Green
        2: (0, 255, 255),   # Cyan
        3: (128, 0, 128),   # Purple
        4: (255, 0, 0),     # Red
        5: (255, 255, 0),   # Yellow
        6: (0, 0, 255),     # Blue
    }

    for r in range(rows):
        for c in range(cols):
            x0, y0 = c * tile_size, r * tile_size
            tile_val = int(board_state[r, c])
            color = tile_colors_rgb.get(tile_val, (128, 128, 128)) # Default to gray
            draw.rectangle([x0, y0, x0 + tile_size, y0 + tile_size], fill=color, outline=(0,0,0))
            try:
                font_size = max(8, tile_size // 2)
                font = ImageFont.truetype("arial.ttf", font_size)
            except IOError:
                try:
                    font = ImageFont.load_default(size=font_size) # For Pillow >= 9.3.0
                except AttributeError: # Fallback for older Pillow versions
                    font = ImageFont.load_default()

            text = COLOR_MAP.get(tile_val, str(tile_val))
            if hasattr(font, 'getbbox'): # For Pillow >= 10.0.0
                bbox = font.getbbox(text)
                text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            else: # For Pillow < 10.0.0
                text_w, text_h = draw.textsize(text, font=font) # type: ignore
            draw.text((x0 + (tile_size - text_w) // 2, y0 + (tile_size - text_h) // 2), text, fill=(0,0,0), font=font)

    # Display moves_left, perf_score at the bottom if provided
    # info_y_start is the y-coordinate for the *top* of the first line of text,
    # relative to the image top (0). It starts after the board area.
    info_y_start = img_height + top_text_padding 
    
    # font_size_info was already calculated above
    try: font_info = ImageFont.truetype("arial.ttf", font_size_info)
    except IOError: 
        try: font_info = ImageFont.load_default(size=font_size_info) # For Pillow >= 9.3.0
        except AttributeError: # Fallback for older Pillow versions
            font_info = ImageFont.load_default()

    current_y = info_y_start
    if moves_left is not None:
        text_content = f"Moves: {moves_left}"
        draw.text((5, current_y), text_content, fill=(0,0,0), font=font_info)
        if hasattr(font_info, 'getbbox'): current_y += font_info.getbbox(text_content)[3] - font_info.getbbox(text_content)[1] + 2
        else: current_y += draw.textsize(text_content, font=font_info)[1] + 2 # type: ignore

    if perf_score is not None:
        text_content = f"Perf: {perf_score:.2f}"
        draw.text((5, current_y), text_content, fill=(0,0,0), font=font_info)
        if hasattr(font_info, 'getbbox'): current_y += font_info.getbbox(text_content)[3] - font_info.getbbox(text_content)[1] + 2
        else: current_y += draw.textsize(text_content, font=font_info)[1] + 2 # type: ignore

    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir: os.makedirs(save_dir, exist_ok=True)
        img.save(save_path)

class CandyCrushEnvWrapper(gym.Env):
    metadata = {"render_modes": ["string", "human", "rgb_array"], "render_fps": 2}

    def __init__(self, 
                 # Parameters for TileMatchEnv core
                 num_rows_override: Optional[int] = None, # Allow overriding from runner
                 num_cols_override: Optional[int] = None,
                 num_colours_override: Optional[int] = None,
                 num_moves_override: Optional[int] = None,
                 # Parameters for GymEnvAdapter
                 game_name_for_adapter: str = "candy_crush",
                 observation_mode_for_adapter: str = "vision", # "vision", "text", "both"
                 agent_cache_dir_for_adapter: str = "cache/candy_crush/default_run",
                 game_specific_config_path_for_adapter: str = "gamingagent/envs/custom_03_candy_crush/game_env_config.json",
                 max_stuck_steps_for_adapter: Optional[int] = None, # For GymEnvAdapter
                 # Other params
                 config_root_dir: Optional[str] = "configs", # For finding game_env_config.json
                 log_root_dir: Optional[str] = "runs_output" # Potentially for replays
                 ):
        super().__init__()

        self.config_root_dir = config_root_dir
        self.log_root_dir = log_root_dir # Stored for potential replay use

        self._game_env_config: Dict[str, Any] = {}
        self._load_candy_crush_config(game_specific_config_path_for_adapter) # Load this first for env_init_kwargs
        
        # --- Start of TileMatchEnv __init__ logic integration ---
        env_init_kwargs = self._game_env_config.get("env_init_kwargs", {})
        
        # Use overrides if provided, else use values from config, else use TileMatchEnv defaults
        self.num_rows = num_rows_override if num_rows_override is not None else env_init_kwargs.get("num_rows", 8)
        self.num_cols = num_cols_override if num_cols_override is not None else env_init_kwargs.get("num_cols", 8)
        self.num_colours = num_colours_override if num_colours_override is not None else env_init_kwargs.get("num_colours", 4)
        self.num_moves = num_moves_override if num_moves_override is not None else env_init_kwargs.get("num_moves", 50)
        
        self.colourless_specials = env_init_kwargs.get("colourless_specials", [])
        self.colour_specials = env_init_kwargs.get("colour_specials", [])
        self.seed_val = env_init_kwargs.get("seed") # Can be None

        # Renderer setup (adapted from TileMatchEnv)
        self.renderer: Optional[Renderer] = None
        # Use render_mode from game_env_config, default to "string" (no pygame) if not specified
        self.internal_render_mode = self._game_env_config.get("render_mode_for_make", "string") 
        self.render_mode = self.internal_render_mode # Expose for runner
        if self.internal_render_mode in ["human", "rgb_array"]:
            self.renderer = Renderer(self.num_rows, self.num_cols, self.num_colours, self.num_moves, render_fps=self.metadata["render_fps"], render_mode=self.internal_render_mode)
        elif self.internal_render_mode == "string":
             # For string rendering, ensure colour_map is initialized
            if not hasattr(self, 'np_random') or self.np_random is None: # np_random is set below
                temp_rng = np.random.default_rng(seed=self.seed_val)
                self.colour_map = temp_rng.choice(range(105, 230), size=self.num_colours + 1, replace=False)

        self.num_colour_specials = len(self.colour_specials)
        self.num_colourless_specials = len(self.colourless_specials)

        # Board and random number generator setup
        self.np_random = np.random.default_rng(seed=self.seed_val)
        self.board = Board(self.num_rows, self.num_cols, self.num_colours, self.colourless_specials, self.colour_specials, self.np_random)
        if self.internal_render_mode == "string" and not hasattr(self, 'colour_map'): # Ensure colour_map if not set during renderer init
            self.colour_map = self.np_random.choice(range(105, 230), size=self.num_colours + 1, replace=False)


        # Action space (from TileMatchEnv)
        self.num_actions = int((self.num_rows * self.num_cols * 2) - self.num_rows - self.num_cols)
        self._action_to_coords = self.board.action_to_coords # TileMatchEnv provides this mapping
        
        self.action_space = Discrete(self.num_actions, seed=self.seed_val if self.seed_val is not None else np.random.randint(1_000_000))

        # Internal observation space (raw, from TileMatchEnv)
        _obs_low_board = np.array([np.zeros((self.num_rows, self.num_cols), dtype=np.int32),
                            np.full((self.num_rows, self.num_cols), - self.num_colourless_specials, dtype=np.int32)])
        _obs_high_board = np.array([np.full((self.num_rows, self.num_cols), self.num_colours, dtype=np.int32),
                             np.full((self.num_rows, self.num_cols), self.num_colour_specials + 2,
                                     dtype=np.int32)])
        self._gym_board_observation_space = Box(
            low=_obs_low_board, high=_obs_high_board,
            shape=(2, self.num_rows, self.num_cols), dtype=np.int32,
            seed=self.seed_val if self.seed_val is not None else np.random.randint(1_000_000)
        )
        self._gym_moves_left_observation_space = Discrete(self.num_moves + 1, seed=self.seed_val if self.seed_val is not None else np.random.randint(1_000_000))
        
        # This is the observation space of the underlying TileMatchEnv, not what the agent sees.
        self.observation_space = gym.spaces.Dict({
            "board": self._gym_board_observation_space,
            "num_moves_left": self._gym_moves_left_observation_space
        })
        
        self.timer: Optional[int] = None
        self.current_score: float = 0.0 # Raw score from eliminations
        # --- End of TileMatchEnv

        # Initialize Adapter
        self.adapter = GymEnvAdapter(
            game_name=game_name_for_adapter,
            observation_mode=observation_mode_for_adapter,
            agent_cache_dir=agent_cache_dir_for_adapter,
            game_specific_config_path=game_specific_config_path_for_adapter,
            max_steps_for_stuck=max_stuck_steps_for_adapter
        )
        
        # Fallback action mapping if adapter's config is empty
        # self.move_to_action_idx from adapter takes precedence if loaded.
        self.env_move_to_action_idx: Dict[str, int] = {}
        self.env_action_idx_to_move: Dict[int, str] = {}
        if self._action_to_coords:
            for idx, coords_pair in enumerate(self._action_to_coords):
                coord1 = tuple(coords_pair[0])
                coord2 = tuple(coords_pair[1])

                c_min = min(coord1, coord2) # Sort by tuple comparison
                c_max = max(coord1, coord2)
                action_str = f"(({c_min[0]},{c_min[1]}),({c_max[0]},{c_max[1]}))"
                self.env_move_to_action_idx[action_str] = idx
                self.env_action_idx_to_move[idx] = action_str
        
        # Attributes for tracking performance score within the env if needed
        self.current_episode_cumulative_perf_score: float = 0.0
        self.last_action_str_for_render: Optional[str] = None # For rendering on image

        print(f"[CandyCrushEnvWrapper] Initialized. Adapter obs mode: {self.adapter.observation_mode}")

    def _load_candy_crush_config(self, game_specific_config_path: str):
        """Loads game_env_config.json to get env_init_kwargs and other settings."""
        config_file_to_load = game_specific_config_path # Directly use the path provided by runner
        
        if os.path.exists(config_file_to_load):
            try:
                with open(config_file_to_load, 'r') as f:
                    self._game_env_config = json.load(f)
                print(f"[CandyCrushEnvWrapper] Loaded game_env_config from: {config_file_to_load}")
            except json.JSONDecodeError as e:
                print(f"[CandyCrushEnvWrapper] ERROR decoding JSON from {config_file_to_load}: {e}. Using defaults.")
                self._game_env_config = {} # Ensure it's a dict
        else:
            print(f"[CandyCrushEnvWrapper] WARNING: Config {config_file_to_load} not found. Using default env_init_kwargs.")
        
        # Ensure env_init_kwargs exists
        if "env_init_kwargs" not in self._game_env_config:
            self._game_env_config["env_init_kwargs"] = {
                "num_rows": 8, "num_cols": 8, "num_colours": 4, "num_moves": 50,
                "colourless_specials": [], "colour_specials": [], "seed": None
            }
        if "tile_size_for_render" not in self._game_env_config:
             self._game_env_config["tile_size_for_render"] = 32
        if "render_mode_for_make" not in self._game_env_config:
             self._game_env_config["render_mode_for_make"] = "string"


    # --- Methods from/inspired by TileMatchEnv ---
    def _set_internal_seed(self, seed: Optional[int]) -> None:
        """Sets the seed for internal randomization aspects like action space and board."""
        actual_seed = seed if seed is not None else np.random.randint(1_000_000)
        self.np_random = np.random.default_rng(seed=actual_seed)
        if self.board:
            self.board.np_random = self.np_random
        if self.action_space: # action_space might not be initialized if called very early
            self.action_space.seed(actual_seed)
        # Re-initialize colour_map for string rendering if needed
        if self.internal_render_mode == "string" or not self.renderer:
            self.colour_map = self.np_random.choice(range(105, 230), size=self.num_colours + 1, replace=False)


    def _get_obs(self) -> Dict[str, Union[np.ndarray, int]]:
        """Generates the raw internal gym-style observation (board state and moves left)."""
        if self.timer is None: # Should be set in reset
            current_moves_left = self.num_moves 
        else:
            current_moves_left = self.num_moves - self.timer
        return OrderedDict([("board", self.board.board.copy()), ("num_moves_left", current_moves_left)])

    def _get_effective_actions(self) -> List[int]:
        """Gets a list of actions that would result in a match or special activation."""
        if self.timer is None or self.timer >= self.num_moves:
            return []
        # TileMatchEnv action_to_coords is a list of tuples of np.arrays. Convert to tuples of tuples for lambda.
        # However, is_move_effective expects np.arrays.
        # _action_to_coords is already List[Tuple[np.ndarray, np.ndarray]]
        action_check = lambda a: is_move_effective(self.board.board, self._action_to_coords[a][0], self._action_to_coords[a][1])
        effective_actions = list(filter(action_check, range(self.num_actions)))
        return effective_actions
    # --- End of TileMatchEnv inspired methods ---

    # --- Gym Core Methods (internal versions) ---
    def _reset_gym(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict, Dict]:
        if seed is not None:
            self.seed_val = seed # Store for potential re-seeding
            self._set_internal_seed(seed)
        elif self.seed_val is not None: # If a seed was set at init
            self._set_internal_seed(self.seed_val)


        self.board.generate_board()
        self.timer = 0
        self.current_score = 0.0
        self.current_episode_cumulative_perf_score = 0.0 # Reset cumulative perf score for the episode
        
        raw_observation = self._get_obs()
        info = {
            'effective_actions': self._get_effective_actions(),
            'score': self.current_score, # This is step score, which is 0 at reset
            'cumulative_raw_score': self.current_score, # Cumulative raw score (eliminations)
            'total_score': self.current_score, # Add for consistent runner access
            'num_moves_left': raw_observation["num_moves_left"]
        }
        
        return raw_observation, info

    def _step_gym(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        if self.timer is None or self.timer >= self.num_moves:
            # This case should ideally be caught before calling _step_gym
            empty_board_obs = np.zeros((2, self.num_rows, self.num_cols), dtype=np.int32)
            raw_obs = OrderedDict([("board", empty_board_obs), ("num_moves_left", 0)])
            return raw_obs, 0.0, True, False, {"score": self.current_score, "effective_actions": [], "cumulative_raw_score": self.current_score, "num_moves_left": 0}

        coord1, coord2 = self._action_to_coords[action] # action is int
        num_eliminations, is_combination_match, num_new_specials, num_specials_activated, shuffled = self.board.move(coord1, coord2)

        self.timer += 1
        terminated = self.timer >= self.num_moves
        
        # Reward for this step is just the number of eliminations
        reward = float(num_eliminations) 
        self.current_score += reward # Accumulate raw score

        raw_observation = self._get_obs()

        info = {
            "score": reward,
            "cumulative_raw_score": self.current_score,
            "total_score": self.current_score,
            "is_combination_match": is_combination_match,
            "num_new_specials": num_new_specials,
            "num_specials_activated": num_specials_activated,
            "shuffled": shuffled,
            "effective_actions": self._get_effective_actions(),
            "num_moves_left": raw_observation["num_moves_left"]
        }
        
        return raw_observation, reward, terminated, False, info

    # Runner-facing Methods
    def reset(self, seed: Optional[int] = None, episode_id: int = 1, options: Optional[dict] = None, max_memory: Optional[int] = 10) -> Tuple[Observation, Dict[str, Any]]:
        self.adapter.reset_episode(episode_id)
        
        gym_options = options if options is not None else {}
        # episode_id is handled by adapter.reset_episode

        raw_obs_dict, info_dict = self._reset_gym(seed=seed, options=gym_options)
        
        # Prepare data for agent observation
        img_path_for_adapter, text_representation_for_adapter = None, None
        board_array_for_render = self.get_board_state(raw_obs_dict, info_dict) # Extracts color board
        
        # Cumulative perf score is 0 at reset start
        initial_perf_score = self.current_episode_cumulative_perf_score 

        if self.adapter.observation_mode in ["vision", "both"] and board_array_for_render is not None:
            img_path_for_adapter = self.adapter._create_agent_observation_path(self.adapter.current_episode_id, self.adapter.current_step_num)
            create_board_image_candy_crush(
                board_array_for_render, 
                img_path_for_adapter, 
                tile_size=self._game_env_config.get("tile_size_for_render", 32), 
                perf_score=initial_perf_score, 
                action_taken_str="Reset", 
                moves_left=raw_obs_dict.get("num_moves_left")
            )
        
        if self.adapter.observation_mode in ["text", "both"]:
            text_parts = []
            if board_array_for_render is not None:
                text_parts.append("Board:")
                for r_idx, row_val in enumerate(board_array_for_render):
                     text_parts.append(f"{r_idx}| {' '.join([COLOR_MAP.get(int(tile), '?') for tile in row_val])}")
            else:
                text_parts.append("Board: [Data not available]")
            text_parts.extend([
                f"Score: {info_dict.get('cumulative_raw_score', 0.0):.0f}", 
                f"Moves Left: {raw_obs_dict.get('num_moves_left', self.num_moves)}"
            ])
            text_representation_for_adapter = "\n".join(text_parts)

        agent_observation = self.adapter.create_agent_observation(
            img_path=img_path_for_adapter,
            text_representation=text_representation_for_adapter,
            max_memory=max_memory
        )
        
        # Store symbolic representation for stuck detection by adapter
        return agent_observation, info_dict

    def _parse_agent_action_str(self, action_str_agent: Optional[str]) -> Tuple[Optional[int], str]:
        """ Parses agent action string to int. Returns (action_int, processed_action_str_for_logging) """
        if action_str_agent is None:
            return None, "None"

        processed_action_str = str(action_str_agent) # Default for logging
        
        # 1. Try adapter's mapping first
        action_int = self.adapter.map_agent_action_to_env_action(action_str_agent)
        if action_int is not None:
            # Adapter uses lowercase keys, so get the original-case version if available from env map
            processed_action_str = self.env_action_idx_to_move.get(action_int, action_str_agent.lower())
            return action_int, processed_action_str

        # 2. Fallback: Try parsing "((r1,c1),(r2,c2))" format using env's internal mapping
        if isinstance(action_str_agent, str):
            match = re.match(r"^\s*\(\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*\)", action_str_agent.strip())
            if match:
                try:
                    r1, c1, r2, c2 = map(int, match.groups())
                    coord1 = (r1, c1)
                    coord2 = (r2, c2)
                    
                    # Canonical key for env_move_to_action_idx (which sorts the two coordinate pairs)
                    coords_pair_sorted_for_key = tuple(sorted((coord1, coord2)))
                    action_key_for_lookup = f"(({coords_pair_sorted_for_key[0][0]},{coords_pair_sorted_for_key[0][1]}),({coords_pair_sorted_for_key[1][0]},{coords_pair_sorted_for_key[1][1]}))"
                    
                    action_int = self.env_move_to_action_idx.get(action_key_for_lookup)
                    if action_int is not None:
                        processed_action_str = action_key_for_lookup # Use the canonical form for logging
                        return action_int, processed_action_str
                except ValueError:
                    pass 
        
        # 3. Fallback: Try direct integer conversion
        if isinstance(action_str_agent, str):
            try:
                action_int_candidate = int(action_str_agent.strip())
                if 0 <= action_int_candidate < self.num_actions:
                    action_int = action_int_candidate
                    processed_action_str = self.env_action_idx_to_move.get(action_int, str(action_int_candidate))
                    return action_int, processed_action_str
            except ValueError:
                pass
        elif isinstance(action_str_agent, int): # If agent directly passes an int
             if 0 <= action_str_agent < self.num_actions:
                 action_int = action_str_agent
                 processed_action_str = self.env_action_idx_to_move.get(action_int, str(action_str_agent))
                 return action_int, processed_action_str
        
        return None, str(action_str_agent) # Action could not be parsed/mapped

    def _calculate_step_perf_score(self, info_from_gym_step: Dict[str, Any]) -> float:
        """Calculates performance score for the current step."""
        num_elim = info_from_gym_step.get("score", 0) # Step reward from _step_gym is num_eliminations
        num_new_specials = info_from_gym_step.get("num_new_specials", 0)
        num_specials_activated = info_from_gym_step.get("num_specials_activated", 0)
        
        step_perf = float(num_elim) + (5.0 * float(num_new_specials)) + (3.0 * float(num_specials_activated))
        self.current_episode_cumulative_perf_score += step_perf
        return step_perf # Return step perf, adapter logs cumulative if it wants to from its side (it does not currently)

    def step(self, agent_action_str: str, thought_process: Optional[str] = None, time_taken_s: Optional[float] = 0.0) -> Tuple[Observation, float, bool, bool, Dict[str, Any], float]:
        self.adapter.increment_step()
        
        action_int, parsed_action_str_for_log = self._parse_agent_action_str(agent_action_str)
        self.last_action_str_for_render = parsed_action_str_for_log # For rendering on image

        raw_obs_dict: Dict
        reward: float
        terminated: bool
        truncated: bool = False # TileMatchEnv doesn't inherently truncate beyond num_moves
        info_dict: Dict[str, Any]

        if action_int is not None:
            raw_obs_dict, reward, terminated, truncated, info_dict = self._step_gym(action_int)
            info_dict['executed_action_int_for_replay'] = action_int
        else:
            # Invalid action from agent: treat as a "null" move
            print(f"[CandyCrushEnvWrapper] Warning: Invalid action '{parsed_action_str_for_log}' (original: '{agent_action_str}'). Null move.")
            self.timer = (self.timer if self.timer is not None else -1) + 1 
            reward = 0.0 # No points for invalid action
            terminated = self.timer >= self.num_moves if self.timer is not None else False
            raw_obs_dict = self._get_obs() 
            info_dict = {
                "score": 0.0, # Step reward
                "cumulative_raw_score": self.current_score,
                "total_score": self.current_score,
                "is_combination_match": False, 
                "num_new_specials": 0,
                "num_specials_activated": 0, 
                "shuffled": False,
                "effective_actions": self._get_effective_actions(), 
                "num_moves_left": raw_obs_dict.get("num_moves_left", 0),
                "invalid_action_taken": True, 
                "original_agent_action": agent_action_str,
                "parsed_action_str": parsed_action_str_for_log,
                'executed_action_int_for_replay': None
            }
        
        # Store raw observation dictionary in info for potential replay from logs
        info_dict['raw_env_observation_for_replay'] = raw_obs_dict.copy()

        # Calculate performance score for this step
        current_step_perf_score = self._calculate_step_perf_score(info_dict)
        
        # Prepare data for agent observation
        img_path_for_adapter, text_representation_for_adapter = None, None
        board_array_for_render = self.get_board_state(raw_obs_dict, info_dict)

        if self.adapter.observation_mode in ["vision", "both"] and board_array_for_render is not None:
            img_path_for_adapter = self.adapter._create_agent_observation_path(self.adapter.current_episode_id, self.adapter.current_step_num)
            create_board_image_candy_crush(
                board_array_for_render,
                img_path_for_adapter,
                tile_size=self._game_env_config.get("tile_size_for_render", 32),
                perf_score=self.current_episode_cumulative_perf_score, # Display cumulative perf on image
                action_taken_str=self.last_action_str_for_render,
                moves_left=raw_obs_dict.get("num_moves_left")
            )

        if self.adapter.observation_mode in ["text", "both"]:
            text_parts = []
            if board_array_for_render is not None:
                text_parts.append("Board:")
                for r_idx, row_val in enumerate(board_array_for_render):
                     text_parts.append(f"{r_idx}| {' '.join([COLOR_MAP.get(int(tile), '?') for tile in row_val])}")
            else:
                text_parts.append("Board: [Data not available]")
            text_parts.extend([
                f"Score: {info_dict.get('cumulative_raw_score', 0.0):.0f}", 
                f"Moves Left: {raw_obs_dict.get('num_moves_left', 0)}",
                f"Last Action: {self.last_action_str_for_render if self.last_action_str_for_render else 'N/A'}"
            ])
            text_representation_for_adapter = "\n".join(text_parts)

        agent_observation = self.adapter.create_agent_observation(
            img_path=img_path_for_adapter,
            text_representation=text_representation_for_adapter
        )
        
        final_terminated, final_truncated = self.adapter.verify_termination(agent_observation, terminated, truncated)

        self.adapter.log_step_data(
            agent_action_str=parsed_action_str_for_log, # Log the processed action string
            thought_process=thought_process if thought_process is not None else "",
            reward=reward, # Step reward
            info=info_dict, # Rich info from gym_step
            terminated=final_terminated,
            truncated=final_truncated,
            time_taken_s=time_taken_s if time_taken_s is not None else 0.0,
            perf_score=current_step_perf_score, # Log step perf score
            agent_observation=agent_observation
        )
        
        # The step perf score is returned, runner will sum it up for total episode perf score.
        return agent_observation, reward, final_terminated, final_truncated, info_dict, current_step_perf_score


    # def game_replay(self, replay_data: Dict[str, Any], output_video_path: Optional[str] = None, frame_duration: float = 0.5) -> None:
    #     # ... (Commented out for now) ...

    # def replay_from_seed_and_actions(self, initial_seed: int, executed_action_ints: List[Optional[int]], output_video_path: Optional[str] = None, frame_duration: float = 0.5, tile_size_for_replay_render: int = 32) -> None:
    #     # ... (Commented out for now) ...

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        self.adapter.close_log_file() # Close current episode log
        # Note: finalize_and_save_summary is typically called by the runner, not here.
        print("[CandyCrushEnvWrapper] Closed.")


    def get_board_state(self, raw_observation: Any, info: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extracts the color board (2D numpy array) from raw gym observation."""
        board_array_3d = None
        if isinstance(raw_observation, dict) and "board" in raw_observation:
            board_array_3d = raw_observation.get("board")
        # raw_observation could also be directly the board if not from _get_obs()
        elif isinstance(raw_observation, np.ndarray) and raw_observation.ndim == 3 and raw_observation.shape[0] == 2:
             board_array_3d = raw_observation

        if isinstance(board_array_3d, np.ndarray) and board_array_3d.ndim == 3 and board_array_3d.shape[0] >=1:
            # TileMatchEnv board obs is (2, rows, cols), first layer is color, second is type
            return board_array_3d[0] # Return the color layer
        elif isinstance(board_array_3d, np.ndarray) and board_array_3d.ndim == 2: # If it's already 2D
            return board_array_3d
        
        print(f"[CandyCrushEnvWrapper] Warning: Could not extract 2D color board from raw_observation: {type(raw_observation)}")
        return None

    def map_env_action_to_agent_action(self, env_action_idx: int) -> str:
        """Maps an internal environment action index to its string representation."""
        # Try adapter's mapping first (which should be loaded from game_env_config.json)
        if env_action_idx in self.adapter.action_idx_to_move:
            return self.adapter.action_idx_to_move[env_action_idx]
        # Fallback to environment's internal generation
        if env_action_idx in self.env_action_idx_to_move:
            return self.env_action_idx_to_move[env_action_idx]
        return f"action_index_{env_action_idx}" # Generic fallback

    def render(self, mode: Optional[str] = None) -> Union[None, np.ndarray, List[np.ndarray]]: # Adjusted return type
        render_mode_to_use = mode if mode is not None else self.internal_render_mode

        if render_mode_to_use == "string":
            if not (hasattr(self, 'board') and self.board and self.board.board is not None):
                print("[CandyCrushEnvWrapper] Board not available for rendering.")
                return None
            # Ensure colour_map is initialized for string rendering
            if not hasattr(self, 'colour_map') or self.colour_map is None:
                if not hasattr(self, 'np_random') or self.np_random is None: 
                    self._set_internal_seed(self.seed_val if self.seed_val is not None else 123)
                self.colour_map = self.np_random.choice(range(105, 230), size=self.num_colours + 1, replace=False)

            # TileMatchEnv board is (2, height, width)
            # board[0] is color, board[1] is type (not used by COLOR_MAP directly here)
            color_board = self.board.board[0] 
            height, width = color_board.shape
            
            # ANSI escape codes for colors
            # Using a simplified version that relies on COLOR_MAP for chars and basic ANSI for background
            # This is a simplified string render, not using tile_match_gym's complex ANSI.
            output_lines = [" " + "-" * (width * 2 + 1)]
            for r_num in range(height):
                line_str = ["| "]
                for c_col in range(width):
                    tile_colour_idx = int(color_board[r_num, c_col])
                    display_char = COLOR_MAP.get(tile_colour_idx, '?')
                    # For simplicity, not attempting complex ANSI colors here like TileMatchEnv's renderer
                    line_str.append(f"{display_char} ") 
                line_str.append("|")
                output_lines.append("".join(line_str))
            output_lines.append(" " + "-" * (width * 2 + 1))

            if self.timer is not None: 
                output_lines.append(f"Moves left: {self.num_moves - self.timer}, Score: {self.current_score:.0f}, Perf: {self.current_episode_cumulative_perf_score:.0f}")
            
            for line in output_lines: print(line)
            return None # String render prints to console

        elif render_mode_to_use in ["human", "rgb_array"] and self.renderer:
            if hasattr(self, 'board') and self.board.board is not None and self.timer is not None:
                # TileMatch Renderer expects the (2,H,W) board and moves left
                return self.renderer.render(self.board.board, self.num_moves - self.timer)
            else:
                # Fallback for rgb_array if state is not ready
                if render_mode_to_use == "rgb_array": 
                    dummy_rows = self.num_rows if hasattr(self, 'num_rows') else 8
                    dummy_cols = self.num_cols if hasattr(self, 'num_cols') else 8
                    return np.zeros((dummy_rows * 32, dummy_cols * 32, 3), dtype=np.uint8)
                return None 
        return None

    # get_current_episode_step_num is handled by adapter's attributes
    # _create_agent_observation_path is handled by adapter