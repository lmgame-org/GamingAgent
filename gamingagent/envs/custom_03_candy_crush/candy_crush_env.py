import gymnasium as gym
import numpy as np
import os
import json
import re
from typing import Any, Dict, Tuple, Optional, List, Union
from PIL import Image, ImageDraw, ImageFont
import hashlib
import imageio
import tempfile
import shutil
from collections import OrderedDict

from gamingagent.modules.core_module import Observation
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
    img_height = rows * tile_size
    img = Image.new('RGB', (img_width, img_height + (tile_size if moves_left is not None or perf_score is not None or action_taken_str is not None else 0)), (200, 200, 200))
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

    # Display moves_left, perf_score, action_taken_str at the bottom if provided
    info_y_start = img_height + 5
    font_size_info = max(12, tile_size // 2)
    try: font_info = ImageFont.truetype("arial.ttf", font_size_info)
    except IOError: 
        try: font_info = ImageFont.load_default(size=font_size_info)
        except AttributeError: font_info = ImageFont.load_default()

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

    if action_taken_str is not None:
        text_content = f"Action: {action_taken_str}"
        draw.text((5, current_y), text_content, fill=(0,0,0), font=font_info)

    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir: os.makedirs(save_dir, exist_ok=True)
        img.save(save_path)

class CandyCrushEnvWrapper(gym.Env):
    metadata = {"render_modes": ["string", "human", "rgb_array"], "render_fps": 2}

    def __init__(self, game_name: str, observation_mode: str, agent_observations_base_dir: str, 
                 env_type: str, config_root_dir: str = "configs", log_root_dir: str = "runs_output"):
        super().__init__()

        self.game_name = game_name
        self.observation_mode = observation_mode
        self.agent_observations_base_dir = agent_observations_base_dir
        self.config_root_dir = config_root_dir
        self.log_root_dir = log_root_dir

        self._game_env_config: Dict[str, Any] = {}
        self._load_candy_crush_config()
        
        # --- Start of TileMatchEnv __init__ logic integration ---
        env_init_kwargs = self._game_env_config.get("env_init_kwargs", {})
        self.num_rows = env_init_kwargs.get("num_rows", 8)
        self.num_cols = env_init_kwargs.get("num_cols", 8)
        self.num_colours = env_init_kwargs.get("num_colours", 4)
        self.num_moves = env_init_kwargs.get("num_moves", 50)
        self.colourless_specials = env_init_kwargs.get("colourless_specials", [])
        self.colour_specials = env_init_kwargs.get("colour_specials", [])
        self.seed_val = env_init_kwargs.get("seed")

        # Renderer setup (adapted from TileMatchEnv)
        self.renderer: Optional[Renderer] = None
        self.internal_render_mode = self._game_env_config.get("render_mode_for_make", "human")
        if self.internal_render_mode == "string":
            pass
        elif self.internal_render_mode in ["human", "rgb_array"]:
            self.renderer = Renderer(self.num_rows, self.num_cols, self.num_colours, self.num_moves, render_fps=self.metadata["render_fps"], render_mode=self.internal_render_mode)

        self.num_colour_specials = len(self.colour_specials)
        self.num_colourless_specials = len(self.colourless_specials)

        # Board and random number generator setup
        self.np_random = np.random.default_rng(seed=self.seed_val)
        self.board = Board(self.num_rows, self.num_cols, self.num_colours, self.colourless_specials, self.colour_specials, self.np_random)

        # Action space (from TileMatchEnv)
        self.num_actions = int((self.num_rows * self.num_cols * 2) - self.num_rows - self.num_cols)
        self._action_to_coords = self.board.action_to_coords
        
        self.action_space = Discrete(self.num_actions, seed=self.seed_val if self.seed_val is not None else np.random.randint(1_000_000))

        # Internal observation space (raw, from TileMatchEnv - not the agent's final observation space)
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
        self.gym_observation_space = gym.spaces.Dict({
            "board": self._gym_board_observation_space,
            "num_moves_left": self._gym_moves_left_observation_space
        })
        
        self.timer: Optional[int] = None
        self.current_score: float = 0.0
        # --- End of TileMatchEnv __init__ logic integration ---

        # Dynamically build action_mapping_config using self._action_to_coords
        self.action_mapping_config = {}
        self.move_to_action_idx = {}
        self.action_idx_to_move = {}
        if self._action_to_coords:
            for idx, coords_pair in enumerate(self._action_to_coords):
                coord1 = tuple(coords_pair[0])
                coord2 = tuple(coords_pair[1])
                c_min = min(coord1, coord2)
                c_max = max(coord1, coord2)
                action_str = f"(({c_min[0]},{c_min[1]}),({c_max[0]},{c_max[1]}))"
                self.action_mapping_config[action_str] = idx
                self.move_to_action_idx[action_str] = idx
                self.action_idx_to_move[idx] = action_str
        else:
            print("[CandyCrushEnvWrapper] WARNING: self._action_to_coords not found. Action mapping will be empty.")

        # Observation space for the agent (image-based) - this remains the same for compatibility
        board_rows_render = self.num_rows
        board_cols_render = self.num_cols
        tile_size_render = self._game_env_config.get("tile_size_for_render", 32)
        info_area_height = tile_size_render
        obs_shape = (board_rows_render * tile_size_render + info_area_height, board_cols_render * tile_size_render, 3)
        self.observation_space = Box(0, 255, shape=obs_shape, dtype=np.uint8)
        
        self._max_unchanged_steps = self._game_env_config.get("max_unchanged_steps_for_termination", 50)
        self._last_board_hash: Optional[str] = None
        self._unchanged_board_count: int = 0

        # Initialize attributes expected by create_board_image_candy_crush and managed by BaseGameEnv
        self.cumulative_perf_score: float = 0.0
        self.last_action_str: Optional[str] = None
        self.current_episode_id: int = 1
        self.current_step_num: int = 0
        
        if self.internal_render_mode == "string" and not hasattr(self, 'colour_map'):
            if not hasattr(self, 'np_random') or self.np_random is None:
                self._set_internal_seed(self.seed_val if self.seed_val is not None else np.random.randint(1_000_000))
            self.colour_map = self.np_random.choice(range(105, 230), size=self.num_colours + 1, replace=False)

        print(f"[CandyCrushEnvWrapper] Initialized (gym.Env based). Agent Obs Space Shape: {self.observation_space.shape if self.observation_space else 'N/A'}")

    def _load_candy_crush_config(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(current_dir, "game_env_config.json")
        if not os.path.exists(config_file) and self.config_root_dir and self.game_name:
            game_specific_config_dir = self.game_name
            if not self.game_name.startswith("custom_"):
                game_specific_config_dir = f"custom_03_{self.game_name}"

            config_file = os.path.join(self.config_root_dir, game_specific_config_dir, "game_env_config.json")

        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                self._game_env_config = json.load(f)
            if "action_mapping" in self._game_env_config and self._game_env_config["action_mapping"]:
                pass
            else:
                 print(f"[CandyCrushEnvWrapper] Using dynamically generated action mapping from _action_to_coords.")
        else:
            print(f"[CandyCrushEnvWrapper] WARNING: Config {config_file} not found. Using default env_init_kwargs.")
            self._game_env_config["env_init_kwargs"] = {
                "num_rows": 8, "num_cols": 8, "num_colours": 4, "num_moves": 50,
                "colourless_specials": [], "colour_specials": [], "seed": None
            }
            self._game_env_config["tile_size_for_render"] = 32
            self._game_env_config["render_mode_for_make"] = "human"
            self._game_env_config["max_unchanged_steps_for_termination"] = 50

    # --- Methods from/inspired by TileMatchEnv ---
    def _set_internal_seed(self, seed: Optional[int]) -> None:
        """Sets the seed for internal randomization aspects like action space and board."""
        actual_seed = seed if seed is not None else np.random.randint(1_000_000)
        self.np_random = np.random.default_rng(seed=actual_seed)
        if self.board:
            self.board.np_random = self.np_random
        self.action_space.seed(actual_seed)
        if self.internal_render_mode == "string":
            self.colour_map = self.np_random.choice(range(105, 230), size=self.num_colours + 1, replace=False)

    def _get_obs(self) -> Dict[str, Union[np.ndarray, int]]:
        """Generates the raw internal gym-style observation (board state and moves left)."""
        if self.timer is None:
            current_moves_left = self.num_moves
        else:
            current_moves_left = self.num_moves - self.timer
        return OrderedDict([("board", self.board.board.copy()), ("num_moves_left", current_moves_left)])

    def _get_effective_actions(self) -> List[int]:
        """Gets a list of actions that would result in a match or special activation."""
        if self.timer is None or self.timer >= self.num_moves:
            return []
        action_check = lambda a: is_move_effective(self.board.board, *self._action_to_coords[a])
        effective_actions = list(filter(action_check, range(self.num_actions)))
        return effective_actions
    # --- End of TileMatchEnv inspired methods ---

    # --- Gym Core Methods (internal versions) ---
    def _reset_gym(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict, Dict]:
        if seed is not None:
            self.seed_val = seed
            self._set_internal_seed(seed)

        self.board.generate_board()
        self.timer = 0
        self.current_score = 0.0
        
        raw_observation = self._get_obs()
        info = {
            'effective_actions': self._get_effective_actions(),
            'score': self.current_score
        }
        
        return raw_observation, info

    def _step_gym(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        if self.timer is None or self.timer >= self.num_moves:
            empty_board_obs = np.zeros((2, self.num_rows, self.num_cols), dtype=np.int32)
            raw_obs = OrderedDict([("board", empty_board_obs), ("num_moves_left", 0)])
            return raw_obs, 0.0, True, False, {"score": self.current_score, "effective_actions": []}

        coord1, coord2 = self._action_to_coords[action]
        num_eliminations, is_combination_match, num_new_specials, num_specials_activated, shuffled = self.board.move(coord1, coord2)

        self.timer += 1
        terminated = self.timer >= self.num_moves
        
        reward = float(num_eliminations)
        self.current_score += reward

        raw_observation = self._get_obs()

        info = {
            "score": self.current_score,
            "is_combination_match": is_combination_match,
            "num_new_specials": num_new_specials,
            "num_specials_activated": num_specials_activated,
            "shuffled": shuffled,
            "effective_actions": self._get_effective_actions(),
            "num_moves_left": raw_observation["num_moves_left"]
        }
        
        return raw_observation, reward, terminated, False, info

    # --- Runner-facing Methods (Compatibility Layer) ---
    def reset(self, seed: Optional[int] = None, episode_id: int = 1, options: Optional[dict] = None) -> Tuple[Observation, Dict[str, Any]]:
        self.current_episode_id = episode_id
        self.current_step_num = 0
        self.cumulative_perf_score = 0.0
        self.last_action_str = None
        
        gym_options = options if options is not None else {}
        if episode_id is not None: gym_options['episode_id'] = episode_id

        raw_obs, info = self._reset_gym(seed=seed, options=gym_options)
        
        agent_observation = self._extract_observation(raw_obs, info)
        
        self._last_board_hash = self._hash_symbolic_obs(agent_observation.symbolic_representation)
        self._unchanged_board_count = 0
        
        info['score'] = self.current_score
        return agent_observation, info

    def step(self, action_str_agent: str, thought_process: Optional[str] = None, time_taken_s: Optional[float] = None) -> Tuple[Observation, float, bool, bool, Dict[str, Any], float]:
        self.current_step_num += 1
        
        action_int: Optional[int] = None
        valid_action_provided_by_agent = False
        # Default to original agent string for logging, in case parsing fails
        processed_action_str_for_logging: str = str(action_str_agent) 

        if isinstance(action_str_agent, str):
            # Regex to find the ((r1,c1),(r2,c2)) pattern. 
            # Captures numbers for r1, c1, r2, c2. Allows for spaces around numbers and commas.
            match = re.match(r"^\s*\(\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*\)", action_str_agent.strip())
            
            if match:
                try:
                    r1, c1, r2, c2 = map(int, match.groups())
                    coord1 = (r1, c1)
                    coord2 = (r2, c2)
                    
                    # Normalize: create the canonical key form used in move_to_action_idx
                    # (e.g., sort coordinates, format as lowercase string with no internal spaces)
                    coords_pair_sorted = tuple(sorted((coord1, coord2)))
                    action_key_for_lookup = f"(({coords_pair_sorted[0][0]},{coords_pair_sorted[0][1]}),({coords_pair_sorted[1][0]},{coords_pair_sorted[1][1]}))"
                    # self.move_to_action_idx keys are already lowercase and no space by default
                    
                    action_int = self.move_to_action_idx.get(action_key_for_lookup)
                    if action_int is not None:
                        processed_action_str_for_logging = action_key_for_lookup # Log the successfully parsed and normalized action
                except ValueError: # Should not happen if regex matches digits, but as safety
                    pass # action_int remains None
            else:
                # Fallback for simple integer actions if string doesn't match coordinate pattern
                try:
                    action_int_candidate = int(action_str_agent.strip())
                    if 0 <= action_int_candidate < self.num_actions:
                         action_int = action_int_candidate
                         # For logging, get the string representation of this int action
                         processed_action_str_for_logging = self.action_idx_to_move.get(action_int, str(action_int_candidate))
                except ValueError:
                    pass # Not an integer, action_int remains None. processed_action_str_for_logging is already original.
        
        elif isinstance(action_str_agent, int): # Direct integer action
             if 0 <= action_str_agent < self.num_actions:
                 action_int = action_str_agent
                 processed_action_str_for_logging = self.action_idx_to_move.get(action_int, str(action_int))
             else: # Integer out of bounds
                 processed_action_str_for_logging = f"Invalid int action: {action_str_agent}"


        if action_int is not None: # This implies 0 <= action_int < num_actions from previous checks
            valid_action_provided_by_agent = True

        if valid_action_provided_by_agent:
            raw_obs_for_step, reward, terminated, truncated, info_from_gym_step = self._step_gym(action_int) # type: ignore
            self.last_action_str = processed_action_str_for_logging 
            info_from_gym_step['executed_action_int_for_replay'] = action_int
        else:
            # Invalid action from agent: treat as a "null" move that consumes a turn
            print(f"[CandyCrushEnvWrapper] Warning: Invalid action '{processed_action_str_for_logging}' provided by agent. Treating as a null move.")
            self.timer = (self.timer if self.timer is not None else -1) + 1 
            reward = 0.0
            terminated = self.timer >= self.num_moves if self.timer is not None else False
            truncated = False 
            raw_obs_for_step = self._get_obs() 
            info_from_gym_step = {
                "score": self.current_score, 
                "is_combination_match": False, "num_new_specials": 0,
                "num_specials_activated": 0, "shuffled": False,
                "effective_actions": self._get_effective_actions(), 
                "num_moves_left": raw_obs_for_step["num_moves_left"],
                "invalid_action_taken": True, "original_agent_action": action_str_agent, # Log original raw string here
                'executed_action_int_for_replay': None
            }
            self.last_action_str = f"Invalid: {processed_action_str_for_logging[:50]}" 
        
        info_from_gym_step['raw_env_observation_for_replay'] = raw_obs_for_step
        current_step_perf_score = self.perf_score(reward, info_from_gym_step)
        self.cumulative_perf_score += current_step_perf_score
        agent_observation = self._extract_observation(raw_obs_for_step, info_from_gym_step)
        terminated, truncated = self._verify_termination(agent_observation, terminated, truncated)
        return agent_observation, reward, terminated, truncated, info_from_gym_step, current_step_perf_score

    def _extract_observation(self, raw_observation: Dict[str, Any], info: Dict[str, Any]) -> Observation:
        img_path_for_agent, text_representation_for_agent = None, None
        board_data_from_obs, num_moves_left_from_obs = None, 0
        if isinstance(raw_observation, dict):
            board_data_from_obs = raw_observation.get("board")
            num_moves_left_from_obs = raw_observation.get("num_moves_left", 0)
        current_score_display = info.get("score", self.current_score) 
        num_moves_left_display = info.get("num_moves_left", num_moves_left_from_obs)
        color_board = None
        if isinstance(board_data_from_obs, np.ndarray) and board_data_from_obs.ndim == 3 and board_data_from_obs.shape[0] >= 1: color_board = board_data_from_obs[0]
        elif isinstance(board_data_from_obs, np.ndarray) and board_data_from_obs.ndim == 2: color_board = board_data_from_obs 
        if self.observation_mode in ["vision", "both"] and color_board is not None:
            img_path_for_agent = self._create_agent_observation_path(self.current_episode_id, self.current_step_num)
            create_board_image_candy_crush(color_board, img_path_for_agent, tile_size=self._game_env_config.get("tile_size_for_render", 32), perf_score=self.cumulative_perf_score, action_taken_str=self.last_action_str, moves_left=num_moves_left_display)
        if self.observation_mode in ["text", "both"]:
            text_parts = []
            if color_board is not None:
                text_parts.append("Board:")
                for r_idx, row_val in enumerate(color_board): text_parts.append(f"{r_idx}| {' '.join([COLOR_MAP.get(int(tile), '?') for tile in row_val])}")
            else: text_parts.append("Board: [Data not available]")
            text_parts.extend([f"Score: {current_score_display}", f"Moves Left: {num_moves_left_display}"])
            if self.last_action_str: text_parts.append(f"Last Action: {self.last_action_str}")
            text_representation_for_agent = "\\\\n".join(text_parts)
        return Observation(img_path=img_path_for_agent, text_data=None, symbolic_representation=text_representation_for_agent)

    def _hash_symbolic_obs(self, symbolic_representation: Optional[str]) -> Optional[str]:
        if not symbolic_representation: return None
        board_lines, is_board_section = [], False
        for line in symbolic_representation.split('\\\\n'):
            if line.startswith("Board:"): is_board_section = True; continue
            if is_board_section and (line.startswith("Score:") or line.startswith("Moves Left:")): is_board_section = False; break
            if is_board_section and line.strip(): board_lines.append(line)
        board_hash_content = "".join(board_lines)
        return hashlib.md5(board_hash_content.encode()).hexdigest() if board_hash_content else None

    def _verify_termination(self, observation: Observation, current_terminated: bool, current_truncated: bool) -> Tuple[bool, bool]:
        if current_terminated or current_truncated: 
            return current_terminated, current_truncated

        current_obs_hash = self._hash_symbolic_obs(observation.symbolic_representation)

        if self._last_board_hash == current_obs_hash and current_obs_hash is not None:
            self._unchanged_board_count += 1
        else:
            self._unchanged_board_count = 0
        self._last_board_hash = current_obs_hash

        if self._unchanged_board_count >= self._max_unchanged_steps:
            print(f"[CandyCrushEnvWrapper] Terminating: board state unchanged for {self._max_unchanged_steps} steps.")
            return True, current_truncated
        return current_terminated, current_truncated

    def game_replay(self, replay_data: Dict[str, Any], output_video_path: Optional[str] = None, frame_duration: float = 0.5) -> None:
        if not replay_data or not replay_data.get("raw_observations"):
            print("[CandyCrushEnvWrapper] No trajectory data (raw_observations) for replay.")
            return
        trajectory_steps_data = []
        num_steps = len(replay_data["raw_observations"])
        for i in range(num_steps):
            step_info = {"raw_env_observation": replay_data["raw_observations"][i], "agent_action": replay_data["agent_actions_str"][i] if i < len(replay_data.get("agent_actions_str", [])) else "N/A", "raw_env_info": replay_data["infos"][i] if i < len(replay_data.get("infos", [])) else {}}
            trajectory_steps_data.append(step_info)
        replay_log_dir = self.log_root_dir if self.log_root_dir and os.path.isdir(self.log_root_dir) else tempfile.mkdtemp()
        if output_video_path is None:
            ep_id_str = f"_e{self.current_episode_id:03d}" if hasattr(self, 'current_episode_id') and self.current_episode_id is not None else ""
            output_video_path = os.path.join(replay_log_dir, f"candy_crush_replay{ep_id_str}.gif")
        temp_frames_dir, frame_files = tempfile.mkdtemp(), []
        print(f"[CandyCrushEnvWrapper] Generating frames for replay from logged states in {temp_frames_dir}...")
        for idx, step_data_from_log in enumerate(trajectory_steps_data):
            raw_obs_from_log, agent_action, info_for_re_render, score_for_display = step_data_from_log.get("raw_env_observation"), step_data_from_log.get("agent_action", "N/A"), step_data_from_log.get("raw_env_info", {}), info_for_re_render.get("score")
            board_array_3d, moves_left_replay = None, None
            if isinstance(raw_obs_from_log, dict):
                board_array_3d, moves_left_replay = raw_obs_from_log.get("board"), raw_obs_from_log.get("num_moves_left")
            color_board_replay = None
            if isinstance(board_array_3d, np.ndarray) and board_array_3d.ndim == 3 and board_array_3d.shape[0] >=1: color_board_replay = board_array_3d[0]
            elif isinstance(board_array_3d, np.ndarray) and board_array_3d.ndim == 2: color_board_replay = board_array_3d
            else: print(f"[CandyCrushEnvWrapper] Warning: raw_env_observation for replay step {idx} not in expected dict format or board is missing. Skipping frame."); continue
            frame_path = os.path.join(temp_frames_dir, f"frame_{idx:04d}.png")
            try:
                create_board_image_candy_crush(color_board_replay, frame_path, tile_size=self._game_env_config.get("tile_size_for_render", 32), perf_score=score_for_display, action_taken_str=str(agent_action), moves_left=moves_left_replay)
                frame_files.append(frame_path)
            except Exception as e: print(f"[CandyCrushEnvWrapper] Error creating board image for replay step {idx}: {e}. Skipping frame."); continue
        if not frame_files: print("[CandyCrushEnvWrapper] No frames generated for replay.")
        else:
            print(f"[CandyCrushEnvWrapper] Compiling {len(frame_files)} frames to {output_video_path}")
            try:
                images_data = [imageio.imread(f) for f in frame_files]
                imageio.mimsave(output_video_path, images_data, duration=frame_duration, subrectangles=True, palettesize=256)
                print(f"[CandyCrushEnvWrapper] Replay video saved to {output_video_path}")
            except Exception: 
                try: imageio.mimsave(output_video_path, images_data, duration=frame_duration); print(f"[CandyCrushEnvWrapper] Replay video saved to {output_video_path} (default GIF writer).")
                except Exception as e2: print(f"[CandyCrushEnvWrapper] Error creating GIF (default writer failed): {e2}")
        try: shutil.rmtree(temp_frames_dir)
        except Exception as e: print(f"[CandyCrushEnvWrapper] Error cleaning temp frame dir {temp_frames_dir}: {e}")
        if replay_log_dir != self.log_root_dir and os.path.isdir(replay_log_dir) and not os.listdir(replay_log_dir):
             try: shutil.rmtree(replay_log_dir); print(f"[CandyCrushEnvWrapper] Cleaned up temporary replay directory: {replay_log_dir}")
             except Exception as e: print(f"[CandyCrushEnvWrapper] Error cleaning temp replay_log_dir {replay_log_dir}: {e}")

    def replay_from_seed_and_actions(self, initial_seed: int, executed_action_ints: List[Optional[int]], output_video_path: Optional[str] = None, frame_duration: float = 0.5, tile_size_for_replay_render: int = 32) -> None:
        print(f"[CandyCrushEnvWrapper] Starting replay from seed {initial_seed} and {len(executed_action_ints)} actions.")
        self.reset(seed=initial_seed, episode_id=9999)
        temp_frames_dir, frame_files = tempfile.mkdtemp(), []
        current_raw_obs, initial_board_color, initial_moves_left, initial_score = self._get_obs(), self.get_board_state(current_raw_obs, {}), current_raw_obs.get("num_moves_left", self.num_moves), self.current_score
        if initial_board_color is not None:
            frame_path = os.path.join(temp_frames_dir, f"frame_{-1:04d}.png")
            create_board_image_candy_crush(initial_board_color, frame_path, tile_size=tile_size_for_replay_render, moves_left=initial_moves_left, perf_score=initial_score, action_taken_str="Initial State")
            frame_files.append(frame_path)
        for idx, action_int in enumerate(executed_action_ints):
            if action_int is None:
                print(f"Replay step {idx}: Null move (original action was invalid).")
                self.timer = (self.timer if self.timer is not None else -1) + 1
                if self.timer >= self.num_moves: print("Replay: Out of moves after a null action."); break 
                current_raw_obs, reward_for_step, info_for_step = self._get_obs(), 0.0, {"score": self.current_score, "num_moves_left": current_raw_obs.get("num_moves_left", 0)}
            elif 0 <= action_int < self.num_actions:
                current_raw_obs, reward_for_step, terminated, truncated, info_for_step = self._step_gym(action_int)
                print(f"Replay step {idx}: Action {action_int}, Reward {reward_for_step}, Score {info_for_step.get('score')}, Moves Left {info_for_step.get('num_moves_left')}")
                if terminated or truncated:
                    print(f"Replay terminated/truncated at step {idx}.")
                    final_color_board = self.get_board_state(current_raw_obs, info_for_step)
                    if final_color_board is not None:
                        frame_path = os.path.join(temp_frames_dir, f"frame_{idx:04d}.png")
                        create_board_image_candy_crush(final_color_board, frame_path, tile_size=tile_size_for_replay_render, moves_left=info_for_step.get("num_moves_left"), perf_score=info_for_step.get("score"), action_taken_str=self.action_idx_to_move.get(action_int, str(action_int)))
                        frame_files.append(frame_path)
                    break
            else: print(f"Replay step {idx}: Invalid action_int {action_int} in log. Skipping."); continue
            color_board_to_render = self.get_board_state(current_raw_obs, info_for_step)
            if color_board_to_render is not None:
                frame_path = os.path.join(temp_frames_dir, f"frame_{idx:04d}.png")
                action_str_for_display = "Null Move" if action_int is None else self.action_idx_to_move.get(action_int, str(action_int))
                create_board_image_candy_crush(color_board_to_render, frame_path, tile_size=tile_size_for_replay_render, moves_left=info_for_step.get("num_moves_left"), perf_score=info_for_step.get("score"), action_taken_str=action_str_for_display)
                frame_files.append(frame_path)
        replay_log_dir = self.log_root_dir if self.log_root_dir and os.path.isdir(self.log_root_dir) else tempfile.mkdtemp()
        if output_video_path is None: output_video_path = os.path.join(replay_log_dir, f"candy_crush_replay_S{initial_seed}.gif")
        if frame_files:
            print(f"[CandyCrushEnvWrapper] Compiling {len(frame_files)} frames for seed-action replay to {output_video_path}")
            try:
                images_data = [imageio.imread(f) for f in frame_files]
                imageio.mimsave(output_video_path, images_data, duration=frame_duration, subrectangles=True, palettesize=256)
                print(f"[CandyCrushEnvWrapper] Seed-action replay video saved to {output_video_path}")
            except Exception: 
                try: imageio.mimsave(output_video_path, images_data, duration=frame_duration); print(f"[CandyCrushEnvWrapper] Seed-action replay video saved to {output_video_path} (default GIF writer).")
                except Exception as e2: print(f"[CandyCrushEnvWrapper] Seed-action replay GIF creation failed: {e2}")
        else: print("[CandyCrushEnvWrapper] No frames generated for seed-action replay.")
        try: shutil.rmtree(temp_frames_dir)
        except Exception: pass
        if replay_log_dir != self.log_root_dir and os.path.isdir(replay_log_dir) and not os.listdir(replay_log_dir):
             try: shutil.rmtree(replay_log_dir); print(f"[CandyCrushEnvWrapper] Cleaned up temporary replay directory for seed-action replay: {replay_log_dir}")
             except Exception: pass

    def perf_score(self, reward: float, info: Dict[str, Any]) -> float:
        return reward

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def get_board_state(self, raw_observation: Any, info: Dict[str, Any]) -> Optional[np.ndarray]:
        board_array_3d = None
        if isinstance(raw_observation, dict) and "board" in raw_observation:
            board_array_3d = raw_observation.get("board")
        elif isinstance(raw_observation, np.ndarray) and raw_observation.ndim ==3:
             board_array_3d = raw_observation

        if isinstance(board_array_3d, np.ndarray) and board_array_3d.ndim == 3 and board_array_3d.shape[0] >=1:
            return board_array_3d[0]
        elif isinstance(board_array_3d, np.ndarray) and board_array_3d.ndim == 2:
            return board_array_3d
        return None

    def map_env_action_to_agent_action(self, env_action_idx: int) -> str:
        if self.action_idx_to_move and env_action_idx in self.action_idx_to_move:
            return self.action_idx_to_move[env_action_idx]
        if self._action_to_coords and env_action_idx < len(self._action_to_coords):
            coords_pair, coord1_tuple, coord2_tuple = self._action_to_coords[env_action_idx], tuple(coords_pair[0]), tuple(coords_pair[1])
            return f"(({coord1_tuple[0]},{coord1_tuple[1]}),({coord2_tuple[0]},{coord2_tuple[1]}))"
        return f"action_index_{env_action_idx}"

    def render_human(self) -> None:
        effective_render_mode = self.internal_render_mode
        if effective_render_mode == "human" and self.renderer:
             self.render(mode="human")
        elif effective_render_mode == "string":
             self.render(mode="string")
        else:
            output = self.render()
            if isinstance(output, np.ndarray) and effective_render_mode == "rgb_array":
                 pass

    def render(self, mode: Optional[str] = None) -> Union[None, np.ndarray]:
        render_mode_to_use = mode if mode is not None else self.internal_render_mode

        if render_mode_to_use == "string":
            if not (hasattr(self, 'board') and self.board and self.board.board is not None):
                print("[CandyCrushEnvWrapper] Board not available for rendering.")
                return None
            if not hasattr(self, 'colour_map') or self.colour_map is None:
                if not hasattr(self, 'np_random') or self.np_random is None: 
                    self._set_internal_seed(self.seed_val if self.seed_val is not None else 123)
                self.colour_map = self.np_random.choice(range(105, 230), size=self.num_colours + 1, replace=False)

            color_fn = lambda id_val, char_val: "\033[48;5;16m" + f"\033[38;5;{self.colour_map[id_val]}m{char_val}\033[0m"
            height, width, output_lines = self.num_rows, self.num_cols, [" " + "-" * (width * 2 + 1)]
            for r_num in range(height):
                line_str = ["| "]
                for c_col in range(width):
                    tile_colour_idx = self.board.board[0, r_num, c_col]
                    display_char = COLOR_MAP.get(int(tile_colour_idx), '?')
                    colored_char = f" {display_char} " if not (0 <= tile_colour_idx < len(self.colour_map)) else color_fn(tile_colour_idx, display_char)
                    line_str.extend([colored_char, "\033[48;5;16m ", "\033[0m"])
                line_str.append("|")
                output_lines.append("".join(line_str))
            output_lines.append(" " + "-" * (width * 2 + 1))
            if self.timer is not None: 
                output_lines.append(f"Moves left: {self.num_moves - self.timer}, Score: {self.current_score}")
            
            for line in output_lines: print(line)
            return None

        elif render_mode_to_use in ["human", "rgb_array"] and self.renderer:
            if hasattr(self, 'board') and self.board.board is not None and self.timer is not None:
                return self.renderer.render(self.board.board, self.num_moves - self.timer)
            else:
                if render_mode_to_use == "rgb_array": return np.zeros((100,100,3), dtype=np.uint8)
                return None 
        return None

    def get_current_episode_step_num(self) -> Tuple[int, int]:
        return self.current_episode_id, self.current_step_num

    def _create_agent_observation_path(self, episode_num: int, step_num: int) -> str:
        if not os.path.exists(self.agent_observations_base_dir):
            try:
                os.makedirs(self.agent_observations_base_dir, exist_ok=True)
            except OSError as e:
                print(f"Error creating agent_observations_base_dir {self.agent_observations_base_dir}: {e}")
                temp_obs_dir = tempfile.mkdtemp(prefix="gagent_obs_")
                print(f"Using temporary directory for observation: {temp_obs_dir}")
                return os.path.join(temp_obs_dir, f"obs_e{episode_num:04d}_s{step_num:05d}.png")

        episode_dir = os.path.join(self.agent_observations_base_dir, f"ep_{episode_num:04d}")
        if not os.path.exists(episode_dir):
            try:
                os.makedirs(episode_dir, exist_ok=True)
            except OSError as e:
                print(f"Error creating episode directory {episode_dir}: {e}")
                return os.path.join(self.agent_observations_base_dir, f"obs_e{episode_num:04d}_s{step_num:05d}.png")
            
        return os.path.join(episode_dir, f"obs_e{episode_num:04d}_s{step_num:05d}.png")