import gymnasium as gym
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
from typing import Any, Dict, Tuple, Callable, Optional
import json
import hashlib # Added for hashing image data

from gamingagent.envs.base_env import BaseGameEnv
from gamingagent.modules.core_module import Observation


def create_board_image_2048(board_powers: np.ndarray, save_path: str, size: int = 400) -> None:
    """
    Create a visualization of the 2048 board from board powers.
    Args:
        board_powers: Numpy array (4x4) of tile powers (0 for empty).
        save_path: Path to save the image.
        size: Image size in pixels.
    """
    cell_size = size // 4
    padding = cell_size // 10
    img = Image.new('RGB', (size, size), (250, 248, 239))  # Beige background
    draw = ImageDraw.Draw(img)

    colors = {
        0: (205, 193, 180), 2: (238, 228, 218), 4: (237, 224, 200),
        8: (242, 177, 121), 16: (245, 149, 99), 32: (246, 124, 95),
        64: (246, 94, 59), 128: (237, 207, 114), 256: (237, 204, 97),
        512: (237, 200, 80), 1024: (237, 197, 63), 2048: (237, 194, 46),
    }
    dark_text_color = (119, 110, 101)
    light_text_color = (249, 246, 242)

    try:
        font = ImageFont.load_default()
    except IOError:
        print("[TwentyFortyEightEnvWrapper] Default font not found for 2048 image. Using a basic PIL font.")
        font = ImageFont.truetype("arial.ttf", 15) if os.path.exists("arial.ttf") else ImageFont.load_default()

    for r_idx in range(4):
        for c_idx in range(4):
            power = int(board_powers[r_idx, c_idx])
            value = 0 if power == 0 else 2 ** power

            x0, y0 = c_idx * cell_size + padding, r_idx * cell_size + padding
            x1, y1 = (c_idx + 1) * cell_size - padding, (r_idx + 1) * cell_size - padding

            cell_color = colors.get(value, (60, 58, 50))  # Default dark for very high values
            draw.rectangle([x0, y0, x1, y1], fill=cell_color)

            if value > 0:
                text = str(value)
                text_color = light_text_color if value > 4 else dark_text_color
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                text_x = x0 + (cell_size - (2 * padding) - text_width) / 2
                text_y = y0 + (cell_size - (2 * padding) - text_height) / 2
                draw.text((text_x, text_y), text, fill=text_color, font=font)
    try:
        img.save(save_path)
    except Exception as e:
        print(f"[TwentyFortyEightEnvWrapper] Error saving 2048 board image to {save_path}: {e}")


class TwentyFortyEightEnvWrapper(BaseGameEnv):
    """Specific environment wrapper for the 2048 game."""

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
                         log_root_dir)

        # Default render_mode from main config (self.config is loaded by super().__init__())
        default_render_mode = self.config.get("render_mode_gym_make", "human")
        self.render_mode_for_gym_make = default_render_mode # Initialize with default from main config

        # For stuck detection
        self._last_observation_hash: Optional[str] = None # Changed from _last_observation_str
        self._unchanged_obs_count: int = 0
        # Default, will be overridden if found in game_specific_config
        self._max_unchanged_steps: int = 10 

        # Load game-specific config from JSON if it exists
        current_dir = os.path.dirname(os.path.abspath(__file__))
        game_specific_config_path = os.path.join(current_dir, "game_env_config.json")
        game_specific_config = {} # Initialize to empty dict

        if os.path.exists(game_specific_config_path):
            try:
                with open(game_specific_config_path, 'r') as f:
                    game_specific_config = json.load(f)

                self.env_id = game_specific_config.get("env_id", self.env_id)
                self.env_init_kwargs = game_specific_config.get("env_init_kwargs", self.env_init_kwargs)
                
                if "action_mapping" in game_specific_config:
                    override_action_map_config = game_specific_config["action_mapping"]
                    if isinstance(override_action_map_config, dict):
                        try:
                            self.move_to_action_idx = {str(k).lower(): int(v) for k, v in override_action_map_config.items()}
                            self.action_idx_to_move = {int(v): str(k).lower() for k, v in override_action_map_config.items()}
                            self.action_mapping_config = override_action_map_config 
                            print(f"[TwentyFortyEightEnvWrapper] Successfully applied action_mapping override from {game_specific_config_path}")
                        except ValueError:
                            print(f"[TwentyFortyEightEnvWrapper] Warning: Values in action_mapping from {game_specific_config_path} must be integers. Using action_mapping from main config (if any).")
                    else:
                        print(f"[TwentyFortyEightEnvWrapper] Warning: action_mapping in {game_specific_config_path} is not a dictionary. Using action_mapping from main config (if any).")

                self.render_mode_for_gym_make = game_specific_config.get("render_mode_gym_make", default_render_mode)
                
                # Load max_unchanged_steps for termination from config
                loaded_max_unchanged = game_specific_config.get("max_unchanged_steps_for_termination")
                if isinstance(loaded_max_unchanged, int) and loaded_max_unchanged > 0:
                    self._max_unchanged_steps = loaded_max_unchanged
                    print(f"[TwentyFortyEightEnvWrapper] Loaded max_unchanged_steps_for_termination: {self._max_unchanged_steps} from config.")
                else:
                    print(f"[TwentyFortyEightEnvWrapper] Using default max_unchanged_steps_for_termination: {self._max_unchanged_steps}. Value in config was missing or invalid: '{loaded_max_unchanged}'.")

                print(f"[TwentyFortyEightEnvWrapper] Successfully loaded and applied game-specific parameters from {game_specific_config_path}")

            except json.JSONDecodeError as e:
                print(f"[TwentyFortyEightEnvWrapper] Error decoding JSON from {game_specific_config_path}: {e}. "
                      f"Using values from main config, default render_mode ('{default_render_mode}'), and default max_unchanged_steps ({self._max_unchanged_steps}).")
            except Exception as e:
                print(f"[TwentyFortyEightEnvWrapper] Error loading or applying game-specific config from {game_specific_config_path}: {e}. "
                      f"Using values from main config, default render_mode ('{default_render_mode}'), and default max_unchanged_steps ({self._max_unchanged_steps}).")
        else:
            print(f"[TwentyFortyEightEnvWrapper] Game-specific config {game_specific_config_path} not found. "
                  f"Using values from main config, default render_mode ('{self.render_mode_for_gym_make}'), and default max_unchanged_steps ({self._max_unchanged_steps}).")

        # Ensure the render_mode_for_make in BaseGameEnv is updated with the one determined here
        self.render_mode_for_make = self.render_mode_for_gym_make
        print(f"[TwentyFortyEightEnvWrapper] Set BaseGameEnv.render_mode_for_make to: {self.render_mode_for_make}")

    def reset(self, seed: Optional[int] = None, episode_id: int = 1) -> Tuple[Observation, Dict[str, Any]]:
        # Reset stuck detection counters
        self._last_observation_hash = None # Changed from _last_observation_str
        self._unchanged_obs_count = 0
        return super().reset(seed, episode_id)

    def get_board_state(self, raw_observation: Any, info: Dict[str, Any]) -> np.ndarray:
        """Extracts the board state (powers of 2) from the info dictionary for 2048.
           This is a helper method specific to this environment wrapper.
        """
        board = info.get('board')
        if board is None:
            print("[TwentyFortyEightEnvWrapper] Warning: 'board' not found in info dict for 2048. Using raw observation.")
            if isinstance(raw_observation, np.ndarray) and raw_observation.shape == (4,4):
                return raw_observation
            else:
                print("[TwentyFortyEightEnvWrapper] Error: Could not determine 2048 board state. Returning empty board.")
                return np.zeros((4,4), dtype=int)
        return np.array(board, dtype=int)

    def extract_observation(self, raw_observation: Any, info: Dict[str, Any]) -> Observation:
        """Creates an Observation object for the agent from the 2048 game state."""
        board_state_powers = self.get_board_state(raw_observation, info)
        
        img_path_for_agent = None
        text_representation_for_agent = None

        current_ep, current_st = self.get_current_episode_step_num()

        if self.observation_mode in ["vision", "both"]:
            img_path_for_agent = self._create_agent_observation_path(current_ep, current_st)
            create_board_image_2048(board_state_powers, img_path_for_agent)
        
        if self.observation_mode in ["text", "both"]:
            text_representation_for_agent = str(board_state_powers.tolist())
            
        return Observation(
            img_path=img_path_for_agent, 
            symbolic_representation=text_representation_for_agent
        )

    def verify_termination(self, observation: Observation, current_terminated: bool, current_truncated: bool) -> Tuple[bool, bool]:
        """Overrides BaseGameEnv.verify_termination to detect stuck states in 2048.
           If the board state (symbolic observation or image hash) remains unchanged for `_max_unchanged_steps`,
           the episode is considered terminated.
        """
        print(f"[TwentyFortyEightEnvWrapper] verify_termination called with current_terminated: {current_terminated}, current_truncated: {current_truncated}, _max_unchanged_steps: {self._max_unchanged_steps}, _unchanged_obs_count: {self._unchanged_obs_count}, _last_observation_hash: {self._last_observation_hash}")
        # If already terminated or truncated by the environment, honor that.
        if current_terminated or current_truncated:
            return current_terminated, current_truncated

        current_obs_hash: Optional[str] = None

        if observation.symbolic_representation:
            current_obs_hash = hashlib.md5(observation.symbolic_representation.encode()).hexdigest()
            # print(f"[TwentyFortyEightEnvWrapper] [Debug] Symbolic hash: {current_obs_hash}")
        elif observation.img_path and os.path.exists(observation.img_path):
            try:
                with Image.open(observation.img_path) as img:
                    # Convert to bytes and hash. Ensure consistent format (e.g., RGB) if necessary for comparison.
                    img_byte_arr = img.tobytes()
                    current_obs_hash = hashlib.md5(img_byte_arr).hexdigest()
                    # print(f"[TwentyFortyEightEnvWrapper] [Debug] Image hash: {current_obs_hash}")
            except Exception as e:
                print(f"[TwentyFortyEightEnvWrapper] Warning: Could not hash image {observation.img_path} for stuck detection: {e}")
                # Cannot determine hash, so cannot determine if stuck based on vision
                return current_terminated, current_truncated
        else:
            # No symbolic representation and no valid image path, cannot determine if stuck
            # print(f"[TwentyFortyEightEnvWrapper] [Debug] No symbolic_representation or valid img_path to hash.")
            return current_terminated, current_truncated

        if self._last_observation_hash == current_obs_hash:
            self._unchanged_obs_count += 1
            # print(f"[TwentyFortyEightEnvWrapper] [Debug] Unchanged obs count: {self._unchanged_obs_count}")
        else:
            self._unchanged_obs_count = 0
        
        self._last_observation_hash = current_obs_hash

        if self._unchanged_obs_count >= self._max_unchanged_steps:
            print(f"[TwentyFortyEightEnvWrapper] Terminating episode due to unchanged observation for {self._max_unchanged_steps} steps.")
            return True, current_truncated # Set terminated to True
        
        return current_terminated, current_truncated
