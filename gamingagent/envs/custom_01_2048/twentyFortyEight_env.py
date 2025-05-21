import gymnasium as gym
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
from typing import Any, Dict, Tuple, Callable
import json

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
        print("Default font not found for 2048 image. Using a basic PIL font.")
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
        print(f"Error saving 2048 board image to {save_path}: {e}")


class TwentyFortyEightEnvWrapper(BaseGameEnv):
    """Specific environment wrapper for the 2048 game."""

    def __init__(self, 
                 game_name: str, 
                 observation_mode: str, 
                 agent_obs_path_creator: Callable[[int, int], str], 
                 env_type: str, 
                 config_root_dir: str = "configs"):
        super().__init__(game_name, observation_mode, agent_obs_path_creator, env_type, config_root_dir)

        # Default render_mode from main config (self.config is loaded by super().__init__())
        default_render_mode = self.config.get("render_mode_gym_make", "human")
        self.render_mode_for_gym_make = default_render_mode # Initialize with default from main config

        current_dir = os.path.dirname(os.path.abspath(__file__))
        game_specific_config_path = os.path.join(current_dir, "game_env_config.json")

        if os.path.exists(game_specific_config_path):
            try:
                with open(game_specific_config_path, 'r') as f:
                    game_specific_config = json.load(f)

                self.env_id = game_specific_config.get("env_id", self.env_id)
                self.env_init_kwargs = game_specific_config.get("env_init_kwargs", self.env_init_kwargs)
                
                # Handle action_mapping override from game_env_config.json
                if "action_mapping" in game_specific_config:
                    override_action_map_config = game_specific_config["action_mapping"]
                    if isinstance(override_action_map_config, dict):
                        # Repopulate move_to_action_idx and action_idx_to_move using the override
                        # Ensure keys are lowercase strings for move_to_action_idx
                        # Also ensure action indices (values in map) are integers.
                        try:
                            self.move_to_action_idx = {str(k).lower(): int(v) for k, v in override_action_map_config.items()}
                            self.action_idx_to_move = {int(v): str(k).lower() for k, v in override_action_map_config.items()}
                            # self.action_mapping_config can also be updated if needed for consistency
                            self.action_mapping_config = override_action_map_config 
                            print(f"Successfully applied action_mapping override from {game_specific_config_path}")
                        except ValueError:
                            print(f"Warning: Values in action_mapping from {game_specific_config_path} must be integers. Using action_mapping from main config (if any).")
                    else:
                        print(f"Warning: action_mapping in {game_specific_config_path} is not a dictionary. Using action_mapping from main config (if any).")

                # Override render_mode if present in JSON, otherwise it stays as default_render_mode
                self.render_mode_for_gym_make = game_specific_config.get("render_mode_gym_make", default_render_mode)
                
                print(f"Successfully loaded and applied game-specific parameters from {game_specific_config_path}")

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from {game_specific_config_path}: {e}. "
                      f"Using values from main config and default render_mode ('{default_render_mode}').")
            except Exception as e:
                print(f"Error loading or applying game-specific config from {game_specific_config_path}: {e}. "
                      f"Using values from main config and default render_mode ('{default_render_mode}').")
        else:
            print(f"Game-specific config {game_specific_config_path} not found. "
                  f"Using values from main config and default render_mode ('{self.render_mode_for_gym_make}').")


    def _initialize_env(self) -> None:
        """Initializes the 2048 Gymnasium environment."""
        if not self.env_id:
            raise ValueError("'env_id' is not set. It must be provided either in the main config.yaml "
                             "or in the game-specific game_env_config.json for 2048.")

        if self.env_type != "custom":
            raise ValueError("2048 environment wrapper currently only supports 'custom' env_type (which internally uses Gymnasium).")
        
        # Use the render_mode loaded in __init__ (from game_env_config.json or default from main config)
        render_mode_for_make = self.render_mode_for_gym_make

        print(f"Initializing 2048 environment ({self.env_id}) with render_mode='{render_mode_for_make}' and kwargs={self.env_init_kwargs}")
        self.env = gym.make(self.env_id, render_mode=render_mode_for_make, **self.env_init_kwargs)
        print("2048 environment initialized.")

    def get_board_state(self, raw_observation: Any, info: Dict[str, Any]) -> np.ndarray:
        """Extracts the board state (powers of 2) from the info dictionary."""
        board = info.get('board')
        if board is None:
            print("Warning: 'board' not found in info dict for 2048. Using raw observation.")
            # Assuming raw_observation is the board if info['board'] is missing.
            # This might need adjustment based on actual env behavior if info['board'] fails.
            if isinstance(raw_observation, np.ndarray) and raw_observation.shape == (4,4):
                return raw_observation
            else:
                # Fallback: return an empty board to prevent crashes, though this is not ideal.
                print("Error: Could not determine 2048 board state. Returning empty board.")
                return np.zeros((4,4), dtype=int)
        return np.array(board, dtype=int) # Ensure it's a numpy array of ints

    def extract_observation(self, raw_observation: Any, info: Dict[str, Any]) -> Observation:
        """Creates an Observation object for the agent from the 2048 game state."""
        board_state_powers = self.get_board_state(raw_observation, info)
        
        img_path_for_agent = None
        text_representation_for_agent = None

        current_ep, current_st = self.get_current_episode_step_num()

        if self.observation_mode in ["vision", "both"]:
            img_path_for_agent = self.agent_obs_path_creator(current_ep, current_st)
            create_board_image_2048(board_state_powers, img_path_for_agent)
        
        if self.observation_mode in ["text", "both"]:
            text_representation_for_agent = str(board_state_powers.tolist())
            
        return Observation(
            img_path=img_path_for_agent, 
            symbolic_representation=text_representation_for_agent
        )
