import gymnasium as gym
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
from typing import Any, Dict, Tuple, Callable, Optional, List
import json
import hashlib # Added for hashing image data
import imageio # Added for video generation
import tempfile # Added for temporary directory
import shutil # Added for removing temporary directory

from gamingagent.envs.base_env import BaseGameEnv
from gamingagent.modules.core_module import Observation


def create_board_image_2048(board_powers: np.ndarray, save_path: str, size: int = 400, perf_score: Optional[float] = None) -> None:
    """Create a visualization of the 2048 board, incorporating new styling and perf_score display."""
    cell_size = size // 4
    padding = cell_size // 10

    img = Image.new('RGB', (size, size), (250, 248, 239)) # Overall image background (can be overridden by grid bg)
    draw = ImageDraw.Draw(img)

    # Color mapping for different tile values (extended from user's example)
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
        4096: (60, 58, 50),      # 4096 (using a default dark for very high values)
        8192: (60, 58, 50)       # 8192 (using a default dark for very high values)
        # Add more if needed, or use a function to generate colors for higher values
    }
    
    dark_text_color = (119, 110, 101)  # For small values (2, 4)
    light_text_color = (249, 246, 242) # For large values (8+)

    # Font handling (from user's example, slightly adapted)
    font = None
    perf_score_display_font = None
    base_font_size = cell_size // 3
    perf_score_font_size = max(15, size // 25) # For the performance score display

    potential_fonts = [
        "arial.ttf", "Arial.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        # "/System/Library/Fonts/SFNSDisplay-Bold.otf",  # macOS example
        # "C:/Windows/Fonts/Arial.ttf",  # Windows example
        # "C:/Windows/Fonts/ArialBd.ttf" # Windows Bold example
    ]

    for font_name in potential_fonts:
        try:
            if font is None: # Load main font first
                font = ImageFont.truetype(font_name, base_font_size)
            if perf_score_display_font is None: # Load perf_score font
                 perf_score_display_font = ImageFont.truetype(font_name, perf_score_font_size)
            if font and perf_score_display_font: # Stop if both found
                break 
        except (OSError, IOError):
            continue
    
    if font is None: # Fallback for main font
        font = ImageFont.load_default()
        print("[TwentyFortyEightEnvWrapper] Main font not found from potential_fonts. Using PIL default.")
    if perf_score_display_font is None: # Fallback for perf_score font
        perf_score_display_font = ImageFont.load_default(size=perf_score_font_size)
        print("[TwentyFortyEightEnvWrapper] Perf score font not found from potential_fonts. Using PIL default.")

    # Draw the background grid (user's style)
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
            
            # Adjust font size based on number length (user's style)
            current_font_size = base_font_size
            if len(text_content) == 3:
                current_font_size = int(base_font_size * 0.8)
            elif len(text_content) >= 4:
                current_font_size = int(base_font_size * 0.65)
            
            # Get font with correct size (attempt to reload if size changed)
            final_font_for_tile = font
            if current_font_size != base_font_size: # If size needs adjustment
                temp_font_found = False
                for font_name in potential_fonts: # Try to load with specific size
                    try:
                        final_font_for_tile = ImageFont.truetype(font_name, current_font_size)
                        temp_font_found = True
                        break
                    except (OSError, IOError):
                        continue
                if not temp_font_found:
                    final_font_for_tile = ImageFont.load_default(size=current_font_size) # Fallback to default with size
            
            # Get text size (user's style, adapted)
            text_width, text_height = 0, 0
            try:
                if hasattr(final_font_for_tile, 'getbbox'): # Newer PIL
                    bbox = final_font_for_tile.getbbox(text_content)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                elif hasattr(final_font_for_tile, 'getsize'): # Older PIL
                    text_width, text_height = final_font_for_tile.getsize(text_content)
                else: # Fallback
                    text_width = len(text_content) * current_font_size // 2
                    text_height = current_font_size
            except Exception as e:
                 print(f"[TwentyFortyEightEnvWrapper] Error getting text size: {e}. Using fallback.")
                 text_width = len(text_content) * current_font_size // 2
                 text_height = current_font_size
            
            cell_center_x = (x0 + x1) // 2
            cell_center_y = (y0 + y1) // 2
            text_x = cell_center_x - text_width // 2
            text_y = cell_center_y - text_height // 2 - (cell_size // 20) # Minor adjustment to text_y for centering
            
            draw.text((text_x, text_y), text_content, fill=current_text_color, font=final_font_for_tile)
            if value >= 8: # Slight bolding (user's style)
                draw.text((text_x + 1, text_y), text_content, fill=current_text_color, font=final_font_for_tile)

    # Draw performance score if provided (retained from previous version)
    if perf_score is not None:
        score_text_content = f"Perf: {perf_score:.2f}"
        score_display_text_color = (10, 10, 10) # Dark color for score text
        score_pos_x = padding 
        score_pos_y = padding // 2 
        try:
            draw.text((score_pos_x, score_pos_y), score_text_content, fill=score_display_text_color, font=perf_score_display_font)
        except Exception as e:
            print(f"[TwentyFortyEightEnvWrapper] Error drawing perf_score on image: {e}")

    try:
        # Ensure the directory exists (user's style)
        save_dir = os.path.dirname(save_path)
        if save_dir: # Only make dirs if dirname is not empty (i.e. not saving to current dir)
            os.makedirs(save_dir, exist_ok=True)
        img.save(save_path)
    except Exception as e:
        print(f"[TwentyFortyEightEnvWrapper] Error saving 2048 board image to {save_path}: {e}") # Retained original more specific error message


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
            textual_representation=text_representation_for_agent
        )

    def verify_termination(self, observation: Observation, current_terminated: bool, current_truncated: bool) -> Tuple[bool, bool]:
        """Overrides BaseGameEnv.verify_termination to detect stuck states in 2048.
           If the board state (textual representation or image hash) remains unchanged for `_max_unchanged_steps`,
           the episode is considered terminated.
        """
        print(f"[TwentyFortyEightEnvWrapper] verify_termination called with current_terminated: {current_terminated}, current_truncated: {current_truncated}, _max_unchanged_steps: {self._max_unchanged_steps}, _unchanged_obs_count: {self._unchanged_obs_count}, _last_observation_hash: {self._last_observation_hash}")
        # If already terminated or truncated by the environment, honor that.
        if current_terminated or current_truncated:
            return current_terminated, current_truncated

        current_obs_hash: Optional[str] = None

        if observation.textual_representation:
            current_obs_hash = hashlib.md5(observation.textual_representation.encode()).hexdigest()
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
            # No textual representation and no valid image path, cannot determine if stuck
            # print(f"[TwentyFortyEightEnvWrapper] [Debug] No textual_representation or valid img_path to hash.")
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

    def game_replay(self, trajectory_data: List[Dict[str, Any]], perf_score_list: List[float],
                    output_video_path: str = "2048_replay.gif", 
                    frame_duration: float = 0.5) -> None:
        """
        Generates a video replay of a game trajectory.

        Args:
            trajectory_data: A list of dictionaries, where each dictionary represents a step
                             and is expected to have a "board" key containing the 4x4 
                             board state (powers of 2).
            perf_score_list: A list of performance scores, one for each step in trajectory_data.
            output_video_path: The path (including filename) to save the generated video.
            frame_duration: The duration (in seconds) each frame should be displayed in the video.
        """
        if not trajectory_data:
            print("[TwentyFortyEightEnvWrapper] No trajectory data provided for replay. Exiting game_replay.")
            return
        
        if len(trajectory_data) != len(perf_score_list):
            print(f"[TwentyFortyEightEnvWrapper] Warning: Mismatch between trajectory data length ({len(trajectory_data)}) and performance score list length ({len(perf_score_list)}). Scores will not be displayed.")
            display_scores = False
        else:
            display_scores = True

        temp_dir = tempfile.mkdtemp()
        frame_files = []
        
        print(f"[TwentyFortyEightEnvWrapper] Generating frames for replay in {temp_dir}...")

        for idx, step_data in enumerate(trajectory_data):
            board_raw = step_data.get("board")
            if board_raw is None:
                print(f"[TwentyFortyEightEnvWrapper] Warning: 'board' key not found in step {idx} of trajectory data. Skipping frame.")
                continue

            try:
                board_powers = np.array(board_raw, dtype=int)
                if board_powers.shape != (4, 4):
                    print(f"[TwentyFortyEightEnvWrapper] Warning: Board in step {idx} does not have shape (4,4). Actual shape: {board_powers.shape}. Skipping frame.")
                    continue
            except Exception as e:
                print(f"[TwentyFortyEightEnvWrapper] Warning: Could not convert board in step {idx} to numpy array: {e}. Skipping frame.")
                continue

            frame_path = os.path.join(temp_dir, f"frame_{idx:04d}.png")
            current_perf_score = perf_score_list[idx] if display_scores and idx < len(perf_score_list) else None
            
            try:
                create_board_image_2048(board_powers, frame_path, perf_score=current_perf_score)
                frame_files.append(frame_path)
            except Exception as e:
                print(f"[TwentyFortyEightEnvWrapper] Error creating board image for step {idx}: {e}. Skipping frame.")

        if not frame_files:
            print("[TwentyFortyEightEnvWrapper] No frames were generated. Cannot create video.")
        else:
            print(f"[TwentyFortyEightEnvWrapper] Compiling {len(frame_files)} frames into video: {output_video_path}")
            try:
                images_data = [imageio.imread(f) for f in frame_files]
                imageio.mimsave(output_video_path, images_data, duration=frame_duration)
                print(f"[TwentyFortyEightEnvWrapper] Replay video saved to {output_video_path}")
            except Exception as e:
                print(f"[TwentyFortyEightEnvWrapper] Error creating video: {e}")
                print(f"[TwentyFortyEightEnvWrapper] Frames are available in {temp_dir} if you want to assemble them manually.")

        # Clean up the temporary directory
        try:
            shutil.rmtree(temp_dir)
            print(f"[TwentyFortyEightEnvWrapper] Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            print(f"[TwentyFortyEightEnvWrapper] Error cleaning up temporary directory {temp_dir}: {e}")


    def perf_score(self, reward: float, info: Dict[str, Any]) -> float:
        """
        Calculates the performance score for a game episode.
        """
        return reward


# Example usage (assuming you have a TwentyFortyEightEnvWrapper instance 'env' and trajectory_data):
# trajectory_example = [
#     {"board": [[0,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,0]]}, # Board with 2s
#     {"board": [[0,0,0,0], [0,1,0,0], [0,0,2,0], [0,0,0,1]]}, # Board with 2s and a 4
#     # ... more steps
# ]
# env.game_replay(trajectory_example, "my_2048_game.gif", frame_duration=0.8)