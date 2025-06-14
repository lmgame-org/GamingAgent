# TODO: Define reward for each step - Yuxuan
import io
import logging
import pickle
from collections import deque
import heapq
from typing import Optional, Dict, Any, Tuple, List
import os
import json
from datetime import datetime
from PIL import Image, ImageDraw
from pyboy import PyBoy

from gamingagent.envs.custom_05_pokemon_red.memory_reader import PokemonRedReader, StatusCondition
from gymnasium import Env, spaces
import numpy as np

from gamingagent.envs.gym_env_adapter import GymEnvAdapter
from gamingagent.modules.core_module import Observation
from gamingagent.envs.custom_05_pokemon_red.navigation_system import NavigationSystem
from gamingagent.envs.custom_05_pokemon_red.navigation_assistant import NavigationAssistant
from gamingagent.envs.custom_05_pokemon_red.reasoning_aids import MetaCritiqueSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PokemonRedEnv(Env):
    def __init__(self, 
                 render_mode: Optional[str] = None,
                 # Pokemon Red specific params from game_env_config.json
                 rom_path: Optional[str] = None,
                 sound: bool = False,
                 max_episode_steps: int = 50000,
                 # Adapter parameters
                 game_name_for_adapter: str = "pokemon_red",
                 observation_mode_for_adapter: str = "vision",
                 agent_cache_dir_for_adapter: str = "cache/pokemon_red/default_run",
                 game_specific_config_path_for_adapter: str = "gamingagent/envs/custom_05_pokemon_red/game_env_config.json",
                 max_stuck_steps_for_adapter: Optional[int] = 20,
                 navigation_enabled: bool = False,
                 model_name: str = "default",
                 vllm_url: Optional[str] = None,
                 modal_url: Optional[str] = None,
                 enable_reasoning_aids=False,
                 runner_log_dir_base: Optional[str] = None,
                 initial_state: Optional[str] = None):
        super().__init__()
        
        # Store model configuration
        self.model_name = model_name
        self.vllm_url = vllm_url
        self.modal_url = modal_url
        
        # Store initial state path
        self.initial_state = initial_state
        
        # Initialize adapter
        self.adapter = GymEnvAdapter(
            game_name=game_name_for_adapter,
            observation_mode=observation_mode_for_adapter,
            agent_cache_dir=agent_cache_dir_for_adapter,
            game_specific_config_path=game_specific_config_path_for_adapter,
            max_steps_for_stuck=max_stuck_steps_for_adapter
        )
        
        # Gymnasium spaces
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(low=0, high=255, shape=(240, 256, 3), dtype=np.uint8)
        
        # Action mapping - ensure this matches game_env_config.json
        self.action_map = {
            "a": 0,
            "b": 1,
            "start": 2,
            "select": 3,
            "up": 4,
            "down": 5,
            "left": 6,
            "right": 7
        }
        
        # Reverse mapping for validation
        self.reverse_action_map = {v: k for k, v in self.action_map.items()}
        
        # Emulator setup
        self.rom_path = rom_path
        self.render_mode = render_mode
        self.sound = sound
        self.max_episode_steps = max_episode_steps
        self.pyboy = None
        
        # Episode tracking
        self.num_env_steps = 0
        self.current_reward_last_step = 0.0
        
        # Initialize emulator if rom_path provided
        if self.rom_path:
            self._init_emulator()

        self.navigation_enabled = navigation_enabled
        if self.navigation_enabled:
            self.navigation_system = NavigationSystem()
            self.navigation_assistant = NavigationAssistant(
                self.navigation_system,
                model_name=self.model_name,
                vllm_url=self.vllm_url,
                modal_url=self.modal_url
            )

        # Initialize meta-critique system if enabled
        self.meta_critique = None
        if enable_reasoning_aids:
            self.enable_reasoning_aids(runner_log_dir_base=runner_log_dir_base)

    def _init_emulator(self):
        """Initialize the PyBoy emulator"""
        if self.render_mode == "human":
            self.pyboy = PyBoy(self.rom_path, cgb=True, sound=self.sound)
        else:
            self.pyboy = PyBoy(self.rom_path, window="null", cgb=True)
            
        # Load initial state if provided
        if self.initial_state and os.path.exists(self.initial_state):
            try:
                self.load_state(self.initial_state)
                logger.info(f"Loaded initial state from {self.initial_state}")
            except Exception as e:
                logger.error(f"Failed to load initial state: {str(e)}")
                logger.exception("Full traceback:")

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None, episode_id: int = 1) -> Tuple[Observation, Dict[str, Any]]:
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        # Reset episode state
        self.adapter.reset_episode(episode_id)
        self.num_env_steps = 0
        self.current_reward_last_step = 0.0
        
        # Initialize emulator if needed
        if not self.pyboy:
            if not self.rom_path:
                raise ValueError("ROM path must be provided either in __init__ or reset")
            self._init_emulator()
            
        self.initialize()
        info = self._get_info()
        
        # Create observation for adapter
        img_path_for_adapter = None
        text_representation_for_adapter = None
        
        if self.adapter.observation_mode in ["vision", "both"]:
            img_path_for_adapter = self.adapter._create_agent_observation_path(
                self.adapter.current_episode_id, self.adapter.current_step_num
            )
            screenshot = self.get_screenshot()
            Image.fromarray(screenshot).save(img_path_for_adapter)
        
        if self.adapter.observation_mode in ["text", "both"]:
            text_representation_for_adapter = self.get_state_from_memory()

        agent_observation = self.adapter.create_agent_observation(
            img_path=img_path_for_adapter,
            text_representation=text_representation_for_adapter
        )
        
        if self.navigation_enabled:
            self._update_navigation_system()
        
        return agent_observation, info

    def _validate_action(self, action_name: str) -> bool:
        """Validate if an action is valid."""
        if not action_name:
            logger.warning("Empty action name provided")
            return False
            
        if action_name not in self.action_map:
            logger.warning(f"Invalid action name: {action_name}")
            logger.info(f"Valid actions are: {list(self.action_map.keys())}")
            return False
            
        return True

    def step(self, agent_action_str: Optional[str] = None, thought_process: str = "", time_taken_s: float = 0.0) -> Tuple[Observation, float, bool, bool, Dict[str, Any], float]:
        """Execute one step in the environment."""
        # Increment step number first
        self.adapter.increment_step()
        
        # Parse action string
        action_name = None
        repeat_count = 1
        
        if agent_action_str:
            # Handle format: "(action, count)" or just "action"
            agent_action_str = agent_action_str.strip()
            if agent_action_str.startswith('(') and agent_action_str.endswith(')'):
                # Parse "(action, count)" format
                try:
                    content = agent_action_str[1:-1]  # Remove parentheses
                    parts = [part.strip() for part in content.split(',')]
                    if len(parts) == 2:
                        action_name = parts[0].strip('"\'')  # Remove quotes if present
                        repeat_count = int(parts[1])
                    else:
                        action_name = content.strip('"\'')
                except (ValueError, IndexError):
                    action_name = agent_action_str
            else:
                action_name = agent_action_str
        
        # Map action string to environment action
        env_action_idx = self.adapter.map_agent_action_to_env_action(action_name)
        
        reward = 0.0
        terminated = False
        truncated = False
        
        if env_action_idx is not None and self.action_space.contains(env_action_idx):
            button = self.reverse_action_map[env_action_idx]
            # Execute the action multiple times if specified
            for _ in range(repeat_count):
                self.press_buttons([button], wait=True)
                if self._check_terminated():
                    terminated = True
                    break
            reward = self._calculate_reward()
        else:
            print(f"[PokemonRedEnv] Action '{agent_action_str}' is skip/invalid. Env not stepped.")
            reward = -0.01

        self.num_env_steps += 1
        truncated = self._check_truncated()
        self.current_reward_last_step = reward
        
        # Get game info and performance score
        info = self._get_info()
        current_perf_score = self.calculate_perf_score(reward, info)
        
        # Create observation for adapter
        img_path_for_adapter = None
        text_representation_for_adapter = None
        
        if self.pyboy:
            try:
                # Get screenshot
                screenshot = self.pyboy.screen.ndarray
                if screenshot is not None:
                    # Save screenshot with grid overlay if in harness mode
                    if hasattr(self, 'harness_mode') and self.harness_mode:
                        img_path_for_adapter = self._save_game_frame(
                            screenshot,
                            self.adapter.current_episode_id,
                            self.adapter.current_step_num
                        )
                    else:
                        # Save screenshot without grid overlay
                        img_path_for_adapter = self.adapter._create_agent_observation_path(
                            self.adapter.current_episode_id,
                            self.adapter.current_step_num
                        )
                        Image.fromarray(screenshot).save(img_path_for_adapter)
                    
                    # Get text representation
                    text_representation_for_adapter = self._get_text_representation()
            except Exception as e:
                logger.error(f"Error getting observation: {str(e)}")
        
        # Create observation using adapter
        observation = self.adapter.create_agent_observation(
            img_path=img_path_for_adapter,
            text_representation=text_representation_for_adapter
        )
        
        # Log step data
        if hasattr(self.adapter, 'log_step_data'):
            self.adapter.log_step_data(
                agent_action_str=agent_action_str,
                thought_process=thought_process,
                reward=reward,
                info=info,
                terminated=terminated,
                truncated=truncated,
                time_taken_s=time_taken_s,
                perf_score=current_perf_score,
                agent_observation=observation
            )
        
        return observation, reward, terminated, truncated, info, current_perf_score

    def _save_game_frame(self, frame, episode_id: int, step_num: int) -> str:
        """Save a game frame with grid overlay."""
        try:
            # Use the adapter's path format
            frame_path = self.adapter._create_agent_observation_path(episode_id, step_num)
            
            # Ensure the directory exists
            save_dir = os.path.dirname(frame_path)
            os.makedirs(save_dir, exist_ok=True)
            
            # Clean up old frames if they exist
            if os.path.exists(frame_path):
                try:
                    os.remove(frame_path)
                except Exception as e:
                    logger.warning(f"Failed to remove old frame {frame_path}: {str(e)}")
            
            # Convert frame to PIL Image if it's not already
            if isinstance(frame, np.ndarray):
                # Ensure we have RGB format
                if frame.shape[-1] == 4:  # If RGBA
                    frame = frame[:, :, :3]  # Convert to RGB
                # Ensure the array is uint8
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)
                frame = Image.fromarray(frame)
            
            # Create a copy for drawing
            frame_with_grid = frame.copy()
            draw = ImageDraw.Draw(frame_with_grid)
            
            # Get frame dimensions
            width, height = frame_with_grid.size
            
            # Draw grid lines (16x16 pixel grid)
            grid_size = 16
            for x in range(0, width, grid_size):
                draw.line([(x, 0), (x, height)], fill=(255, 0, 0), width=1)
            for y in range(0, height, grid_size):
                draw.line([(0, y), (width, y)], fill=(255, 0, 0), width=1)
            
            # Get collision map and draw terrain info
            collision_map = self._get_collision_map()
            if collision_map is not None:
                # Scale collision map to match grid
                scaled_map = self._downsample_array(collision_map)
                for y in range(scaled_map.shape[0]):
                    for x in range(scaled_map.shape[1]):
                        # Draw terrain label (P for passable, I for impassable)
                        label = "P" if scaled_map[y, x] == 0 else "I"
                        color = (0, 255, 0) if scaled_map[y, x] == 0 else (255, 0, 0)
                        draw.text((x * grid_size + 2, y * grid_size + 2), label, fill=color)
            
            # Draw player position
            player_pos = self._get_player_position()
            if player_pos:
                x, y = player_pos
                # Draw a red dot at player position
                draw.ellipse([(x * grid_size + 4, y * grid_size + 4), 
                             (x * grid_size + 12, y * grid_size + 12)], 
                            fill=(255, 0, 0))
            
            # Save the frame with grid overlay
            try:
                # First try to save as PNG with compression
                frame_with_grid.save(frame_path, format='PNG', optimize=True)
            except Exception as e:
                logger.error(f"Failed to save as PNG: {str(e)}")
                try:
                    # If PNG fails, try JPEG with lower quality
                    frame_path = frame_path.replace('.png', '.jpg')
                    frame_with_grid.save(frame_path, format='JPEG', quality=85, optimize=True)
                except Exception as e2:
                    logger.error(f"Failed to save as JPEG: {str(e2)}")
                    return ""
            
            logger.debug(f"Saved frame with grid overlay to {frame_path}")
            return frame_path
            
        except Exception as e:
            logger.error(f"Error saving game frame: {str(e)}")
            logger.exception("Full traceback:")
            return ""

    def _get_in_combat(self) -> bool:
        """Check if the player is in combat."""
        try:
            # Check battle state from memory reader
            battle_state = self._get_battle_state()
            return battle_state.get('in_battle', False)
        except Exception as e:
            logger.error(f"Error checking combat state: {str(e)}")
            return False

    def _get_observation(self) -> Observation:
        """Get the current observation from the environment."""
        try:
            # Get the current frame
            frame = self.get_screenshot()
            
            # Save frame to file
            img_path = self._save_game_frame(frame, self.adapter.current_episode_id, self.adapter.current_step_num)
            
            # Get textual representation
            textual_representation = self.get_state_from_memory()
            
            # Create observation
            return Observation(
                img_path=img_path,
                textual_representation=textual_representation
            )
            
        except Exception as e:
            logger.error(f"Error getting observation: {str(e)}")
            logger.exception("Full traceback:")
            return Observation(
                img_path="",
                textual_representation="Error getting observation"
            )

    def _calculate_step_perf_score(self, info: Dict[str, Any]) -> float:
        """Calculate performance score for the current step."""
        # Default to reward as performance score
        return float(info.get('reward', 0.0))

    def _get_player_position(self) -> Tuple[int, int]:
        """Get player's current position with validation."""
        try:
            if not self._verify_game_running():
                return (0, 0)
                
            # Use PokemonRedReader to get coordinates
            reader = PokemonRedReader(self.pyboy.memory)
            x, y = reader.read_coordinates()
            
            # Validate coordinates
            if not isinstance(x, (int, np.integer)) or not isinstance(y, (int, np.integer)):
                logger.warning(f"Invalid coordinate types: X={type(x)}, Y={type(y)}")
                return (0, 0)
                
            # Convert to int if numpy types
            x = int(x)
            y = int(y)
            
            # Verify values are reasonable (0-20 range)
            if 0 <= y < 20 and 0 <= x < 20:
                logger.debug(f"Found valid position: ({x}, {y})")
                return (x, y)
            
            logger.warning(f"Coordinates out of bounds: X={x}, Y={y}")
            return (0, 0)
            
        except Exception as e:
            logger.warning(f"Error getting player position: {str(e)}")
            logger.exception("Full traceback:")
            return (0, 0)

    def _get_player_direction(self) -> str:
        """Get player's current direction."""
        try:
            if not self._verify_game_running():
                return "unknown"
                
            # Get direction from sprite pattern
            game_area = self.pyboy.game_wrapper.game_area()
            direction = self._get_direction(game_area)
            
            if direction in ["up", "down", "left", "right"]:
                return direction
                
            logger.warning(f"Invalid direction value: {direction}")
            return "unknown"
            
        except Exception as e:
            logger.warning(f"Error getting player direction: {str(e)}")
            return "unknown"

    def _get_direction(self, array):
        """Determine the player's facing direction from the sprite pattern with validation."""
        try:
            if array is None or not isinstance(array, np.ndarray):
                logger.warning("Invalid input array for direction detection")
                return "no direction found"
                
            rows, cols = array.shape
            logger.debug(f"Game area shape: {array.shape}")
            
            # Validate array dimensions
            if rows < 2 or cols < 2:
                logger.warning(f"Array too small for direction detection: {array.shape}")
                return "no direction found"
            
            # Log the entire game area for debugging
            logger.debug("Game area contents:")
            for i in range(rows):
                row = array[i].tolist()
                logger.debug(f"Row {i}: {row}")

            # Define direction patterns with more detailed logging
            direction_patterns = {
                "down": [0, 1, 2, 3],
                "up": [4, 5, 6, 7],
                "right": [9, 8, 11, 10],
                "left": [8, 9, 10, 11]
            }

            # Check center area first (most likely to contain player)
            center_row = rows // 2
            center_col = cols // 2
            
            # Check a 3x3 area around center
            for i in range(max(0, center_row-1), min(rows-1, center_row+2)):
                for j in range(max(0, center_col-1), min(cols-1, center_col+2)):
                    if i+1 >= rows or j+1 >= cols:
                        continue
                        
                    try:
                        # Get 2x2 grid and ensure it's a list of integers
                        grid = array[i:i+2, j:j+2].flatten()
                        grid_values = [int(x) for x in grid]
                        logger.debug(f"Checking grid at ({i},{j}): {grid_values}")
                        
                        # Check each direction pattern
                        for direction, pattern in direction_patterns.items():
                            if grid_values == pattern:
                                logger.debug(f"Found {direction.upper()} pattern at position ({i},{j})")
                                logger.debug(f"Pattern values: {pattern}")
                                logger.debug(f"Grid values: {grid_values}")
                                return direction
                    except Exception as e:
                        logger.warning(f"Error checking grid at ({i},{j}): {str(e)}")
                        continue

            # If no pattern found in center area, check entire grid
            for i in range(rows - 1):
                for j in range(cols - 1):
                    try:
                        # Get 2x2 grid and ensure it's a list of integers
                        grid = array[i:i+2, j:j+2].flatten()
                        grid_values = [int(x) for x in grid]
                        
                        # Check each direction pattern
                        for direction, pattern in direction_patterns.items():
                            if grid_values == pattern:
                                logger.debug(f"Found {direction.upper()} pattern at position ({i},{j})")
                                logger.debug(f"Pattern values: {pattern}")
                                logger.debug(f"Grid values: {grid_values}")
                                return direction
                    except Exception as e:
                        logger.warning(f"Error checking grid at ({i},{j}): {str(e)}")
                        continue

            # If no pattern is found, log the entire grid for debugging
            logger.warning("No direction pattern found in game area")
            logger.debug("Full game area grid:")
            for i in range(rows):
                for j in range(cols):
                    logger.debug(f"Position ({i},{j}): {array[i,j]}")
            return "no direction found"
            
        except Exception as e:
            logger.error(f"Error determining direction: {str(e)}")
            logger.exception("Full traceback:")
            return "no direction found"

    def _get_collision_map(self) -> Optional[np.ndarray]:
        """Get collision map for current location."""
        try:
            if not self._verify_game_running():
                return None
                
            # Get collision data from game wrapper
            collision_map = self.pyboy.game_wrapper.game_area_collision()
            if collision_map is None:
                logger.warning("Failed to get collision data from game")
                return None
                
            # Downsample the collision map to 10x10
            downsampled = self._downsample_array(collision_map)
            
            # Convert to binary collision map (0 = collision, 1 = walkable)
            binary_map = np.zeros((10, 10), dtype=np.uint8)
            for y in range(10):
                for x in range(10):
                    binary_map[y, x] = 1 if downsampled[y, x] == 0 else 0
                    
            return binary_map
            
        except Exception as e:
            logger.warning(f"Failed to create collision map: {str(e)}")
            return None

    def _get_map_data(self) -> Optional[np.ndarray]:
        """Get the current map data."""
        try:
            if not self._verify_game_running():
                return None
                
            # Get map dimensions from game wrapper
            try:
                map_data = self.pyboy.game_wrapper.game_area()
                if map_data is None:
                    logger.warning("Failed to get map data from game wrapper")
                    return None
                    
                # Convert to numpy array if not already
                if not isinstance(map_data, np.ndarray):
                    map_data = np.array(map_data)
                    
                # Ensure we have valid dimensions
                if map_data.size == 0 or map_data.shape[0] == 0 or map_data.shape[1] == 0:
                    logger.warning("Invalid map dimensions from game wrapper")
                    return None
                    
                return map_data
                
            except Exception as e:
                logger.warning(f"Error getting map data from game wrapper: {str(e)}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get map data: {str(e)}")
            logger.exception("Full traceback:")
            return None

    def _calculate_reward(self) -> float:
        """Calculate reward based on game state"""
        return 0.0

    def _check_terminated(self) -> bool:
        """Check if episode should terminate"""
        return False

    def _check_truncated(self) -> bool:
        """Check if episode should truncate"""
        return self.num_env_steps >= self.max_episode_steps

    def calculate_perf_score(self, reward: float, info: Dict[str, Any]) -> float:
        """Calculate performance score for this step"""
        return reward

    def _get_info(self):
        """Get current game state information."""
        try:
            if not self._verify_game_running():
                return {
                    "player_name": "",
                    "rival_name": "",
                    "money": 0,
                    "location": "unknown",
                    "coordinates": (0, 0),
                    "valid_moves": [],
                    "badges": [],
                    "inventory": {},
                    "dialog": "",
                    "pokemon_party": [],
                    "battle_state": {"in_battle": False}
                }
            
            # Get basic info
            reader = PokemonRedReader(self.pyboy.memory)
            info = {
                "player_name": reader.read_player_name(),
                "rival_name": reader.read_rival_name(),
                "money": reader.read_money(),
                "location": reader.read_location(),
                "coordinates": reader.read_coordinates(),
                "valid_moves": self._get_valid_moves(),
                "inventory": self._get_inventory(),
                "dialog": self._get_dialog(),
                "pokemon_party": self._get_pokemon_party(),
                "battle_state": self._get_battle_state()
            }
            
            # Add quest state if available
            quest_state = self._get_quest_state()
            if quest_state:
                info["quest_state"] = quest_state
                
            # Add game progress if available
            game_progress = self._get_game_progress()
            if game_progress:
                info["game_progress"] = game_progress
                
            return info
            
        except Exception as e:
            logger.error(f"Error getting game info: {str(e)}")
            logger.exception("Full traceback:")
            return {
                "player_name": "",
                "rival_name": "",
                "money": 0,
                "location": "unknown",
                "coordinates": (0, 0),
                "valid_moves": [],
                "badges": [],
                "inventory": {},
                "dialog": "",
                "pokemon_party": [],
                "battle_state": {"in_battle": False}
            }

    def _get_dialog(self) -> str:
        """Get current dialog text."""
        if not self.pyboy:
            return ""
            
        try:
            # Use PokemonRedReader for consistent dialog reading
            reader = PokemonRedReader(self.pyboy.memory)
            dialog = reader.read_dialog()
            
            # Validate dialog
            if not dialog:
                return ""
                
            return dialog.strip()
            
        except Exception as e:
            logger.warning(f"Error getting dialog: {e}")
            logger.exception("Full traceback:")
            return ""

    def _get_menu_state(self) -> str:
        """Get current menu state."""
        if not self.pyboy:
            return "none"
            
        try:
            # Get menu state from memory using PokemonRedReader
            reader = PokemonRedReader(self.pyboy.memory)
            menu_state = self._get_memory_value(0xD057)
            
            # Validate menu state
            if not isinstance(menu_state, (int, np.integer)):
                logger.warning(f"Invalid menu state type: {type(menu_state)}")
                return "none"
                
            menu_states = {
                0: "none",
                1: "main",
                2: "pokemon",
                3: "item",
                4: "save",
                5: "option"
            }
            
            return menu_states.get(int(menu_state), "none")
            
        except Exception as e:
            logger.warning(f"Error getting menu state: {e}")
            logger.exception("Full traceback:")
            return "none"

    def _get_battle_state(self) -> Dict[str, Any]:
        """Get current battle state."""
        if not self.pyboy:
            return {}
            
        try:
            reader = PokemonRedReader(self.pyboy.memory)
            # Get battle state from memory
            battle_state = self._get_memory_value(0xD057)
            if battle_state == 0:
                return {}
                
            return {
                'in_battle': True,
                'battle_type': self._get_memory_value(0xD057),
                'enemy_pokemon': self._get_memory_value(0xCFE5),
                'enemy_level': self._get_memory_value(0xCFE6),
                'enemy_hp': self._get_memory_value(0xCFE7),
                'enemy_max_hp': self._get_memory_value(0xCFE8)
            }
        except Exception as e:
            logger.warning(f"Error getting battle state: {e}")
            return {}

    def _get_text_based_map(self) -> Optional[np.ndarray]:
        """Get text-based representation of current map with validation."""
        try:
            if not self._verify_game_running():
                return None
                
            # Get map data
            map_data = self._get_map_data()
            if map_data is None:
                logger.warning("No map data available for text-based map")
                return None
                
            # Get player position
            player_x, player_y = self._get_player_position()
            
            # Create text-based map (using actual dimensions from map_data)
            rows, cols = map_data.shape
            text_map = np.full((rows, cols), '.', dtype='U1')
            
            # Map tile values to characters
            tile_map = {
                0: '.',  # Walkable
                1: '#',  # Wall
                2: 'D',  # Door
                3: 'T',  # Tree
                4: 'G',  # Grass
                5: 'W',  # Water
                6: 'B',  # Building
                7: 'N',  # NPC
                8: 'I',  # Item
                9: 'P'   # Pokemon
            }
            
            # Convert map data to text representation
            for y in range(rows):
                for x in range(cols):
                    tile = int(map_data[y, x])  # Ensure integer index
                    text_map[y, x] = tile_map.get(tile, '?')
            
            # Add player position if within bounds
            if 0 <= player_y < rows and 0 <= player_x < cols:
                text_map[player_y, player_x] = '@'
            
            logger.debug("Successfully created text-based map")
            return text_map
            
        except Exception as e:
            logger.warning(f"Error creating text-based map: {str(e)}")
            logger.exception("Full traceback:")
            return None

    def render(self, mode='rgb_array'):
        """Render the environment"""
        if mode == 'rgb_array':
            return self.get_screenshot()
        else:
            raise NotImplementedError(f"Render mode {mode} not supported")

    def close(self):
        """Close the environment"""
        if self.pyboy:
            self.pyboy.stop()
        self.adapter.close_log_file()
        print("[PokemonRedEnv] Closed.")

    # ===================== Emulator Methods =====================

    def tick(self, frames):
        """Advance the emulator by the specified number of frames"""
        for _ in range(frames):
            self.pyboy.tick()

    def initialize(self):
        """Initialize the emulator"""
        self.pyboy.set_emulation_speed(0)
        for _ in range(60):
            self.tick(60)
        self.pyboy.set_emulation_speed(1)

    def get_screenshot(self):
        """Get the current screenshot as numpy array"""
        if not self.pyboy:
            return np.zeros((144, 160, 3), dtype=np.uint8)
            
        try:
            # Get screenshot from PyBoy
            screen = self.pyboy.screen.ndarray
            if screen is None:
                logger.warning("Failed to get screenshot from PyBoy")
                return np.zeros((144, 160, 3), dtype=np.uint8)
            
            # Ensure the array is in the correct format
            if not isinstance(screen, np.ndarray):
                screen = np.array(screen)
            
            # Ensure the array has the correct shape and type
            if screen.shape != (144, 160, 4):  # PyBoy returns RGBA
                logger.warning(f"Invalid screenshot shape: {screen.shape}")
                return np.zeros((144, 160, 3), dtype=np.uint8)
            
            # Convert RGBA to RGB
            screen = screen[:, :, :3]
            
            # Ensure the array is uint8
            if screen.dtype != np.uint8:
                screen = screen.astype(np.uint8)
            
            return screen
            
        except Exception as e:
            logger.error(f"Error getting screenshot: {str(e)}")
            logger.exception("Full traceback:")
            return np.zeros((144, 160, 3), dtype=np.uint8)

    def load_state(self, state_filename):
        """Load a state from file"""
        self.pyboy.load_state(open(state_filename, "rb"))

    def save_state(self, state_filename):
        """Save the complete state of the emulator to a file"""
        if not self.pyboy:
            raise RuntimeError("Environment not initialized. Call reset() first.")
            
        with open(state_filename, "wb") as f:
            self.pyboy.save_state(f)
        
        return f"State saved successfully to {state_filename}"

    def press_buttons(self, buttons, wait=True):
        """Press a sequence of buttons on the Game Boy"""
        results = []
        
        for button in buttons:
            if button not in ["a", "b", "start", "select", "up", "down", "left", "right"]:
                results.append(f"Invalid button: {button}")
                continue
                
            self.pyboy.button_press(button)
            self.tick(10)
            self.pyboy.button_release(button)
            
            if wait:
                self.tick(120)
            else:
                self.tick(10)
                
            results.append(f"Pressed {button}")
        
        return "\n".join(results)

    def get_coordinates(self):
        """Returns the player's current coordinates"""
        reader = PokemonRedReader(self.pyboy.memory)
        return reader.read_coordinates()

    def get_active_dialog(self):
        """Returns the active dialog text"""
        reader = PokemonRedReader(self.pyboy.memory)
        dialog = reader.read_dialog()
        if dialog:
            return dialog
        return None

    def get_location(self):
        """Returns the player's current location name"""
        try:
            if not self._verify_game_running():
                return "unknown"
                
            reader = PokemonRedReader(self.pyboy.memory)
            location = reader.read_location()
            
            # Validate location
            if not location or location == "UNKNOWN":
                logger.warning("Invalid location returned from memory")
                return "unknown"
                
            return location
            
        except Exception as e:
            logger.error(f"Error reading location: {str(e)}")
            logger.exception("Full traceback:")
            return "unknown"

    def _downsample_array(self, arr):
        """Downsample an array by averaging 2x2 blocks with validation."""
        try:
            if not isinstance(arr, np.ndarray):
                raise ValueError(f"Input must be numpy array, got {type(arr)}")
                
            rows, cols = arr.shape
            if rows % 2 != 0 or cols % 2 != 0:
                raise ValueError(f"Input array dimensions must be even, got {arr.shape}")
                
            # Calculate output dimensions
            out_rows = rows // 2
            out_cols = cols // 2
            
            # Create output array
            result = np.zeros((out_rows, out_cols), dtype=arr.dtype)
            
            # Reshape and average
            for i in range(out_rows):
                for j in range(out_cols):
                    block = arr[i*2:i*2+2, j*2:j*2+2]
                    result[i, j] = np.mean(block)
                    
            return result
            
        except Exception as e:
            logger.error(f"Error in downsampling: {str(e)}")
            logger.exception("Full traceback:")
            raise

    def _get_valid_moves(self) -> List[str]:
        """Get list of valid moves based on current position and collision map."""
        try:
            if not self._verify_game_running():
                logger.error("Game is not running")
                return []
                
            # Get position from PokemonRedReader
            reader = PokemonRedReader(self.pyboy.memory)
            if not hasattr(reader, 'verify_memory_access') or not reader.verify_memory_access():
                logger.error("Failed to access game memory for position")
                return []
                
            x, y = reader.read_coordinates()
            
            # Scale coordinates to 10x10 grid
            x = min(9, max(0, x // 2))
            y = min(9, max(0, y // 2))
            
            # Get collision map
            collision_map = self._get_collision_map()
            if collision_map is None:
                logger.error("Failed to get collision map")
                return []
                
            valid_moves = []
            
            # Check each direction
            if y > 0 and collision_map[y-1, x] == 0:
                valid_moves.append("up")
            if y < 9 and collision_map[y+1, x] == 0:
                valid_moves.append("down")
            if x > 0 and collision_map[y, x-1] == 0:
                valid_moves.append("left")
            if x < 9 and collision_map[y, x+1] == 0:
                valid_moves.append("right")
                
            if not valid_moves:
                logger.warning(f"No valid moves found at position ({x}, {y})")
                
            return valid_moves
            
        except Exception as e:
            logger.error(f"Error getting valid moves: {str(e)}")
            logger.exception("Full traceback:")
            return []

    def _can_move_between_tiles(self, tile1: int, tile2: int, tileset: str) -> bool:
        """Check if movement between two tiles is allowed based on tile pair collision data"""
        TILE_PAIR_COLLISIONS_LAND = [
            ("CAVERN", 288, 261), ("CAVERN", 321, 261), ("FOREST", 304, 302),
            ("CAVERN", 298, 261), ("CAVERN", 261, 289), ("FOREST", 338, 302),
            ("FOREST", 341, 302), ("FOREST", 342, 302), ("FOREST", 288, 302),
            ("FOREST", 350, 302), ("FOREST", 351, 302),
        ]

        TILE_PAIR_COLLISIONS_WATER = [
            ("FOREST", 276, 302), ("FOREST", 328, 302), ("CAVERN", 276, 261),
        ]

        for ts, t1, t2 in TILE_PAIR_COLLISIONS_LAND + TILE_PAIR_COLLISIONS_WATER:
            if ts == tileset:
                if (tile1 == t1 and tile2 == t2) or (tile1 == t2 and tile2 == t1):
                    return False

        return True

    def get_sprites(self, debug=False):
        """Get the location of all sprites on the screen"""
        sprites_by_y = {}

        for i in range(40):
            sp = self.pyboy.get_sprite(i)
            if sp.on_screen:
                x = int(sp.x / 160 * 10)
                y = int(sp.y / 144 * 9)
                orig_y = sp.y

                if orig_y not in sprites_by_y:
                    sprites_by_y[orig_y] = []
                sprites_by_y[orig_y].append((x, y, i))

        y_positions = sorted(sprites_by_y.keys())
        bottom_sprite_tiles = set()

        if debug:
            print("\nSprites grouped by original Y:")
            for orig_y in y_positions:
                sprites = sprites_by_y[orig_y]
                print(f"Y={orig_y}:")
                for x, grid_y, i in sprites:
                    print(f"  Sprite {i}: x={x}, grid_y={grid_y}")

        SPRITE_HEIGHT = 8

        for i in range(len(y_positions) - 1):
            y1 = y_positions[i]
            y2 = y_positions[i + 1]

            if y2 - y1 == SPRITE_HEIGHT:
                sprites_at_y1 = {s[0]: s for s in sprites_by_y[y1]}
                sprites_at_y2 = {s[0]: s for s in sprites_by_y[y2]}

                for x in sprites_at_y2:
                    if x in sprites_at_y1:
                        bottom_sprite = sprites_at_y2[x]
                        bottom_sprite_tiles.add((x, bottom_sprite[1]))
                        if debug:
                            print(f"\nMatched sprites at x={x}, Y1={y1}, Y2={y2}")

        return bottom_sprite_tiles

    def find_path(self, target_row: int, target_col: int) -> tuple[str, list[str]]:
        """Finds the most efficient path from the player's current position to the target position"""
        collision_map = self.pyboy.game_wrapper.game_area_collision()
        terrain = self._downsample_array(collision_map)
        sprite_locations = self.get_sprites()

        full_map = self.pyboy.game_wrapper._get_screen_background_tilemap()
        reader = PokemonRedReader(self.pyboy.memory)
        tileset = reader.read_tileset()

        start = (4, 4)
        end = (target_row, target_col)

        if not (0 <= target_row < 9 and 0 <= target_col < 10):
            return "Invalid target coordinates", []

        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, end)}

        closest_point = start
        min_distance = heuristic(start, end)

        def reconstruct_path(current):
            path = []
            while current in came_from:
                prev = came_from[current]
                if prev[0] < current[0]:
                    path.append("down")
                elif prev[0] > current[0]:
                    path.append("up")
                elif prev[1] < current[1]:
                    path.append("right")
                else:
                    path.append("left")
                current = prev
            path.reverse()
            return path

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == end:
                path = reconstruct_path(current)
                is_wall = terrain[end[0]][end[1]] == 0
                if is_wall:
                    return (
                        f"Partial Success: Your target location is a wall. In case this is intentional, attempting to navigate there.",
                        path,
                    )
                else:
                    return (
                        f"Success: Found path to target at ({target_row}, {target_col}).",
                        path,
                    )

            current_distance = heuristic(current, end)
            if current_distance < min_distance:
                closest_point = current
                min_distance = current_distance

            if (abs(current[0] - end[0]) + abs(current[1] - end[1])) == 1 and terrain[end[0]][end[1]] == 0:
                path = reconstruct_path(current)
                if end[0] > current[0]:
                    path.append("down")
                elif end[0] < current[0]:
                    path.append("up")
                elif end[1] > current[1]:
                    path.append("right")
                else:
                    path.append("left")
                return (
                    f"Success: Found path to position adjacent to wall at ({target_row}, {target_col}).",
                    path,
                )

            for dr, dc, direction in [
                (1, 0, "down"), (-1, 0, "up"), (0, 1, "right"), (0, -1, "left"),
            ]:
                neighbor = (current[0] + dr, current[1] + dc)

                if not (0 <= neighbor[0] < 9 and 0 <= neighbor[1] < 10):
                    continue
                if terrain[neighbor[0]][neighbor[1]] == 0 and neighbor != end:
                    continue
                if (neighbor[1], neighbor[0]) in sprite_locations and neighbor != end:
                    continue

                current_tile = full_map[current[0] * 2 + 1][current[1] * 2]
                neighbor_tile = full_map[neighbor[0] * 2 + 1][neighbor[1] * 2]
                if not self._can_move_between_tiles(current_tile, neighbor_tile, tileset):
                    continue

                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        if closest_point != start:
            path = reconstruct_path(closest_point)
            return (
                f"Partial Success: Could not reach the exact target, but found a path to the closest reachable point.",
                path,
            )

        return (
            "Failure: No path is visible to the chosen location. You may need to explore a totally different path to get where you're trying to go.",
            [],
        )

    def get_state_from_memory(self) -> str:
        """Reads the game state from memory and returns a string representation"""
        try:
            if not self.pyboy:
                logger.error("PyBoy instance not initialized")
                return "Error: Game not initialized"
                
            reader = PokemonRedReader(self.pyboy.memory)
            
            # Verify memory access
            if not reader._verify_memory_access():
                logger.error("Failed to verify memory access")
                return "Error: Cannot access game memory"
                
            memory_str = ""

            # Read basic info with safe memory access
            name = reader.read_player_name()
            rival_name = reader.read_rival_name()
            money = reader.read_money()
            location = reader.read_location()
            coords = reader.read_coordinates()
            
            # Get valid moves with error handling
            try:
                valid_moves = self._get_valid_moves()
                valid_moves_str = ", ".join(valid_moves) if valid_moves else "None"
            except Exception as e:
                logger.warning(f"Failed to get valid moves: {str(e)}")
                valid_moves_str = "None"

            # Get badges with error handling
            try:
                badges = reader.read_badges()
                badges_str = ", ".join(badges) if badges else "None"
            except Exception as e:
                logger.warning(f"Failed to get badges: {str(e)}")
                badges_str = "None"

            # Get inventory with error handling
            try:
                inventory = reader.read_items()
                inventory_str = "\n".join(f"  {item} x{qty}" for item, qty in inventory)
            except Exception as e:
                logger.warning(f"Failed to get inventory: {str(e)}")
                inventory_str = "  None"

            # Build the state string
            memory_str += f"Player: {name}\n"
            memory_str += f"Rival: {rival_name}\n"
            memory_str += f"Money: ${money}\n"
            memory_str += f"Location: {location}\n"
            memory_str += f"Coordinates: {coords}\n"
            memory_str += f"Valid Moves: {valid_moves_str}\n"
            memory_str += f"Badges: {badges_str}\n"
            memory_str += "Inventory:\n"
            memory_str += inventory_str

            return memory_str
            
        except Exception as e:
            logger.error(f"Error getting game state: {str(e)}")
            logger.exception("Full traceback:")
            return "Error: Failed to read game state"

    def _get_memory_value(self, address: int) -> int:
        """Get value from memory address with validation."""
        if not self.pyboy:
            return 0
            
        try:
            # Validate memory address (PyBoy memory is 0x0000-0xFFFF)
            if not (0 <= address <= 0xFFFF):
                logger.warning(f"Invalid memory address: {hex(address)}")
                return 0
                
            # Use PokemonRedReader for consistent memory access
            reader = PokemonRedReader(self.pyboy.memory)
            data = reader._safe_read_memory(address)
            
            if data is None:
                return 0
                
            value = data[0]
            
            # Ensure value is within valid range (0-255 for single byte)
            if not (0 <= value <= 255):
                logger.warning(f"Invalid memory value at {hex(address)}: {value}")
                return 0
                
            return value
            
        except Exception as e:
            logger.error(f"Error reading memory at {hex(address)}: {str(e)}")
            return 0

    def _verify_game_running(self) -> bool:
        """Verify that the game is running and memory is accessible."""
        try:
            if not self.pyboy:
                return False
                
            # Check if we can read from a known good address
            # D35E: Current map number (should always be readable)
            map_id = self._get_memory_value(0xD35E)
            
            # Just verify we can read memory, don't check value
            # Map ID 0 is valid (PALLET TOWN)
            return True
        except Exception as e:
            logger.warning(f"Error verifying game state: {e}")
            return False

    def _get_inventory(self) -> Dict[str, int]:
        """Get current inventory items and quantities."""
        if not self.pyboy:
            return {}
            
        try:
            reader = PokemonRedReader(self.pyboy.memory)
            items = reader.read_items()
            return {item: qty for item, qty in items}
        except Exception as e:
            logger.warning(f"Error getting inventory: {e}")
            return {}

    def _get_pokemon_party(self) -> List[Dict[str, Any]]:
        """Get current Pokemon party information."""
        if not self.pyboy:
            return []
            
        try:
            reader = PokemonRedReader(self.pyboy.memory)
            party = reader.read_party_pokemon()
            return [{
                'species': pokemon.species_name,
                'level': pokemon.level,
                'current_hp': pokemon.current_hp,
                'max_hp': pokemon.max_hp,
                'status': pokemon.status.get_status_name(),
                'moves': [{'name': move, 'pp': pp} for move, pp in zip(pokemon.moves, pokemon.move_pp)]
            } for pokemon in party]
        except Exception as e:
            logger.warning(f"Error getting Pokemon party: {e}")
            return []

    def _get_quest_state(self) -> Dict[str, Any]:
        """Get current quest state."""
        if not self.pyboy:
            return {}
            
        try:
            reader = PokemonRedReader(self.pyboy.memory)
            badges = reader.read_badges()
            return {f"badge_{i+1}": badge in badges for i, badge in enumerate([
                "BOULDER", "CASCADE", "THUNDER", "RAINBOW",
                "SOUL", "MARSH", "VOLCANO", "EARTH"
            ])}
        except Exception as e:
            logger.warning(f"Error getting quest state: {e}")
            return {}

    def _get_game_progress(self) -> Dict[str, Any]:
        """Get current game progress."""
        if not self.pyboy:
            return {}
            
        try:
            reader = PokemonRedReader(self.pyboy.memory)
            return {
                'badges': reader.read_badges(),
                'pokemon_seen': reader.read_pokedex_caught_count(),
                'pokemon_caught': reader.read_pokedex_caught_count(),
                'current_map': reader.read_location(),
                'player_level': self._get_player_level()
            }
        except Exception as e:
            logger.warning(f"Error getting game progress: {e}")
            return {}

    def _get_player_level(self) -> int:
        """Get player's level."""
        if not self.pyboy:
            return 0
            
        try:
            reader = PokemonRedReader(self.pyboy.memory)
            party = reader.read_party_pokemon()
            if party:
                return party[0].level
            return 0
        except Exception as e:
            logger.warning(f"Error getting player level: {e}")
            return 0

    def _update_navigation_system(self):
        """Update navigation system with current collision map and player position."""
        try:
            if not self.pyboy:
                logger.error("Cannot update navigation system: PyBoy not initialized")
                return

            reader = PokemonRedReader(self.pyboy.memory)
            location = self.get_location()
            coords = reader.read_coordinates()
            
            logger.info(f"Navigation System State:")
            logger.info(f"- Location: {location}")
            logger.info(f"- Coordinates: {coords}")
            logger.info(f"- Navigation Enabled: {self.navigation_enabled}")
            logger.info(f"- Navigation System Initialized: {hasattr(self, 'navigation_system')}")
            
            # Create and update collision map
            collision_map = self._create_collision_map()
            if collision_map is not None:
                # Update navigation system
                self.navigation_system.update_collision_map(location, collision_map)
                
                # Get ASCII map for logging
                ascii_map = self.navigation_system.get_ascii_map(location)
                
                # Update memory module if available
                if hasattr(self, 'memory_module'):
                    self.memory_module.update_navigation_memory(location, collision_map)
                    self.memory_module.add_explored_area(location, coords)
                    self.memory_module.add_navigation_history(
                        action=self._get_direction(self.pyboy.game_wrapper.game_area()),
                        location=location,
                        coords=coords
                    )
                
                # Enhanced logging
                logger.info(f"Navigation Update:")
                logger.info(f"- Location: {location}")
                logger.info(f"- Coordinates: {coords}")
                logger.info(f"- Collision Map:\n{ascii_map}")
                
                # Log valid moves
                valid_moves = self._get_valid_moves()
                logger.info(f"- Valid Moves: {valid_moves}")
                
                # Log navigation history
                if hasattr(self, 'memory_module'):
                    history = self.memory_module.get_navigation_history()
                    logger.info(f"- Navigation History: {history}")
                    
                # Log dialog state
                dialog = reader.read_dialog()
                if dialog:
                    logger.info(f"- Active Dialog: {dialog}")
            else:
                logger.warning("Failed to create collision map")
            
        except Exception as e:
            logger.error(f"Error updating navigation system: {e}")
            logger.exception("Full traceback:")

    def _create_collision_map(self):
        """Create a collision map for navigation."""
        try:
            if not self._verify_game_running():
                return None
                
            # Get map data from game wrapper
            map_data = self._get_map_data()
            if map_data is None:
                logger.warning("Failed to get map data for collision map")
                return None
                
            # Get collision data from game wrapper
            collision_data = self.pyboy.game_wrapper.game_area_collision()
            if collision_data is None:
                logger.warning("Failed to get collision data")
                return None
                
            # Convert to numpy array if not already
            if not isinstance(collision_data, np.ndarray):
                collision_data = np.array(collision_data)
                
            # Ensure we have valid dimensions
            if collision_data.size == 0 or collision_data.shape[0] == 0 or collision_data.shape[1] == 0:
                logger.warning("Invalid collision map dimensions")
                return None
                
            # Create binary collision map (0 = walkable, 1 = collision)
            collision_map = np.zeros_like(collision_data, dtype=bool)
            collision_map[collision_data > 0] = True
            
            return collision_map
            
        except Exception as e:
            logger.error(f"Failed to create collision map: {str(e)}")
            logger.exception("Full traceback:")
            return None

    def get_navigation_advice(self, goal_location: str = None, goal_coords: Tuple[int, int] = None) -> str:
        """Get navigation advice from the navigation assistant."""
        if not self.navigation_enabled:
            return "Navigation assistance is not enabled."
        
        try:
            advice = self.navigation_assistant.get_navigation_advice(
                location=self.get_location(),
                navigation_goal=f"Goal: {goal_location} at {goal_coords}" if goal_location and goal_coords else "Explore the area"
            )
            
            # Log navigation advice
            logger.info(f"Navigation Advice:\n{advice}")
            
            return advice
        except Exception as e:
            logger.error(f"Error getting navigation advice: {str(e)}")
            return "Failed to get navigation advice."

    def auto_path_to_location(self, goal_location: str = None, goal_coords: Tuple[int, int] = None) -> List[str]:
        """Get automatic path to a location using the navigation system."""
        if not self.navigation_enabled:
            return []
        
        try:
            path = self.navigation_assistant.auto_path_to_location(
                location=self.get_location(),
                goal_coords=goal_coords
            )
            
            # Log path
            logger.info(f"Auto-generated path to {goal_location or 'current goal'}:")
            logger.info(f"Path: {' -> '.join(path) if path else 'No path found'}")
            
            return path
        except Exception as e:
            logger.error(f"Error generating auto path: {str(e)}")
            return []

    def add_location_label(self, location: str, coords: Tuple[int, int], label: str):
        if not self.navigation_enabled:
            return
        self.navigation_system.add_location_label(location, coords, label)

    def _update_meta_critique(self, action: str, observation: str):
        """Update meta-critique system with current state."""
        if not self.adapter or not hasattr(self.adapter, 'meta_critique_system'):
            logger.error("Meta-critique system not properly initialized")
            return False
            
        try:
            # Get current state
            current_location = self.get_location()
            if not current_location or current_location == "unknown":
                logger.error("Failed to get valid location")
                return False
                
            current_inventory = self._get_inventory()
            current_pokemon = self._get_pokemon_party()
            current_quest = self._get_quest_state()
            current_progress = self._get_game_progress()
            current_map = self._get_text_based_map()
            current_screenshot = self.get_screenshot() if self.pyboy else None
            
            # Update meta-critique system with current state
            self.adapter.meta_critique_system.mark_checkpoint(
                location=current_location,
                inventory=current_inventory,
                pokemon=current_pokemon,
                quest_state=current_quest,
                game_progress=current_progress
            )
            
            logger.info(f"Successfully updated meta-critique system with location: {current_location}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating meta-critique: {str(e)}")
            logger.exception("Full traceback:")
            return False

    def _verify_state_consistency(self, observation: str):
        """Verify that the game state is consistent with the observation."""
        if not hasattr(self.adapter, 'meta_critique_system'):
            return
            
        current_location = self.get_location()
        expected_location = self.adapter.meta_critique_system.get_current_location()
        
        if current_location != expected_location:
            logger.warning(f"Location mismatch: Expected {expected_location}, got observation: {observation}")
            logger.warning("Game state inconsistency detected")
            
            # Update location in meta-critique system
            self.adapter.meta_critique_system.update_location(current_location)
            
            # Update navigation system if enabled
            if self.navigation_enabled and current_location:
                current_map = self._get_map_data()
                if current_map is not None:
                    self.navigation_system.update_collision_map(
                        location=current_location,
                        collision_map=current_map
                    )
                    
                    # Update memory module if available
                    if hasattr(self.adapter, 'memory_module'):
                        self.adapter.memory_module.location_maps[current_location] = current_map
                        
                        # Update explored areas
                        player_pos = self._get_player_position()
                        if player_pos:
                            self.adapter.memory_module.add_explored_area(current_location, player_pos)
                            self.adapter.memory_module.add_navigation_history(
                                action="location_update",
                                location=current_location,
                                coords=player_pos
                            )

    def get_context_summary(self) -> str:
        """Get the current game context summary for the LLM."""
        if not self.meta_critique:
            logger.warning("Meta-critique system not enabled, cannot get context summary")
            return "Meta-critique system not enabled"
            
        logger.info("Getting context summary from meta-critique system")
        summary = self.meta_critique.get_context_summary()
        logger.info("Context summary retrieved successfully")
        return summary

    def get_meta_critique(self, action: str, observation: str) -> str:
        """Get a meta-critique of the current action and observation."""
        if not self.meta_critique:
            logger.warning("Meta-critique system not enabled, cannot get meta-critique")
            return "Meta-critique system not enabled"
            
        logger.info(f"Getting meta-critique for action: {action}")
        critique = self.meta_critique.get_meta_critique(action, observation)
        logger.info("Meta-critique retrieved successfully")
        return critique

    def enable_reasoning_aids(self, runner_log_dir_base: str = None):
        """Enable meta-critique and reasoning aids system."""
        if not self.adapter:
            logger.warning("Cannot enable reasoning aids: adapter not initialized")
            return
            
        try:
            # Initialize meta-critique system in adapter
            self.adapter.meta_critique_system = MetaCritiqueSystem(
                checkpoint_file="checkpoints.json",
                model_name="default",
                vllm_url=None,
                modal_url=None,
                runner_log_dir_base=self.adapter.agent_cache_dir
            )
            
            # Set self.meta_critique to reference the adapter's meta-critique system
            self.meta_critique = self.adapter.meta_critique_system
            
            # Get current state
            current_location = self.get_location()
            current_inventory = self._get_inventory()
            current_pokemon = self._get_pokemon_party()
            current_quest = self._get_quest_state()
            current_progress = self._get_game_progress()
            
            # Mark initial checkpoint with current state
            # Exclude text_based_map and screenshot as they're not essential for reasoning
            self.adapter.meta_critique_system.mark_checkpoint(
                location=current_location,
                inventory=current_inventory,
                pokemon=current_pokemon,
                quest_state=current_quest,
                game_progress=current_progress
            )
            
            logger.info("Reasoning aids system enabled and initialized")
        except Exception as e:
            logger.error(f"Failed to enable reasoning aids: {str(e)}")
            logger.exception("Full traceback:")

    def _get_text_representation(self) -> str:
        """Get a text representation of the current game state."""
        try:
            if not self._verify_game_running():
                return "Error: Game not running"
                
            # Get basic game state
            reader = PokemonRedReader(self.pyboy.memory)
            location = reader.read_location()
            coords = reader.read_coordinates()
            dialog = reader.read_dialog()
            
            # Get valid moves
            valid_moves = self._get_valid_moves()
            valid_moves_str = ", ".join(valid_moves) if valid_moves else "None"
            
            # Get text-based map
            text_map = self._get_text_based_map()
            map_str = ""
            if text_map is not None:
                map_str = "\n".join(["".join(row) for row in text_map])
            
            # Build text representation
            text_rep = f"Location: {location}\n"
            text_rep += f"Coordinates: {coords}\n"
            text_rep += f"Valid Moves: {valid_moves_str}\n"
            if dialog:
                text_rep += f"Dialog: {dialog}\n"
            if map_str:
                text_rep += f"Map:\n{map_str}\n"
            
            return text_rep
            
        except Exception as e:
            logger.error(f"Error getting text representation: {str(e)}")
            logger.exception("Full traceback:")
            return "Error: Failed to get text representation"