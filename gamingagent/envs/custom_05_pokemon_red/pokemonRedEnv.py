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
from PIL import Image, ImageDraw, ImageFont
from pyboy import PyBoy
import time
import cv2

from gamingagent.envs.custom_05_pokemon_red.memory_reader import PokemonRedReader, StatusCondition, MapLocation
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
                 initial_state: Optional[str] = None,
                 harness_mode: bool = False):
        super().__init__()
        
        # Store model configuration
        self.model_name = model_name
        self.vllm_url = vllm_url
        self.modal_url = modal_url
        
        # Store initial state path
        self.initial_state = initial_state
        
        # Store harness mode
        self.harness_mode = harness_mode
        
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
        self.observation_space = spaces.Box(low=0, high=255, shape=(144, 160, 3), dtype=np.uint8)
        
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
                # Add delay to allow game to initialize
                time.sleep(1.0)
                # Verify game is running
                if not self._verify_game_running():
                    logger.error("Game failed to initialize properly")
                    raise RuntimeError("Game initialization failed")
            except Exception as e:
                logger.error(f"Failed to load initial state: {str(e)}")
                logger.exception("Full traceback:")

    def _verify_game_running(self) -> bool:
        """Verify that the game is running and memory is accessible"""
        try:
            # Try to read a few known memory locations that should be initialized
            # Check map ID (D35E) and player coordinates (D361, D362)
            map_id = self._get_memory_value(0xD35E)
            x_coord = self._get_memory_value(0xD362)
            y_coord = self._get_memory_value(0xD361)
            
            # Check if any reads failed
            if any(x is None for x in [map_id, x_coord, y_coord]):
                logger.error("Failed to read critical memory addresses")
                return False
                
            # Check if values are within expected ranges
            if not (0 <= map_id <= 0xFF):
                logger.error(f"Invalid map ID: {map_id}")
                return False
                
            if not (0 <= x_coord <= 0xFF):
                logger.error(f"Invalid X coordinate: {x_coord}")
                return False
                
            if not (0 <= y_coord <= 0xFF):
                logger.error(f"Invalid Y coordinate: {y_coord}")
                return False
                
            # If we can read these values without error, game is running
            return True
        except Exception as e:
            logger.error(f"Game verification failed: {str(e)}")
            return False

    def _get_memory_value(self, address: int, max_retries: int = 3) -> Optional[int]:
        """Safely read a single byte from memory with retries"""
        for attempt in range(max_retries):
            try:
                if not (0 <= address <= 0xFFFF):
                    logger.warning(f"Invalid memory address: {hex(address)}")
                    return None
                    
                # Add a small delay before reading
                time.sleep(0.01)  # 10ms delay
                
                # Try to read the memory
                value = self.pyboy.memory[address]
                if value is None:
                    logger.warning(f"Failed to read memory at {hex(address)}")
                    return None
                    
                return value
            except Exception as e:
                logger.warning(f"Memory read attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(0.1)  # 100ms delay before retry
                continue
        return None

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
                    logger.info(f"Harness mode: {self.harness_mode}")  # Debug log
                    if hasattr(self, 'harness_mode') and self.harness_mode:
                        logger.info("Using _save_game_frame for grid overlay")  # Debug log
                        img_path_for_adapter = self._save_game_frame(
                            screenshot,
                            self.adapter.current_episode_id,
                            self.adapter.current_step_num
                        )
                        logger.info(f"Saved frame with grid to: {img_path_for_adapter}")  # Debug log
                    else:
                        logger.info("Using basic screenshot save without grid")  # Debug log
                        # Save screenshot without grid overlay
                        img_path_for_adapter = self.adapter._create_agent_observation_path(
                            self.adapter.current_episode_id,
                            self.adapter.current_step_num
                        )
                        Image.fromarray(screenshot).save(img_path_for_adapter)
                        logger.info(f"Saved basic frame to: {img_path_for_adapter}")  # Debug log
                    
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
        """Save a game frame with grid overlay (robust, reference-style)."""
        try:
            frame_path = self.adapter._create_agent_observation_path(episode_id, step_num)
            logger.info(f"Saving frame to: {frame_path}")
            save_dir = os.path.dirname(frame_path)
            os.makedirs(save_dir, exist_ok=True)
            if os.path.exists(frame_path):
                try:
                    os.remove(frame_path)
                except Exception as e:
                    logger.warning(f"Failed to remove old frame {frame_path}: {str(e)}")

            # Convert frame to numpy array if it's not already
            if isinstance(frame, np.ndarray):
                # Convert RGBA to RGB if needed
                if frame.shape[-1] == 4:
                    frame = frame[:, :, :3]
                
                # Log original frame shape
                logger.info(f"Original frame shape: {frame.shape}")
                
                # Target dimensions for Game Boy screen (scaled up for better visibility)
                target_width = 320  # 2x original width
                target_height = 288  # 2x original height
                
                # Only scale if dimensions don't match
                if frame.shape[:2] != (target_height, target_width):
                    logger.info("Scaling frame to target dimensions")
                    # Get current dimensions
                    h, w = frame.shape[:2]
                    
                    # Calculate scaling factors to fill the target dimensions
                    scale_x = target_width / w
                    scale_y = target_height / h
                    
                    # Use the larger scale to ensure we fill the space
                    scale = max(scale_x, scale_y)
                    
                    # Calculate new dimensions
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    
                    # Resize the frame
                    resized_frame = cv2.resize(frame, (new_w, new_h))
                    
                    # Calculate crop dimensions to center the resized image
                    crop_x = (new_w - target_width) // 2
                    crop_y = (new_h - target_height) // 2
                    
                    # Add extra padding at the top to center the game screen
                    extra_padding = 32  # Add 32 pixels of padding at the top
                    crop_y += extra_padding
                    
                    # Crop the resized image to target dimensions
                    if crop_x > 0 and crop_y > 0:
                        frame = resized_frame[crop_y:crop_y+target_height, crop_x:crop_x+target_width]
                    else:
                        # If no cropping needed, just copy the resized image
                        frame = resized_frame[:target_height, :target_width]
                else:
                    logger.info("Using original frame dimensions")
                
                logger.info(f"Final frame shape: {frame.shape}")
                
                # Convert to PIL Image
                img = Image.fromarray(frame)
                logger.info(f"PIL Image size: {img.size}")

                # --- Grid overlay logic ---
                try:
                    if not hasattr(self, 'pyboy') or self.pyboy is None:
                        logger.error("PyBoy instance not found")
                        return None
                        
                    # Get player position and location using direct memory access
                    x = self._get_memory_value(0xD362)
                    y = self._get_memory_value(0xD361)
                    map_id = self._get_memory_value(0xD35E)
                    
                    # Default values if memory read fails
                    if x is None: x = 0
                    if y is None: y = 0
                    if map_id is None: map_id = 0
                        
                    # Get location name from map ID
                    try:
                        location = MapLocation(map_id).name.replace("_", " ")
                    except ValueError:
                        location = "unknown"
                        
                    logger.info(f"Player position: ({x}, {y}), Location: {location}")

                    # Get collision map and sprite locations
                    try:
                        collision_map = self._get_collision_map()
                        terrain = collision_map  # Use collision map
                        sprite_locations = self._get_sprite_locations()
                        
                        # Log collision map info for debugging
                        logger.info(f"Collision map shape: {terrain.shape}")
                        logger.info(f"Collision map unique values: {np.unique(terrain)}")
                        logger.info(f"Collision map sample: {terrain[:5, :5]}")
                        
                    except Exception as e:
                        logger.error(f"Failed to get game data: {str(e)}")
                        terrain = None
                        sprite_locations = set()

                    # Draw grid lines
                    shape = img.size
                    draw = ImageDraw.Draw(img)
                    
                    # Calculate tile size based on actual collision map dimensions
                    # Game Boy screen is 160x144 pixels, with 20x18 tiles
                    # Scale up tile size to match new dimensions
                    tile_width = 16  # 320/20 = 16 pixels per tile
                    tile_height = 16  # 288/18 = 16 pixels per tile
                    
                    logger.info(f"Image size: {shape}, Tile size: {tile_width}x{tile_height}")
                    
                    # Draw vertical lines (20 columns = 21 lines)
                    for x in range(0, shape[0], tile_width):
                        draw.line(((x, 0), (x, shape[1])), fill=(255, 0, 0))
                    # Draw horizontal lines (18 rows = 19 lines)
                    for y in range(0, shape[1], tile_height):
                        draw.line(((0, y), (shape[0], y)), fill=(255, 0, 0))

                    # Add minimal labels
                    for row in range(min(terrain.shape[0], 18)):  # Limit to 18 rows
                        for col in range(min(terrain.shape[1], 20)):  # Limit to 20 columns
                            # Calculate text position with padding
                            # Center the text in each tile
                            # Adjust text position to align with game pixels
                            text_x = col * tile_width + (tile_width // 2) - 6  # Center horizontally
                            text_y = row * tile_height + (tile_height // 2) - 6  # Center vertically
                            
                            # Ensure text stays within bounds
                            if text_x < 0 or text_x >= shape[0] or text_y < 0 or text_y >= shape[1]:
                                continue
                            
                            # Draw the label with smaller font
                            try:
                                font = ImageFont.truetype("arial.ttf", 12)  # Larger font size for better visibility
                            except:
                                # Fallback to default font if arial.ttf not available
                                font = ImageFont.load_default()
                                
                            # Draw text with black outline for better visibility
                            outline_color = (0, 0, 0)  # Black outline
                            text_color = (255, 0, 0)   # Red text
                            
                            # Simple label based on collision and sprites
                            if terrain is not None and (col, row) not in sprite_locations:
                                # In Pokemon Red, 0 means walkable, 1 means collision
                                if terrain[row][col] == 1:
                                    label = "X"  # Impassable
                                else:
                                    label = "O"  # Passable
                            elif (col, row) in sprite_locations:
                                label = "S"  # Sprite/NPC
                            else:
                                label = "?"
                                
                            # Draw outline
                            for dx, dy in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                                draw.text(
                                    (text_x + dx, text_y + dy),
                                    label,
                                    fill=outline_color,
                                    font=font
                                )
                            
                            # Draw main text
                            draw.text(
                                (text_x, text_y),
                                label,
                                fill=text_color,
                                font=font
                            )

                except Exception as e:
                    logger.error(f"Grid overlay failed: {str(e)}")
                    logger.exception("Full traceback:")
                # --- End grid overlay logic ---

                try:
                    # Save with original size and no modifications
                    img.save(frame_path)
                    logger.info(f"Successfully saved frame to {frame_path}")
                    return frame_path
                except Exception as e:
                    logger.error(f"Failed to save frame: {str(e)}")
                    return None
            else:
                logger.warning(f"Invalid frame type: {type(frame)}")
                return None
        except Exception as e:
            logger.error(f"Failed to save game frame: {str(e)}")
            logger.exception("Full traceback:")
            return None

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

    def _get_player_position(self) -> Optional[tuple[int, int]]:
        """Get player's current position."""
        try:
            if not self._verify_game_running():
                return None
                
            # Use PokemonRedReader for memory access
            reader = PokemonRedReader(self.pyboy.memory)
            if not reader._verify_memory_access():
                logger.warning("Memory access verification failed")
                return None
                
            # Read X and Y coordinates
            x_data = reader._safe_read_memory(0xD362)
            y_data = reader._safe_read_memory(0xD361)
            
            if x_data is None or y_data is None or len(x_data) == 0 or len(y_data) == 0:
                logger.warning("Failed to read player coordinates")
                return None
                
            x, y = x_data[0], y_data[0]
            
            # Validate coordinates
            if not (0 <= x <= 0xFF and 0 <= y <= 0xFF):
                logger.warning(f"Invalid player coordinates: ({x}, {y})")
                return None
                
            return (x, y)
            
        except Exception as e:
            logger.error(f"Failed to access game memory for position: {str(e)}")
            return None

    def _get_player_direction(self) -> str:
        """Get player's current direction."""
        try:
            if not self._verify_game_running():
                return "down"
                
            # Use PokemonRedReader for memory access
            reader = PokemonRedReader(self.pyboy.memory)
            if not reader._verify_memory_access():
                logger.warning("Memory access verification failed")
                return "down"
                
            # Read direction from memory
            direction_data = reader._safe_read_memory(0xD367)
            if direction_data is None or len(direction_data) == 0:
                logger.warning("Failed to read player direction")
                return "down"
                
            direction = direction_data[0]
            
            # Map direction value to string
            direction_map = {
                0: "down",
                1: "up",
                2: "left",
                3: "right"
            }
            
            return direction_map.get(direction, "down")
            
        except Exception as e:
            logger.error(f"Failed to access game memory for direction: {str(e)}")
            return "down"

    def _get_direction(self, array) -> str:
        """Get direction from array of values."""
        try:
            if not isinstance(array, np.ndarray) or array.size == 0:
                return "down"
                
            # Get the center of the array
            center_y, center_x = array.shape[0] // 2, array.shape[1] // 2
            
            # Check each direction
            if array[center_y-1, center_x] == 0:  # Up
                return "up"
            elif array[center_y+1, center_x] == 0:  # Down
                return "down"
            elif array[center_y, center_x-1] == 0:  # Left
                return "left"
            elif array[center_y, center_x+1] == 0:  # Right
                return "right"
                
            return "down"
            
        except Exception as e:
            logger.error(f"Failed to get direction from array: {str(e)}")
            logger.exception("Full traceback:")
            return "down"

    def _get_map_data(self) -> Optional[np.ndarray]:
        """Get the map data for the current area."""
        try:
            if not self._verify_game_running():
                return None
                
            # Create a 20x18 array for the map (Game Boy screen size in tiles)
            map_data = np.zeros((18, 20), dtype=np.uint8)
            
            # Read map data from memory
            # In Pokemon Red, map data is stored in VRAM at 0x9800-0x9BFF
            for y in range(18):
                for x in range(20):
                    # Calculate VRAM address for this tile
                    vram_addr = 0x9800 + (y * 32) + x
                    tile_id = self._get_memory_value(vram_addr)
                    if tile_id is not None:
                        map_data[y, x] = tile_id
                    else:
                        logger.warning(f"Failed to read tile at ({x}, {y})")
                        return None
                        
            return map_data
            
        except Exception as e:
            logger.error(f"Error getting map data: {str(e)}")
            logger.exception("Full traceback:")
            return None

    def _get_collision_map(self) -> Optional[np.ndarray]:
        """Get the collision map for the current area."""
        try:
            if not self._verify_game_running():
                return None
                
            # Get map data
            map_data = self._get_map_data()
            if map_data is None:
                logger.warning("Failed to get map data")
                return None
                
            # Create collision map based on map data
            # In Pokemon Red, tiles 0x00-0x0F are walkable, 0x10-0xFF are not
            collision_map = np.zeros_like(map_data, dtype=np.uint8)
            
            # Set collision for non-walkable tiles
            collision_map[map_data >= 0x10] = 1
            
            # Log collision map info for debugging
            logger.info(f"Collision map shape: {collision_map.shape}")
            logger.info(f"Collision map unique values: {np.unique(collision_map)}")
            logger.info(f"Collision map sample: {collision_map[:5, :5]}")
            
            return collision_map
            
        except Exception as e:
            logger.error(f"Error getting collision map: {str(e)}")
            logger.exception("Full traceback:")
            return None

    def _get_sprite_locations(self) -> set[tuple[int, int]]:
        """Get locations of all sprites in the current map."""
        try:
            if not hasattr(self, 'pyboy') or self.pyboy is None:
                return set()
                
            # Get sprite data from memory
            sprite_data = []
            
            # Pokemon Red uses OAM (Object Attribute Memory) at 0xFE00-0xFE9F
            # Each sprite takes 4 bytes: Y, X, tile, attributes
            for i in range(0, 0xA0, 4):
                sprite_y = self._get_memory_value(0xFE00 + i)
                sprite_x = self._get_memory_value(0xFE01 + i)
                sprite_tile = self._get_memory_value(0xFE02 + i)
                
                # Valid sprite if tile is non-zero and coordinates are within bounds
                if sprite_tile != 0 and 0 < sprite_x < 160 and 0 < sprite_y < 144:
                    # Convert to tile coordinates
                    tile_x = sprite_x // 8
                    tile_y = sprite_y // 8
                    sprite_data.append((tile_x, tile_y))
                    
            return set(sprite_data)
            
        except Exception as e:
            logger.warning(f"Failed to get sprite locations: {str(e)}")
            return set()

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
            
            # Log original screen dimensions
            logger.info(f"Original screen shape: {screen.shape}")
            
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
            
            # Log screen info for debugging
            logger.info(f"Processed screen shape: {screen.shape}")
            logger.info(f"Screen min/max values: {screen.min()}/{screen.max()}")
            
            # Instead of removing borders, ensure we have the full screen
            # This maintains consistent dimensions for grid overlay
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
        """Find a path to the target coordinates using A* algorithm."""
        try:
            # Get collision map
            collision_map = self._get_collision_map()
            if collision_map is None:
                return "No collision map available", []
                
            # Get current position
            current_pos = self._get_player_position()
            if current_pos is None:
                return "Could not determine current position", []
                
            current_row, current_col = current_pos
            
            # Validate target coordinates
            if not (0 <= target_row < collision_map.shape[0] and 0 <= target_col < collision_map.shape[1]):
                return f"Target coordinates ({target_row}, {target_col}) out of bounds", []
                
            # Check if target is walkable
            if collision_map[target_row, target_col] == 0:
                return f"Target position ({target_row}, {target_col}) is not walkable", []
                
            # Initialize A* search
            start = (current_row, current_col)
            goal = (target_row, target_col)
            
            # Priority queue for A* search
            frontier = []
            heapq.heappush(frontier, (0, start))
            
            # Keep track of where we came from
            came_from = {start: None}
            
            # Keep track of cost so far
            cost_so_far = {start: 0}
            
            def heuristic(a, b):
                """Manhattan distance heuristic."""
                return abs(a[0] - b[0]) + abs(a[1] - b[1])
            
            def reconstruct_path(current):
                """Reconstruct path from start to current position."""
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                
                # Convert path to actions
                actions = []
                for i in range(len(path) - 1):
                    current = path[i]
                    next_pos = path[i + 1]
                    
                    # Determine direction
                    if next_pos[0] < current[0]:
                        actions.append("up")
                    elif next_pos[0] > current[0]:
                        actions.append("down")
                    elif next_pos[1] < current[1]:
                        actions.append("left")
                    elif next_pos[1] > current[1]:
                        actions.append("right")
                
                return actions
            
            # A* search
            while frontier:
                _, current = heapq.heappop(frontier)
                
                if current == goal:
                    actions = reconstruct_path(current)
                    return "Path found", actions
                
                # Check all neighbors
                for next_row, next_col in [
                    (current[0] - 1, current[1]),  # up
                    (current[0] + 1, current[1]),  # down
                    (current[0], current[1] - 1),  # left
                    (current[0], current[1] + 1)   # right
                ]:
                    # Check bounds
                    if not (0 <= next_row < collision_map.shape[0] and 0 <= next_col < collision_map.shape[1]):
                        continue
                        
                    # Check if walkable
                    if collision_map[next_row, next_col] == 0:
                        continue
                        
                    next_pos = (next_row, next_col)
                    new_cost = cost_so_far[current] + 1
                    
                    if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                        cost_so_far[next_pos] = new_cost
                        priority = new_cost + heuristic(goal, next_pos)
                        heapq.heappush(frontier, (priority, next_pos))
                        came_from[next_pos] = current
            
            return "No path found", []
            
        except Exception as e:
            logger.error(f"Error finding path: {str(e)}")
            logger.exception("Full traceback:")
            return f"Error finding path: {str(e)}", []

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

    def _save_game_state(self, episode_id: int, step_num: int) -> str:
        """Save the current game state as an image."""
        try:
            if not hasattr(self, 'pyboy') or self.pyboy is None:
                logger.error("PyBoy instance not found")
                return None
                
            # Get the screen image
            screen = self.pyboy.screen_image()
            if screen is None:
                logger.error("Failed to get screen image")
                return None
                
            # Convert to numpy array
            screen_array = np.array(screen)
            logger.info(f"Original screen shape: {screen_array.shape}")
            
            # Convert RGBA to RGB if needed
            if screen_array.shape[-1] == 4:
                screen_array = screen_array[:, :, :3]
                logger.info(f"Processed screen shape: {screen_array.shape}")
            
            # Log pixel value range
            logger.info(f"Screen min/max values: {screen_array.min()}/{screen_array.max()}")
            
            # Target dimensions for Game Boy screen (scaled up for better visibility)
            target_width = 320  # 2x original width
            target_height = 288  # 2x original height
            
            # Get current dimensions
            h, w = screen_array.shape[:2]
            
            # Calculate scaling factors to fill the target dimensions
            scale_x = target_width / w
            scale_y = target_height / h
            
            # Use the larger scale to ensure we fill the space
            scale = max(scale_x, scale_y)
            
            # Calculate new dimensions
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Resize the frame
            resized_screen = cv2.resize(screen_array, (new_w, new_h))
            
            # Calculate crop dimensions to center the resized image
            crop_x = (new_w - target_width) // 2
            crop_y = (new_h - target_height) // 2
            
            # Crop the resized image to target dimensions
            if crop_x > 0 and crop_y > 0:
                new_frame = resized_screen[crop_y:crop_y+target_height, crop_x:crop_x+target_width]
            else:
                # If no cropping needed, just copy the resized image
                new_frame = resized_screen[:target_height, :target_width]
            
            # Convert to PIL Image
            img = Image.fromarray(new_frame)
            
            # Save the image
            frame_path = self.adapter._create_agent_observation_path(episode_id, step_num)
            save_dir = os.path.dirname(frame_path)
            os.makedirs(save_dir, exist_ok=True)
            
            if os.path.exists(frame_path):
                try:
                    os.remove(frame_path)
                except Exception as e:
                    logger.warning(f"Failed to remove old frame {frame_path}: {str(e)}")
            
            img.save(frame_path)
            logger.info(f"Successfully saved game state to {frame_path}")
            return frame_path
            
        except Exception as e:
            logger.error(f"Failed to save game state: {str(e)}")
            logger.exception("Full traceback:")
            return None