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

from .memory_reader import PokemonRedReader, StatusCondition
from PIL import Image
from pyboy import PyBoy

from gymnasium import Env, spaces
import numpy as np

from gamingagent.envs.gym_env_adapter import GymEnvAdapter
from gamingagent.modules.core_module import Observation
from gamingagent.envs.custom_05_pokemon_red.navigation_system import NavigationSystem
from gamingagent.envs.custom_05_pokemon_red.navigation_assistant import NavigationAssistant
from .reasoning_aids import MetaCritiqueSystem

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
                 runner_log_dir_base: Optional[str] = None):
        super().__init__()
        
        # Store model configuration
        self.model_name = model_name
        self.vllm_url = vllm_url
        self.modal_url = modal_url
        
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
        
        # Action mapping
        self.action_map = {
            0: "a", 1: "b", 2: "start", 3: "select",
            4: "up", 5: "down", 6: "left", 7: "right"
        }
        
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

    def step(self, agent_action_str: Optional[str] = None, thought_process: str = "", time_taken_s: float = 0.0) -> Tuple[Observation, float, bool, bool, Dict[str, Any], float]:
        """Execute one step in the environment.
        
        Args:
            agent_action_str: The action string from the agent
            thought_process: The agent's thought process
            time_taken_s: Time taken by the agent to decide on the action
            
        Returns:
            Tuple containing:
            - Observation: The next observation
            - float: The reward
            - bool: Whether the episode is terminated
            - bool: Whether the episode is truncated
            - Dict[str, Any]: Additional information
            - float: Performance score for this step
        """
        # Parse action string
        base_action_name = "noop"
        frame_count = 1
        
        if agent_action_str:
            # Handle tuple format (action, repeat_count)
            if isinstance(agent_action_str, tuple):
                base_action_name, frame_count = agent_action_str
            elif agent_action_str.startswith("(") and agent_action_str.endswith(")"):
                try:
                    content = agent_action_str[1:-1]  # Remove parentheses
                    parts = [part.strip() for part in content.split(',')]
                    if len(parts) == 2:
                        base_action_name = parts[0].strip('"\'')  # Remove quotes if present
                        frame_count = int(parts[1])
                except (ValueError, IndexError):
                    logger.warning(f"Invalid action format: {agent_action_str}")
                    base_action_name = agent_action_str
            else:
                base_action_name = agent_action_str.strip("()\\\' ")
        
        # Execute action using PyBoy
        if self.pyboy:
            # Map action string to button press
            if base_action_name in self.adapter.move_to_action_idx:
                button = base_action_name  # Use the action name directly
                # Press and hold button for frame_count frames
                self.pyboy.button_press(button)
                for _ in range(frame_count):
                    self.pyboy.tick()
                self.pyboy.button_release(button)
                # Add a small delay after action
                self.pyboy.tick(10)
            else:
                logger.warning(f"Invalid action: {base_action_name}")
                return self._get_observation(), 0.0, False, False, self._get_info(), 0.0
        
        # Get observation
        observation = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check termination conditions
        terminated = self._check_terminated()
        truncated = self._check_truncated()
        
        # Get info
        info = self._get_info()
        
        # Update game state in adapter
        if hasattr(self.adapter, 'game_state'):
            current_location = self.get_location()
            current_inventory = self._get_inventory()
            current_pokemon = self._get_pokemon_party()
            current_quest = self._get_quest_state()
            current_progress = self._get_game_progress()
            current_map = self._get_map_data()
            
            self.adapter.game_state.update({
                'location': current_location,
                'inventory': current_inventory,
                'pokemon': current_pokemon,
                'quest_state': current_quest,
                'game_progress': current_progress,
                'map_data': current_map
            })
            
            # Update navigation system if enabled
            if self.navigation_enabled and current_location:
                # Get collision map
                if current_map is not None:
                    # Update navigation system
                    self.navigation_system.update_collision_map(
                        location=current_location,
                        collision_map=current_map
                    )
                    
                    # Update memory module if available
                    if hasattr(self.adapter, 'memory_module'):
                        # Update location map
                        self.adapter.memory_module.location_maps[current_location] = current_map
                        
                        # Update explored areas
                        player_pos = self._get_player_position()
                        if player_pos:
                            # Mark current position as explored
                            self.adapter.memory_module.add_explored_area(current_location, player_pos)
                            
                            # Add to navigation history
                            self.adapter.memory_module.add_navigation_history(
                                action=agent_action_str,
                                location=current_location,
                                coords=player_pos
                            )
                            
                            # Update location labels if needed
                            if hasattr(self.adapter, 'location_labels'):
                                labels = self.adapter.location_labels.get(current_location, {})
                                if labels:
                                    self.navigation_system.location_labels[current_location] = labels
                                    self.adapter.memory_module.location_labels[current_location] = labels
        
        # Update meta-critique if enabled
        if self.enable_reasoning_aids:
            self._update_meta_critique(agent_action_str, str(observation))
        
        # Calculate performance score for this step
        current_step_perf_score = self._calculate_step_perf_score(info)
        
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
                perf_score=current_step_perf_score,
                agent_observation=observation
            )
        
        return observation, reward, terminated, truncated, info, current_step_perf_score

    def _save_game_frame(self, frame, episode_id: int, step_num: int) -> str:
        """Save the current game frame to a file.
        
        Args:
            frame: The game frame to save
            episode_id: Current episode ID
            step_num: Current step number
            
        Returns:
            str: Path to the saved frame
        """
        # Create directory if it doesn't exist
        os.makedirs(self.adapter.agent_cache_dir, exist_ok=True)
        
        # Create path for the frame
        img_path = self.adapter._create_agent_observation_path(episode_id, step_num)
        
        # Save frame
        Image.fromarray(frame).save(img_path)
        
        return img_path

    def _get_observation(self) -> Observation:
        """Get the current observation from the environment."""
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

    def _calculate_step_perf_score(self, info: Dict[str, Any]) -> float:
        """Calculate performance score for the current step."""
        # Default to reward as performance score
        return float(info.get('reward', 0.0))

    def _get_player_position(self) -> Tuple[int, int]:
        """Get player's current position."""
        if not self.pyboy:
            return (0, 0)
            
        try:
            # Get player position from memory
            # 0xC104: Player's X position
            # 0xC106: Player's Y position
            x = self._get_memory_value(0xC104)
            y = self._get_memory_value(0xC106)
            return (x, y)
        except Exception as e:
            logger.warning(f"Error getting player position: {e}")
            return (0, 0)

    def _get_map_data(self) -> List[List[int]]:
        """Get the current map data as a 2D array."""
        try:
            # Get map data from the game state
            if hasattr(self.adapter, 'game_state'):
                return self.adapter.game_state.get('map_data', [])
            return []
        except Exception as e:
            logger.error(f"Error getting map data: {e}")
            return []

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

    def _get_info(self) -> Dict[str, Any]:
        """Get current game state information."""
        if not self.pyboy:
            return {}
            
        try:
            # Get basic info
            player_name = self._get_player_name()
            rival_name = self._get_rival_name()
            money = self._get_money()
            location = self.get_location()
            x, y = self._get_player_position()
            direction = self._get_player_direction()
            
            # Get valid moves
            valid_moves = self._get_valid_moves()
            valid_moves_str = ", ".join(valid_moves) if valid_moves else "None"
            
            # Get Pokemon party info
            party_info = self._get_party_info()
            
            # Get inventory info
            inventory = self._get_inventory()
            
            # Get badges info
            badges = self._get_badges()
            
            # Get current dialog/menu text
            dialog_text = self._get_dialog_text()
            
            # Get battle info if in battle
            battle_info = self._get_battle_info() if self._is_in_battle() else None
            
            # Get game state
            game_state = self._get_game_state()
            
            # Get navigation info
            navigation_info = self._get_navigation_info()
            
            # Get memory info
            memory_info = self._get_memory_info()
            
            # Get meta-critique info if available
            meta_critique = self._get_meta_critique() if hasattr(self, '_get_meta_critique') else None
            
            # Combine all info
            info = {
                "player_name": player_name,
                "rival_name": rival_name,
                "money": money,
                "location": location,
                "coordinates": (x, y),
                "direction": direction,
                "valid_moves": valid_moves_str,
                "party": party_info,
                "inventory": inventory,
                "badges": badges,
                "dialog_text": dialog_text,
                "battle_info": battle_info,
                "game_state": game_state,
                "navigation_info": navigation_info,
                "memory_info": memory_info,
                "meta_critique": meta_critique
            }
            
            return info
        except Exception as e:
            logger.warning(f"Error getting game state info: {e}")
            return {}

    def _get_dialog(self) -> str:
        """Get current dialog text."""
        if not self.pyboy:
            return ""
            
        try:
            # Get dialog from memory
            dialog = ""
            # Read dialog from memory addresses
            for i in range(20):  # Pokemon Red dialog is 20 characters
                char = self._get_memory_value(0xC4A0 + i)
                if char:
                    dialog += chr(char)
            return dialog.strip()
        except Exception as e:
            logger.warning(f"Error getting dialog: {e}")
            return ""

    def _get_menu_state(self) -> str:
        """Get current menu state."""
        if not self.pyboy:
            return "none"
            
        try:
            # Get menu state from memory
            menu_state = self._get_memory_value(0xD057)
            menu_states = {
                0: "none",
                1: "main",
                2: "pokemon",
                3: "item",
                4: "save",
                5: "option"
            }
            return menu_states.get(menu_state, "none")
        except Exception as e:
            logger.warning(f"Error getting menu state: {e}")
            return "none"

    def _get_battle_state(self) -> Dict[str, Any]:
        """Get current battle state."""
        if not self.pyboy:
            return {}
            
        try:
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

    def _get_text_based_map(self) -> str:
        """Get text-based representation of current map."""
        if not self.pyboy:
            return ""
            
        try:
            # Get map from memory
            map_id = self._get_memory_value(0xD35E)
            map_names = {
                0: "PALLET TOWN",
                1: "VIRIDIAN CITY",
                2: "PEWTER CITY",
                # Add more maps as needed
            }
            return map_names.get(map_id, f"MAP_{map_id}")
        except Exception as e:
            logger.warning(f"Error getting text-based map: {e}")
            return ""

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
            return np.zeros((240, 256, 3), dtype=np.uint8)
        return np.array(self.pyboy.screen.ndarray)

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
        reader = PokemonRedReader(self.pyboy.memory)
        return reader.read_location()

    def _get_direction(self, array):
        """Determine the player's facing direction from the sprite pattern"""
        rows, cols = array.shape

        for i in range(rows - 1):
            for j in range(cols - 1):
                grid = array[i : i + 2, j : j + 2].flatten()

                if list(grid) == [0, 1, 2, 3]:
                    return "down"
                elif list(grid) == [4, 5, 6, 7]:
                    return "up"
                elif list(grid) == [9, 8, 11, 10]:
                    return "right"
                elif list(grid) == [8, 9, 10, 11]:
                    return "left"

        return "no direction found"

    def _downsample_array(self, arr):
        """Downsample an 18x20 array to 9x10 by averaging 2x2 blocks"""
        if arr.shape != (18, 20):
            raise ValueError("Input array must be 18x20")
        return arr.reshape(9, 2, 10, 2).mean(axis=(1, 3))

    def _get_collision_map(self) -> List[List[int]]:
        """Get collision map for current area."""
        if not self.pyboy:
            return [[0] * 20 for _ in range(20)]
            
        try:
            # Get collision map from memory
            # 0xD530: Start of collision map (20x20 grid)
            collision_map = []
            for y in range(20):
                row = []
                for x in range(20):
                    tile = self._get_memory_value(0xD530 + y * 20 + x)
                    # In Pokemon Red, 0 means walkable, non-zero means collision
                    row.append(0 if tile == 0 else 1)
                collision_map.append(row)
            return collision_map
        except Exception as e:
            logger.warning(f"Error getting collision map: {e}")
            return [[0] * 20 for _ in range(20)]

    def _get_valid_moves(self) -> List[str]:
        """Get valid moves based on collision map and player position."""
        if not self.pyboy:
            return []
            
        try:
            # Get collision map and player position
            collision_map = self._get_collision_map()
            x, y = self._get_player_position()
            
            # Check each direction
            valid_moves = []
            if y > 0 and collision_map[y-1][x] == 0:
                valid_moves.append("up")
            if y < 19 and collision_map[y+1][x] == 0:
                valid_moves.append("down")
            if x > 0 and collision_map[y][x-1] == 0:
                valid_moves.append("left")
            if x < 19 and collision_map[y][x+1] == 0:
                valid_moves.append("right")
                
            return valid_moves
        except Exception as e:
            logger.warning(f"Error getting valid moves: {e}")
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
        reader = PokemonRedReader(self.pyboy.memory)
        memory_str = ""

        name = reader.read_player_name()
        if name == "NINTEN":
            name = "Not yet set"
        rival_name = reader.read_rival_name()
        if rival_name == "SONY":
            rival_name = "Not yet set"

        valid_moves = self._get_valid_moves()
        valid_moves_str = ", ".join(valid_moves) if valid_moves else "None"

        memory_str += f"Player: {name}\n"
        memory_str += f"Rival: {rival_name}\n"
        memory_str += f"Money: ${reader.read_money()}\n"
        memory_str += f"Location: {reader.read_location()}\n"
        memory_str += f"Coordinates: {reader.read_coordinates()}\n"
        memory_str += f"Valid Moves: {valid_moves_str}\n"
        memory_str += f"Badges: {', '.join(reader.read_badges())}\n"

        memory_str += "Inventory:\n"
        for item, qty in reader.read_items():
            memory_str += f"  {item} x{qty}\n"

        dialog = reader.read_dialog()
        if dialog:
            memory_str += f"Dialog: {dialog}\n"
        else:
            memory_str += "Dialog: None\n"

        memory_str += "\nPokemon Party:\n"
        for pokemon in reader.read_party_pokemon():
            memory_str += f"\n{pokemon.nickname} ({pokemon.species_name}):\n"
            memory_str += f"Level {pokemon.level} - HP: {pokemon.current_hp}/{pokemon.max_hp}\n"
            memory_str += f"Types: {pokemon.type1.name}{', ' + pokemon.type2.name if pokemon.type2 else ''}\n"
            for move, pp in zip(pokemon.moves, pokemon.move_pp, strict=True):
                memory_str += f"- {move} (PP: {pp})\n"
            if pokemon.status != StatusCondition.NONE:
                memory_str += f"Status: {pokemon.status.get_status_name()}\n"

        return memory_str

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
        """Create a collision map from the current game state."""
        try:
            if not self.pyboy:
                logger.error("Cannot create collision map: PyBoy not initialized")
                return None

            # Check if we're in a valid game state
            reader = PokemonRedReader(self.pyboy.memory)
            dialog = reader.read_dialog()
            if dialog and "NEW GAME" in dialog:
                logger.info("Game is in title screen state - collision map not available yet")
                return None

            # Get collision data
            collision_map = self.pyboy.game_wrapper.game_area_collision()
            if collision_map is None:
                logger.error("Failed to get collision data from game")
                return None

            # Downsample the collision map
            downsampled_terrain = self._downsample_array(collision_map)
            
            # Get sprite locations
            sprite_locations = self.get_sprites()
            
            # Get player direction
            game_area = self.pyboy.game_wrapper.game_area()
            direction = self._get_direction(game_area)
            if direction == "no direction found":
                logger.warning("Could not determine player direction for collision map")
                return None

            # Direction symbols
            direction_chars = {"up": "↑", "down": "↓", "left": "←", "right": "→"}
            player_char = direction_chars.get(direction, "P")

            # Create the ASCII map
            horizontal_border = "+" + "-" * 10 + "+"
            lines = [horizontal_border]

            # Create each row
            for i in range(9):
                row = "|"
                for j in range(10):
                    if i == 4 and j == 4:
                        # Player position with direction
                        row += player_char
                    elif (j, i) in sprite_locations:
                        # Sprite position
                        row += "S"
                    else:
                        # Terrain representation
                        if downsampled_terrain[i][j] == 0:
                            row += "█"  # Wall
                        else:
                            row += "·"  # Path
                row += "|"
                lines.append(row)

            # Add bottom border
            lines.append(horizontal_border)

            # Add legend
            lines.extend([
                "",
                "Legend:",
                "█ - Wall/Obstacle",
                "· - Path/Walkable",
                "S - Sprite",
                f"{direction_chars['up']}/{direction_chars['down']}/{direction_chars['left']}/{direction_chars['right']} - Player (facing direction)",
            ])

            # Join all lines with newlines
            ascii_map = "\n".join(lines)
            
            # Log the collision map
            logger.info(f"Generated collision map:\n{ascii_map}")
            
            # Return the downsampled terrain for navigation
            return downsampled_terrain
            
        except Exception as e:
            logger.error(f"Error creating collision map: {e}")
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
        """Update the meta-critique system with current game state."""
        if not hasattr(self.adapter, 'meta_critique_system'):
            return
            
        # Get current game state
        current_location = self.get_location()
        current_inventory = self._get_inventory()
        current_pokemon = self._get_pokemon_party()
        current_quest = self._get_quest_state()
        current_progress = self._get_game_progress()
        current_map = self._get_map_data()
        
        # Update meta-critique system
        self.adapter.meta_critique_system.update_location(current_location)
        self.adapter.meta_critique_system.record_event(
            action=action,
            observation=observation
        )
        
        # Update navigation system if enabled
        if self.navigation_enabled and current_location:
            # Get collision map
            if current_map is not None:
                # Update navigation system
                self.navigation_system.update_collision_map(
                    location=current_location,
                    collision_map=current_map
                )
                
                # Update memory module if available
                if hasattr(self.adapter, 'memory_module'):
                    # Update location map
                    self.adapter.memory_module.location_maps[current_location] = current_map
                    
                    # Update explored areas
                    player_pos = self._get_player_position()
                    if player_pos:
                        # Mark current position as explored
                        self.adapter.memory_module.add_explored_area(current_location, player_pos)
                        
                        # Add to navigation history
                        self.adapter.memory_module.add_navigation_history(
                            action=action,
                            location=current_location,
                            coords=player_pos
                        )
                        
                        # Update location labels if needed
                        if hasattr(self.adapter, 'location_labels'):
                            labels = self.adapter.location_labels.get(current_location, {})
                            if labels:
                                self.navigation_system.location_labels[current_location] = labels
                                self.adapter.memory_module.location_labels[current_location] = labels
        
        # Save checkpoint
        checkpoint_data = {
            'location': current_location,
            'inventory': current_inventory,
            'pokemon': current_pokemon,
            'quest_state': current_quest,
            'game_progress': current_progress,
            'map_data': current_map,
            'recent_events': self.adapter.meta_critique_system.get_recent_events(),
            'conversation_history': self.adapter.meta_critique_system.get_conversation_history(),
            'text_based_map': self.adapter.meta_critique_system.get_text_based_map(),
            'screenshot': self.get_screenshot() if self.pyboy else None,
            'timestamp': datetime.now().isoformat(),
            'step_count': self.num_env_steps,
            'episode_count': self.adapter.current_episode_id if hasattr(self.adapter, 'current_episode_id') else 0
        }
        
        # Save checkpoint to file
        checkpoint_file = os.path.join(self.adapter.log_dir, "checkpoints.json")
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
            
            # Load existing checkpoints if file exists
            existing_checkpoints = {}
            if os.path.exists(checkpoint_file):
                try:
                    with open(checkpoint_file, 'r') as f:
                        existing_checkpoints = json.load(f)
                except json.JSONDecodeError:
                    logger.warning("Error reading existing checkpoints file, starting fresh")
            
            # Add new checkpoint
            checkpoint_id = f"checkpoint_{len(existing_checkpoints)}"
            existing_checkpoints[checkpoint_id] = checkpoint_data
            
            # Save updated checkpoints
            with open(checkpoint_file, 'w') as f:
                json.dump(existing_checkpoints, f, indent=2)
                
            logger.info(f"Saved checkpoint {checkpoint_id} to {checkpoint_file}")
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
            logger.exception("Full traceback:")
        
        # Verify state consistency
        self._verify_state_consistency(observation)

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
        """Enable reasoning aids system."""
        if not hasattr(self.adapter, 'meta_critique_system'):
            logger.warning("Meta-critique system not available in adapter")
            return False
            
        try:
            # Initialize meta-critique system
            self.adapter.meta_critique_system.initialize(
                location=self.get_location(),
                inventory=self._get_inventory(),
                pokemon=self._get_pokemon_party(),
                quest_state=self._get_quest_state(),
                game_progress=self._get_game_progress()
            )
            
            # Initialize navigation system if enabled
            if self.navigation_enabled:
                self.navigation_system = NavigationSystem()
                current_location = self.get_location()
                if current_location:
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
                                    action="init",
                                    location=current_location,
                                    coords=player_pos
                                )
            
            logger.info("Reasoning aids system enabled successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error enabling reasoning aids: {e}")
            logger.exception("Full traceback:")
            return False

    def _get_memory_value(self, address: int) -> int:
        """Get value from memory address."""
        if not self.pyboy:
            return 0
            
        try:
            return self.pyboy.memory[address]
        except Exception as e:
            logger.warning(f"Error reading memory at {address}: {e}")
            return 0

    def _get_player_direction(self) -> str:
        """Get player's current direction."""
        if not self.pyboy:
            return "down"
            
        try:
            # Get player direction from memory
            # 0xC109: Player direction (0=down, 1=up, 2=left, 3=right)
            direction = self._get_memory_value(0xC109)
            direction_map = {0: "down", 1: "up", 2: "left", 3: "right"}
            return direction_map.get(direction, "down")
        except Exception as e:
            logger.warning(f"Error getting player direction: {e}")
            return "down"

    def _get_inventory(self) -> Dict[str, int]:
        """Get current inventory items and quantities."""
        if not self.pyboy:
            return {}
            
        try:
            # Get inventory from memory
            inventory = {}
            # Read inventory items from memory addresses
            for i in range(20):  # Pokemon Red has 20 inventory slots
                item_id = self._get_memory_value(0xD31D + i)
                if item_id:
                    item_name = self._get_item_name(item_id)
                    if item_name:
                        inventory[item_name] = 1  # Default quantity
            return inventory
        except Exception as e:
            logger.warning(f"Error getting inventory: {e}")
            return {}

    def _get_pokemon_data(self, slot: int) -> Dict[str, Any]:
        """Get Pokemon data for a specific party slot."""
        if not self.pyboy:
            return None
            
        try:
            # Base address for Pokemon data
            base_addr = 0xD16B + (slot * 44)  # Each Pokemon takes 44 bytes
            
            # Read Pokemon data
            species = self._get_memory_value(base_addr)
            if not species:
                return None
                
            # Get Pokemon name
            name = self._get_pokemon_name(species)
            
            # Get level
            level = self._get_memory_value(base_addr + 1)
            
            # Get HP
            current_hp = self._get_memory_value(base_addr + 2)
            max_hp = self._get_memory_value(base_addr + 3)
            
            # Get status
            status = self._get_memory_value(base_addr + 4)
            
            # Get moves
            moves = []
            for i in range(4):
                move_id = self._get_memory_value(base_addr + 5 + i)
                if move_id:
                    move_name = self._get_move_name(move_id)
                    pp = self._get_memory_value(base_addr + 9 + i)
                    moves.append({
                        'name': move_name,
                        'pp': pp
                    })
            
            return {
                'species': name,
                'level': level,
                'current_hp': current_hp,
                'max_hp': max_hp,
                'status': self._get_status_name(status),
                'moves': moves
            }
        except Exception as e:
            logger.warning(f"Error getting Pokemon data: {e}")
            return None

    def _get_pokemon_name(self, species_id: int) -> str:
        """Get Pokemon name from species ID."""
        # Map of species IDs to names
        pokemon_names = {
            1: "BULBASAUR",
            2: "IVYSAUR",
            3: "VENUSAUR",
            4: "CHARMANDER",
            5: "CHARMELEON",
            6: "CHARIZARD",
            7: "SQUIRTLE",
            8: "WARTORTLE",
            9: "BLASTOISE",
            # Add more Pokemon as needed
        }
        return pokemon_names.get(species_id, f"POKEMON_{species_id}")

    def _get_move_name(self, move_id: int) -> str:
        """Get move name from move ID."""
        # Map of move IDs to names
        move_names = {
            1: "POUND",
            2: "KARATE CHOP",
            3: "DOUBLESLAP",
            4: "COMET PUNCH",
            5: "MEGA PUNCH",
            # Add more moves as needed
        }
        return move_names.get(move_id, f"MOVE_{move_id}")

    def _get_status_name(self, status_id: int) -> str:
        """Get status condition name from status ID."""
        status_names = {
            0: "OK",
            1: "POISON",
            2: "BURN",
            3: "FREEZE",
            4: "SLEEP",
            5: "PARALYZE"
        }
        return status_names.get(status_id, "UNKNOWN")

    def _get_badges(self) -> List[str]:
        """Get list of obtained badges."""
        if not self.pyboy:
            return []
            
        try:
            badges = []
            badge_names = [
                "BOULDER",
                "CASCADE",
                "THUNDER",
                "RAINBOW",
                "SOUL",
                "MARSH",
                "VOLCANO",
                "EARTH"
            ]
            
            for i, name in enumerate(badge_names):
                if self._get_memory_value(0xD57A + i):
                    badges.append(name)
            return badges
        except Exception as e:
            logger.warning(f"Error getting badges: {e}")
            return []

    def _get_pokemon_seen(self) -> int:
        """Get number of Pokemon seen."""
        if not self.pyboy:
            return 0
            
        try:
            return self._get_memory_value(0xD2F7)
        except Exception as e:
            logger.warning(f"Error getting Pokemon seen: {e}")
            return 0

    def _get_pokemon_caught(self) -> int:
        """Get number of Pokemon caught."""
        if not self.pyboy:
            return 0
            
        try:
            return self._get_memory_value(0xD2F8)
        except Exception as e:
            logger.warning(f"Error getting Pokemon caught: {e}")
            return 0

    def _get_current_map(self) -> str:
        """Get current map name."""
        if not self.pyboy:
            return ""
            
        try:
            map_id = self._get_memory_value(0xD35E)
            map_names = {
                0: "PALLET TOWN",
                1: "VIRIDIAN CITY",
                2: "PEWTER CITY",
                # Add more maps as needed
            }
            return map_names.get(map_id, f"MAP_{map_id}")
        except Exception as e:
            logger.warning(f"Error getting current map: {e}")
            return ""

    def _get_player_level(self) -> int:
        """Get player's level."""
        if not self.pyboy:
            return 0
            
        try:
            return self._get_memory_value(0xD18C)
        except Exception as e:
            logger.warning(f"Error getting player level: {e}")
            return 0

    def _get_pokemon_party(self) -> List[Dict[str, Any]]:
        """Get current Pokemon party information."""
        if not self.pyboy:
            return []
            
        try:
            party = []
            # Read party Pokemon from memory
            for i in range(6):  # Pokemon Red has 6 party slots
                pokemon_data = self._get_pokemon_data(i)
                if pokemon_data:
                    party.append(pokemon_data)
            return party
        except Exception as e:
            logger.warning(f"Error getting Pokemon party: {e}")
            return []

    def _get_quest_state(self) -> Dict[str, Any]:
        """Get current quest state."""
        if not self.pyboy:
            return {}
            
        try:
            # Get quest flags from memory
            quest_flags = {}
            # Read quest flags from memory addresses
            for i in range(8):  # Pokemon Red has 8 gym badges
                badge_flag = self._get_memory_value(0xD57A + i)
                if badge_flag:
                    quest_flags[f"badge_{i+1}"] = True
            return quest_flags
        except Exception as e:
            logger.warning(f"Error getting quest state: {e}")
            return {}

    def _get_game_progress(self) -> Dict[str, Any]:
        """Get current game progress."""
        if not self.pyboy:
            return {}
            
        try:
            # Get game progress from memory
            progress = {
                'badges': self._get_badges(),
                'pokemon_seen': self._get_pokemon_seen(),
                'pokemon_caught': self._get_pokemon_caught(),
                'current_map': self._get_current_map(),
                'player_level': self._get_player_level()
            }
            return progress
        except Exception as e:
            logger.warning(f"Error getting game progress: {e}")
            return {}

    def _get_item_name(self, item_id: int) -> str:
        """Get item name from item ID."""
        # Map of item IDs to names
        item_names = {
            1: "MASTER BALL",
            2: "ULTRA BALL",
            3: "GREAT BALL",
            4: "POKE BALL",
            5: "TOWN MAP",
            6: "BICYCLE",
            7: "SURFBOARD",
            8: "SAFARI BALL",
            9: "POKEDEX",
            10: "POTION",
            # Add more items as needed
        }
        return item_names.get(item_id, f"ITEM_{item_id}")

    def _get_player_name(self) -> str:
        """Get player's name."""
        if not self.pyboy:
            return ""
            
        try:
            # Get player name from memory
            # 0xD158: Start of player name (10 characters)
            name = ""
            for i in range(10):
                char = self._get_memory_value(0xD158 + i)
                if char == 0x50:  # End of name marker
                    break
                name += chr(char)
            return name.strip()
        except Exception as e:
            logger.warning(f"Error getting player name: {e}")
            return ""

    def _get_rival_name(self) -> str:
        """Get rival's name."""
        if not self.pyboy:
            return ""
            
        try:
            # Get rival name from memory
            # 0xD34A: Start of rival name (10 characters)
            name = ""
            for i in range(10):
                char = self._get_memory_value(0xD34A + i)
                if char == 0x50:  # End of name marker
                    break
                name += chr(char)
            return name.strip()
        except Exception as e:
            logger.warning(f"Error getting rival name: {e}")
            return ""

    def _get_money(self) -> int:
        """Get player's money."""
        if not self.pyboy:
            return 0
            
        try:
            # Get money from memory
            # 0xD347: Money (3 bytes, BCD format)
            money = 0
            for i in range(3):
                byte = self._get_memory_value(0xD347 + i)
                money = (money * 100) + ((byte >> 4) * 10) + (byte & 0x0F)
            return money
        except Exception as e:
            logger.warning(f"Error getting money: {e}")
            return 0

    def _get_reasoning_aid(self) -> Dict[str, Any]:
        """Get reasoning aid information."""
        if not self.pyboy:
            return {}
            
        try:
            # Get basic game state
            game_state = self._get_game_state()
            location = self.get_location()
            x, y = self._get_player_position()
            direction = self._get_player_direction()
            valid_moves = self._get_valid_moves()
            
            # Get navigation info
            navigation_info = self._get_navigation_info()
            
            # Get memory info
            memory_info = self._get_memory_info()
            
            # Get checkpoint info
            checkpoint_info = {
                "current_location": location,
                "coordinates": (x, y),
                "direction": direction,
                "valid_moves": valid_moves,
                "game_state": game_state,
                "navigation_info": navigation_info,
                "memory_info": memory_info
            }
            
            # Log checkpoint if enabled
            if hasattr(self, 'checkpoint_logger') and self.checkpoint_logger:
                self.checkpoint_logger.log_checkpoint(checkpoint_info)
            
            return {
                "game_state": game_state,
                "location": location,
                "coordinates": (x, y),
                "direction": direction,
                "valid_moves": valid_moves,
                "navigation_info": navigation_info,
                "memory_info": memory_info,
                "checkpoint_info": checkpoint_info
            }
        except Exception as e:
            logger.warning(f"Error getting reasoning aid: {e}")
            return {}