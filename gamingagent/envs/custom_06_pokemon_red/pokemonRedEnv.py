# TODO: Define reward for each step - Yuxuan
import io
import pickle
from collections import deque
import heapq
from typing import Optional, Dict, Any, Tuple, List
import os
import base64

from .memory_reader import PokemonRedReader, StatusCondition
from .full_collision_map import LocationCollisionMap
from PIL import Image, ImageDraw
from pyboy import PyBoy

from gymnasium import Env, spaces
import numpy as np

from gamingagent.envs.gym_env_adapter import GymEnvAdapter
from gamingagent.modules.core_module import Observation


class PokemonRedEnv(Env):
    def __init__(self, 
                 render_mode: Optional[str] = None,
                 # Pokemon Red specific params from game_env_config.json
                 rom_path: Optional[str] = None,
                 sound: bool = False,
                 # Adapter parameters
                 game_name_for_adapter: str = "pokemon_red",
                 observation_mode_for_adapter: str = "vision",
                 agent_cache_dir_for_adapter: str = "cache/pokemon_red/default_run",
                 game_specific_config_path_for_adapter: str = "gamingagent/envs/custom_06_pokemon_red/game_env_config.json",
                 max_stuck_steps_for_adapter: Optional[int] = 20,
                 harness: bool = False):
        super().__init__()
        
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
        self.pyboy = None
        
        # Episode tracking
        self.num_env_steps = 0
        self.current_reward_last_step = 0.0
        
        # Universal collision map tracking per episode (ignoring location differences)
        self.universal_collision_map: Optional[LocationCollisionMap] = None
        
        # Harness mode for enhanced image processing
        self.harness = harness
        
        # Initialize tracking data for enhanced processing (only used in harness mode)
        self.location_history = set()  # Track visited locations
        self.label_archive = {}  # Store location labels
        self.location_tracker = {}  # Track explored tiles per location
        
        # Additional tracking variables to match sample code
        self.location_tracker_activated = False
        self.last_location = None
        
        # Initialize emulator if rom_path provided
        if self.rom_path:
            self._init_emulator()

    def _init_emulator(self):
        """Initialize the PyBoy emulator"""
        if self.render_mode == "human":
            self.pyboy = PyBoy(self.rom_path, cgb=True, sound=self.sound)
        else:
            self.pyboy = PyBoy(self.rom_path, window="null", cgb=True)

    def reset(self, *, max_memory: Optional[int] = 10, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None, episode_id: int = 1) -> Tuple[Observation, Dict[str, Any]]:
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        # Reset episode state
        self.adapter.reset_episode(episode_id)
        self.num_env_steps = 0
        self.current_reward_last_step = 0.0
        
        # Reset universal collision map for new episode
        self.universal_collision_map = None
        
        # Reset tracking variables for harness mode
        if self.harness:
            self.location_history = set()
            self.location_tracker_activated = False
            self.last_location = None
        
        # Always restart the emulator for a clean reset
        if self.pyboy:
            self.pyboy.stop()
            self.pyboy = None
            
        # Initialize emulator fresh
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
            self._save_processed_screenshot(img_path_for_adapter)
        
        if self.adapter.observation_mode in ["text", "both"]:
            # Get basic game state from memory
            game_state = self.get_state_from_memory()
            
            if self.harness:
                # Enhanced text representation for harness mode includes full collision map
                location = self.get_location()
                coords = self.get_coordinates()
                self.update_full_collision_map(location, coords)
                self.save_collision_map_to_file(location)
                
                # Load the full collision map from file
                full_collision_map_text = self.get_full_collision_map_from_file(location)
                if full_collision_map_text:
                    # Split based on the marker to get only the detailed map section
                    if "=== DETAILED MAP ===" in full_collision_map_text:
                        detailed_map = full_collision_map_text.split("=== DETAILED MAP ===")[1].strip()
                        text_representation_for_adapter = f"{game_state}\n\nDetailed Location Map:\n{detailed_map}\n\nThis detailed map shows exact coordinate information and movement costs for {location}."
                    else:
                        # Fallback if marker not found
                        text_representation_for_adapter = f"{game_state}\n\nFull Location Map:\n{full_collision_map_text}\n\nThis shows the complete explored map for {location} with coordinate information and movement costs."
                else:
                    # Fallback to regular collision map
                    collision_map = self.get_collision_map()
                    text_representation_for_adapter = f"{game_state}\n\nSpatial Map:\n{collision_map}\n\nThe spatial map shows your current surroundings in a 9x10 grid. You are always at the center (position 4,4)."
            else:
                # Standard text representation
                # Get collision map for spatial awareness
                collision_map = self.get_collision_map()
                
                # Update and save full collision map
                location = self.get_location()
                coords = self.get_coordinates()
                self.update_full_collision_map(location, coords)
                self.save_collision_map_to_file(location)
                
                # Combine game state with collision map and explanation
                text_representation_for_adapter = f"{game_state}\n\nSpatial Map:\n{collision_map}\n\nThe spatial map shows your current surroundings in a 9x10 grid. You are always at the center (position 4,4). Use this map to understand your environment and plan your movements. Walls (█) block movement, paths (·) are walkable, sprites (S) are NPCs or objects, and the arrow shows which direction you're facing."

        agent_observation = self.adapter.create_agent_observation(
            img_path=img_path_for_adapter,
            text_representation=text_representation_for_adapter,
            max_memory=max_memory,
        )
        
        return agent_observation, info

    def step(self, agent_action_str: Optional[str], thought_process: str = "", time_taken_s: float = 0.0) -> Tuple[Observation, float, bool, bool, Dict[str, Any], float]:
        """Execute one step in the environment"""
        self.adapter.increment_step()
        
        # Parse action string to extract action and repeat count
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
            button = self.action_map[env_action_idx]
            # Execute the action multiple times if specified
            for _ in range(repeat_count):
                self.press_buttons([button], wait=True)
                if self._check_terminated():
                    terminated = True
                    break
            reward = self._calculate_reward()
        else:
            print(f"[PokemonRedEnv] Action '{agent_action_str}' (parsed: '{action_name}', count: {repeat_count}) is skip/invalid. Env not stepped.")
            reward = -0.01

        self.num_env_steps += 1
        truncated = self._check_truncated()
        self.current_reward_last_step = reward
        
        # Get game info and performance score
        info = self._get_info()
        current_perf_score = self.calculate_perf_score(reward, info)
        
        # Track location changes for harness mode
        if self.harness:
            location = self.get_location()
            coords = self.get_coordinates()
            
            # Update location tracking similar to sample code
            if self.location_tracker_activated and coords[0] >= 0 and coords[1] >= 0:
                cols = self.location_tracker.setdefault(location, [])
                # Expand the tracker array as needed (similar to sample code)
                if coords[0] > len(cols) - 1:
                    if len(cols) == 0:
                        cols.extend(list() for _ in range(0, coords[0] + 1))
                    else:
                        cols.extend([False for _ in range(0, len(cols[0]))] for _ in range(len(cols), coords[0] + 1))
                if len(cols) > 0 and coords[1] > len(cols[0]) - 1:
                    for col in cols:
                        col.extend(False for _ in range(len(col), coords[1] + 1))
                cols[coords[0]][coords[1]] = True
            
            # Update location history
            self.location_history.add((location, coords))
            
            # Track location changes
            if self.last_location != location:
                self.last_location = location
        
        # Create observation for adapter
        img_path_for_adapter = None
        text_representation_for_adapter = None
        
        if self.adapter.observation_mode in ["vision", "both"]:
            img_path_for_adapter = self.adapter._create_agent_observation_path(
                self.adapter.current_episode_id, self.adapter.current_step_num
            )
            self._save_processed_screenshot(img_path_for_adapter)
        
        if self.adapter.observation_mode in ["text", "both"]:
            # Get basic game state from memory
            game_state = self.get_state_from_memory()
            
            if self.harness:
                # Enhanced text representation for harness mode includes full collision map
                location = self.get_location()
                coords = self.get_coordinates()
                self.update_full_collision_map(location, coords)
                self.save_collision_map_to_file(location)
                
                # Load the full collision map from file
                full_collision_map_text = self.get_full_collision_map_from_file(location)
                if full_collision_map_text:
                    # Split based on the marker to get only the detailed map section
                    if "=== DETAILED MAP ===" in full_collision_map_text:
                        detailed_map = full_collision_map_text.split("=== DETAILED MAP ===")[1].strip()
                        text_representation_for_adapter = f"{game_state}\n\nDetailed Location Map:\n{detailed_map}\n\nThis detailed map shows exact coordinate information and movement costs for {location}."
                    else:
                        # Fallback if marker not found
                        text_representation_for_adapter = f"{game_state}\n\nFull Location Map:\n{full_collision_map_text}\n\nThis shows the complete explored map for {location} with coordinate information and movement costs."
                else:
                    # Fallback to regular collision map
                    collision_map = self.get_collision_map()
                    text_representation_for_adapter = f"{game_state}\n\nSpatial Map:\n{collision_map}\n\nThe spatial map shows your current surroundings in a 9x10 grid. You are always at the center (position 4,4)."
            else:
                # Standard text representation
                # Get collision map for spatial awareness
                collision_map = self.get_collision_map()
                
                # Update and save full collision map
                location = self.get_location()
                coords = self.get_coordinates()
                self.update_full_collision_map(location, coords)
                self.save_collision_map_to_file(location)
                
                # Combine game state with collision map and explanation
                text_representation_for_adapter = f"{game_state}\n\nSpatial Map:\n{collision_map}\n\nThe spatial map shows your current surroundings in a 9x10 grid. You are always at the center (position 4,4). Use this map to understand your environment and plan your movements. Walls (█) block movement, paths (·) are walkable, sprites (S) are NPCs or objects, and the arrow shows which direction you're facing."

        agent_observation = self.adapter.create_agent_observation(
            img_path=img_path_for_adapter,
            text_representation=text_representation_for_adapter
        )
        
        # Check for stuck detection
        final_terminated, final_truncated = self.adapter.verify_termination(
            agent_observation, terminated, truncated
        )


        # Log step data
        self.adapter.log_step_data(
            agent_action_str=agent_action_str,
            thought_process=thought_process,
            reward=reward,
            info=info,
            terminated=final_terminated,
            truncated=final_truncated,
            time_taken_s=time_taken_s,
            perf_score=current_perf_score,
            agent_observation=agent_observation
        )

        return agent_observation, reward, final_terminated, final_truncated, info, current_perf_score

    def _calculate_reward(self) -> float:
        """Calculate reward based on game state"""
        return 0.0

    def _check_terminated(self) -> bool:
        """Check if episode should terminate"""
        return False

    def _check_truncated(self) -> bool:
        """Check if episode should truncate - controlled by runner instead"""
        return False

    def calculate_perf_score(self, reward: float, info: Dict[str, Any]) -> float:
        """Calculate performance score for this step"""
        return reward

    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about the game state"""
        if not self.pyboy:
            return {}
            
        try:
            info = {
                'coordinates': self.get_coordinates(),
                'location': self.get_location(),
                'valid_moves': self.get_valid_moves(),
                'dialog': self.get_active_dialog(),
                'steps': self.num_env_steps
            }
        except Exception as e:
            print(f"[PokemonRedEnv] Warning: Error getting game info: {e}")
            info = {'steps': self.num_env_steps}
            
        return info

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
        
        # Skip game start sequence
        self._skip_intro_sequence()

    def _skip_intro_sequence(self):
        """Skip the game intro sequence by pressing start/A buttons"""
        print("[PokemonRedEnv] Skipping intro sequence...")
        
        # Wait a bit for the game to fully load
        self.tick(60)
        
        # Press start button 3 times to skip the intro screens
        for i in range(3):
            print(f"[PokemonRedEnv] Pressing start button {i+1}/3")
            self.press_buttons(["a"], wait=True)
            # Add extra wait time between presses
            self.tick(60)
        
        print("[PokemonRedEnv] Intro sequence skipped.")


    def get_screenshot(self):
        """Get the current screenshot as numpy array"""
        if not self.pyboy:
            return np.zeros((240, 256, 3), dtype=np.uint8)
        return np.array(self.pyboy.screen.ndarray)

    def get_screenshot_base64(self, screenshot: Image.Image, upscale=1, add_coords: bool=True,
                             player_coords: Optional[Tuple[int, int]]=None, location: Optional[str]=None, 
                             relative_square_size=8):
        """Convert PIL image to base64 string."""
        # Resize if needed
        if upscale > 1:
            new_size = (screenshot.width * upscale, screenshot.height * upscale)
            screenshot = screenshot.resize(new_size)

        past_locations = self.location_history
        location_labels = self.label_archive.get(location)
        if location_labels is None:
            # this sucks man
            for key, value in self.label_archive.items():
                if location and location.lower() == key.lower():
                    location_labels = value
                    break
        if location_labels is None:
            location_labels = {}
        local_location_tracker = self.location_tracker.get(location, [])

        collision_map = self.pyboy.game_wrapper.game_area_collision()
        downsampled_terrain = self._downsample_array(collision_map)

        sprite_locations = self.get_sprites()

        if not self.get_in_combat():
            shape = screenshot.size
            # Draw some eye-searing lines across the image that nonetheless might make it more obvious to the LLM that this is a grid.
            for x in range(0, shape[0], shape[0]//10):
                ImageDraw.Draw(screenshot).line(((x, 0), (x, shape[1] - 1)), fill=(255, 0, 0))
            for y in range(0, shape[1], shape[1]//9):
                ImageDraw.Draw(screenshot).line(((0, y), (shape[0] - 1, y)), fill=(255, 0, 0))

            # add coordinate labels (note: if scale is too small it may be unreadable)
            # The assumption is the central square is the player's current location, which is 4, 4
            # Rows 0 - 8, Cols 0 - 9
            if add_coords:
                assert player_coords is not None
                tile_size = 16 * upscale
                mid_length = tile_size/2
                for row in range(0, 9):
                    # For bad legacy reasons location labels is row first
                    real_row = player_coords[1] + row - 4
                    local_cols = location_labels.get(real_row, {})
                    for col in range(0, 10):
                        if row == 4 and col == 4:
                            continue  # Skip the player themselves.
                        real_col = player_coords[0] + col - 4
                        label = local_cols.get(real_col, "")
                        tile_label = f"{str(real_col)}, {str(real_row)}"
                        if label:
                            tile_label += "\n" + label
                        if (col, row) not in sprite_locations:
                            if downsampled_terrain[row][col] == 0:
                                tile_label += "\n" + "IMPASSABLE"
                            else:
                                if local_location_tracker and real_col > -1 and real_row > -1 and real_col < len(local_location_tracker) and real_row < len(local_location_tracker[real_col]) and local_location_tracker[real_col][real_row]:
                                    tile_label += "\n" + "EXPLORED"
                                elif (location, (real_col, real_row)) in past_locations:
                                    tile_label += "\n" + "RECENTLY\nVISITED"
                                else:
                                    tile_label += "\n" + "CHECK\nHERE"
                        else:
                            # ImageDraw.Draw(screenshot).rectangle(((col * tile_size + (relative_square_size - 1)*mid_length/relative_square_size, row * tile_size + (relative_square_size - 1)*mid_length/relative_square_size), (col * tile_size + (relative_square_size + 1)*mid_length/relative_square_size, row * tile_size + (relative_square_size + 1)*mid_length/relative_square_size)), (255, 0, 255))
                            tile_label += "\n" + "NPC/OBJECT"
                        font_size = 8
                        # The original SimpleAgent only uses larger font for GEMINI model
                        # For better text readability, we'll keep the smaller font even with upscale=4
                        ImageDraw.Draw(screenshot).text((col * tile_size + mid_length/2, row * tile_size + mid_length/2), tile_label, (255, 0, 0), font_size=font_size)

        # Convert to base64
        buffered = io.BytesIO()
        screenshot.save(buffered, format="PNG")
        return base64.standard_b64encode(buffered.getvalue()).decode()

    def _save_processed_screenshot(self, img_path_for_adapter: str):
        """
        Save screenshot with optional enhanced processing based on harness mode.
        Also saves the original screenshot for video generation.
        
        Args:
            img_path_for_adapter: Path where the image should be saved
        """
        screenshot = self.get_screenshot()
        
        # Always save original screenshot for video generation
        original_img_path = img_path_for_adapter.replace('.png', '_original.png')
        Image.fromarray(screenshot).save(original_img_path)
        
        if self.harness:
            # Enhanced processing for harness mode with overlays and coordinate labels
            screenshot_pil = Image.fromarray(screenshot)
            player_coords = self.get_coordinates()
            location = self.get_location()
            
            # Update location tracking
            self.location_history.add((location, player_coords))
            
            # Process the image with enhanced overlays
            processed_screenshot = self.get_screenshot_base64(
                screenshot_pil, 
                upscale=4, 
                add_coords=True, 
                player_coords=player_coords, 
                location=location
            )
            
            # Save the enhanced image
            enhanced_img_data = base64.b64decode(processed_screenshot)
            with open(img_path_for_adapter, 'wb') as f:
                f.write(enhanced_img_data)
        else:
            # Standard processing - just save the raw screenshot
            Image.fromarray(screenshot).save(img_path_for_adapter)

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

    def get_collision_map(self):
        """
        Creates a simple ASCII map showing player position, direction, terrain and sprites.
        Returns:
            str: A string representation of the ASCII map with legend
        """
        # Get the terrain and movement data
        full_map = self.pyboy.game_wrapper.game_area()
        collision_map = self.pyboy.game_wrapper.game_area_collision()
        downsampled_terrain = self._downsample_array(collision_map)

        # Get sprite locations
        sprite_locations = self.get_sprites()

        # Get character direction from the full map
        direction = self._get_direction(full_map)
        if direction == "no direction found":
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
        lines.extend(
            [
                "",
                "Legend:",
                "█ - Wall/Obstacle",
                "· - Path/Walkable",
                "S - Sprite",
                f"{direction_chars['up']}/{direction_chars['down']}/{direction_chars['left']}/{direction_chars['right']} - Player (facing direction)",
            ]
        )

        # Join all lines with newlines
        return "\n".join(lines)

    def get_valid_moves(self):
        """
        Returns a list of valid moves (up, down, left, right) based on the collision map.
        Returns:
            list[str]: List of valid movement directions
        """
        # Get collision map
        collision_map = self.pyboy.game_wrapper.game_area_collision()
        terrain = self._downsample_array(collision_map)

        # Player is always at position (4,4) in the 9x10 downsampled map
        valid_moves = []

        # We need to check sprites too as they will block traversal
        sprites = self.get_sprites()

        # Special casing for warp tiles. If they're at a 0-coordinate we can safely assume the warp transition direction.
        # otherwise I haven't figured out how to figure it out so we just tell the model all directions are valid and just
        # deal with it.
        reader = PokemonRedReader(self.pyboy.memory)
        warp_coords = reader.get_warps()

        # We need absolute coordinates to check warp.
        player_coords = reader.read_coordinates()
        if player_coords in warp_coords:
            if player_coords[0] and player_coords[1]:  # They're both not 9
                return ["up", "down", "left", "right"]  # I have no idea which directions are valid warps so we just fallback on yielding everything. Probably not even worth checking sprites.
            if not player_coords[0]:
                valid_moves.append("left")
            if not player_coords[1]:  # there is a literal corner case where both are 0, but that never happens in Pokemon Red.
                valid_moves.append("up")
        # Check each direction
        if terrain[3][4] != 0 and (4, 3) not in sprites:  # Up
            valid_moves.append("up")
        if terrain[5][4] != 0 and (4, 5) not in sprites:  # Down
            valid_moves.append("down")
        if terrain[4][3] != 0 and (3, 4) not in sprites:  # Left
            valid_moves.append("left")
        if terrain[4][5] != 0 and (5, 4) not in sprites:  # Right
            valid_moves.append("right")

        return valid_moves

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
        """
        Get the location of all of the sprites on the screen.
        returns set of coordinates that are (column, row)
        """
        # Group sprites by their exact Y coordinate
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

        # Sort Y coordinates
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

        # First, group sprites by X coordinate for each Y level
        for i in range(len(y_positions) - 1):
            y1 = y_positions[i]
            y2 = y_positions[i + 1]

            if y2 - y1 == SPRITE_HEIGHT:
                # Group sprites by X coordinate at each Y level
                sprites_at_y1 = {s[0]: s for s in sprites_by_y[y1]}  # x -> sprite info
                sprites_at_y2 = {s[0]: s for s in sprites_by_y[y2]}

                # Only match sprites that share the same X coordinate
                for x in sprites_at_y2:
                    if x in sprites_at_y1:  # If there's a matching top sprite at this X
                        bottom_sprite = sprites_at_y2[x]
                        bottom_sprite_tiles.add((x, bottom_sprite[1]))
                        if debug:
                            print(f"\nMatched sprites at x={x}, Y1={y1}, Y2={y2}")

        return bottom_sprite_tiles

    def find_path(self, target_row: int, target_col: int) -> tuple[str, list[str]]:
        """
        Finds the most efficient path from the player's current position (4,4) to the target position.
        If the target is unreachable, finds path to nearest accessible spot.
        Allows ending on a wall tile if that's the target.
        Takes into account terrain, sprite collisions, and tile pair collisions.

        Args:
            target_row: Row index in the 9x10 downsampled map (0-8)
            target_col: Column index in the 9x10 downsampled map (0-9)

        Returns:
            tuple[str, list[str]]: Status message and sequence of movements
        """
        # Get collision map, terrain, and sprites
        collision_map = self.pyboy.game_wrapper.game_area_collision()
        terrain = self._downsample_array(collision_map)
        sprite_locations = self.get_sprites()

        # Get full map for tile values and current tileset
        full_map = self.pyboy.game_wrapper._get_screen_background_tilemap()
        reader = PokemonRedReader(self.pyboy.memory)
        tileset = reader.read_tileset()

        # Start at player position (always 4,4 in the 9x10 grid)
        start = (4, 4)
        end = (target_row, target_col)

        # Validate target position
        if not (0 <= target_row < 9 and 0 <= target_col < 10):
            return "Invalid target coordinates", []

        # A* algorithm
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, end)}

        # Track closest reachable point
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

            # Check if we've reached target
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

            # Track closest point
            current_distance = heuristic(current, end)
            if current_distance < min_distance:
                closest_point = current
                min_distance = current_distance

            # If we're next to target and target is a wall, we can end here
            if (abs(current[0] - end[0]) + abs(current[1] - end[1])) == 1 and terrain[
                end[0]
            ][end[1]] == 0:
                path = reconstruct_path(current)
                # Add final move onto wall
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

            # Check all four directions
            for dr, dc, direction in [
                (1, 0, "down"),
                (-1, 0, "up"),
                (0, 1, "right"),
                (0, -1, "left"),
            ]:
                neighbor = (current[0] + dr, current[1] + dc)

                # Check bounds
                if not (0 <= neighbor[0] < 9 and 0 <= neighbor[1] < 10):
                    continue
                # Skip walls unless it's the final destination
                if terrain[neighbor[0]][neighbor[1]] == 0 and neighbor != end:
                    continue
                # Skip sprites unless it's the final destination
                if (neighbor[1], neighbor[0]) in sprite_locations and neighbor != end:
                    continue

                # Check tile pair collisions
                # Get bottom-left tile of each 2x2 block
                current_tile = full_map[current[0] * 2 + 1][
                    current[1] * 2
                ]  # Bottom-left tile of current block
                neighbor_tile = full_map[neighbor[0] * 2 + 1][
                    neighbor[1] * 2
                ]  # Bottom-left tile of neighbor block
                if not self._can_move_between_tiles(
                    current_tile, neighbor_tile, tileset
                ):
                    continue

                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        # If target unreachable, return path to closest point
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
    
    def get_in_combat(self) -> bool:
        """Check if the player is currently in combat"""
        reader = PokemonRedReader(self.pyboy.memory)
        return reader.read_in_combat()

    def get_state_from_memory(self) -> str:
        """
        Reads the game state from memory and returns a string representation of it.
        """
        reader = PokemonRedReader(self.pyboy.memory)
        memory_str = ""

        name = reader.read_player_name()
        if name == "NINTEN":
            name = "Not yet set"
        rival_name = reader.read_rival_name()
        if rival_name == "SONY":
            rival_name = "Not yet set"

        # Get valid moves
        valid_moves = self.get_valid_moves()
        valid_moves_str = ", ".join(valid_moves) if valid_moves else "None"

        location = reader.read_location()
        coords = reader.read_coordinates()  # This comes out col, row

        memory_str += f"Player: {name}\n"
        memory_str += f"Rival: {rival_name}\n"
        memory_str += f"Money: ${reader.read_money()}\n"
        memory_str += f"RAM Location: {location}\n"
        memory_str += f"Coordinates (Horizontal Position/column left-to-right, Vertical Position/row top-to-bottom): {coords}\n"
        memory_str += f"Valid Moves: {valid_moves_str}\n"
        memory_str += f"Badges: {', '.join(reader.read_badges())}\n"

        # Inventory
        memory_str += "Inventory:\n"
        for item, qty in reader.read_items():
            memory_str += f"  {item} x{qty}\n"

        # Dialog
        dialog = reader.read_dialog()
        if dialog:
            memory_str += f"Dialog: {dialog}\n"
        else:
            memory_str += "Dialog: None\n"

        # Party Pokemon
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

    def stop(self):
        """Stop the environment - placeholder for compatibility"""
        self.close()

    # ===================== Full Collision Map Methods =====================

    def _create_collision_map_path(self, location: str) -> str:
        """
        Generate a file path for a collision map text file.
        
        Args:
            location: Location name
            
        Returns:
            File path for the collision map
        """
        # Clean location name for filename
        safe_location = "".join(c if c.isalnum() or c in '-_' else '_' for c in location)
        
        # Create collision_maps directory in agent cache dir (outside observations)
        collision_maps_dir = os.path.join(self.adapter.agent_cache_dir, "collision_maps")
        os.makedirs(collision_maps_dir, exist_ok=True)
        
        return os.path.join(collision_maps_dir, f"collision_map_{safe_location}.txt")

    def _create_episode_collision_map_path(self) -> str:
        """
        Generate a file path for the universal episode collision map text file.
        
        Returns:
            File path for the episode collision map
        """
        # Create collision_maps directory in agent cache dir
        collision_maps_dir = os.path.join(self.adapter.agent_cache_dir, "collision_maps")
        os.makedirs(collision_maps_dir, exist_ok=True)
        
        # Format episode ID with zero padding (e.g., episode_01, episode_02, etc.)
        episode_str = f"episode_{self.adapter.current_episode_id:02d}"
        
        return os.path.join(collision_maps_dir, f"full_collision_map_{episode_str}.txt")

    def update_full_collision_map(self, location: str, coords: Tuple[int, int]) -> str:
        """
        Update the universal collision map for the current episode and return ASCII representation.
        Ignores location differences and maintains one unified map per episode.
        
        Args:
            location: Current location name (ignored for universal map)
            coords: Current player coordinates (col, row)
            
        Returns:
            ASCII representation of the updated collision map
        """
        # Get collision map and sprite data
        collision_map = self.pyboy.game_wrapper.game_area_collision()
        downsampled_terrain = self._downsample_array(collision_map)
        sprite_locations = self.get_sprites()
        
        # Get local location tracker if available (combine all locations for universal map)
        local_location_tracker = None
        if self.harness and self.location_tracker:
            # For universal map, we could combine all location trackers, but for simplicity keep it simple
            local_location_tracker = self.location_tracker.get(location, [])
        
        # Get or create universal collision map for this episode
        if self.universal_collision_map is None:
            self.universal_collision_map = LocationCollisionMap(
                downsampled_terrain, 
                sprite_locations, 
                coords
            )
            return self.universal_collision_map.to_ascii(local_location_tracker)
        else:
            self.universal_collision_map.update_map(
                downsampled_terrain, 
                sprite_locations, 
                coords
            )
            return self.universal_collision_map.to_ascii(local_location_tracker)

    def get_full_collision_map_ascii(self, location: str, local_location_tracker: Optional[list] = None) -> Optional[str]:
        """
        Get ASCII representation of the universal collision map for the current episode.
        
        Args:
            location: Location name (ignored for universal map)
            local_location_tracker: Optional tracker for visited locations
            
        Returns:
            ASCII representation or None if no map exists
        """
        if self.universal_collision_map is not None:
            return self.universal_collision_map.to_ascii(local_location_tracker)
        return None

    def save_collision_map_to_file(self, location: str, local_location_tracker: Optional[list] = None):
        """
        Save universal collision map to an episode-specific text file in the collision maps directory.
        
        Args:
            location: Location name (ignored for universal map)
            local_location_tracker: Optional tracker for visited locations
        """
        if self.universal_collision_map is None:
            return
            
        # Create episode-specific collision map file path
        collision_map_path = self._create_episode_collision_map_path()
        
        # Save the collision map using the to_ascii method with file path
        self.universal_collision_map.to_ascii(
            local_location_tracker,
            save_file_path=collision_map_path
        )

    def get_full_collision_map_from_file(self, location: str) -> Optional[str]:
        """
        Load universal collision map representation from a saved episode-specific text file.
        
        Args:
            location: Location name (ignored for universal map)
            
        Returns:
            String representation of the collision map or None if not found
        """
        collision_map_path = self._create_episode_collision_map_path()
        return LocationCollisionMap.load_from_file(collision_map_path)

    def load_collision_map_from_file(self, location: str) -> Optional[str]:
        """
        Load universal collision map representation from an episode-specific text file.
        
        Args:
            location: Location name (ignored for universal map)
            
        Returns:
            String representation of the collision map or None if not found
        """
        collision_map_path = self._create_episode_collision_map_path()
        return LocationCollisionMap.load_from_file(collision_map_path)

    def get_collision_map_file_path(self, location: str) -> str:
        """
        Get the file path for the universal episode collision map.
        
        Args:
            location: Location name (ignored for universal map)
            
        Returns:
            File path for the episode collision map
        """
        return self._create_episode_collision_map_path()

    def get_all_location_labels(self, location: str) -> List[Tuple[Tuple[int, int], str]]:
        """
        Get all labels for a given location (matches sample code approach).
        
        Args:
            location: Location name
            
        Returns:
            List of tuples containing coordinates and labels
        """
        all_labels: List[Tuple[Tuple[int, int], str]] = []
        this_location = self.label_archive.get(location.lower())
        
        if this_location is None:
            # Case-insensitive lookup
            for key, value in self.label_archive.items():
                if location.lower() == key.lower():
                    this_location = value
                    break
                    
        if this_location is not None and this_location:
            max_row = max(this_location.keys())
            for nearby_row in range(max_row + 1):
                this_row = this_location.get(nearby_row)
                if this_row is not None:
                    max_col = max(this_row.keys())
                    for nearby_col in range(max_col + 1):
                        this_col = this_row.get(nearby_col)
                        if this_col is not None:
                            all_labels.append(((nearby_col, nearby_row), this_col))
                            
        return all_labels