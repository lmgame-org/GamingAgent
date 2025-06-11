import io
import logging
import pickle
from collections import deque
import heapq
from typing import Optional, Dict, Any, Tuple

from .memory_reader import PokemonRedReader, StatusCondition
from PIL import Image
from pyboy import PyBoy

from gymnasium import Env, spaces
import numpy as np

from gamingagent.envs.gym_env_adapter import GymEnvAdapter
from gamingagent.modules.core_module import Observation

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
                 max_stuck_steps_for_adapter: Optional[int] = 20):
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
        self.max_episode_steps = max_episode_steps
        self.pyboy = None
        
        # Episode tracking
        self.num_env_steps = 0
        self.current_reward_last_step = 0.0
        
        # Initialize emulator if rom_path provided
        if self.rom_path:
            self._init_emulator()

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
        
        return agent_observation, info

    def step(self, agent_action_str: Optional[str], thought_process: str = "", time_taken_s: float = 0.0) -> Tuple[Observation, float, bool, bool, Dict[str, Any], float]:
        """Execute one step in the environment"""
        self.adapter.increment_step()
        
        # Map action string to environment action
        env_action_idx = self.adapter.map_agent_action_to_env_action(agent_action_str)
        
        reward = 0.0
        terminated = False
        truncated = False
        
        if env_action_idx is not None and self.action_space.contains(env_action_idx):
            button = self.action_map[env_action_idx]
            self.press_buttons([button], wait=True)
            reward = self._calculate_reward()
            terminated = self._check_terminated()
        else:
            print(f"[PokemonRedEnv] Action '{agent_action_str}' (mapped to {env_action_idx}) is skip/invalid. Env not stepped.")
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
        """Check if episode should truncate"""
        return self.num_env_steps >= self.max_episode_steps

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
            logger.warning(f"Error getting game info: {e}")
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

    # Emulator Methods

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

    def get_collision_map(self):
        """Creates a simple ASCII map showing player position, direction, terrain and sprites"""
        full_map = self.pyboy.game_wrapper.game_area()
        collision_map = self.pyboy.game_wrapper.game_area_collision()
        downsampled_terrain = self._downsample_array(collision_map)

        sprite_locations = self.get_sprites()
        direction = self._get_direction(full_map)
        
        if direction == "no direction found":
            return None

        direction_chars = {"up": "↑", "down": "↓", "left": "←", "right": "→"}
        player_char = direction_chars.get(direction, "P")

        horizontal_border = "+" + "-" * 10 + "+"
        lines = [horizontal_border]

        for i in range(9):
            row = "|"
            for j in range(10):
                if i == 4 and j == 4:
                    row += player_char
                elif (j, i) in sprite_locations:
                    row += "S"
                else:
                    if downsampled_terrain[i][j] == 0:
                        row += "█"
                    else:
                        row += "·"
            row += "|"
            lines.append(row)

        lines.append(horizontal_border)
        lines.extend([
            "",
            "Legend:",
            "█ - Wall/Obstacle",
            "· - Path/Walkable",
            "S - Sprite",
            f"{direction_chars['up']}/{direction_chars['down']}/{direction_chars['left']}/{direction_chars['right']} - Player (facing direction)",
        ])

        return "\n".join(lines)

    def get_valid_moves(self):
        """Returns a list of valid moves based on the collision map"""
        collision_map = self.pyboy.game_wrapper.game_area_collision()
        terrain = self._downsample_array(collision_map)

        valid_moves = []

        if terrain[3][4] != 0:
            valid_moves.append("up")
        if terrain[5][4] != 0:
            valid_moves.append("down")
        if terrain[4][3] != 0:
            valid_moves.append("left")
        if terrain[4][5] != 0:
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

        valid_moves = self.get_valid_moves()
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