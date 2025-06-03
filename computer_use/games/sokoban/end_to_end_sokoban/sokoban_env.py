# filename: gamingagent/envs/sokoban_env.py
#!/usr/bin/env python

# --- Add local gym-sokoban to path ---
import sys
import os
# Get the directory containing this file (envs/)
envs_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to gamingagent/
gamingagent_dir = os.path.dirname(envs_dir)
# Go up one level to the project root (assuming GamingAgent/)
project_root = os.path.dirname(gamingagent_dir)
# Construct the path to the local gym-sokoban
local_sokoban_path = os.path.join(project_root, 'gym-sokoban')

# Insert the local path at the beginning of sys.path if it exists
if os.path.isdir(local_sokoban_path):
    if local_sokoban_path not in sys.path:
        sys.path.insert(0, local_sokoban_path)
    # print(f"DEBUG: Added local gym-sokoban path: {local_sokoban_path}")
else:
    print(f"DEBUG: Local gym-sokoban path not found at {local_sokoban_path}. Using installed version.")
# --- End Add local path ---

# Now the regular imports start
import gymnasium as gym
# Try importing gym_sokoban, handle potential import error
try:
    import gym_sokoban
    from gym_sokoban.envs.sokoban_env import SokobanEnv, ACTION_LOOKUP, CHANGE_COORDINATES
    from gym_sokoban.envs.render_utils import room_to_rgb
except ImportError:
    print("Warning: gym_sokoban not found. Please install it: pip install gym-sokoban")
    # Define dummy classes/variables if needed for type hinting or basic structure
    # This allows the file to be imported even if gym_sokoban isn't installed yet,
    # but it will fail at runtime if used.
    class SokobanEnv(gym.Env): pass
    ACTION_LOOKUP = {}
    CHANGE_COORDINATES = {}
    def room_to_rgb(room_state, room_fixed): return None

import numpy as np
import sys
import pygame # For render mode check
import json # Added missing import for potential future use loading dims/levels

# Mapping from your level file characters to gym-sokoban internal state codes.
# See gym_sokoban.envs.room_utils for original code definitions:
# wall=0, floor=1, box_target=2, box_on_target=3, box_not_on_target=4, player=5
CHAR_TO_STATE = {
    '#': (0, 0),  # Wall (Fixed=Wall, State=Wall)
    ' ': (1, 1),  # Floor (Fixed=Floor, State=Floor)
    '?': (2, 2),  # Target (Fixed=Target, State=Target) - Assuming '?' is your target char
    '$': (1, 4),  # Box on Floor (Fixed=Floor, State=BoxOffTarget)
    '*': (2, 3),  # Box on Target (Fixed=Target, State=BoxOnTarget)
    '@': (1, 5),  # Player on Floor (Fixed=Floor, State=Player)
    '+': (2, 5)   # Player on Target (Fixed=Target, State=Player)
}

# Reverse mapping for rendering
STATE_TO_CHAR = {v[1]: k for k, v in CHAR_TO_STATE.items() if v is not None}
# Add mappings for fixed elements if needed, handling potential overlaps carefully
STATE_TO_CHAR[0] = '#' # Wall state
STATE_TO_CHAR[1] = ' ' # Floor state
STATE_TO_CHAR[2] = '?' # Target state (when empty)

# Define colors for rendering (RGB tuples)
COLOR_MAPPING = {
    0: (64, 64, 64),    # Wall (Dark Gray)
    1: (240, 240, 240), # Floor (Near White)
    2: (255, 180, 180), # Target (Light Red)
    3: (200, 0, 0),     # Box on Target (Dark Red)
    4: (210, 105, 30),  # Box Off Target (Chocolate Brown)
    5: (0, 100, 255)    # Player (Blue)
}
DEFAULT_COLOR = (0, 0, 0) # Black for any unexpected values

class CustomSokobanEnv(SokobanEnv):
    """
    Sokoban Environment that allows loading specific levels from a file
    or generating random levels using the gym-sokoban library.
    Uses the Gymnasium API.
    """
    metadata = {
        'render_modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array', 'raw'],
        'render_fps': 4 # Default FPS for rendering
    }
    original_tile_size = 32

    # --- Define default asset paths relative to this file's directory --- #
    _ASSETS_DIR = os.path.join(os.path.dirname(__file__), 'assets')
    DEFAULT_LEVEL_FILE = os.path.join(_ASSETS_DIR, 'levels')
    DEFAULT_IMAGE_DIR = os.path.join(_ASSETS_DIR, 'images')
    # --- End asset paths --- #

    def __init__(self, level_file=DEFAULT_LEVEL_FILE, image_dir=DEFAULT_IMAGE_DIR, **kwargs):
        """
        Initialize the custom environment.

        Args:
            level_file (str): Path to the file containing specific level layouts.
            image_dir (str): Path to the directory containing custom image files.
            **kwargs: Arguments to pass to the parent SokobanEnv constructor
                      (e.g., render_mode, max_steps, dim_room).
        """
        self.level_file = level_file
        self._init_kwargs = kwargs.copy()
        render_mode = kwargs.pop('render_mode', None)

        # Initialize parent class *without* resetting yet and without render_mode.
        super().__init__(reset=False, **kwargs)

        # Set the render_mode attribute *after* parent initialization
        self.render_mode = render_mode

        # --- Ensure Pygame is initialized early --- Needed for image loading
        if not pygame.get_init():
             pygame.init()
        # --- Initialize display module unconditionally --- #
        if not pygame.display.get_init():
             pygame.display.init()
             print("[DEBUG] CustomSokobanEnv.__init__: pygame.display.init() was called.")
        else:
             print("[DEBUG] CustomSokobanEnv.__init__: pygame.display was already initialized.")
        # --- End display initialization --- #

        self.screen = None
        self.clock = None

        # Initialize image attributes
        # Store paths first, load later
        self.image_dir = image_dir
        self.img_paths = {
            'wall': os.path.join(self.image_dir, 'wall.png'),
            'floor': os.path.join(self.image_dir, 'floor.png'),
            'box': os.path.join(self.image_dir, 'box.png'),
            'box_docked': os.path.join(self.image_dir, 'box_docked.png'),
            'worker': os.path.join(self.image_dir, 'worker.png'),
            'worker_docked': os.path.join(self.image_dir, 'worker_dock.png'),
            'docker': os.path.join(self.image_dir, 'dock.png')
        }
        self.img_originals = {} # Will store loaded original surfaces
        self.img_scaled = {}   # Will store loaded scaled surfaces

        # Initialize state related to images/rendering
        self.images_loaded = False
        self.current_tile_size = self.original_tile_size

        # Loading will happen on the first call to _scale_images.

    def _load_images(self):
        """Loads the original images from paths. Called by _scale_images if needed."""
        if self.images_loaded: # Don't reload if already loaded
             return True
        try:
            self.img_originals['wall'] = pygame.image.load(self.img_paths['wall']).convert_alpha()
            self.img_originals['floor'] = pygame.image.load(self.img_paths['floor']).convert_alpha()
            self.img_originals['box'] = pygame.image.load(self.img_paths['box']).convert_alpha()
            self.img_originals['box_docked'] = pygame.image.load(self.img_paths['box_docked']).convert_alpha()
            self.img_originals['worker'] = pygame.image.load(self.img_paths['worker']).convert_alpha()
            self.img_originals['worker_docked'] = pygame.image.load(self.img_paths['worker_docked']).convert_alpha()
            self.img_originals['docker'] = pygame.image.load(self.img_paths['docker']).convert_alpha()
            self.images_loaded = True
            # print("DEBUG: Custom images loaded successfully.")
            return True
        except pygame.error as e:
            # Fallback to default gym-sokoban colors if images fail
            print(f"Error loading images from {self.image_dir}: {e}", file=sys.stderr)
            print("Rendering will fall back to default gym-sokoban colors.", file=sys.stderr)
            self.images_loaded = False
            return False
        except FileNotFoundError as e:
             # Fallback to default gym-sokoban colors if images fail
             print(f"Error finding image file in {self.image_dir}: {e}", file=sys.stderr)
             print("Rendering will fall back to default gym-sokoban colors.", file=sys.stderr)
             self.images_loaded = False
             return False

    def _scale_images(self, scale_factor):
        """Loads images if needed, then scales them."""
        new_tile_size = int(self.original_tile_size * scale_factor)
        if new_tile_size <= 0: # Prevent invalid size
            new_tile_size = 1

        # If images are already loaded and scale hasn't changed, do nothing
        if self.images_loaded and new_tile_size == self.current_tile_size:
             return

        # Load original images if they haven't been loaded yet
        if not self.images_loaded:
             if not self._load_images():
                  # Loading failed, ensure flag is False and return
                  self.images_loaded = False
                  return # Cannot scale if loading failed

        # Now, scale the loaded original images
        self.current_tile_size = new_tile_size
        try:
             self.img_scaled['wall'] = pygame.transform.scale(self.img_originals['wall'], (self.current_tile_size, self.current_tile_size))
             self.img_scaled['floor'] = pygame.transform.scale(self.img_originals['floor'], (self.current_tile_size, self.current_tile_size))
             self.img_scaled['box'] = pygame.transform.scale(self.img_originals['box'], (self.current_tile_size, self.current_tile_size))
             self.img_scaled['box_docked'] = pygame.transform.scale(self.img_originals['box_docked'], (self.current_tile_size, self.current_tile_size))
             self.img_scaled['worker'] = pygame.transform.scale(self.img_originals['worker'], (self.current_tile_size, self.current_tile_size))
             self.img_scaled['worker_docked'] = pygame.transform.scale(self.img_originals['worker_docked'], (self.current_tile_size, self.current_tile_size))
             self.img_scaled['docker'] = pygame.transform.scale(self.img_originals['docker'], (self.current_tile_size, self.current_tile_size))
        except pygame.error as e:
             # Fallback if scaling fails
             print(f"Error scaling images: {e}", file=sys.stderr)
             self.img_scaled = {} # Clear scaled images on error
             self.images_loaded = False # Treat as not loaded if scaling fails

    def _load_level_from_file(self, level_index):
        """Loads a specific level layout from the level file."""
        if level_index < 1:
             raise ValueError(f"ERROR: Level {level_index} is out of range (must be >= 1)")

        level_found = False
        level_layout = []
        num_boxes = 0
        try:
            with open(self.level_file, 'r') as file:
                for line in file:
                    stripped_line = line.strip()
                    if not level_found:
                        # Check if the line marks the start of the desired level
                        if stripped_line.startswith("Level ") and stripped_line.split(" ")[1] == str(level_index):
                            level_found = True
                    elif stripped_line == "": # End of level definition block
                        if level_found:
                            break # Found the level and its end
                    elif level_found:
                        # Parse the row character by character
                        row = []
                        clean_line = line.rstrip('\n')
                        for char in clean_line:
                             if char not in CHAR_TO_STATE:
                                 raise ValueError(f"ERROR: Level {level_index} has invalid value '{char}'")
                             row.append(char)
                             if char in ['$', '*']:
                                 num_boxes += 1
                        level_layout.append(row)

            if not level_found:
                raise ValueError(f"ERROR: Level {level_index} not found in {self.level_file}")
            if not level_layout:
                raise ValueError(f"ERROR: Level {level_index} is empty in {self.level_file}")

        except FileNotFoundError:
            raise FileNotFoundError(f"ERROR: Levels file not found at {self.level_file}")
        except Exception as e:
            # Catch other potential errors during file reading/parsing
            raise RuntimeError(f"Error reading level {level_index}: {e}")

        return level_layout, num_boxes

    def _parse_level_layout(self, layout):
        """Parses the character layout into numeric arrays and sets env state."""
        if not layout:
            raise ValueError("Cannot parse empty layout.")

        rows = len(layout)
        max_cols = max(len(row) for row in layout) if rows > 0 else 0
        self.dim_room = (rows, max_cols)

        room_fixed = np.full(self.dim_room, 1, dtype=np.uint8) # Default to Floor
        room_state = np.full(self.dim_room, 1, dtype=np.uint8) # Default to Floor
        boxes_on_target = 0
        num_boxes = 0
        player_pos = None

        for r, row_data in enumerate(layout):
            for c, char in enumerate(row_data):
                fixed_val, state_val = CHAR_TO_STATE[char]
                room_fixed[r, c] = fixed_val
                room_state[r, c] = state_val
                if state_val == 5: # Player state
                    if player_pos is not None:
                         print(f"Warning: Multiple players found in level. Using last one at ({r},{c}).", file=sys.stderr)
                    player_pos = np.array([r, c])
                elif state_val == 3: # Box on Target state
                    boxes_on_target += 1
                    num_boxes += 1
                elif state_val == 4: # Box off Target state
                    num_boxes += 1

            # Pad shorter rows with floor (assuming floor is state 1)
            for c in range(len(row_data), max_cols):
                 room_fixed[r, c] = 1
                 room_state[r, c] = 1

        if player_pos is None:
            raise ValueError("Player ('@' or '+') not found in the level layout.")

        # --- Update internal environment state --- #
        self.player_position = player_pos
        self.num_boxes = num_boxes
        self.boxes_on_target = boxes_on_target
        self.room_fixed = room_fixed
        self.room_state = room_state
        self.box_mapping = {} # Reset box mapping (might not be needed by base env)

        # --- Update observation space based on dimensions --- #
        # Render scale is handled by render(), obs space needs defined pixel dimensions
        obs_scale = 1 if (self.render_mode and self.render_mode.startswith('tiny')) else 16
        screen_height, screen_width = (self.dim_room[0] * obs_scale, self.dim_room[1] * obs_scale)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(screen_height, screen_width, 3), dtype=np.uint8
        )
        # Action space is fixed (Discrete(9)) from parent, no update needed

        # Defer image scaling until the first render() call

    def reset(self, *, seed=None, options=None, **kwargs):
        """Resets the environment. Loads specific level if options={'level_index': N}."""
        super(SokobanEnv, self).reset(seed=seed) # Grandparent reset for seeding

        level_index = options.get('level_index') if options else None

        if level_index is not None:
            # --- Load Specific Level --- #
            try:
                layout, _ = self._load_level_from_file(level_index)
                self._parse_level_layout(layout)
                self.num_env_steps = 0
                self.reward_last = 0
                # Call render() to generate the initial observation based on the loaded state
                observation = self.render()
                info = self._get_info()
                return observation, info

            except (ValueError, FileNotFoundError, RuntimeError) as e:
                # Fallback to random level generation if specific level fails
                print(f"Error loading level {level_index}: {e}", file=sys.stderr)
                print("Falling back to random level generation.", file=sys.stderr)
                # Parent reset returns a tuple (obs, info)
                reset_result = super().reset()
                # Ensure it's a tuple of length 2 before returning
                if isinstance(reset_result, tuple) and len(reset_result) == 2:
                    return reset_result
                else:
                    # Handle unexpected return from parent if necessary
                    print("Warning: Fallback super().reset() did not return (obs, info). Adapting.", file=sys.stderr)
                    obs = reset_result # Assume it's just obs
                    info = self._get_info()
                    return obs, info
        else:
            # --- Random Level Generation --- #
            # Parent reset returns a tuple (obs, info)
            reset_result = super().reset()
            # Ensure it's a tuple of length 2 before returning
            if isinstance(reset_result, tuple) and len(reset_result) == 2:
                return reset_result
            else:
                # Handle unexpected return from parent
                print("Warning: Random super().reset() did not return (obs, info). Adapting.", file=sys.stderr)
                # Ensure scaling happens before rendering random state
                # Determine appropriate scale for the observation space
                obs_scale_factor = self.observation_space.shape[0] / (self.dim_room[0] * self.original_tile_size) if self.dim_room[0] > 0 else 1.0
                self._scale_images(obs_scale_factor) # Scale based on obs space
                obs = self.render() # Render the randomly generated state
                info = self._get_info()
                return obs, info

    def _get_info(self):
        """Returns the info dictionary (currently basic)."""
        print(f"[DEBUG] CustomSokobanEnv._get_info(): self.boxes_on_target = {getattr(self, 'boxes_on_target', 'N/A')}, self.num_boxes = {getattr(self, 'num_boxes', 'N/A')}")
        return {
            "action.name": ACTION_LOOKUP.get(0, "NoOp"), # Default NoOp for reset state
            "action.moved_player": False,
            "action.moved_box": False,
            "steps": self.num_env_steps,
            "boxes_on_target": self.boxes_on_target,
        }

    def get_char_matrix(self):
        """Returns the current game state as a 2D list of characters."""
        if self.room_state is None or self.room_fixed is None:
            # Handle case where reset hasn't been called or failed
            print("Warning: Environment state not available for get_char_matrix.", file=sys.stderr)
            return None

        rows, cols = self.dim_room
        char_matrix = [[' ' for _ in range(cols)] for _ in range(rows)]

        for r in range(rows):
            for c in range(cols):
                state_val = self.room_state[r, c]
                fixed_val = self.room_fixed[r, c]
                char_to_draw = ' ' # Default

                # Determine character based on state and fixed background (same logic as render)
                if state_val == 5: # Player
                    char_to_draw = '+' if fixed_val == 2 else '@'
                elif state_val == 3: # Box on Target
                    char_to_draw = '*'
                elif state_val == 4: # Box off Target
                     char_to_draw = '$'
                elif fixed_val == 0: # Wall
                     char_to_draw = '#'
                elif fixed_val == 2: # Target (empty)
                     char_to_draw = '?'
                # No need for explicit floor check, default is ' '
                # else: # Fallback for unexpected values
                #      char_to_draw = STATE_TO_CHAR.get(state_val, ' ')

                char_matrix[r][c] = char_to_draw
        return char_matrix

    # Overridden render method to handle custom images and modes
    def render(self, *args, **kwargs):
        """Renders the environment using custom images if available."""
        # --- Determine the effective render mode --- #
        # Prioritize mode passed in kwargs, fallback to instance attribute
        mode = kwargs.get('mode', self.render_mode)
        # --- End mode determination ---

        if mode is None:
            # Ensure observation space matches expectations even if not rendering
            if hasattr(self, 'dim_room'):
                 scale = 16 # Default scale for non-tiny obs
                 screen_height, screen_width = (self.dim_room[0] * scale, self.dim_room[1] * scale)
                 self.observation_space = gym.spaces.Box(
                     low=0, high=255, shape=(screen_height, screen_width, 3), dtype=np.uint8
                 )
            return np.zeros(self.observation_space.shape, dtype=np.uint8) # Return black screen matching obs space

        # Determine scale factor based on mode
        scale_factor = 1.0 # Default scale
        if 'tiny' in mode:
            # Assume tiny means 1 pixel per tile for simplicity
            if self.original_tile_size > 0:
                 scale_factor = 1.0 / self.original_tile_size
            else:
                 scale_factor = 1.0 # Fallback
        elif 'rgb_array' in mode: # Calculate scale for target size
            try:
                rows, cols = self.dim_room
                max_dim = max(rows, cols)
                if max_dim > 0 and self.original_tile_size > 0:
                    target_size = 1500.0
                    scale_factor = target_size / (max_dim * self.original_tile_size)
                    print(f"[DEBUG] Scaling rgb_array to target ~{int(target_size)}px. Rows: {rows}, Cols: {cols}, MaxDim: {max_dim}, OrigTile: {self.original_tile_size}, ScaleFactor: {scale_factor:.2f}")
                else:
                    scale_factor = 1.0 # Fallback if dimensions are invalid
            except AttributeError:
                 # Fallback if dim_room not set
                 print("Warning: dim_room not available for dynamic scaling, using default scale 1.0.", file=sys.stderr)
                 scale_factor = 1.0
        # else: scale_factor remains 1.0 for standard 'human' mode

        # Ensure images are loaded and scaled correctly for the current mode
        self._scale_images(scale_factor)

        # Get room dimensions and tile size
        try:
            (rows, cols) = self.dim_room
            tile_size = self.current_tile_size
        except AttributeError:
             # Handle case where reset hasn't been called yet or failed
             print("Warning: Environment dimensions not set, cannot render.", file=sys.stderr)
             return np.zeros((100, 100, 3), dtype=np.uint8) # Fallback

        screen_width, screen_height = cols * tile_size, rows * tile_size

        # Create surface to draw on
        surface = pygame.Surface((screen_width, screen_height))
        surface.fill((255, 255, 255)) # White background

        # Draw using custom images if loaded and scaled successfully
        if self.images_loaded and self.img_scaled:
            for r in range(rows):
                for c in range(cols):
                    state_val = self.room_state[r, c]
                    fixed_val = self.room_fixed[r, c]

                    char_to_draw = ' '
                    # Determine character based on state and fixed background
                    if state_val == 5: char_to_draw = '+' if fixed_val == 2 else '@'
                    elif state_val == 3: char_to_draw = '*'
                    elif state_val == 4: char_to_draw = '$'
                    elif fixed_val == 0: char_to_draw = '#'
                    elif fixed_val == 2: char_to_draw = '?'
                    # Floor (' ') is handled by default

                    # Blit the corresponding image
                    img_to_blit = None
                    if char_to_draw == '#': img_to_blit = self.img_scaled.get('wall')
                    elif char_to_draw == ' ': img_to_blit = self.img_scaled.get('floor')
                    elif char_to_draw == '?': img_to_blit = self.img_scaled.get('docker')
                    elif char_to_draw == '$': img_to_blit = self.img_scaled.get('box')
                    elif char_to_draw == '*': img_to_blit = self.img_scaled.get('box_docked')
                    elif char_to_draw == '@': img_to_blit = self.img_scaled.get('worker')
                    elif char_to_draw == '+': img_to_blit = self.img_scaled.get('worker_docked')

                    if img_to_blit:
                        surface.blit(img_to_blit, (c * tile_size, r * tile_size))
                    else:
                        # Fallback color if image missing (should not happen if loading ok)
                        pygame.draw.rect(surface, DEFAULT_COLOR, (c * tile_size, r * tile_size, tile_size, tile_size))
        else:
            # Fallback to simple colors if images failed loading/scaling
            for r in range(rows):
                 for c in range(cols):
                      state_val = self.room_state[r, c]
                      color = COLOR_MAPPING.get(state_val, DEFAULT_COLOR)
                      pygame.draw.rect(surface, color, (c * tile_size, r * tile_size, tile_size, tile_size))

        if 'human' in mode:
            if self.screen is None:
                pygame.display.init()
                self.screen = pygame.display.set_mode((screen_width, screen_height))
                pygame.display.set_caption("Sokoban")
            if self.clock is None:
                self.clock = pygame.time.Clock()

            self.screen.blit(surface, (0, 0))
            pygame.event.pump() # Process Pygame events
            pygame.display.flip() # Update the display
            self.clock.tick(self.metadata['render_fps']) # Control frame rate
            return None # Human mode doesn't return the array

        elif 'rgb_array' in mode:
            # Convert the Pygame surface to a NumPy array
            if surface is None:
                 # Handle case where surface creation failed
                 print("Warning: Render surface is None, cannot return rgb_array.", file=sys.stderr)
                 if hasattr(self, 'observation_space'):
                      return np.zeros(self.observation_space.shape, dtype=np.uint8)
                 else:
                      return np.zeros((1,1,3), dtype=np.uint8)
            try:
                 # Transpose needed for gym: (width, height, channels) -> (height, width, channels)
                 return pygame.surfarray.array3d(surface).transpose(1, 0, 2)
            except IndexError:
                 # surfarray can fail if surface is 0-size.
                 print("Warning: pygame.surfarray.array3d failed (potential 0-size surface).", file=sys.stderr)
                 if hasattr(self, 'observation_space'):
                      return np.zeros(self.observation_space.shape, dtype=np.uint8)
                 else:
                      return np.zeros((1,1,3), dtype=np.uint8)
        else:
             super().render() # Call parent render for any other modes it might support

    def step(self, action):
        """
        Steps the environment using the given action.
        Overrides the parent step method to ensure compatibility with the
        5-value return tuple (observation, reward, terminated, truncated, info)
        expected by Gymnasium.
        """
        # gym-sokoban's step method already returns 5 values:
        # observation, self.reward_last, done, truncated, info

        assert self.action_space.contains(action), f"Invalid action: {action}"

        # Call the inherited step method directly
        try:
            observation, reward, terminated, truncated, info = super().step(action)

            # Manually add the correct boxes_on_target count to the info dict
            # The parent calculates it in _calc_reward and stores it in self.boxes_on_target
            info['boxes_on_target'] = self.boxes_on_target 

            # Add current step count to info dictionary
            info['steps'] = self.num_env_steps

            return observation, reward, terminated, truncated, info

        except ValueError as e:
             # Handle potential unpack error if parent *does* return 4 values unexpectedly
             if "not enough values to unpack" in str(e):
                 print("Warning: Parent SokobanEnv.step() returned 4 values. Adapting.", file=sys.stderr)
                 observation, reward, done, info = super().step(action)
                 # Determine terminated/truncated based on 'done' and parent's info key
                 truncated = info.get('maxsteps_used', False)
                 terminated = done and not truncated
                 info['steps'] = self.num_env_steps
                 return observation, reward, terminated, truncated, info
             else:
                 raise e # Re-raise other ValueErrors
        except Exception as e:
             # Catch any other unexpected error during parent step
             print(f"Error during parent step execution: {e}", file=sys.stderr)
             raise e

    def close(self):
        """Closes the environment and Pygame window."""
        # Only quit pygame if it was initialized
        if pygame.get_init():
            try:
                display_initialized = hasattr(self, 'screen') and self.screen is not None
                if not display_initialized:
                     # Check if display was initialized independently
                     display_initialized = pygame.display.get_init()

                if display_initialized:
                     pygame.display.quit()
                pygame.quit()
            except Exception as e:
                 print(f"Error closing pygame: {e}", file=sys.stderr)

        # Ensure attributes are reset even if pygame closing failed
        self.screen = None
        self.clock = None
        # Consider calling super().close() if the parent has cleanup logic