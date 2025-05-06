from tile_match_gym.tile_match_env import TileMatchEnv
import numpy as np
from collections import OrderedDict

class CandyCrushObservationWrapper:
    """Observation wrapper for Candy Crush environment."""
    
    def __init__(self, env, num_colours=4):
        self.env = env
        self.num_rows = env.num_rows
        self.num_cols = env.num_cols
        self.num_colours = num_colours
        self.global_num_colourless_specials = 0  # Update as needed
        self.global_num_colour_specials = 0  # Update as needed
        self.num_type_slices = 0  # Update as needed
        self.type_slices = []  # Update as needed

    def observation(self, obs) -> dict:
        """Apply one-hot encoding to the observation."""
        board = obs["board"]
        ohe_board = self._one_hot_encode_board(board)
        return OrderedDict([("board", ohe_board), ("num_moves_left", obs["num_moves_left"])])
    
    def _one_hot_encode_board(self, board: np.ndarray) -> np.ndarray:
        """One-hot encode the board."""
        tile_colours = board[0]
        rows, cols = np.indices(tile_colours.shape)
        colour_ohe = np.zeros((1 + self.num_colours, self.num_rows, self.num_cols)) # Remove colourless slice after encoding
        colour_ohe[tile_colours.flatten(), rows.flatten(), cols.flatten()] = 1
        ohe_board = colour_ohe[1:]

        # Only keep the types for the specials that are in the environment (absence of any 1 means ordinary)
        if self.num_type_slices > 0:
            tile_types = board[1] + self.global_num_colourless_specials
            type_ohe = np.zeros((2 + self.global_num_colour_specials + self.global_num_colourless_specials, self.num_rows, self.num_cols)) # +1 for ordinary, +1 for empty
            type_ohe[tile_types.flatten(), rows.flatten(), cols.flatten()] = 1
            type_ohe = type_ohe[self.type_slices]
            ohe_board = np.concatenate([ohe_board, type_ohe], axis=0) # 1 + num_colours + num_colourless_specials + num_colour_specials.
        
        return ohe_board

def decode_ohe_board(ohe_board, num_colours=4):
    """
    Decode a one-hot encoded board back to a 2D array of color indices.
    
    Args:
        ohe_board: One-hot encoded board from the observation wrapper
        num_colours: Number of colors in the game
        
    Returns:
        2D array with color indices
    """
    # If ohe_board has shape [num_colours, rows, cols]
    if len(ohe_board.shape) == 3:
        # Get the indices of the maximum values along the color dimension
        # This will tell us which color is present at each position
        decoded_board = np.zeros((ohe_board.shape[1], ohe_board.shape[2]), dtype=int)
        
        # For each position, find which color channel has a 1
        for r in range(ohe_board.shape[1]):
            for c in range(ohe_board.shape[2]):
                for color in range(num_colours):
                    if color < ohe_board.shape[0] and ohe_board[color, r, c] == 1:
                        decoded_board[r, c] = color + 1  # +1 because colors start at 1
                        break
        
        return decoded_board
    
    # If the shape is not as expected, return the input
    return ohe_board

def convert_obs_to_2d_array(obs):
    """Convert observation to a more readable 2D array showing the tile colors."""
    # Extract the board from the observation
    board = obs["board"]
    
    # If it's already a 2D array (raw observation)
    if len(board.shape) == 3 and board.shape[0] <= 2:
        # The first channel contains tile colors
        return board[0].astype(int)
    
    # If it's a one-hot encoded board from the wrapper
    return decode_ohe_board(board)

def print_board(board):
    """Print the board in a grid format with column and row numbers."""
    rows, cols = board.shape
    
    # Print column numbers
    print("  ", end="")
    for c in range(cols):
        print(f"{c}", end=" ")
    print("\n  " + "-" * (cols * 2))
    
    # Create color representation
    color_names = {0: "E", 1: "G", 2: "C", 3: "P", 4: "R", 5: "T", 6: "B"}
    
    # Print rows with row numbers
    for r in range(rows):
        print(f"{r}|", end="")
        for c in range(cols):
            color_id = board[r, c]
            color_symbol = color_names.get(color_id, str(color_id))
            print(f"{color_symbol:2}", end="")
        print()
    print()

def coords_to_action_index(coord1, coord2, env):
    """
    Convert a pair of coordinates to the corresponding action index.
    
    Args:
        coord1: First coordinate tuple (row, col)
        coord2: Second coordinate tuple (row, col)
        env: Environment with _action_to_coords mapping
        
    Returns:
        Action index (integer) corresponding to the coordinates
    """
    # Make sure the coordinates are tuples
    coord1 = tuple(coord1)
    coord2 = tuple(coord2)
    
    # Sort the coordinates to match the order in the action space
    if coord1 > coord2:
        coord1, coord2 = coord2, coord1
        
    # Look up the action index
    for action, (c1, c2) in enumerate(env._action_to_coords):
        if (c1, c2) == (coord1, coord2):
            return action
    
    # If no match is found
    return None

# Create the environment
env = TileMatchEnv(
  num_rows=8, 
  num_cols=8, 
  num_colours=4, 
  num_moves=30, 
  colourless_specials=[], 
  colour_specials=[], 
  seed=2,
  render_mode="human"
)

# Create the observation wrapper
wrapped_env = CandyCrushObservationWrapper(env)

# Reset the environment
obs, _ = env.reset()
print("Initial board (raw):")
board_2d = convert_obs_to_2d_array(obs)
print_board(board_2d)

# Convert to wrapped observation
wrapped_obs = wrapped_env.observation(obs)
print("\nWrapped observation:")
wrapped_board_2d = convert_obs_to_2d_array(wrapped_obs)
print_board(wrapped_board_2d)
print(f"Moves left: {wrapped_obs['num_moves_left']}")

print(f"Action space size: {len(env._action_to_coords)}")


print(env._action_to_coords)
# Test the coord-to-action mapping
test_coords = [(0, 0), (1, 0)]
action_idx = coords_to_action_index(test_coords[0], test_coords[1], env)
print(f"Coordinates {test_coords} map to action index {action_idx}")

test_coords = [(3, 4), (3, 5)]
action_idx = coords_to_action_index(test_coords[0], test_coords[1], env)
print(f"Coordinates {test_coords} map to action index {action_idx}")

# Game loop
while True:
    action = env.action_space.sample()
    coord1, coord2 = env._action_to_coords[action]
    print(f"\nSwapping tiles at {coord1} and {coord2}")
    print(f"Action: {action}")
    
    # Take a step in the environment
    next_obs, reward, done, truncated, info = env.step(action)
    
    # Get raw board representation
    raw_board_2d = convert_obs_to_2d_array(next_obs)
    print("Board after move (raw):")
    print_board(raw_board_2d)
    
    # Convert to wrapped observation
    wrapped_next_obs = wrapped_env.observation(next_obs)
    wrapped_board_2d = convert_obs_to_2d_array(wrapped_next_obs)
    print("Board after move (wrapped):")
    print_board(wrapped_board_2d)
    
    print(f"Reward: {reward}, Moves left: {next_obs['num_moves_left']}")
    
    env.render()
    if done:
        print("\nGame over!")
        break
    else:
      obs = next_obs