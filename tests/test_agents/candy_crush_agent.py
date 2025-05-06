from tile_match_gym.tile_match_env import TileMatchEnv
import numpy as np
import os
import json
import argparse
import time
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import asyncio
from collections import OrderedDict

# Import our agent modules
from tests.test_agents.modules.base_module import Base_module
from tests.test_agents.modules.perception import PerceptionModule
from tests.test_agents.modules.memory import MemoryModule
from tests.test_agents.modules.reasoning import ReasoningModule
from tools.serving import APIManager

# Create cache directory for this run
CACHE_DIR = os.path.join("cache", "candy_crush_experiments", datetime.now().strftime("%Y%m%d_%H%M%S"))
BOARD_IMG_PATH = os.path.join(CACHE_DIR, "board_latest.png")
GAME_LOG_FILE = os.path.join(CACHE_DIR, "game_log.jsonl")
DATA_LOG_FILE = os.path.join(CACHE_DIR, "data_log.jsonl")
MEMORY_FILE = os.path.join(CACHE_DIR, "memory.json")
os.makedirs(CACHE_DIR, exist_ok=True)

# Mapping of color names
COLOR_NAMES = {0: "E", 1: "G", 2: "C", 3: "P", 4: "R"}

# Helper function to convert NumPy types to Python native types for JSON serialization
def convert_numpy_types(obj):
    """Convert NumPy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [convert_numpy_types(x) for x in obj.tolist()]
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

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

class CandyCrushAgent:
    """Agent for playing Candy Crush style games."""
    
    def __init__(self, model_name="claude-3-7-sonnet-latest", use_base_module=False):
        """Initialize the agent with either base module or full pipeline."""
        self.model_name = model_name
        self.use_base_module = use_base_module
        self.last_action = None
        
        if use_base_module:
            print(f"Using Base Module with model: {model_name}")
            self.base_module = Base_module(model_name=model_name)
        else:
            print(f"Using full pipeline (Perception + Memory + Reasoning) with model: {model_name}")
            self.perception_module = PerceptionModule(model_name=model_name)
            self.memory_module = MemoryModule(
                model_name=model_name,
                memory_file=MEMORY_FILE
            )
            self.reasoning_module = ReasoningModule(model_name=model_name)
    
    async def get_action(self, observation, env, info=None, max_retries_const=1):
        """Get the next action from the agent."""
        try:
            if self.use_base_module:
                # Base module: Create an image from the observation for the vision model
                board_2d = convert_obs_to_2d_array(observation)
                create_board_image(board_2d, BOARD_IMG_PATH)
                
                print("\n" + "="*80)
                action_plan = self.base_module.process_observation(board_2d, info)
                print("\nACTION PLAN:")
                print(f"Coordinates: {action_plan.get('coords', None)}")
                print(f"Thought: {action_plan['thought']}")
                print("="*80 + "\n")
                
                # Convert coordinates to action index
                if 'coords' in action_plan and len(action_plan['coords']) == 2:
                    coord1, coord2 = action_plan['coords']
                    action_index = coords_to_action_index(coord1, coord2, env)
                    action_plan['action_index'] = action_index
                
                self.last_action = action_plan.get('coords', None)
                return action_plan, None, None
            else:
                # Full pipeline approach
                # Step 1: Perception - Convert observation to 2D board
                board_2d = convert_obs_to_2d_array(observation)
                perception_data = {
                    "board": board_2d.tolist(),
                    "num_rows": board_2d.shape[0],
                    "num_cols": board_2d.shape[1],
                    "num_moves_left": observation.get("num_moves_left", 0),
                    "special_tiles": []  # Could be enhanced to detect special tiles
                }
                
                # Save image for visualization
                create_board_image(board_2d, BOARD_IMG_PATH)
                
                # Print perception data
                print("\n" + "="*80)
                print("PERCEPTION DATA:")
                print(f"Board shape: {board_2d.shape}")
                print(f"Moves left: {observation.get('num_moves_left', 'Unknown')}")
                print("Board state:")
                print_board(board_2d)
                
                # Step 2: Memory - Add to memory and generate reflection with retry logic
                max_retries = max_retries_const
                retry_count = 0
                memory_summary = None
                self.memory_module.add_game_state(perception_data, self.last_action)
                
                while memory_summary is None and retry_count < max_retries:
                    memory_summary = self.memory_module.get_memory_summary()
                    
                    if memory_summary is None or len(memory_summary) == 0:
                        retry_count += 1
                        print(f"Memory retrieval attempt {retry_count}/{max_retries} failed. Retrying...")
                        await asyncio.sleep(2)  # Short delay before retry
                
                # Print memory summary and reflection
                if memory_summary:
                    print("\nMEMORY SUMMARY:")
                    print(f"Memory entries: {len(memory_summary)}")
                    if len(memory_summary) > 0 and 'reflection' in memory_summary[-1]:
                        print(f"Latest reflection: {memory_summary[-1]['reflection']}")
                
                # Step 3: Reasoning - Plan the next action with retry logic
                max_retries = max_retries_const
                retry_count = 0
                action_plan = None
                
                while action_plan is None and retry_count < max_retries:
                    # Call the async plan_action method 
                    action_plan = await self.reasoning_module.plan_action(
                        current_perception=perception_data,
                        memory_summary=memory_summary,
                        img_path=None,  # No need for image since we're using text
                        max_retries=3
                    )
                    
                    if action_plan is None or 'coords' not in action_plan:
                        retry_count += 1
                        print(f"Action planning attempt {retry_count}/{max_retries} failed. Retrying...")
                        await asyncio.sleep(2)  # Short delay before retry
                
                # If still no valid action plan after retries, create a fallback
                if action_plan is None or 'coords' not in action_plan:
                    print("All reasoning attempts failed. Using fallback action.")
                    # Generate a random valid action as fallback
                    rows, cols = board_2d.shape
                    action_plan = {
                        "coords": [(np.random.randint(0, rows-1), np.random.randint(0, cols)), 
                                  (np.random.randint(0, rows-1), np.random.randint(0, cols))],
                        "thought": "Fallback action after failed reasoning attempts"
                    }
                
                # Convert coordinates to action index
                if 'coords' in action_plan and len(action_plan['coords']) == 2:
                    coord1, coord2 = action_plan['coords']
                    action_index = coords_to_action_index(coord1, coord2, env)
                    action_plan['action_index'] = action_index
                
                # Print action plan
                print("\nACTION PLAN:")
                print(f"Coordinates: {action_plan.get('coords', None)}")
                print(f"Action index: {action_plan.get('action_index', None)}")
                print(f"Thought: {action_plan['thought']}")
                print("="*80 + "\n")
                
                # Store this action for the next iteration
                self.last_action = action_plan.get('coords', None)
                
                return action_plan, perception_data, memory_summary
        
        except Exception as e:
            print(f"Error in get_action: {e}")
            # Return a fallback action on error
            rows, cols = observation["board"][0].shape if "board" in observation else (8, 8)
            fallback_coords = [(0, 0), (0, 1)]  # Simple fallback
            self.last_action = fallback_coords
            return {
                "coords": fallback_coords, 
                "thought": f"Error occurred: {str(e)}"
            }, None, None
    
    def parse_coords_to_action(self, coords, env):
        """Convert coordinates to action index for the environment."""
        if not coords or len(coords) != 2:
            return None
        
        try:
            coord1, coord2 = coords
            return coords_to_action_index(coord1, coord2, env)
        except Exception as e:
            print(f"Error converting coords to action: {e}")
            return None

def create_board_image(board, save_path, size=400):
    """Create a visualization of the Candy Crush board."""
    rows, cols = board.shape
    cell_size = size // max(rows, cols)
    padding = cell_size // 10
    
    # Create a new image with a dark background
    img = Image.new('RGB', (cols * cell_size, rows * cell_size), (50, 50, 50))
    draw = ImageDraw.Draw(img)
    
    # Color mapping for different tile values
    colors = {
        0: (100, 100, 100),    # Empty
        1: (100, 200, 100),    # Green
        2: (100, 200, 220),    # Cyan
        3: (180, 100, 200),    # Purple
        4: (220, 100, 100),    # Red
        5: (200, 200, 100),    # Yellow
        6: (200, 150, 100),    # Orange
    }
    
    try:
        # Use default font
        font = ImageFont.load_default()
        
        # Draw each cell
        for row in range(rows):
            for col in range(cols):
                # Get tile value
                value = int(board[row][col])
                
                # Calculate position
                x0 = col * cell_size + padding
                y0 = row * cell_size + padding
                x1 = (col + 1) * cell_size - padding
                y1 = (row + 1) * cell_size - padding
                
                # Draw cell background
                cell_color = colors.get(value, (200, 200, 200))
                draw.rectangle([x0, y0, x1, y1], fill=cell_color)
                
                # Draw the value/symbol
                text = COLOR_NAMES.get(value, str(value))
                
                # Center text
                cell_width = x1 - x0
                cell_height = y1 - y0
                text_width = len(text) * 8
                text_x = x0 + (cell_width - text_width) // 2
                text_y = y0 + (cell_height - 16) // 2
                
                # Draw text
                draw.text((text_x, text_y), text, fill=(255, 255, 255), font=font)
        
        # Save the image
        img.save(save_path)
        
    except Exception as e:
        print(f"Error creating board image: {e}")

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

def convert_obs_to_2d_array(obs):
    """Convert observation to a more readable 2D array showing the tile colors."""
    # Extract the board from the observation
    board = obs["board"]
    
    # If it's already a 2D array (raw observation)
    if len(board.shape) == 3 and board.shape[0] <= 2:
        # The first channel contains tile colors
        return board[0].astype(int)
    
    # If it's a one-hot encoded board
    return np.argmax(board, axis=0) if len(board.shape) == 3 else board

def print_board(board):
    """Print the board in a grid format with column and row numbers."""
    rows, cols = board.shape
    
    # Print column numbers
    print("  ", end="")
    for c in range(cols):
        print(f"{c}", end=" ")
    print("\n  " + "-" * (cols * 2))
    
    # Create color representation
    color_names = COLOR_NAMES
    
    # Print rows with row numbers
    for r in range(rows):
        print(f"{r}|", end="")
        for c in range(cols):
            color_id = board[r, c]
            color_symbol = color_names.get(color_id, str(color_id))
            print(f"{color_symbol:2}", end="")
        print()
    print()

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

def log_step(step, action_plan, perception_data, memory_summary, action, reward, info, observation, run_id=1):
    """Log comprehensive step information to a JSON Lines file."""
    # Convert numpy arrays and other types for JSON serialization
    if isinstance(observation, np.ndarray):
        observation = observation.tolist()
    
    # Create comprehensive log entry
    log_entry = {
        "run_id": run_id,
        "step": step,
        "timestamp": datetime.now().isoformat(),
        "coords": action_plan.get("coords", []),
        "action_index": int(action) if action is not None else None,
        "thought": action_plan.get("thought", ""),
        "reward": float(reward),
        "score": info.get('score', 0) if info else 0,
        "info": {k: v for k, v in info.items() if isinstance(v, (int, float, str, bool, list, dict))} if info else {},
        "perception_data": perception_data,
        "memory_summary": memory_summary[-1] if memory_summary and len(memory_summary) > 0 else None
    }
    
    # Convert any remaining NumPy types
    log_entry = convert_numpy_types(log_entry)
    
    # Write to log file
    with open(GAME_LOG_FILE, 'a') as f:
        json.dump(log_entry, f)
        f.write('\n')

def log_raw_data(step, action, reward, info, observation, run_id=1):
    """Log raw data from the environment to a separate JSON Lines file."""
    # Convert info dictionary for JSON serialization
    if isinstance(info, dict):
        # Make a copy to avoid modifying the original
        info_copy = {}
        for k, v in info.items():
            if isinstance(v, np.ndarray):
                info_copy[k] = v.tolist()
            elif isinstance(v, (int, float, str, bool, list, dict)):
                info_copy[k] = v
            else:
                info_copy[k] = str(v)
    else:
        info_copy = str(info)
    
    # Create simple data log entry
    data_entry = {
        "run_id": run_id,
        "step": step,
        "timestamp": datetime.now().isoformat(),
        "action": int(action) if action is not None else None,
        "reward": float(reward),
        "info": info_copy,
        "observation": observation.tolist() if isinstance(observation, np.ndarray) else observation
    }
    
    # Convert any remaining NumPy types
    data_entry = convert_numpy_types(data_entry)
    
    # Write to data log file
    with open(DATA_LOG_FILE, 'a') as f:
        json.dump(data_entry, f)
        f.write('\n')

async def run_agent(args):
    """Run the Candy Crush agent with the specified parameters."""
    # Write experiment info
    with open(os.path.join(CACHE_DIR, "experiment_info.json"), 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "model": args.model,
            "use_base_module": args.base,
            "use_random": args.random,
            "num_runs": args.num_runs,
            "cache_dir": CACHE_DIR,
            "env_config": {
                "num_rows": args.rows,
                "num_cols": args.cols,
                "num_colours": args.colours,
                "num_moves": args.moves
            }
        }, f, indent=2)
    
    # Create a summary file for tracking scores
    summary_file = os.path.join(CACHE_DIR, "runs_summary.jsonl")
    all_scores = []
    
    # Initialize the agent (shared across all runs)
    agent = None
    if not args.random:
        agent = CandyCrushAgent(
            model_name=args.model,
            use_base_module=args.base
        )
    else:
        print("Running with random actions")
    
    # Run for the specified number of games
    for run_id in range(1, args.num_runs + 1):
        print(f"\n========== STARTING RUN {run_id}/{args.num_runs} ==========\n")
        
        # Create the Candy Crush environment
        env = TileMatchEnv(
            num_rows=args.rows, 
            num_cols=args.cols, 
            num_colours=args.colours, 
            num_moves=args.moves, 
            colourless_specials=[], 
            colour_specials=[], 
            seed=args.seed + run_id - 1,  # Different seed for each run
            render_mode="human"
        )
        
        # Create the observation wrapper
        wrapped_env = CandyCrushObservationWrapper(env, num_colours=args.colours)
        
        # Reset the environment
        obs, _ = env.reset()
        wrapped_obs = wrapped_env.observation(obs)
        
        # Run the game loop
        done = False
        total_reward = 0
        step = 0
        
        # Max attempts to get a valid action
        max_retries = 3
        
        while not done:
            # For random actions
            if args.random:
                # Sample a random action
                action = env.action_space.sample()
                coord1, coord2 = env._action_to_coords[action]
                # Create a basic action plan for logging
                action_plan = {
                    "coords": [coord1, coord2],
                    "thought": "Random action"
                }
                perception_data = None
                memory_summary = None
            else:
                # Get action from agent with retry logic
                retry_count = 0
                valid_action = False
                
                while not valid_action and retry_count < max_retries:
                    # Get action from the agent
                    action_plan, perception_data, memory_summary = await agent.get_action(
                        observation=wrapped_obs, 
                        env=env, 
                        info={"num_moves_left": wrapped_obs["num_moves_left"]}
                    )
                    
                    # Convert coordinates to action index
                    if 'coords' in action_plan and len(action_plan['coords']) == 2:
                        action = agent.parse_coords_to_action(action_plan['coords'], env)
                        if action is not None:
                            valid_action = True
                        else:
                            print(f"Invalid coordinates: {action_plan['coords']}. Retrying...")
                            retry_count += 1
                    else:
                        print(f"No valid coordinates in action plan. Retrying...")
                        retry_count += 1
                
                # If we still don't have a valid action after retries, use a random action
                if not valid_action:
                    print("Exhausted retry attempts, using random action")
                    action = env.action_space.sample()
                    coord1, coord2 = env._action_to_coords[action]
                    action_plan = {
                        "coords": [coord1, coord2],
                        "thought": "Random action after failed attempts"
                    }
            
            # Take the action in the environment
            next_obs, reward, done, truncated, info = env.step(action)
            wrapped_next_obs = wrapped_env.observation(next_obs)
            
            # Log comprehensive information
            log_step(step, action_plan, perception_data, memory_summary, action, reward, info, wrapped_next_obs, run_id)
            
            # Log raw data to separate file
            log_raw_data(step, action, reward, info, wrapped_next_obs, run_id)
            
            # Update for next iteration
            obs = next_obs
            wrapped_obs = wrapped_next_obs
            total_reward += reward
            step += 1
            
            # Print progress
            print(f"Run {run_id}, Step: {step}, Action: {action}, Coords: {action_plan.get('coords')}, Reward: {reward}, Total Reward: {total_reward}, Moves left: {wrapped_obs['num_moves_left']}")
            
            # Done condition (moves exhausted or terminated)
            done = done or truncated or wrapped_obs["num_moves_left"] <= 0
        
        # Close the environment
        env.close()
        
        # Log run summary
        run_summary = {
            "run_id": int(run_id),
            "score": float(total_reward),
            "steps": int(step),
            "timestamp": datetime.now().isoformat()
        }
        
        with open(summary_file, 'a') as f:
            json.dump(run_summary, f)
            f.write('\n')
        
        # Store score for overall summary
        all_scores.append(total_reward)
        
        # Print run summary
        print(f"\nRun {run_id} finished after {step} steps")
        print(f"Total reward: {total_reward}")
        print(f"========== COMPLETED RUN {run_id}/{args.num_runs} ==========\n")
    
    # Print overall summary
    if args.num_runs > 1:
        print("\n===== EXPERIMENT SUMMARY =====")
        print(f"Total runs: {args.num_runs}")
        print(f"Average total reward: {sum(all_scores) / len(all_scores):.2f}")
        print(f"Highest total reward: {max(all_scores)}")
        print(f"Lowest total reward: {min(all_scores)}")
        print(f"All total rewards: {all_scores}")
    
    return all_scores

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run the Candy Crush AI Agent')
    
    parser.add_argument('--model', type=str, default="claude-3-5-sonnet-latest",
                        help='Model name to use (default: claude-3-7-sonnet-latest)')
    
    parser.add_argument('--base', action='store_true',
                        help='Use the simplified Base_module (default: False)')
    
    parser.add_argument('--random', action='store_true',
                        help='Use random actions instead of AI agent (default: False)')
    
    parser.add_argument('--num_runs', type=int, default=1,
                        help='Number of game runs to perform (default: 1)')
    
    parser.add_argument('--rows', type=int, default=8,
                        help='Number of rows in the board (default: 8)')
    
    parser.add_argument('--cols', type=int, default=8,
                        help='Number of columns in the board (default: 8)')
    
    parser.add_argument('--colours', type=int, default=4,
                        help='Number of different tile colors (default: 4)')
    
    parser.add_argument('--moves', type=int, default=30,
                        help='Number of moves per game (default: 30)')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for the environment (default: 42)')
    
    return parser.parse_args()

async def main():
    """Main entry point for running the agent."""
    args = parse_args()
    scores = await run_agent(args)
    if len(scores) == 1:
        print(f"Final total reward: {scores[0]}")
    else:
        print(f"Average total reward: {sum(scores) / len(scores):.2f}")

if __name__ == "__main__":
    asyncio.run(main()) 