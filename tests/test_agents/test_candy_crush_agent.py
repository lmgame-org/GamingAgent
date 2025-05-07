from tile_match_gym.tile_match_env import TileMatchEnv
import numpy as np
from collections import OrderedDict
import os
import json
import argparse
import time
from datetime import datetime
import asyncio
from PIL import Image

# Import our custom modules for Candy Crush
from tests.test_agents.modules.candy_crush_modules import (
    CandyCrushPerceptionModule,
    CandyCrushMemoryModule,
    CandyCrushReasoningModule,
    CandyCrushBaseModule
)

# Create cache directory for this run
CACHE_DIR = os.path.join("cache", "candy_crush_experiments", datetime.now().strftime("%Y%m%d_%H%M%S"))
BOARD_IMG_PATH = os.path.join(CACHE_DIR, "board_latest.png")
GAME_LOG_FILE = os.path.join(CACHE_DIR, "game_log.jsonl")
DATA_LOG_FILE = os.path.join(CACHE_DIR, "data_log.jsonl")
MEMORY_FILE = os.path.join(CACHE_DIR, "memory.json")
os.makedirs(CACHE_DIR, exist_ok=True)

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
        "move": action_plan.get("move", "unknown"),
        "action_index": int(action) if action is not None else None,
        "thought": action_plan.get("thought", ""),
        "reward": float(reward),
        "score": info.get('score', 0),
        "info": {k: v for k, v in info.items() if isinstance(v, (int, float, str, bool, list, dict))},
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

class CandyCrushAgent:
    """A Candy Crush agent that uses either base module or perception+memory+reasoning modules."""
    
    def __init__(self, model_name="claude-3-7-sonnet-latest", use_base_module=False):
        """Initialize the agent with either base module or full pipeline."""
        self.model_name = model_name
        self.use_base_module = use_base_module
        self.last_action = None
        
        if use_base_module:
            print(f"Using Base Module with model: {model_name}")
            self.base_module = CandyCrushBaseModule(model_name=model_name)
        else:
            print(f"Using full pipeline (Perception + Memory + Reasoning) with model: {model_name}")
            self.perception_module = CandyCrushPerceptionModule(model_name=model_name)
            self.memory_module = CandyCrushMemoryModule(
                model_name=model_name,
                memory_file=MEMORY_FILE
            )
            self.reasoning_module = CandyCrushReasoningModule(model_name=model_name)
    
    async def get_action(self, observation, info=None, max_retries_const=1):
        """Get the next action from the agent."""
        try:
            if self.use_base_module:
                # Base module: Use RGB image directly if available, otherwise use info
                print("\n" + "="*80)
                action_plan = self.base_module.process_observation(observation, info)
                print("\nACTION PLAN:")
                print(f"Action: {action_plan['move']}")
                print(f"Thought: {action_plan['thought']}")
                print("="*80 + "\n")
                self.last_action = action_plan["move"]
                return action_plan, None, None
            else:
                # Full pipeline approach
                # Step 1: Perception - Analyze the game board
                perception_data = self.perception_module.analyze_board(observation, info)
                
                # Print perception data
                print("\n" + "="*80)
                print("PERCEPTION DATA:")
                print(f"Potential matches: {len(perception_data.get('potential_matches', []))}")
                print(f"Highest color: {perception_data.get('highest_color', 'Unknown')}")
                print(f"Empty spaces: {len(perception_data.get('empty_spaces', []))}")
                

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
                        await asyncio.sleep(1)  # Short delay before retry
                
 
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
                        img_path=BOARD_IMG_PATH,
                        max_retries=3  # Handle retries at this level, not inside plan_action
                    )
                    
                    if action_plan is None or 'move' not in action_plan:
                        retry_count += 1
                        print(f"Action planning attempt {retry_count}/{max_retries} failed. Retrying...")
                        await asyncio.sleep(1)  # Short delay before retry
                
                # If still no valid action plan after retries, create a fallback
                if action_plan is None or 'move' not in action_plan:
                    print("All reasoning attempts failed. Using fallback action.")
                    # Try to get a match from perception module if available
                    if perception_data and 'potential_matches' in perception_data and perception_data['potential_matches']:
                        match = perception_data['potential_matches'][0]
                        fallback_move = (match['coord1'], match['coord2'])
                    else:
                        # Default fallback
                        fallback_move = ((0, 0), (0, 1))
                    
                    action_plan = {
                        "move": fallback_move,
                        "thought": "Fallback action after failed reasoning attempts"
                    }
                
                # Print action plan
                print("\nACTION PLAN:")
                print(f"Action: {action_plan['move']}")
                print(f"Thought: {action_plan['thought']}")
                print("="*80 + "\n")
                
                # Store this action for the next iteration
                self.last_action = action_plan["move"]
                
                return action_plan, perception_data, memory_summary
        
        except Exception as e:
            print(f"Error in get_action: {e}")
            # Return a fallback action on error
            fallback_move = ((0, 0), (0, 1))
            self.last_action = fallback_move
            return {
                "move": fallback_move, 
                "thought": f"Error occurred: {str(e)}"
            }, None, None
    
    def parse_move_to_action(self, move, env):
        """Convert move coordinates to action index for the environment."""
        if not move or len(move) != 2:
            print(f"Invalid move format: {move}")
            return None
            
        try:
            coord1, coord2 = move
            # Call coords_to_action_index to get the action index
            action_idx = coords_to_action_index(coord1, coord2, env)
            
            if action_idx is None:
                print(f"Could not find action index for coordinates: {move}")
                # Fall back to a random valid action
                action_idx = env.action_space.sample()
            
            return action_idx
            
        except Exception as e:
            print(f"Error parsing move to action: {e}")
            return None

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
            "cache_dir": CACHE_DIR
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
        print(f"\n{'*' * 80}")
        if args.base:
            print(f"RUNNING IN BASE MODULE MODE - Using model: {args.model}")
            print("Base module uses vision API to analyze rendered game images")
        else:
            print(f"RUNNING IN FULL PIPELINE MODE - Using model: {args.model}")
            print("Full pipeline uses Perception → Memory → Reasoning modules")
        print(f"{'*' * 80}\n")
    else:
        print("\n" + "*" * 80)
        print("RUNNING WITH RANDOM ACTIONS - No AI model used")
        print("*" * 80 + "\n")
    
    # Run for the specified number of games
    for run_id in range(1, args.num_runs + 1):
        print(f"\n========== STARTING RUN {run_id}/{args.num_runs} ==========\n")
        
        # Create the Candy Crush environment with appropriate render mode
        env = TileMatchEnv(
            num_rows=8, 
            num_cols=8, 
            num_colours=4, 
            num_moves=50, 
            colourless_specials=[], 
            colour_specials=[], 
            seed=run_id,
            render_mode="human"
        )
        
        # Create the observation wrapper
        wrapped_env = CandyCrushObservationWrapper(env)
        
        # Reset the environment
        obs, _ = env.reset()
        wrapped_obs = wrapped_env.observation(obs)
        
        # Make sure rendering window appears
        env.render()
        
        # Print initial board state
        print("Initial board state:")
        board_2d = convert_obs_to_2d_array(obs)
        print_board(board_2d)
        
        # Run the game loop
        done = False
        terminated = False
        truncated = False
        total_reward = 0
        step = 0
        
        # Max attempts to get a valid action
        max_retries = 3
        
        # Track when game is stuck
        stuck_count = 0
        max_stuck_actions = 50  # Maximum number of unchanged board states before terminating
        
        while not done:
            # For random actions
            if args.random:
                # Sample a random action
                action = env.action_space.sample()
                # Create a basic action plan for logging
                action_plan = {
                    "move": env._action_to_coords[action],
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
                    if args.base:
                        # For base agent, we need a rendered frame image
                        try:
                            # Render the game to get the frame
                            frame = env.render()
                            
                            # If we didn't get a valid frame, use a fallback
                            if frame is None or not isinstance(frame, np.ndarray):
                                # Create a basic color representation of the board
                                board_2d = convert_obs_to_2d_array(obs)
                                height, width = board_2d.shape
                                frame = np.zeros((height*50, width*50, 3), dtype=np.uint8)
                                
                                # Fill with colors based on board values
                                colors = {
                                    0: (200, 200, 200),  # Empty - gray
                                    1: (0, 255, 0),      # Green
                                    2: (0, 255, 255),    # Cyan
                                    3: (255, 0, 255),    # Purple
                                    4: (255, 0, 0),      # Red
                                }
                                
                                for r in range(height):
                                    for c in range(width):
                                        color_idx = int(board_2d[r, c])
                                        color = colors.get(color_idx, (128, 128, 128))
                                        # Fill a square
                                        frame[r*50:(r+1)*50, c*50:(c+1)*50] = color
                            
                            # Pass the frame along with the image path in info
                            modified_info = dict(obs)
                            modified_info['img_path'] = BOARD_IMG_PATH
                            
                            # Ensure the image is saved before passing to the base module
                            os.makedirs(os.path.dirname(BOARD_IMG_PATH), exist_ok=True)
                            Image.fromarray(frame).save(BOARD_IMG_PATH)
                            print(f"Saved board image to {BOARD_IMG_PATH}")
                            
                            action_plan, perception_data, memory_summary = await agent.get_action(frame, modified_info)
                        except Exception as e:
                            print(f"Error getting frame for base module: {e}")
                            # Fallback to default action
                            action_plan = {
                                "move": ((0, 0), (0, 1)),
                                "thought": f"Error getting frame: {str(e)}"
                            }
                            perception_data = None
                            memory_summary = None
                    else:
                        # For full pipeline, pass the observation
                        action_plan, perception_data, memory_summary = await agent.get_action(obs["board"], obs)
                    
                    # Convert move to action index
                    action = agent.parse_move_to_action(action_plan["move"], env)
                    
                    # Check if we got a valid action
                    if action is not None:
                        valid_action = True
                    else:
                        print(f"Skipping action attempt {retry_count + 1}/{max_retries}")
                        retry_count += 1
                        # Wait a moment before trying again
                        await asyncio.sleep(1)
                
                # If we still don't have a valid action after retries, use a random action
                if not valid_action:
                    print("Exhausted retry attempts, using random action")
                    action = env.action_space.sample()
                    # Update action_plan for logging
                    action_plan["move"] = env._action_to_coords[action]
                    action_plan["thought"] += " (Random action after retries)"
            
            # Store the current board state before taking action
            current_board = None
            if 'board' in obs:
                current_board = obs['board'][0].copy()  # First channel is tile colors
            
            # Take the action in the environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Explicitly render the environment to ensure the window is shown
            env.render()
            
            # Convert to wrapped observation
            wrapped_next_obs = wrapped_env.observation(next_obs)
            
            # Check if the board state changed after the action
            if 'board' in next_obs and current_board is not None:
                new_board = next_obs['board'][0]
                if np.array_equal(current_board, new_board) and reward == 0:
                    stuck_count += 1
                    print(f"Board unchanged after action. Stuck count: {stuck_count}/{max_stuck_actions}")
                    if stuck_count >= max_stuck_actions:
                        print(f"Board hasn't changed for {max_stuck_actions} consecutive actions. Terminating run.")
                        done = True
                else:
                    # Reset stuck counter if board has changed
                    stuck_count = 0
            
            # Log comprehensive information with run_id
            log_step(step, action_plan, perception_data, memory_summary, action, reward, info, next_obs, run_id)
            
            # Log raw data to separate file with run_id
            log_raw_data(step, action, reward, info, next_obs, run_id)
            
            total_reward += reward
            step += 1
            
            # Print board state after action
            print(f"\nBoard after move (step {step}):")
            board_2d = convert_obs_to_2d_array(next_obs)
            print_board(board_2d)
            print(f"Action: {action}, Reward: {reward}, Total Reward: {total_reward}, Moves left: {next_obs['num_moves_left']}")
            
            # Check if game is done
            done = done or terminated or truncated
            
            # Update current observation
            obs = next_obs
        
        # Close the environment
        env.close()
        
        # Get final score
        all_scores.append(total_reward)  # Store total reward as the score
        
        # Log run summary
        run_summary = {
            "run_id": int(run_id),
            "score": float(total_reward),
            "steps": int(step),
            "timestamp": datetime.now().isoformat(),
            "termination_reason": "Normal completion" if stuck_count < max_stuck_actions else f"Terminated due to board being stuck for {max_stuck_actions} actions"
        }
        
        with open(summary_file, 'a') as f:
            json.dump(run_summary, f)
            f.write('\n')
        
        # Print final results for this run
        termination_reason = "Normal completion" if stuck_count < max_stuck_actions else f"Terminated due to board being stuck for {max_stuck_actions} actions"
        
        print(f"\nRun {run_id} finished after {step} steps")
        print(f"Total reward: {total_reward}")
        print(f"Termination reason: {termination_reason}")
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
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    scores = asyncio.run(run_agent(args))
    if len(scores) == 1:
        print(f"Final total reward: {scores[0]}")
    else:
        print(f"Average total reward: {sum(scores) / len(scores):.2f}")