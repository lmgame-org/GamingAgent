import gymnasium as gym
import numpy as np
import os
import json
import argparse
import time
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import asyncio

# Import our agent modules
from tests.test_agents.modules.base_module import Base_module
from tests.test_agents.modules.perception import PerceptionModule
from tests.test_agents.modules.memory import MemoryModule
from tests.test_agents.modules.reasoning import ReasoningModule
from tools.serving import APIManager

# Create cache directory for this run
CACHE_DIR = os.path.join("cache", "2048_experiments", datetime.now().strftime("%Y%m%d_%H%M%S"))
BOARD_IMG_PATH = os.path.join(CACHE_DIR, "board_latest.png")
GAME_LOG_FILE = os.path.join(CACHE_DIR, "game_log.jsonl")
DATA_LOG_FILE = os.path.join(CACHE_DIR, "data_log.jsonl")
MEMORY_FILE = os.path.join(CACHE_DIR, "memory.json")
os.makedirs(CACHE_DIR, exist_ok=True)

# Mapping of moves to action indices for the environment
move_to_action = {
    "up": 0,
    "right": 1, 
    "down": 2,
    "left": 3
}

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

class Agent2048:
    """A simple 2048 agent that uses either base module or perception+memory+reasoning modules."""
    
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
    
    async def get_action(self, observation, info=None, max_retries_const=1):
        """Get the next action from the agent."""
        try:
            if self.use_base_module:
                # Base module: Use RGB image directly
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
                # Step 1: Perception - Simply convert info["board"] to actual 2048 values
                perception_data = self.perception_module.analyze_board(observation, info)
                
                # Print perception data
                print("\n" + "="*80)
                print("PERCEPTION DATA:")
                print(f"Highest tile: {perception_data.get('highest_tile', 'Unknown')} (2^{perception_data.get('highest_tile_power', 0)})")
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
                    # Call the async plan_action method directly
                    action_plan = await self.reasoning_module.plan_action(
                        current_perception=perception_data,
                        memory_summary=memory_summary,
                        img_path=BOARD_IMG_PATH,
                        max_retries=3  # Handle retries at this level, not inside plan_action
                    )
                    
                    if action_plan is None or 'move' not in action_plan:
                        retry_count += 1
                        print(f"Action planning attempt {retry_count}/{max_retries} failed. Retrying...")
                        await asyncio.sleep(2)  # Short delay before retry
                
                # If still no valid action plan after retries, create a fallback
                if action_plan is None or 'move' not in action_plan:
                    print("All reasoning attempts failed. Using fallback action.")
                    action_plan = {
                        "move": "skip",  # Simple fallback
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
            self.last_action = "skip"
            return {
                "move": "skip", 
                "thought": f"Error occurred: {str(e)}"
            }, None, None
    
    def parse_move_to_action(self, move):
        """Convert move string to action index for the environment."""
        move_str = move.lower() if isinstance(move, str) else "skip"
        
        # Only return a valid action if the move is in our mapping
        if move_str in move_to_action:
            return move_to_action[move_str]
        
        # For skip or any invalid move, return None to indicate no valid action
        print(f"Invalid or skip action: {move_str}")
        return None

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
    
    # Save the latest board image for reference (if info contains board)
    if info and 'board' in info:
        try:
            # Create board image from info['board']
            board = info['board']
            create_board_image(board, BOARD_IMG_PATH)
        except Exception as e:
            print(f"Warning: Could not save board image: {e}")

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

def create_board_image(board, save_path, size=400):
    """Create a visualization of the 2048 board."""
    cell_size = size // 4
    padding = cell_size // 10
    
    # Create a new image with a beige background
    img = Image.new('RGB', (size, size), (250, 248, 239))
    draw = ImageDraw.Draw(img)
    
    # Color mapping for different tile values
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
    }
    
    # Text colors
    dark_text = (119, 110, 101)  # For small values
    light_text = (249, 246, 242) # For large values
    
    try:
        # Use default font
        font = ImageFont.load_default()
        
        # Draw each cell
        for row in range(4):
            for col in range(4):
                # Get power value and convert to actual 2048 value
                power = int(board[row][col])
                value = 0 if power == 0 else 2**power
                
                # Calculate position
                x0 = col * cell_size + padding
                y0 = row * cell_size + padding
                x1 = (col + 1) * cell_size - padding
                y1 = (row + 1) * cell_size - padding
                
                # Draw cell background
                cell_color = colors.get(value, (60, 58, 50))  # Default to dark color for large values
                draw.rectangle([x0, y0, x1, y1], fill=cell_color)
                
                # Skip text for empty cells
                if value == 0:
                    continue
                
                # Choose text color based on value
                text_color = light_text if value > 4 else dark_text
                
                # Draw the value
                text = str(value)
                
                # Simple centering of text with larger size estimation
                cell_width = x1 - x0
                cell_height = y1 - y0
                
                # Increase effective text size by adjusting the spacing
                # Estimate text width based on character count and make it centered
                text_width = len(text) * 8  # Larger estimate for character width
                text_x = x0 + (cell_width - text_width) // 2
                text_y = y0 + (cell_height - 16) // 2  # Even larger estimate for text height
                
                # Draw text (default font) - just once, no bolding effect
                # Draw text twice with slight offset to increase the apparent size
                draw.text((text_x, text_y), text, fill=text_color, font=font)
                draw.text((text_x+1, text_y), text, fill=text_color, font=font)
        
        # Save the image
        img.save(save_path)
        
    except Exception as e:
        print(f"Error creating board image: {e}")

async def run_agent(args):
    """Run the 2048 agent with the specified parameters."""
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
        agent = Agent2048(
            model_name=args.model,
            use_base_module=args.base
        )
    else:
        print("Running with random actions")
    
    # Run for the specified number of games
    for run_id in range(1, args.num_runs + 1):
        print(f"\n========== STARTING RUN {run_id}/{args.num_runs} ==========\n")
        
        # Create the 2048 environment with appropriate render mode
        if args.base:
            # For base agent: Need RGB array to save images and pass to vision model
            render_mode = "rgb_array"
        else:
            # For full pipeline: Human rendering for visualization
            render_mode = "human"
        
        env = gym.make("gymnasium_2048:gymnasium_2048/TwentyFortyEight-v0", 
                     size=4, max_pow=16, render_mode=render_mode)
        
        # Reset the environment
        observation, info = env.reset()
        
        # Run the game loop
        done = False
        total_reward = 0
        step = 0
        
        # Max attempts to get a valid action
        max_retries = 3
        
        # Track when game is stuck
        stuck_count = 0
        max_stuck_actions = 10  # Maximum number of unchanged board states before terminating
        
        while not done:
            # Save board image if needed
            if args.base or args.random:
                try:
                    # Save the current frame (for vision API or documentation)
                    frame = env.render()
                    if frame is not None:
                        # Save to the board image path
                        Image.fromarray(frame).save(BOARD_IMG_PATH)
                except Exception as e:
                    print(f"Warning: Could not save frame: {e}")
            
            # For random actions
            if args.random:
                # Sample a random action
                action = env.action_space.sample()
                # Create a basic action plan for logging
                action_plan = {
                    "move": list(move_to_action.keys())[list(move_to_action.values()).index(action)],
                    "thought": "Random action"
                }
                perception_data = None
                memory_summary = None
            else:
                # Get action from agent with retry logic for skips
                retry_count = 0
                valid_action = False
                
                while not valid_action and retry_count < max_retries:
                    # Get action from the agent
                    if args.base:
                        # For base agent, pass the RGB frame as observation
                        frame = env.render()
                        action_plan, perception_data, memory_summary = await agent.get_action(frame, info)
                    else:
                        # For full pipeline, pass the observation and info
                        action_plan, perception_data, memory_summary = await agent.get_action(observation, info)
                    
                    # Convert move to action index
                    action = agent.parse_move_to_action(action_plan["move"])
                    
                    # Check if we got a valid action
                    if action is not None:
                        valid_action = True
                    else:
                        print(f"Skipping action attempt {retry_count + 1}/{max_retries}")
                        retry_count += 1
                        # Wait a moment before trying again
                        # This helps if the error is temporary (e.g., API rate limit)
                        await asyncio.sleep(1)
                
                # If we still don't have a valid action after retries, use a random action
                if not valid_action:
                    print("Exhausted retry attempts, using random action")
                    action = env.action_space.sample()
                    # Update action_plan for logging
                    action_plan["move"] = list(move_to_action.keys())[list(move_to_action.values()).index(action)]
                    action_plan["thought"] += " (Random action after retries)"
            
            # Store the current board state before taking action
            current_board = None
            if 'board' in info:
                current_board = info['board'].tolist()
            
            # Skip taking an action if the move was "skip"
            if action_plan["move"].lower() == "skip":
                print("SKIPPING THIS ACTION - continuing to next iteration")
                continue

            # Take the action in the environment
            observation, reward, terminated, truncated, info = env.step(action)
            
            # Check if the board state changed after the action
            if 'board' in info and current_board is not None:
                new_board = info['board'].tolist()
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
            log_step(step, action_plan, perception_data, memory_summary, action, reward, info, observation, run_id)
            
            # Log raw data to separate file with run_id
            log_raw_data(step, action, reward, info, observation, run_id)
            
            total_reward += reward
            step += 1
            
            # Check if game is done
            done = done or terminated or truncated
            
            # Print some information
            max_tile_power = info.get('max', 0)
            max_tile = 2 ** max_tile_power if max_tile_power > 0 else 0
            print(f"Run {run_id}, Step: {step}, Action: {action_plan['move']}, Reward: {reward}, Total Reward: {total_reward}, Max Tile: {max_tile}")
        
        # Close the environment
        env.close()
        
        # Get final score and max tile and ensure they're Python native types
        max_tile_power = int(info.get('max', 0))
        max_tile = 2 ** max_tile_power if max_tile_power > 0 else 0
        all_scores.append(total_reward)  # Store total reward as the score
        
        # Log run summary - ensure all values are Python native types
        run_summary = {
            "run_id": int(run_id),
            "score": float(total_reward),
            "steps": int(step),
            "max_tile": int(max_tile),
            "max_tile_power": int(max_tile_power),
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
        print(f"Max tile achieved: {max_tile} (2^{max_tile_power})")
        print(f"Termination reason: {termination_reason}")
        print(f"========== COMPLETED RUN {run_id}/{args.num_runs} ==========\n")
    
    # Print overall summary
    if args.num_runs > 1:
        # Calculate max tiles achieved across runs
        max_tiles = []
        for run_id in range(1, args.num_runs + 1):
            with open(summary_file, 'r') as f:
                for line in f:
                    summary = json.loads(line)
                    if summary.get('run_id') == run_id:
                        max_tiles.append(summary.get('max_tile', 0))
                        break
        
        print("\n===== EXPERIMENT SUMMARY =====")
        print(f"Total runs: {args.num_runs}")
        print(f"Average total reward: {sum(all_scores) / len(all_scores):.2f}")
        print(f"Highest total reward: {max(all_scores)}")
        print(f"Lowest total reward: {min(all_scores)}")
        print(f"Average max tile: {sum(max_tiles) / len(max_tiles):.2f}")
        print(f"Highest max tile: {max(max_tiles)}")
        print(f"All total rewards: {all_scores}")
    
    return all_scores

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run the 2048 AI Agent')
    
    parser.add_argument('--model', type=str, default="claude-3-7-sonnet-latest",
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