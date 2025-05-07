import cv2
import gymnasium as gym
import numpy as np
import time
import os
import json
import argparse
from datetime import datetime
import asyncio
from PIL import Image

"""
Tetris Agent Runner with Simple Random Mode

This script runs the Tetris game with three possible modes:
1. Full pipeline mode (Perception + Memory + Reasoning)
2. Base module mode (simplified, using vision API)
3. Random action mode - uses action_space.sample() for testing

Usage:
    python test_tetris_agent.py [options]

Options:
    --model MODEL    Model name to use for AI agent (default: claude-3-7-sonnet-latest)
    --base           Use base module instead of full pipeline (default: False)
    --random         Use random actions instead of AI agent (default: False)
    --num_runs N     Number of game runs to perform (default: 1)

Examples:
    # Run with full pipeline AI agent (default)
    python test_tetris_agent.py

    # Run with random actions
    python test_tetris_agent.py --random
"""

from tetris_gymnasium.envs.tetris import Tetris

# Import our custom modules for Tetris
from tests.test_agents.modules.tetris_modules import (
    TetrisPerceptionModule,
    TetrisMemoryModule,
    TetrisReasoningModule,
    TetrisBaseModule,
    tetris_actions
)

# Create cache directory for this run
CACHE_DIR = os.path.join("cache", "tetris_experiments", datetime.now().strftime("%Y%m%d_%H%M%S"))
BOARD_IMG_PATH = os.path.join(CACHE_DIR, "board_latest.png")
GAME_LOG_FILE = os.path.join(CACHE_DIR, "game_log.jsonl")
DATA_LOG_FILE = os.path.join(CACHE_DIR, "data_log.jsonl")
MEMORY_FILE = os.path.join(CACHE_DIR, "memory.json")
os.makedirs(CACHE_DIR, exist_ok=True)

# Default fallback action
DEFAULT_ACTION = [7]  # no_operation

def print_board(board):
    """Print a simplified text version of the board for debugging"""
    # Create a copy to visualize, skipping border and showing only the game area
    # Assuming borders are at indices [0-3, 14-17] horizontally and [20-23] vertically
    vis_board = board[0:20, 4:14].copy()
    
    # Replace numbers with symbols for better visualization
    symbols = {0: '.', 1: '#', 2: 'I', 3: 'O', 4: 'T', 5: 'S', 6: 'Z', 7: 'J', 8: 'L'}
    
    for row in vis_board:
        print(''.join([symbols.get(cell, str(cell)) for cell in row]))

def piece_to_symbol(pieces):
    """Convert piece numbers to symbols and return formatted string"""
    symbols = {0: '.', 2: 'I', 3: 'O', 4: 'T', 5: 'S', 6: 'Z', 7: 'J', 8: 'L'}
    if isinstance(pieces, np.ndarray):
        return '\n'.join([''.join([symbols.get(int(cell), str(cell)) for cell in row]) for row in pieces])
    return symbols.get(int(pieces), str(pieces))

def extract_active_tetromino(board, active_mask):
    """Extract the active tetromino from the board using the mask"""
    # Use element-wise multiplication to extract the active tetromino
    active_piece = board * active_mask
    
    # Find the bounds of the active piece
    non_zero = np.nonzero(active_mask)
    if len(non_zero[0]) == 0:
        return None, None, None, None
    
    min_row, max_row = np.min(non_zero[0]), np.max(non_zero[0])
    min_col, max_col = np.min(non_zero[1]), np.max(non_zero[1])
    
    # Extract the piece and its position
    piece = active_piece[min_row:max_row+1, min_col:max_col+1]
    
    return piece, min_row, min_col, active_piece

def get_piece_id(piece):
    """Identify the piece type from the piece matrix"""
    unique_values = np.unique(piece)
    unique_values = unique_values[unique_values > 0]  # Ignore empty cells
    if len(unique_values) == 0:
        return None
    return unique_values[0]  # Return the first non-zero value

def rotate_piece(piece, piece_id):
    """Rotate a tetromino piece clockwise using np.rot90, matching the gymnasium environment"""
    # Simply use np.rot90 with k=1 for clockwise rotation
    # This matches the Tetris gymnasium environment's rotate function
    return np.rot90(piece, k=1)

async def generate_rotated_and_dropped_boards(board, active_mask, active_orientation=0):
    """
    Generate potential boards with the active piece rotated and dropped
    using simple rotation and collision detection
    
    Args:
        board: The game board
        active_mask: Mask of the active tetromino
        active_orientation: Not used in this simpler implementation
    
    Returns:
        List of dictionaries with potential board states
    """
    # Extract the active tetromino
    piece, row, col, active_piece_on_board = extract_active_tetromino(board, active_mask)
    if piece is None:
        return []
    
    # Get the piece ID (type)
    piece_id = get_piece_id(piece)
    if piece_id is None:
        return []
    
    # Determine max rotations based on piece type
    # I, S, Z have 2 rotations
    # T, L, J have 4 rotations
    # O has 1 rotation (no change)
    max_rotations = 1  # Default for O piece
    if piece_id in [2]:  # I piece
        max_rotations = 2
    elif piece_id in [5, 6]:  # S and Z pieces
        max_rotations = 2
    elif piece_id in [4, 7, 8]:  # T, J, L pieces
        max_rotations = 4
    
    # Clean the board by removing the active piece
    clean_board = board.copy()
    clean_board[active_mask > 0] = 0
    
    results = []
    
    # First, add the original position (no rotation, no drop)
    original_board = clean_board.copy()
    
    # Place the original piece on the board
    for r in range(piece.shape[0]):
        for c in range(piece.shape[1]):
            if piece[r, c] > 0:
                if (row+r < original_board.shape[0] and 
                    col+c < original_board.shape[1]):
                    original_board[row+r, col+c] = piece[r, c]
    
    # Add original position board
    results.append({
        'rotation': 0,
        'board': original_board,
        'row': row,
        'col': col,
        'description': "Original position"
    })
    
    # Skip rotation for O piece (ID 3)
    if piece_id == 3:
        return results
    
    # For each possible rotation (starting from 1)
    current_piece = piece.copy()
    for rotation in range(1, max_rotations):
        # Rotate the piece (apply rotation multiple times for higher rotation states)
        current_piece = rotate_piece(current_piece, piece_id)
        
        # Try wall kicks for rotation (simple implementation)
        # Each tuple is (col_offset, row_offset)
        wall_kicks = [(0, 0), (-1, 0), (1, 0), (0, -1)]
        
        # Try each wall kick until a valid rotation is found
        rotation_applied = False
        for col_offset, row_offset in wall_kicks:
            new_row = row + row_offset
            new_col = col + col_offset
            
            # Check if rotation is valid at this position
            valid = True
            for r in range(current_piece.shape[0]):
                for c in range(current_piece.shape[1]):
                    if current_piece[r, c] > 0:
                        board_r = new_row + r
                        board_c = new_col + c
                        
                        # Check bounds and collision
                        if (board_r < 0 or board_r >= board.shape[0] or
                            board_c < 0 or board_c >= board.shape[1] or
                            clean_board[board_r, board_c] > 0):
                            valid = False
                            break
                if not valid:
                    break
            
            if valid:
                # Create board with rotated piece
                rotated_board = clean_board.copy()
                
                # Place rotated piece at the appropriate position
                for r in range(current_piece.shape[0]):
                    for c in range(current_piece.shape[1]):
                        if current_piece[r, c] > 0:
                            if (new_row+r < rotated_board.shape[0] and 
                                new_col+c < rotated_board.shape[1]):
                                rotated_board[new_row+r, new_col+c] = current_piece[r, c]
                
                # Add to results
                results.append({
                    'rotation': rotation,
                    'board': rotated_board,
                    'row': new_row,
                    'col': new_col,
                    'description': f"Rotation {rotation}"
                })
                
                # We found a valid rotation, mark as applied and break out of wall kicks loop
                rotation_applied = True
                break
        
        # If we couldn't apply this rotation, we might still want to continue to the next rotation state
        if not rotation_applied:
            continue
    
    return results

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
    
    # Create comprehensive log entry
    log_entry = {
        "run_id": run_id,
        "step": step,
        "timestamp": datetime.now().isoformat(),
        "action_sequence": action_plan.get("action_sequence", []),
        "action": action,
        "thought": action_plan.get("thought", ""),
        "reward": float(reward),
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
        "action": action,
        "reward": float(reward),
        "info": info_copy
    }
    
    # Convert any remaining NumPy types
    data_entry = convert_numpy_types(data_entry)
    
    # Write to data log file
    with open(DATA_LOG_FILE, 'a') as f:
        json.dump(data_entry, f)
        f.write('\n')

class TetrisAgent:
    """A Tetris agent that uses either base module or perception+memory+reasoning modules."""
    
    def __init__(self, model_name="claude-3-7-sonnet-latest", use_base_module=False):
        """Initialize the agent with either base module or full pipeline."""
        self.model_name = model_name
        self.use_base_module = use_base_module
        self.last_action = None
        
        if use_base_module:
            print(f"Using Base Module with model: {model_name}")
            self.base_module = TetrisBaseModule(model_name=model_name)
        else:
            print(f"Using full pipeline (Perception + Memory + Reasoning) with model: {model_name}")
            self.perception_module = TetrisPerceptionModule(model_name=model_name)
            self.memory_module = TetrisMemoryModule(
                model_name=model_name,
                memory_file=MEMORY_FILE
            )
            self.reasoning_module = TetrisReasoningModule(model_name=model_name)
    
    async def get_action(self, observation, info=None, max_retries_const=1):
        """Get the next action from the agent."""
        try:
            if self.use_base_module:
                # Base module: Use RGB image directly if available, otherwise use info
                print("\n" + "="*80)
                action_plan = await self.base_module.process_observation(observation, info)
                print("\nACTION PLAN:")
                print(f"Action sequence: {action_plan['action_sequence']}")
                print(f"Thought: {action_plan['thought']}")
                print("="*80 + "\n")
                self.last_action = action_plan["action_sequence"]
                return action_plan, None, None
            else:
                # Full pipeline approach
                # Step 1: Perception - Analyze the game board
                perception_data = self.perception_module.analyze_board(observation, info)
                
                # Print perception data
                print("\n" + "="*80)
                print("PERCEPTION DATA:")
                print(f"Board shape: {perception_data.get('board_shape', None)}")
                print(f"Active piece: {perception_data.get('active_piece', None)}")
                print(f"Active position: {perception_data.get('active_position', None)}")
                print(f"Potential states: {len(perception_data.get('potential_states', []))}")
                
                # Step 2: Memory - Add to memory and generate reflection
                max_retries = max_retries_const
                retry_count = 0
                memory_summary = None
                
                # Add the current game state to memory
                self.memory_module.add_game_state(perception_data, self.last_action)
                
                # Try to update reflection asynchronously (non-blocking)
                asyncio.create_task(self.memory_module.update_reflection(perception_data, self.last_action))
                
                # Try to get memory summary
                while memory_summary is None and retry_count < max_retries:
                    memory_summary = self.memory_module.get_memory_summary()
                    
                    if memory_summary is None or len(memory_summary) == 0:
                        retry_count += 1
                        print(f"Memory retrieval attempt {retry_count}/{max_retries} failed. Retrying...")
                        await asyncio.sleep(1)  # Short delay before retry
                
                # Print memory summary
                print("\nMEMORY SUMMARY:")
                print(f"Memory entries: {len(memory_summary) if memory_summary else 0}")
                if memory_summary and len(memory_summary) > 0 and 'reflection' in memory_summary[-1]:
                    print(f"Latest reflection: {memory_summary[-1]['reflection']}")
                else:
                    print("No reflection available")
                
                # Step 3: Reasoning - Plan the next action
                max_retries = max_retries_const
                retry_count = 0
                action_plan = None
                
                # Try to get action plan                
                while action_plan is None and retry_count < max_retries:
                    try:
                        # Call the async plan_action method
                        action_plan = await self.reasoning_module.plan_action(
                            current_perception=perception_data,
                            memory_summary=memory_summary,
                            img_path=BOARD_IMG_PATH,
                            max_retries=1  # We'll handle retries here
                        )
                        
                        if action_plan is None or 'action_sequence' not in action_plan:
                            retry_count += 1
                            print(f"Action planning attempt {retry_count}/{max_retries} failed. Retrying...")
                            await asyncio.sleep(1)  # Short delay before retry
                    except Exception as e:
                        retry_count += 1
                        print(f"Error in reasoning module: {e}")
                        await asyncio.sleep(1)  # Short delay before retry
                
                # If still no valid action plan after retries, use fallback
                if action_plan is None or 'action_sequence' not in action_plan:
                    print("All reasoning attempts failed. Using fallback action.")
                    action_plan = {
                        "action_sequence": DEFAULT_ACTION,
                        "thought": "Fallback action after failed reasoning attempts"
                    }
                
                # Print action plan
                print("\nACTION PLAN:")
                print(f"Action sequence: {action_plan['action_sequence']}")
                print(f"Thought: {action_plan['thought']}")
                print("="*80 + "\n")
                
                # Store this action for the next iteration
                self.last_action = action_plan["action_sequence"]
                
                return action_plan, perception_data, memory_summary
        
        except Exception as e:
            print(f"Error in get_action: {e}")
            # Return a fallback action on error - use no_operation
            fallback_sequence = DEFAULT_ACTION  # no_operation
            self.last_action = fallback_sequence
            return {
                "action_sequence": fallback_sequence, 
                "thought": f"Error occurred: {str(e)}"
            }, None, None

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run the Tetris AI Agent')
    
    parser.add_argument('--model', type=str, default="claude-3-7-sonnet-latest",
                        help='Model name to use (default: claude-3-7-sonnet-latest)')
    
    parser.add_argument('--base', action='store_true',
                        help='Use the simplified Base_module (default: False)')
    
    parser.add_argument('--random', action='store_true',
                        help='Use random actions instead of AI agent (default: False)')
    
    parser.add_argument('--num_runs', type=int, default=1,
                        help='Number of game runs to perform (default: 1)')
    
    return parser.parse_args()

async def run_agent(args):
    """Run the Tetris agent with the specified parameters."""
    try:
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
        
        # Initialize the agent
        agent = None
        if not args.random:
            try:
                agent = TetrisAgent(
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
            except Exception as agent_error:
                print(f"Error initializing agent: {agent_error}")
                raise
        else:
            print("\n" + "*" * 80)
            print("RUNNING WITH RANDOM ACTIONS - No AI model used")
            print("*" * 80 + "\n")
        
        # Run for the specified number of games
        for run_id in range(1, args.num_runs + 1):
            print(f"\n========== STARTING RUN {run_id}/{args.num_runs} ==========\n")
            
            # Create the Tetris environment
            env = None
            try:
                env = gym.make("tetris_gymnasium/Tetris", render_mode="human")
                # Reset the environment
                obs, info = env.reset()
                
                # Make sure rendering window appears
                env.render()
                cv2.waitKey(100)
            except Exception as env_error:
                print(f"Error creating or resetting environment: {env_error}")
                continue  # Skip this run and try the next one
            
            # Run the game loop
            terminated = False
            truncated = False
            total_reward = 0
            step = 0
            
            # Max attempts to get a valid action
            max_retries = 3
            
            # Track when game is stuck
            stuck_count = 0
            max_stuck_actions = 50  # Maximum number of unchanged board states before terminating
            
            # Initialize current action sequence state
            current_sequence = []
            sequence_index = 0
            
            while not (terminated or truncated):
                # If no action sequence is queued or we're done with the current sequence
                if len(current_sequence) == 0 or sequence_index >= len(current_sequence):
                    if args.random:
                        # Simple random action - just sample a single action
                        current_sequence = [env.action_space.sample()]
                        
                        # Create a basic action plan for logging
                        action_plan = {
                            "action_sequence": current_sequence,
                            "thought": "Random action"
                        }
                        perception_data = None
                        memory_summary = None
                    else:
                        # Get action sequence from agent with retry logic
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
                                        # Create a basic representation of the board
                                        frame = np.zeros((400, 300, 3), dtype=np.uint8)
                                    
                                    # Pass the frame along with the image path in info
                                    modified_info = dict(info)
                                    modified_info['img_path'] = BOARD_IMG_PATH
                                    
                                    # Ensure the image is saved before passing to the base module
                                    os.makedirs(os.path.dirname(BOARD_IMG_PATH), exist_ok=True)
                                    Image.fromarray(frame).save(BOARD_IMG_PATH)
                                    print(f"Saved board image to {BOARD_IMG_PATH}")
                                    
                                    try:
                                        action_plan, perception_data, memory_summary = await agent.get_action(frame, modified_info)
                                    except Exception as api_error:
                                        print(f"Error calling AI API: {api_error}")
                                        # Fallback to default action - no_operation
                                        action_plan = {
                                            "action_sequence": DEFAULT_ACTION,
                                            "thought": f"API error: {str(api_error)}"
                                        }
                                        perception_data = None
                                        memory_summary = None
                                except Exception as e:
                                    print(f"Error getting frame for base module: {e}")
                                    # Fallback to default action - no_operation
                                    action_plan = {
                                        "action_sequence": DEFAULT_ACTION,
                                        "thought": f"Error getting frame: {str(e)}"
                                    }
                                    perception_data = None
                                    memory_summary = None
                            else:
                                # For full pipeline, pass the observation
                                try:
                                    action_plan, perception_data, memory_summary = await agent.get_action(obs, info)
                                except Exception as api_error:
                                    print(f"Error calling AI API: {api_error}")
                                    # Fallback to default action - no_operation
                                    action_plan = {
                                        "action_sequence": DEFAULT_ACTION,
                                        "thought": f"API error: {str(api_error)}"
                                    }
                                    perception_data = None
                                    memory_summary = None
                            
                            # Check if we got a valid action sequence
                            if action_plan and 'action_sequence' in action_plan and len(action_plan['action_sequence']) > 0:
                                current_sequence = action_plan['action_sequence']
                                valid_action = True
                            else:
                                print(f"Skipping action attempt {retry_count + 1}/{max_retries}")
                                retry_count += 1
                                # Wait a moment before trying again
                                await asyncio.sleep(1)
                        
                        # If we still don't have a valid action after retries, use no_operation
                        if not valid_action:
                            print("Exhausted retry attempts, using fallback action sequence")
                            current_sequence = DEFAULT_ACTION  # no_operation
                            # Update action_plan for logging
                            if not action_plan:
                                action_plan = {}
                            action_plan["action_sequence"] = current_sequence
                            action_plan["thought"] = "Fallback action sequence after retries"
                    
                    # Reset sequence index to start executing the new sequence
                    sequence_index = 0
                
                # Take the next action in the sequence
                action = current_sequence[sequence_index]
                sequence_index += 1
                
                # Store the current board state before taking action
                current_board = None
                if 'board' in obs:
                    current_board = obs['board'].copy()
                
                # Take the action in the environment
                next_obs, reward, terminated, truncated, info = env.step(action)
                env.render()
                # Add waitKey to visualize gameplay with delay
                cv2.waitKey(100)
                
                # Log comprehensive information with run_id
                # Use the current action rather than the whole sequence for this step's log
                current_action_name = list(tetris_actions.keys())[list(tetris_actions.values()).index(action)]
                print(f"Taking action: {current_action_name} (code: {action})")
                log_step(step, action_plan, perception_data, memory_summary, current_action_name, reward, info, next_obs, run_id)
                
                # Log raw data to separate file with run_id
                log_raw_data(step, action, reward, info, next_obs, run_id)
                
                total_reward += reward
                step += 1
                
                # Print board state after action
                print(f"\nBoard after action (step {step}):")
                print_board(next_obs['board'])
                print(f"Action: {current_action_name} ({action}), Reward: {reward}, Total Reward: {total_reward}")
                
                # Check if the board state changed after the action
                if 'board' in next_obs and current_board is not None:
                    new_board = next_obs['board']
                    if np.array_equal(current_board, new_board) and reward == 0:
                        stuck_count += 1
                        print(f"Board unchanged after action. Stuck count: {stuck_count}/{max_stuck_actions}")
                        if stuck_count >= max_stuck_actions:
                            print(f"Board hasn't changed for {max_stuck_actions} consecutive actions. Terminating run.")
                            terminated = True
                    else:
                        # Reset stuck counter if board has changed
                        stuck_count = 0
                
                # Explicitly render the environment
                env.render()
                # Add waitKey to visualize gameplay with delay
                cv2.waitKey(100)
                
                # Update current observation and info
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
    except Exception as e:
        print(f"Error in run_agent: {e}")
        return []

if __name__ == "__main__":
    args = parse_args()
    scores = asyncio.run(run_agent(args))
    if len(scores) == 1:
        print(f"Final total reward: {scores[0]}")
    else:
        print(f"Average total reward: {sum(scores) / len(scores):.2f}")
