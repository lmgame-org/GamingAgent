import cv2
import gymnasium as gym
import numpy as np
import os
import json
import argparse
import time
from datetime import datetime
import asyncio
from PIL import Image

from tetris_gymnasium.envs.tetris import Tetris
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

async def run_actions(env, actions):
    """
    Execute a list of actions on the environment sequentially.
    
    Args:
        env: The Tetris environment
        actions: List of actions to execute
        
    Returns:
        The final observation, reward, terminated, truncated, and info
    """
    obs = None
    reward = 0
    terminated = False
    truncated = False
    info = {}
    
    for action in actions:
        if terminated or truncated:
            break
        
        obs, step_reward, terminated, truncated, info = env.step(action)
        reward += step_reward
        
    return obs, reward, terminated, truncated, info

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
            self.perception_module = None  # Will be initialized with each observation
            self.memory_module = TetrisMemoryModule(
                model_name=model_name,
                memory_file=MEMORY_FILE,
                max_memory=10  # Store last 5 states
            )
            self.reasoning_module = TetrisReasoningModule(model_name=model_name)
    
    async def get_action(self, observation, info=None, max_retries_const=1):
        """Get the next action from the agent."""
        try:
            if self.use_base_module:
                # Base module: Use RGB image directly
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
                perception_data = self.perception_module.get_perception_data()
                
                # Print perception data
                print("\n" + "="*80)
                print("PERCEPTION DATA:")
                print(f"Current board state:\n{perception_data.get('board', 'Unknown')}")
                print(f"Next pieces: {perception_data.get('next_pieces', 'Unknown')}")
                print(f"Potential states: {len(perception_data.get('potential_states', []))} states")
                
                # Prepare simplified state for memory
                memory_input_state = {
                    'board_text': perception_data.get('board', 'No board data'),
                    'next_pieces_text': perception_data.get('next_pieces', 'No next pieces data')
                }
                
                # Step 2: Memory - Add to memory and generate reflection with retry logic
                max_retries = max_retries_const
                retry_count = 0
                memory_summary = None
                await self.memory_module.add_game_state(memory_input_state, self.last_action)
                
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
                    # Reasoning module still gets the full perception_data for detailed planning
                    action_plan = await self.reasoning_module.plan_action(
                        current_perception=perception_data, 
                        memory_summary=memory_summary,
                        img_path=BOARD_IMG_PATH,
                        max_retries=3  # Handle retries at this level, not inside plan_action
                    )
                    
                    if action_plan is None or 'action_sequence' not in action_plan:
                        retry_count += 1
                        print(f"Action planning attempt {retry_count}/{max_retries} failed. Retrying...")
                        await asyncio.sleep(1)  # Short delay before retry
                
                # If still no valid action plan after retries, create a fallback
                if action_plan is None or 'action_sequence' not in action_plan:
                    print("All reasoning attempts failed. Using fallback action.")
                    action_plan = {
                        "action_sequence": [7],  # no_operation
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
            # Return a fallback action on error
            self.last_action = [7]  # no_operation
            return {
                "action_sequence": [7], 
                "thought": f"Error occurred: {str(e)}"
            }, None, None

def log_step(step, action_plan, perception_data, memory_summary, action, reward, info, observation, run_id=1):
    """Log comprehensive step information to a JSON Lines file."""
    # Convert numpy arrays and other types for JSON serialization
    if isinstance(observation, dict) and 'board' in observation and isinstance(observation['board'], np.ndarray):
        observation_copy = observation.copy()
        observation_copy['board'] = observation['board'].tolist()
        observation = observation_copy
    elif isinstance(observation, np.ndarray):
        observation = observation.tolist()
    
    # Create comprehensive log entry
    log_entry = {
        "run_id": run_id,
        "step": step,
        "timestamp": datetime.now().isoformat(),
        "action_sequence": action_plan.get("action_sequence", []),
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

async def run_agent(args):
    """Run the Tetris agent with the specified parameters."""
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
        agent = TetrisAgent(
            model_name=args.model,
            use_base_module=args.base
        )
    else:
        print("Running with random actions")
    
    # Run for the specified number of games
    for run_id in range(1, args.num_runs + 1):
        print(f"\n========== STARTING RUN {run_id}/{args.num_runs} ==========\n")
        
        # Create the Tetris environment with appropriate render mode
        if args.base:
            # For base agent: Need RGB array to save images and pass to vision model
            render_mode = "rgb_array"
        else:
            # For full pipeline: Human rendering for visualization
            render_mode = "human"
        
        env = gym.make("tetris_gymnasium/Tetris", render_mode=render_mode)
        
        # Reset the environment
        obs, info = env.reset(seed=42)
        
        # Run the game loop
        terminated = False
        truncated = False
        all_rewards = 0
        step_count = 0
        
        # Max attempts to get a valid action
        max_retries = 3
        
        # Track when game is stuck
        stuck_count = 0
        max_stuck_actions = 10  # Maximum number of unchanged board states before terminating
        
        while not terminated and not truncated:
            # Render to update display
            env.render()

            key = cv2.waitKey(100)
            
            # Save board image if needed
            if args.base:
                try:
                    # Save the current frame (for vision API or documentation)
                    frame = env.render()
                    if frame is not None:
                        # Add the image path to info dictionary
                        modified_info = dict(info) if info is not None else {}
                        modified_info['img_path'] = BOARD_IMG_PATH
                        
                        # Add piece queue information
                        if 'queue' in obs:
                            # Create a temporary perception module to use its methods
                            temp_perception = TetrisPerceptionModule(model_name=agent.model_name, observation=obs)
                            modified_info['piece_queue_symbols'] = temp_perception.get_next_pieces()
                            modified_info['next_piece_queue'] = obs['queue']
                        
                        # Ensure the image is saved before passing to the base module
                        Image.fromarray(frame).save(BOARD_IMG_PATH)
                except Exception as e:
                    print(f"Warning: Could not save frame: {e}")
            
            # For random actions
            if args.random:
                # Sample a random action
                action = env.action_space.sample()
                # Create a basic action plan for logging
                action_plan = {
                    "action_sequence": [action],
                    "thought": "Random action"
                }
                perception_data = None
                memory_summary = None
            else:
                # Get action from agent with retry logic
                retry_count = 0
                valid_action_sequence = False
                
                while not valid_action_sequence and retry_count < max_retries:
                    # Get action from the agent
                    if args.base:
                        # For base agent, pass the RGB frame as observation
                        frame = env.render()
                        # Add the image path to info dictionary
                        modified_info = dict(info) if info is not None else {}
                        modified_info['img_path'] = BOARD_IMG_PATH
                        
                        # Add piece queue information
                        if 'queue' in obs:
                            # Create a temporary perception module to use its methods
                            temp_perception = TetrisPerceptionModule(model_name=agent.model_name, observation=obs)
                            modified_info['piece_queue_symbols'] = temp_perception.get_next_pieces()
                            modified_info['next_piece_queue'] = obs['queue']
                        
                        # Ensure the image is saved before passing to the base module
                        Image.fromarray(frame).save(BOARD_IMG_PATH)
                        action_plan, perception_data, memory_summary = await agent.get_action(frame, modified_info)
                    else:
                        # For full pipeline, initialize perception with current observation
                        agent.perception_module = TetrisPerceptionModule(model_name=agent.model_name, observation=obs)
                        action_plan, perception_data, memory_summary = await agent.get_action(obs, info)
                    
                    # Check if we got a valid action sequence
                    action_sequence = action_plan.get("action_sequence", [])
                    if action_sequence and isinstance(action_sequence, list) and len(action_sequence) > 0:
                        valid_action_sequence = True
                    else:
                        print(f"Skipping action attempt {retry_count + 1}/{max_retries}")
                        retry_count += 1
                        # Wait a moment before trying again
                        await asyncio.sleep(1)
                
                # If we still don't have a valid action sequence after retries, use a random action
                if not valid_action_sequence:
                    print("Exhausted retry attempts, using random action")
                    action_sequence = [env.action_space.sample()]
                    # Update action_plan for logging
                    action_plan["action_sequence"] = action_sequence
                    action_plan["thought"] += " (Random action after retries)"
            
            # Execute the action sequence
            action_sequence = action_plan.get("action_sequence", [7])  # Default to no_operation
            
            # Execute the sequence of actions
            if not action_sequence:
                action_sequence = [7]  # Default to no_operation if action_sequence is empty
                
            obs, reward, terminated, truncated, info = await run_actions(env, action_sequence)
            
            # Log comprehensive information with run_id
            log_step(step_count, action_plan, perception_data, memory_summary, action_sequence[0], reward, info, obs, run_id)
            
            # Log raw data to separate file with run_id
            log_raw_data(step_count, action_sequence[0], reward, info, obs, run_id)
            
            all_rewards += reward
            step_count += 1
            
            # Print some information
            lines_cleared = info.get('lines_cleared', 0)
            print(f"Run {run_id}, Step: {step_count}, Actions: {action_sequence}, Reward: {reward}, Total Reward: {all_rewards}, Lines Cleared: {lines_cleared}")
            
            # Wait to see movement
            key = cv2.waitKey(100)
            
            # Allow early exit with 'q' key
            if key == ord('q'):
                break
        
        # Close the environment
        env.close()
        
        # Get final score and ensure it's a Python native type
        all_scores.append(float(all_rewards))
        
        # Log run summary
        run_summary = {
            "run_id": int(run_id),
            "score": float(all_rewards),
            "steps": int(step_count),
            "timestamp": datetime.now().isoformat(),
            "termination_reason": "Normal completion" if step_count < max_stuck_actions else f"Terminated due to inactivity for {max_stuck_actions} actions"
        }
        
        with open(summary_file, 'a') as f:
            json.dump(run_summary, f)
            f.write('\n')
        
        # Print final results for this run
        print(f"\nRun {run_id} finished after {step_count} steps")
        print(f"Total reward: {all_rewards}")
        print(f"Lines cleared: {info.get('lines_cleared', 0)}")
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
    parser = argparse.ArgumentParser(description='Run the Tetris AI Agent')
    
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