import numpy as np
import os
import json
import time
from datetime import datetime
import argparse
from PIL import Image

# Import modules
from .base_module import Base_module
from .perception import PerceptionModule
from .memory import MemoryModule
from .reasoning import ReasoningModule

# Cache directories and file paths
CACHE_DIR = os.path.join("cache", "2048_experiments", datetime.now().strftime("%Y%m%d_%H%M%S"))
BOARD_IMG_PATH = os.path.join(CACHE_DIR, "board_latest.png")
EXPERIMENT_INFO_FILE = os.path.join(CACHE_DIR, "experiment_info.json")
os.makedirs(CACHE_DIR, exist_ok=True)

class Agent2048:
    def __init__(self, model_name="claude-3-7-sonnet-latest", use_base_module=False,
                 reasoning_effort="high", thinking=True):
        """
        Initialize the 2048 Agent with either the full pipeline or simplified Base_module.
        
        Args:
            model_name (str): Name of the model to use for inference
            use_base_module (bool): Whether to use the simplified Base_module instead of the full pipeline
            reasoning_effort (str): Level of reasoning effort (low, medium, high)
            thinking (bool): Whether to enable thinking mode
        """
        self.model_name = model_name
        self.last_action = None  # Store the last action taken
        self.use_base_module = use_base_module
        self.reasoning_effort = reasoning_effort
        self.thinking = thinking
        
        # Initialize modules
        if use_base_module:
            print(f"Using simplified Base_module with model: {model_name}")
            self.base_module = Base_module(
                model_name=model_name,
                reasoning_effort=reasoning_effort,
                thinking=thinking
            )
        else:
            print(f"Using full pipeline (Perception + Memory + Reasoning) with model: {model_name}")
            self.perception_module = PerceptionModule(model_name=model_name)
            self.memory_module = MemoryModule(model_name=model_name)
            self.reasoning_module = ReasoningModule(
                model_name=model_name,
                reasoning_effort=reasoning_effort,
                thinking=thinking
            )
            
        # Store experiment info
        self._store_experiment_info()
        
    def _store_experiment_info(self):
        """Store information about the current experiment in a JSON file."""
        experiment_info = {
            "timestamp": datetime.now().isoformat(),
            "model_name": self.model_name,
            "game_name": "2048",
            "use_base_module": self.use_base_module,
            "reasoning_effort": self.reasoning_effort,
            "thinking": self.thinking,
            "cache_directory": CACHE_DIR
        }
        
        # Write to JSON file
        with open(EXPERIMENT_INFO_FILE, 'w') as f:
            json.dump(experiment_info, f, indent=2)
        
        print(f"Experiment info saved to {EXPERIMENT_INFO_FILE}")
        
    def get_action(self, observation, info=None, reward=0):
        """
        Process observation to get game action using either full pipeline or Base_module.
        
        Args:
            observation: The game observation (board matrix)
            info: Additional information from the environment
            reward: The current reward from the environment
            
        Returns:
            dict: A dictionary containing move and thought
        """
        try:
            if self.use_base_module:
                # Simplified direct approach
                print("\n" + "="*80)
                action_plan = self.base_module.process_observation(observation, info)
                print("\nACTION PLAN:")
                print(f"Action: {action_plan['move']}")
                print(f"Thought: {action_plan['thought']}")
                print("="*80 + "\n")
                self.last_action = action_plan["move"]
                return action_plan
            else:
                # Full pipeline approach
                # Step 1: Perception - Analyze the board
                perception_data = self.perception_module.analyze_board(observation, info)
                
                # Print perception data in a readable format
                print("\n" + "="*80)
                print("PERCEPTION DATA:")
                print(f"Highest tile: {perception_data.get('highest_tile', 'Unknown')}")
                print(f"Empty spaces: {len(perception_data.get('empty_spaces', []))}")
                if 'game_state' in perception_data:
                    print(f"Analysis: {perception_data['game_state'].get('analysis', 'None')}")
                    print(f"Strategy: {perception_data['game_state'].get('strategy', 'None')}")
                
                # Step 2: Memory - Add to memory and get summary
                self.memory_module.add_game_state(perception_data, self.last_action)
                memory_summary = self.memory_module.get_memory_summary()
                
                # Print memory summary
                if memory_summary:
                    print("\nMEMORY SUMMARY:")
                    print(f"Memory entries: {len(memory_summary)}")
                    if len(memory_summary) > 0:
                        latest_entry = memory_summary[-1]
                        if 'reflection' in latest_entry and latest_entry['reflection']:
                            print(f"Latest reflection: {latest_entry['reflection']}")
                        if 'last_action' in latest_entry and latest_entry['last_action']:
                            print(f"Previous action: {latest_entry['last_action']}")
                
                # Step 3: Reasoning - Plan the next action
                action_plan = self.reasoning_module.plan_action(
                    current_perception=perception_data,
                    memory_summary=memory_summary,
                    img_path=BOARD_IMG_PATH
                )
                
                # Print action plan
                print("\nACTION PLAN:")
                print(f"Action: {action_plan['move']}")
                print(f"Thought: {action_plan['thought']}")
                print("="*80 + "\n")
                
                # Store this action for the next iteration
                self.last_action = action_plan["move"]
                
                return action_plan
            
        except Exception as e:
            print(f"Error in get_action: {e}")
            # Fallback to skip action on error
            self.last_action = "skip"
            return {
                "move": "skip",
                "thought": f"Error occurred: {str(e)}"
            }
        
    def parse_move_to_action_index(self, move):
        """
        Convert a move string to an action index for the environment.
        
        Args:
            move (str): Move string ('up', 'down', 'left', 'right', or 'skip')
            
        Returns:
            int: Action index for the environment
        """
        # Map move strings to action indices (0: up, 1: right, 2: down, 3: left)
        move_to_action = {
            "up": 0,
            "right": 1, 
            "down": 2,
            "left": 3,
            "skip": 0  # Use 'up' as default for 'skip' action
        }
        
        move_str = move.lower() if isinstance(move, str) else "skip"
        
        # Handle skip action specially - print a message and use 'up' as the action
        if move_str == "skip":
            print("SKIPPING ACTION - using 'up' as default")
            
        return move_to_action.get(move_str, 0)  # Default to 'up' (0) if move is invalid


def run_agent(args):
    """Run the 2048 agent with the specified parameters."""
    import gymnasium as gym
    
    # Create the 2048 environment
    env = gym.make("gymnasium_2048:gymnasium_2048/TwentyFortyEight-v0", 
                  size=4, max_pow=16, render_mode="human")
    
    # Initialize the agent
    agent = Agent2048(
        model_name=args.model,
        use_base_module=args.base,
        reasoning_effort=args.reasoning_effort,
        thinking=args.thinking
    )
    
    # Reset the environment
    observation, info = env.reset()
    
    # Setup logging
    log_file = os.path.join(CACHE_DIR, "game_log.jsonl")
    frames_dir = os.path.join(CACHE_DIR, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    # Run the game loop
    done = False
    total_reward = 0
    step = 0
    
    while not done:
        # Save the current frame as an image
        try:
            # Get and save the board image first (important!)
            frame = env.render(mode="rgb_array")
            if frame is not None:
                frame_img = Image.fromarray(frame)
                frame_path = os.path.join(frames_dir, f"frame_{step:04d}.png")
                frame_img.save(frame_path)
                
                # Also save a copy to the expected board image path to ensure it exists
                frame_img.save(BOARD_IMG_PATH)
        except Exception as e:
            print(f"Warning: Could not save frame: {e}")
            
        # Get action from agent
        action_plan = agent.get_action(observation, info, reward=0)
        
        # Convert move to action index
        action = agent.parse_move_to_action_index(action_plan["move"])
        
        # Take the action in the environment
        observation, reward, terminated, truncated, info = env.step(action)
        
        # Log the step
        log_step(log_file, step, action_plan["move"], action, reward, info, observation)
        
        total_reward += reward
        step += 1
        
        # Check if game is done
        done = terminated or truncated
        
        # Print some information
        print(f"Step: {step}, Action: {action_plan['move']}, Reward: {reward}, Score: {info.get('score', 0)}")
    
    # Save the final frame
    try:
        frame = env.render(mode="rgb_array")
        if frame is not None:
            frame_img = Image.fromarray(frame)
            frame_img.save(os.path.join(frames_dir, f"frame_final.png"))
    except Exception as e:
        print(f"Warning: Could not save final frame: {e}")
    
    # Close the environment
    env.close()
    
    # Print final results
    print(f"\nGame finished after {step} steps")
    print(f"Total reward: {total_reward}")
    print(f"Final score: {info.get('score', 0)}")
    
    return info.get('score', 0)

def log_step(log_file, step, move, action, reward, info, observation):
    """Log step information to a JSON Lines file."""
    # Convert numpy arrays to lists
    if isinstance(observation, np.ndarray):
        observation = observation.tolist()
    
    # Create log entry
    log_entry = {
        "step": step,
        "move": move,
        "action": int(action),
        "reward": float(reward),
        "info": {k: v for k, v in info.items() if isinstance(v, (int, float, str, bool, list, dict))},
        "observation": observation
    }
    
    # Write to log file
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run the 2048 AI Agent')
    
    parser.add_argument('--model', type=str, default="claude-3-7-sonnet-latest",
                        help='Model name to use for reasoning (default: claude-3-7-sonnet-latest)')
    
    parser.add_argument('--base', action='store_true',
                        help='Use the simplified Base_module (default: False)')
    
    parser.add_argument('--reasoning-effort', type=str, default="high", choices=["low", "medium", "high"],
                        help='Level of reasoning effort (default: high)')
    
    parser.add_argument('--thinking', action='store_true',
                        help='Enable thinking mode (default: False)')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_agent(args) 