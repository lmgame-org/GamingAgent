import retro
import time
import os
import pickle
import numpy as np
from PIL import Image
import json
from datetime import datetime
from tools.serving import APIManager
import asyncio


CACHE_DIR = os.path.join("cache", "super_mario_bros_experiments", datetime.now().strftime("%Y%m%d_%H%M%S"))
OBSERVATION_IMG_PATH = os.path.join(CACHE_DIR, "obs_latest.png")

all_actions = {
    "[NOOP]":             [0, 0, 0, 0, 0, 0, 0, 0, 0],  # Do nothing
    "[right]":            [0, 0, 0, 0, 0, 0, 0, 1, 0],  # Move right
    "[right,A]":       [0, 0, 0, 0, 0, 0, 0, 1, 1],  # Move right and jump
    "[right,B]":       [1, 0, 0, 0, 0, 0, 0, 1, 0],  # Move right and run
    "[right,A,B]":  [1, 0, 0, 0, 0, 0, 0, 1, 1],  # Move right, jump, and run
    "[A]":                [0, 0, 0, 0, 0, 0, 0, 0, 1],  # Jump in place
    "[left]":             [0, 0, 0, 0, 0, 0, 1, 0, 0],  # Move left
}

class SuperMarioBrosAgent:
    def __init__(self, model_name="claude-3-7-sonnet-latest", img_path=OBSERVATION_IMG_PATH):
        """
        Initialize the Super Mario Bros Agent.
        
        Args:
            model_name (str): Name of the model to use for inference
            img_path (str): Path where to save observation images for API calls
        """
        self.model_name = model_name
        self.api_manager = APIManager(game_name="super_mario_bros")
        self.img_path = img_path
        self.system_prompt = """You are an AI agent navigating the Super Mario Bros environment using the OpenAI Gym interface.  
Your objective is to guide Mario through levels by selecting appropriate actions from a simplified action space.

Super Mario Bros Quick Guide:
Primary Goal: Survive as long as possible and make it to the end of the level (move right until the flag).
Secondary Goal: Collect coins and defeat enemies when possible.

Observation Space:
- You receive RGB images representing the current frame of the game.
- The game runs at 30 FPS (frames per second).

Action Space:
You may select from 7 discrete actions, each corresponding to specific button combinations:

| Action Index | Button Combination    | Description                        |
|--------------|-----------------------|------------------------------------|
| 0            | ['NOOP']              | Do nothing                         |
| 1            | ['right']             | Move right                         |
| 2            | ['right', 'A']        | Move right and jump                |
| 3            | ['right', 'B']        | Move right and run                 |
| 4            | ['right', 'A', 'B']   | Move right, jump, and run          |
| 5            | ['A']                 | Jump in place                      |
| 6            | ['left']              | Move left                          |

Important: Think about future frames when deciding on actions!

Action Planning:
- For each screenshot, you need to plan actions for multiple future frames.
- You can provide either:
  * Long action sequence (1 second = 30 frames): e.g., move: ([right], 30)
  * Short action sequence (0.5 seconds = 15 frames): e.g., move: ([right], 15)

Your response format should contain:
1. thought: [Your reasoning about the game state and planned actions]
2. move: ([action_name], frame_count)

Example responses:
- thought: Mario needs to jump over a gap. I'll have him run and jump.
  move: ([right,A,B], 15)

- thought: There's a Goomba ahead. Mario should jump on it to defeat it.
  move: ([right,A], 30)

Focus on making strategic decisions that help Mario progress through the level safely and efficiently.
Do not discuss reward calculations in your response.
"""
        
    def parse_agent_response(self, response):
        """
        Parse LLM response to extract thought and move components.
        
        Args:
            response (str): Raw response from LLM with thought and move
            
        Returns:
            dict: Dictionary with 'thought' and 'move' keys
                  where 'move' is a tuple (action_name, frame_count)
        """
        move = None
        thought = None
        
        # Look for thought: and move: in the response
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith("thought:"):
                thought = line[len("thought:"):].strip()
            elif line.startswith("move:"):
                # Extract the move pattern and count
                move_text = line[len("move:"):].strip()
                # Expected format: move: ([action_name], frame_count)
                # Extract action and frame count using regex
                import re
                move_match = re.search(r'\((\[.*?\]),\s*(\d+)\)', move_text)
                if move_match:
                    action_name = move_match.group(1)
                    frame_count = int(move_match.group(2))
                    
                    # Normalize action name to match the all_actions dictionary keys
                    # Remove spaces between commas
                    action_name = action_name.replace(", ", ",")
                    
                    # Check if the normalized action exists in all_actions
                    if action_name not in all_actions:
                        print(f"Warning: Unknown action '{action_name}', defaulting to '[right]'")
                        action_name = "[right]"
                    
                    move = (action_name, frame_count)
        
        # If parsing failed, use default values
        if move is None:
            move = ("[right]", 15)
            print(f"Failed to parse move from response: {response}")
        
        if thought is None:
            thought = "No explicit thought provided in response"
            print(f"Failed to parse thought from response: {response}")
        
        return {
            "move": move,
            "thought": thought
        }
        
    def get_action(self, observation, reward):
        """
        Process observation and use API manager to get game action.
        
        Args:
            observation: The game observation (RGB image)
            reward: The current reward from the environment
            
        Returns:
            dict: A dictionary containing move and thought
        """
        try:
            # User prompt - simple instruction to analyze and act
            user_prompt = "Here's the current game state. What action should Mario take next?"
            
            # Call API with the image
            if self.api_manager:
                # Use API manager for vision API call if available
                response, _ = self.api_manager.vision_text_completion(
                    model_name=self.model_name,
                    system_prompt=self.system_prompt,
                    prompt=user_prompt,
                    image_path=self.img_path
                )
            else:
                # Fallback: just return a default action
                print("Warning: No API manager available, returning default action")
                return {
                    "move": ("[right,A,B]", 15),
                    "thought": "No API manager available, using default action"
                }
            
            # Parse the response to extract thought and move
            return self.parse_agent_response(response)
            
        except Exception as e:
            print(f"Error in get_action: {e}")
            # Fallback to a safe default action
            return {
                "move": ("[right]", 15),
                "thought": f"Error occurred: {str(e)}"
            }

def get_mario_position(env):
    """
    Calculate Mario's true position from RAM addresses.
    Based on SuperMarioBrosEnv._x_position calculation.
    
    Args:
        env: The game environment
        
    Returns:
        int: Mario's x position in the level
    """
    # Get RAM directly from the emulator
    ram = env.get_ram()
    # Page number (coarse position) at 0x6d (109 decimal)
    page = ram[0x6d]
    # Fine position within page at 0x86 (134 decimal)
    x_pos_fine = ram[0x86]
    # Combined position - equivalent to SuperMarioBrosEnv's calculation
    x_pos = (page * 0x100) + x_pos_fine
    # print(f"DEBUG: page={page}, x_pos_fine={x_pos_fine}, x_pos={x_pos}")
    return x_pos

async def run_actions(env, actions, fps=30):
    """
    Run a sequence of actions on the given env at the specified frames per second.
    """
    sleep_time = 1.0 / fps
    for idx, action in enumerate(actions):
        observation, reward, terminated, truncated, info = env.step(action)
        # Add x_position to info
        info['x_pos'] = get_mario_position(env)
        # log each step's data
        log_to_jsonl(observation, info, idx, reward, terminated, truncated)
        env.render()
        await asyncio.sleep(sleep_time)
        if terminated or truncated:
            env.reset()
    return observation, reward, terminated, truncated, info

def convert_to_json_serializable(obj):
    """Convert NumPy types to Python natives for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(i) for i in obj]
    return obj

def log_to_jsonl(observation, info, count, reward, terminated, truncated, json_file=None):
    """
    Append observation, info, reward, and termination flags to a JSON Lines log file.
    """
    if json_file is None:
        json_file = os.path.join(CACHE_DIR, 'data_log.jsonl')
    # Overwrite a single observation image each time
    img_path = os.path.join(CACHE_DIR, "obs_latest.png")
    Image.fromarray(observation).save(img_path)
    # Create record and convert NumPy types to Python natives
    record = convert_to_json_serializable({
        'count': count,
        'info': info,
        'reward': reward,
        'terminated': terminated,
        'truncated': truncated
    })
    with open(json_file, 'a') as f:
        json.dump(record, f)
        f.write('\n')

async def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    env = retro.make(
        game="SuperMarioBros-Nes",
        obs_type=retro.Observations.IMAGE,
        render_mode='human'
    )
    env.reset()
    
    # Print controller button mapping
    print("Buttons:", env.buttons)
    
    # Print 10 random action samples
    print("Action space samples:")
    for i in range(10):
        action = env.action_space.sample()
        print(f"Sample {i}: {action}")

    agent = SuperMarioBrosAgent()
    
    # First execute a 10-frame run+jump burst, then fall back to default NOOP loop
    # run_actions(env, [all_actions["[right,A,B]"]] * 30, fps=30)
    # Temporarily commenting out the game loop

    count = 0
    default_action = all_actions["[NOOP]"]
    observation, reward, terminated, truncated, info = env.step(default_action)
    log_to_jsonl(observation, info, count, reward, terminated, truncated)
    count += 1
    
    while True:
        llm_action_response = agent.get_action(observation, reward)
        print(f"action: {llm_action_response['move']}; count: {count}")
        
        # Get the action and frame count from the response
        action_name, frame_count = llm_action_response['move']
        # Convert action name to actual action array
        action = all_actions[action_name]
        # Create list of actions to run
        actions = [action] * frame_count
        
        # Run the actions and wait for completion
        
        observation, reward, terminated, truncated, info = await run_actions(env, actions, fps=30)
        
        # Add x_position to info
        info['x_pos'] = get_mario_position(env)
        # Append to JSONL log with reward and termination status
        log_to_jsonl(observation, info, count, reward, terminated, truncated)
        env.render()
        count += 1
        if terminated or truncated:
            env.reset()

    
    env.close()


if __name__ == "__main__":
    asyncio.run(main())