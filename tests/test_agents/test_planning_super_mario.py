import retro
import time
import os
import pickle
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json
from datetime import datetime
from tools.serving import APIManager
import asyncio
from collections import deque
import argparse
import random
import atexit  # Add atexit module for cleanup functions


CACHE_DIR = os.path.join("cache", "super_mario_bros_experiments", datetime.now().strftime("%Y%m%d_%H%M%S"))
OBSERVATION_IMG_PATH = os.path.join(CACHE_DIR, "obs_latest.png")
GRID_IMG_PATH = os.path.join(CACHE_DIR, "obs_grid_latest.png")
MEMORY_FILE = os.path.join(CACHE_DIR, "memory.json")
EXPERIMENT_INFO_FILE = os.path.join(CACHE_DIR, "experiment_info.json")
MAX_SHORT_SIDE = 768
MAX_LONG_SIDE = 2000

all_actions = {
    "[NOOP]":             [0, 0, 0, 0, 0, 0, 0, 0, 0],  # Do nothing
    "[right]":            [0, 0, 0, 0, 0, 0, 0, 1, 0],  # Move right
    "[right,A]":       [0, 0, 0, 0, 0, 0, 0, 1, 1],  # Move right and jump
    "[right,B]":       [1, 0, 0, 0, 0, 0, 0, 1, 0],  # Move right and run
    "[right,A,B]":  [1, 0, 0, 0, 0, 0, 0, 1, 1],  # Move right, jump, and run
    "[A]":                [0, 0, 0, 0, 0, 0, 0, 0, 1],  # Jump in place
    "[left]":             [0, 0, 0, 0, 0, 0, 1, 0, 0],  # Move left
}

def resize_image_with_aspect_ratio(img, max_short_side=MAX_SHORT_SIDE, max_long_side=MAX_LONG_SIDE):
    """
    Resize an image to fit within the specified dimensions while maintaining aspect ratio.
    
    Args:
        img: PIL Image to resize
        max_short_side: Maximum size for the short side
        max_long_side: Maximum size for the long side
        
    Returns:
        PIL Image resized to fit within the constraints
    """
    width, height = img.size
    
    # Determine short and long sides
    if width <= height:
        short_side, long_side = width, height
        is_width_shorter = True
    else:
        short_side, long_side = height, width
        is_width_shorter = False
    
    # Calculate scale factor based on short side
    scale_short = max_short_side / short_side
    
    # Check if scaling by short side would exceed long side limit
    if long_side * scale_short > max_long_side:
        # If so, scale by long side instead
        scale_factor = max_long_side / long_side
    else:
        scale_factor = scale_short
    
    # Calculate new dimensions
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    # Resize the image
    return img.resize((new_width, new_height), Image.LANCZOS)

class PerceptionModule:
    def __init__(self, model_name="claude-3-7-sonnet-latest"):
        """
        Initialize the Perception Module for analyzing game states.
        
        Args:
            model_name (str): Name of the vision model to use
        """
        self.model_name = model_name
        self.api_manager = APIManager(game_name="super_mario_bros")
        self.system_prompt = """You are a computer vision system analyzing frames from Super Mario Bros.
"""

    def analyze_frame(self, observation, img_path):
        """
        Analyze the current frame to identify game elements and their positions.
        
        Args:
            observation: The game observation (RGB image)
            img_path: Path to save the image
            
        Returns:
            dict: A dictionary containing the analyzed game state in a structured format
        """
        try:
            # Convert the observation to a PIL Image
            img = Image.fromarray(observation)
            
            # Resize the image maintaining aspect ratio
            img_resized = resize_image_with_aspect_ratio(img)
            
            # Save the resized original observation image
            img_resized.save(OBSERVATION_IMG_PATH)
            
            # Create a copy and draw a 5x5 grid on it
            img_with_grid = img.copy()
            img_with_grid = self._add_grid_to_image(img_with_grid)
            
            # Resize the grid image using the same aspect ratio
            img_with_grid_resized = resize_image_with_aspect_ratio(img_with_grid)
            
            # Save the resized image with grid
            img_with_grid_resized.save(GRID_IMG_PATH)
            
            user_prompt = """Analyze this Super Mario Bros frame with the 5x5 grid overlay and identify game elements in each grid cell.
                Your task is to identify and locate game elements in a 5x5 grid overlay on the screen.
            
                Identify the following elements and their approximate positions in (x,y) grid coordinates:
                - Mario (player character)
                - Pipes (green obstacles)
                - Goombas (brown mushroom enemies)
                - Koopas (turtle enemies)
                - Gaps/pits (areas where Mario can fall)
                - Question blocks (blocks with ? that can be hit)
                - Brick blocks (breakable blocks)
                - Coins
                - Power-ups (if visible)
                - Flag pole (end of level)

                IMPORTANT: Use a 5x5 grid system where (0,0) is the top-left corner and (4,4) is the bottom-right.
                        
                Your response must be in valid JSON format with the following structure:
                {
                "mario": {"x": int, "y": int},
                "environment": {
                    "pipes": [{"x": int, "y": int, "height": "small|medium|large"}],
                    "goombas": [{"x": int, "y": int, "distance": "very_close|close|medium|far"}],
                    "koopas": [{"x": int, "y": int, "distance": "very_close|close|medium|far"}],
                    "gaps": [{"x": int, "width": "small|medium|large"}],
                    "question_blocks": [{"x": int, "y": int}],
                    "brick_blocks": [{"x": int, "y": int}],
                    "coins": [{"x": int, "y": int}],
                    "power_ups": [{"x": int, "y": int, "type": "mushroom|flower|star"}],
                    "flag_pole": {"x": int, "y": int} or null
                },
                "game_state": {
                    "scroll_direction": "right|left|stationary",
                    "mario_state": "small|big|fire|invincible",
                    "immediate_threats": ["goomba"|"koopa"|"gap"|"pipe"],
                    "obstacles_ahead": ["goomba"|"koopa"|"gap"|"pipe"]
                }
                }

                Ensure all coordinates are integers within the 0-4 range for the 5x5 grid.
                If an element is not present, include it as an empty array or null as appropriate.
                For immediate_threats, only include elements that pose an immediate danger to Mario.
                """
            
            response, _ = self.api_manager.vision_text_completion(
                model_name=self.model_name,
                system_prompt=self.system_prompt,
                prompt=user_prompt,
                image_path=GRID_IMG_PATH
            )
            
            # Extract and parse JSON from the response
            try:
                # Find JSON content in the response (might be surrounded by markdown or other text)
                import re
                json_match = re.search(r'```json\n(.*?)\n```', response, re.DOTALL)
                if json_match:
                    json_content = json_match.group(1)
                else:
                    json_content = response
                
                # Parse the JSON
                perception_data = json.loads(json_content)
                return perception_data
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON from perception module: {e}")
                print(f"Raw response: {response}")
                # Return a minimal valid structure if parsing fails
                return {
                    "mario": {"x": 2, "y": 3},
                    "environment": {
                        "pipes": [], "goombas": [], "koopas": [], "gaps": [],
                        "question_blocks": [], "brick_blocks": [], "coins": [],
                        "power_ups": [], "flag_pole": None
                    },
                    "game_state": {
                        "scroll_direction": "right",
                        "mario_state": "small",
                        "immediate_threats": [],
                        "obstacles_ahead": []
                    }
                }
                
        except Exception as e:
            print(f"Error in perception module: {e}")
            # Return a minimal valid structure on error
            return {
                "mario": {"x": 2, "y": 3},
                "environment": {
                    "pipes": [], "goombas": [], "koopas": [], "gaps": [],
                    "question_blocks": [], "brick_blocks": [], "coins": [],
                    "power_ups": [], "flag_pole": None
                },
                "game_state": {
                    "scroll_direction": "right",
                    "mario_state": "small",
                    "immediate_threats": [],
                    "obstacles_ahead": []
                }
            }
            
    def _add_grid_to_image(self, img):
        """
        Add a 5x5 grid overlay to the image.
        
        Args:
            img: PIL Image object
            
        Returns:
            PIL Image with grid overlay
        """
        try:
            # Create a copy of the image to draw on
            draw = ImageDraw.Draw(img)
            
            # Get image dimensions
            width, height = img.size
            
            # Calculate grid cell size
            cell_width = width // 5
            cell_height = height // 5
            
            # Draw horizontal grid lines
            for i in range(1, 5):
                y = i * cell_height
                draw.line([(0, y), (width, y)], fill=(255, 0, 0), width=2)
                
            # Draw vertical grid lines
            for i in range(1, 5):
                x = i * cell_width
                draw.line([(x, 0), (x, height)], fill=(255, 0, 0), width=2)
                
            # Add coordinates to cells (optional)
            # Try to load a font (fallback to default if not available)
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()
                
            # Add coordinate labels to cells
            for y in range(5):
                for x in range(5):
                    coord_text = f"({x},{y})"
                    text_x = x * cell_width + 5
                    text_y = y * cell_height + 5
                    draw.text((text_x, text_y), coord_text, fill=(255, 255, 0), font=font)
            
            return img
        except Exception as e:
            print(f"Error adding grid to image: {e}")
            # Return the original image if grid addition fails
            return img


class MemoryModule:
    def __init__(self, memory_file=MEMORY_FILE, max_memory=10, model_name="claude-3-7-sonnet-latest"):
        """
        Initialize the Memory Module for tracking game state history.
        
        Args:
            memory_file (str): Path to the memory JSON file
            max_memory (int): Maximum number of game states to remember
            model_name (str): Name of the model to use for reflections
        """
        self.memory_file = memory_file
        self.max_memory = max_memory
        self.memory = deque(maxlen=max_memory)
        self.model_name = model_name
        self.api_manager = APIManager(game_name="super_mario_bros")
        
        # Create the memory file directory if it doesn't exist
        os.makedirs(os.path.dirname(memory_file), exist_ok=True)
        
        # Load existing memory if available
        self.load_memory()
        
    def load_memory(self):
        """Load memory from the memory file if it exists."""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r') as f:
                    memory_data = json.load(f)
                    # Convert to deque with max length
                    self.memory = deque(memory_data, maxlen=self.max_memory)
        except Exception as e:
            print(f"Error loading memory: {e}")
            self.memory = deque(maxlen=self.max_memory)
            
    def save_memory(self):
        """Save the current memory to the memory file."""
        try:
            with open(self.memory_file, 'w') as f:
                json.dump(list(self.memory), f, indent=2)
        except Exception as e:
            print(f"Error saving memory: {e}")
            
    def generate_reflection(self, current_perception, last_action):
        """
        Generate a reflection on the current state by comparing it with previous states.
        
        Args:
            current_perception (dict): The current perceived game state
            last_action (tuple): The previous action taken (action_name, frame_count)
            
        Returns:
            str: A reflection on the current state and how it relates to previous states and actions
        """
        try:
            # If there are not enough previous states, return a default reflection
            if len(self.memory) < 1:
                return "Not enough history to generate a meaningful reflection."
            
            # Get the previous state
            previous_state = self.memory[-1]["game_state"]
            previous_action = self.memory[-1].get("last_action", None)
            
            # Ensure last_action is not None to avoid calculation errors
            if last_action is None:
                last_action = ("[NOOP]", 0)
            
            system_prompt = """You are an analytical assistant for a Super Mario Bros AI agent. Our task is to generate a brief, insightful reflection on the game state changes and the effectiveness of recent actions.
Focus on strategic insights and patterns that would help the agent make better decisions.
Keep your reflections short, precise, and actionable.
"""
            
            # Format information for the prompt - safely handle None values
            prev_action_name = previous_action[0] if previous_action else "None"
            prev_action_frames = previous_action[1] if previous_action else 0
            
            user_prompt = f"""Please analyze the following game states and actions to generate a brief reflection:

Previous Game State:
{json.dumps(previous_state, indent=2)}

Previous Action: {prev_action_name} for {prev_action_frames} frames

Current Game State:
{json.dumps(current_perception, indent=2)}

Last Action: {last_action[0]} for {last_action[1]} frames

Focus your reflection on:
1. How the game state changed after the last action
2. Whether the action was effective for the situation
3. Patterns or issues to be aware of
4. Any strategic insights for future actions

Keep your reflection under 100 words and focus only on the most important insights."""
            
            # Make the API call for reflection
            response, _ = self.api_manager.text_completion(
                model_name=self.model_name,
                system_prompt=system_prompt,
                prompt=user_prompt
            )
            
            return response.strip()
            
        except Exception as e:
            print(f"Error generating reflection: {str(e)}")
            return f"No reflection available: {str(e)[:50]}"
            
    def add_game_state(self, game_state, action=None, timestamp=None):
        """
        Add a new game state to memory.
        
        Args:
            game_state (dict): The perceived game state to add
            action (tuple, optional): Action taken in the previous state (action_name, frame_count)
            timestamp (float, optional): Timestamp for the game state
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Generate reflection if we have at least one previous state
        reflection = None
        if len(self.memory) > 0:
            reflection = self.generate_reflection(game_state, action)
            
        # Add timestamp, action and reflection to the game state
        memory_entry = {
            "timestamp": timestamp,
            "game_state": game_state,
            "last_action": action,
            "reflection": reflection
        }
        
        # Add to memory
        self.memory.append(memory_entry)
        
        # Save updated memory
        self.save_memory()
        
    def get_memory_summary(self):
        """
        Get a summary of the memory for the reasoning module.
        
        Returns:
            list: List of memory entries
        """
        return list(self.memory)


class ReasoningModule:
    def __init__(self, model_name="claude-3-7-sonnet-latest", reasoning_effort="high", thinking=True):
        """
        Initialize the Reasoning Module for action planning.
        
        Args:
            model_name (str): Name of the model to use for reasoning
        """
        self.model_name = model_name
        self.reasoning_effort = reasoning_effort
        self.thinking = thinking
        self.api_manager = APIManager(game_name="super_mario_bros")
        # Simplified system prompt with strict output instructions
        self.system_prompt = """You are an intelligent AI player playing Super Mario Bros. Your goal is to help Mario progress through the level safely and efficiently.

IMPORTANT: You MUST format your response using EXACTLY these lines:
thought: [Your reasoning about the game state]
move: ([action_name], frame_count)

Do not include # or any other prefix. Start directly with "thought:" followed by your analysis."""
        
        # Keep the detailed prompt as a user prompt component
        self.action_prompt = """Super Mario Bros Quick Guide:
Primary Goal: Survive as long as possible and make it to the end of the level (move right until the flag).
Secondary Goal: Collect coins and defeat enemies when possible, and hit question mark blocks when safe.

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
  * Short action sequence (0-15 frames): e.g., move: ([right], 15)
  * Long action sequence (16-30 frames): e.g., move: ([right], 30)

Key Strategies:
- Approaching gaps: Be extremely cautious. Use short sequences first to prepare positioning, then commit to a jump with enough momentum.
- Enemies: Use defensive jumps when enemies are near. If unsure, move back or jump in place.
- Obstacles: Predict what's coming and prepare actions accordingly.
- Speed management: Don't move too fast as unseen enemies may appear from off-screen.
- Defensive play: When in doubt, take a defensive approach (move left or jump in place).

Your response format should contain:
1. thought: [Your reasoning about the game state and planned actions]
2. move: ([action_name], frame_count)

Example responses:
- thought: I see a gap ahead. I need to get the right momentum before jumping.
  move: ([right,B], 15)

- thought: I see multiple enemies clustering ahead. Taking a defensive position.
  move: ([left], 8)

- thought: There's a tall pipe ahead. I need a long, high jump to clear it completely.
  move: ([right,A,B], 30)

- thought: I'm at the edge of a large gap. Need to execute a powerful jump immediately to clear it.
  move: ([right,A,B], 30)

- thought: Mario is falling down and close to the ground but not fully landed yet. I need a small frame skip before taking further actions.
  move: ([NOOP], 5)

Focus on making strategic decisions that help Mario progress through the level safely and efficiently.
Do not discuss reward calculations in your response.
"""

    async def plan_action(self, current_perception, memory_summary, img_path, max_retries=3):
        """
        Plan the next action based on current perception and memory.
        
        Args:
            current_perception (dict): Current perceived game state
            memory_summary (list): Summary of past game states
            img_path (str): Path to the current observation image
            max_retries (int): Maximum number of retry attempts if response is empty
            
        Returns:
            dict: A dictionary containing move and thought
        """
        try:
            # Prepare memory context for the prompt
            memory_context = self._prepare_memory_context(memory_summary)
            
            # Create combined user prompt with perception and memory data
            user_prompt = f"""{self.action_prompt}

Here's the current game state:
{json.dumps(current_perception, indent=2)}

Memory of recent states:
{memory_context}

Based on this information and the current image, what action should Mario take next?

IMPORTANT - FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
thought: [your analysis here]
move: ([action_name], frame_count)

Do NOT use # or any other prefix. Start directly with "thought:" followed by your analysis.
Only use available actions: [NOOP], [right], [right,A], [right,B], [right,A,B], [A], [left]
Frame count must be between 1-30.
"""
            
            # Implement retry mechanism for API call
            retry_count = 0
            response = None
            parsed_response = None
            
            while (response is None or parsed_response is None or 
                  'move' not in parsed_response or parsed_response['move'] is None) and retry_count < max_retries:
                
                if retry_count > 0:
                    print(f"Retry attempt {retry_count}/{max_retries} for reasoning module...")
                    await asyncio.sleep(2)  # Short delay before retry
                
                # Use the grid image for the API call
                response, _ = self.api_manager.vision_text_completion(
                    model_name=self.model_name,
                    system_prompt=self.system_prompt,
                    prompt=user_prompt,
                    image_path=img_path,
                    thinking=self.thinking,
                    reasoning_effort=self.reasoning_effort,
                    token_limit=100000 if "gemini-2.5-flash" not in self.model_name else 30000
                )
                
                # Parse the response
                parsed_response = self._parse_response(response)
                
                # Check if we got a valid response
                if parsed_response is None or 'move' not in parsed_response or parsed_response['move'] is None:
                    retry_count += 1
                else:
                    break
            
            # If all retries failed, return a fallback action
            if parsed_response is None or 'move' not in parsed_response or parsed_response['move'] is None:
                print("All reasoning attempts failed. Using fallback action.")
                return {
                    "move": ("[NOOP]", 1),  # Safe fallback
                    "thought": "Fallback action after failed reasoning attempts"
                }
                
            return parsed_response
            
        except Exception as e:
            print(f"Error in reasoning module: {e}")
            # Return a default action on error
            return {
                "move": ("[NOOP]", 1),
                "thought": f"Error occurred in reasoning: {str(e)}"
            }
            
    def _prepare_memory_context(self, memory_summary):
        """
        Prepare a concise summary of memory for the prompt.
        
        Args:
            memory_summary (list): List of memory entries
            
        Returns:
            str: A concise summary of memory
        """
        if not memory_summary:
            return "No memory of past states available."
            
        # Take up to the last 3 memory entries to keep context concise
        recent_memory = memory_summary[-3:]
        
        # Create a summary string
        summary_parts = []
        for idx, entry in enumerate(recent_memory):
            timestamp = entry.get("timestamp", "unknown_time")
            game_state = entry.get("game_state", {})
            last_action = entry.get("last_action", None)
            reflection = entry.get("reflection", None)
            
            # Extract key information
            mario_pos = game_state.get("mario", {})
            game_state_info = game_state.get("game_state", {})
            immediate_threats = game_state_info.get("immediate_threats", [])
            obstacles_ahead = game_state_info.get("obstacles_ahead", [])
            
            summary = f"State {len(memory_summary) - len(recent_memory) + idx + 1}/{len(memory_summary)}:\n"
            summary += f"- Mario at grid ({mario_pos.get('x', '?')},{mario_pos.get('y', '?')})\n"
            summary += f"- Immediate threats: {', '.join(immediate_threats) if immediate_threats else 'none'}\n"
            summary += f"- Obstacles ahead: {', '.join(obstacles_ahead) if obstacles_ahead else 'none'}\n"
            if last_action:
                action_name, frame_count = last_action
                summary += f"- Last action: {action_name} for {frame_count} frames\n"
            if reflection:
                summary += f"- Reflection: {reflection}\n"
            
            summary_parts.append(summary)
            
        return "\n".join(summary_parts)
            
    def _parse_response(self, response):
        """
        Parse the reasoning response to extract thought and move.
        
        Args:
            response (str): Response from the reasoning model
            
        Returns:
            dict: Dictionary with thought and move
        """
        move = None
        thought = None
        
        # Look for thought: and move: in the response (with or without # prefix)
        lines = response.strip().split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Match both "thought:" and "# thought:" patterns
            if line.startswith("thought:") or line.startswith("# thought:"):
                prefix_len = line.find("thought:") + len("thought:")
                
                # If this is the last line, just use it
                if i == len(lines) - 1:
                    thought = line[prefix_len:].strip()
                else:
                    # If not the last line, collect all lines until we hit a move: line
                    thought_lines = []
                    thought_lines.append(line[prefix_len:].strip())
                    
                    j = i + 1
                    while j < len(lines) and not (lines[j].strip().startswith("move:") or lines[j].strip().startswith("# move:")):
                        thought_lines.append(lines[j].strip())
                        j += 1
                    
                    thought = " ".join(thought_lines).strip()
            
            # Match both "move:" and "# move:" patterns  
            elif line.startswith("move:") or line.startswith("# move:"):
                prefix_len = line.find("move:") + len("move:")
                move_text = line[prefix_len:].strip()
                
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
        if move is None or thought is None:
            print(f"Failed to parse response: {response}")

        if move is None:
            move = ("[NOOP]", 1)
            print(f"Failed to parse move from response: {response}")
        
        if thought is None:
            thought = "No explicit thought provided in response"
            print(f"Failed to parse thought from response: {response}")
        
        return {
            "move": move,
            "thought": thought
        }


class Base_module:
    """
    A simplified module that directly processes observation images and returns actions.
    This module skips separate perception and memory stages used in the full pipeline.
    """
    def __init__(self, model_name="claude-3-7-sonnet-latest", reasoning_effort="high", thinking=True):
        """
        Initialize the Base Module for direct action planning.
        
        Args:
            model_name (str): Name of the model to use for reasoning
            reasoning_effort (str): Reasoning effort level for compatible models
            thinking (bool): Whether to enable thinking mode for compatible models
        """
        self.model_name = model_name
        self.reasoning_effort = reasoning_effort
        self.thinking = thinking
        self.api_manager = APIManager(game_name="super_mario_bros")
        self.last_action = None
        
        # System prompt with strict output instructions
        self.system_prompt = """You are an intelligent AI player playing Super Mario Bros. Your goal is to help Mario progress through the level safely and efficiently.

IMPORTANT: You MUST format your response using EXACTLY these lines:
thought: [Your reasoning about the game state]
move: ([action_name], frame_count)

Do not include # or any other prefix. Start directly with "thought:" followed by your analysis."""
        
        # Prompt for direct observation analysis and action planning
        self.action_prompt = """Super Mario Bros Quick Guide:
Primary Goal: Help Mario progress through the level (move right) while avoiding hazards.
Secondary Goal: Collect coins and power-ups when safe to do so.

Action Space:
You may select from 7 discrete actions:

| Action        | Description                        |
|--------------|------------------------------------|
| [NOOP]        | Do nothing                         |
| [right]       | Move right                         |
| [right,A]     | Move right and jump                |
| [right,B]     | Move right and run                 |
| [right,A,B]   | Move right, jump, and run          |
| [A]           | Jump in place                      |
| [left]        | Move left                          |

For each decision, you need to plan an action and specify how many frames (1-30) to perform it.
Example: move: ([right,A], 15)

Important: Think about future frames when deciding on actions!

Action Planning:
- For each screenshot, you need to plan actions for multiple future frames.
- You can provide either:
  * Short action sequence (0-15 frames): e.g., move: ([right], 15)
  * Long action sequence (16-30 frames): e.g., move: ([right], 30)

Key Strategies:
- Approaching gaps: Be cautious. Use short sequences to prepare, then commit to a jump.
- Enemies: Jump over enemies when close. If unsure, move back or jump in place.
- Speed management: Don't rush as unseen enemies may appear from off-screen.
- When Mario is falling: Use [NOOP] until he lands safely.

Your response must contain:
1. thought: [Your reasoning about the game state]
2. move: ([action_name], frame_count)

Example responses:
- thought: I see a gap ahead. I need to get the right momentum before jumping.
  move: ([right,B], 15)

- thought: I see multiple enemies clustering ahead. Taking a defensive position.
  move: ([left], 8)

- thought: There's a tall pipe ahead. I need a long, high jump to clear it completely.(at least more than 25 frames can handle this tall pipe)
  move: ([right,A,B], 30)

- thought: I'm at the edge of a large gap. Need to execute a powerful jump immediately to clear it.
  move: ([right,A,B], 30)

- thought: Mario is falling down and close to the ground but not fully landed yet. I need a small frame skip before taking further actions.
  move: ([NOOP], 5)

Focus on making safe, strategic decisions to help Mario progress.
"""

    def process_observation(self, observation, img_path):
        """
        Process the observation directly to plan the next action.
        
        Args:
            observation: The game observation (RGB image)
            img_path: Path to save the observation image
            
        Returns:
            dict: A dictionary containing move and thought
        """
        try:
            # Save the observation image for the vision model
            img = Image.fromarray(observation)
            img_resized = resize_image_with_aspect_ratio(img)
            img_resized.save(img_path)
            
            # Create a simplified user prompt
            user_prompt = f"""Analyze this Super Mario Bros screenshot and choose the best action to take.

Look for:
- Mario's position
- Nearby enemies (Goombas, Koopas)
- Pits/gaps
- Pipes and other obstacles
- Power-ups

{self.action_prompt}

Based on the current game state in the image, what action should Mario take next?

IMPORTANT - FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
thought: [your analysis here]
move: ([action_name], frame_count)

Only use available actions: [NOOP], [right], [right,A], [right,B], [right,A,B], [A], [left]
Frame count must be between 1-30.
"""
            
            # Make API call with vision model
            response, _ = self.api_manager.vision_text_completion(
                model_name=self.model_name,
                system_prompt=self.system_prompt,
                prompt=user_prompt,
                image_path=img_path,
                thinking=self.thinking,
                reasoning_effort=self.reasoning_effort,
                token_limit=100000
            )
            
            # Parse the response
            return self._parse_response(response)
            
        except Exception as e:
            print(f"Error in Base_module: {e}")
            # Return a default action on error
            return {
                "move": ("[NOOP]", 1),
                "thought": f"Error occurred in Base_module: {str(e)}"
            }
    
    def _parse_response(self, response):
        """
        Parse the response to extract thought and move.
        
        Args:
            response (str): Response from the model
            
        Returns:
            dict: Dictionary with thought and move
        """
        move = None
        thought = None
        
        # Look for thought: and move: in the response (with or without # prefix)
        lines = response.strip().split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Match both "thought:" and "# thought:" patterns
            if line.startswith("thought:") or line.startswith("# thought:"):
                prefix_len = line.find("thought:") + len("thought:")
                
                # If this is the last line, just use it
                if i == len(lines) - 1:
                    thought = line[prefix_len:].strip()
                else:
                    # If not the last line, collect all lines until we hit a move: line
                    thought_lines = []
                    thought_lines.append(line[prefix_len:].strip())
                    
                    j = i + 1
                    while j < len(lines) and not (lines[j].strip().startswith("move:") or lines[j].strip().startswith("# move:")):
                        thought_lines.append(lines[j].strip())
                        j += 1
                    
                    thought = " ".join(thought_lines).strip()
            
            # Match both "move:" and "# move:" patterns  
            elif line.startswith("move:") or line.startswith("# move:"):
                prefix_len = line.find("move:") + len("move:")
                move_text = line[prefix_len:].strip()
                
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
        if move is None or thought is None:
            print(f"Failed to parse response: {response}")

        if move is None:
            move = ("[NOOP]", 1)
            print(f"Failed to parse move from response: {response}")
        
        if thought is None:
            thought = "No explicit thought provided in response"
            print(f"Failed to parse thought from response: {response}")
        
        return {
            "move": move,
            "thought": thought
        }


class SuperMarioBrosAgent:
    def __init__(self, model_name="claude-3-7-sonnet-latest", img_path=GRID_IMG_PATH, use_base_module=False):
        """
        Initialize the Super Mario Bros Agent with either the full pipeline or simplified Base_module.
        
        Args:
            model_name (str): Name of the model to use for inference
            img_path (str): Path where to save observation images for API calls
            use_base_module (bool): Whether to use the simplified Base_module instead of the full pipeline
        """
        self.model_name = model_name
        self.img_path = img_path
        self.last_action = None  # Store the last action taken
        self.use_base_module = use_base_module
        
        # Initialize modules
        if use_base_module:
            print(f"Using simplified Base_module with model: {model_name}")
            self.base_module = Base_module(model_name=model_name)
        else:
            print(f"Using full pipeline (Perception + Memory + Reasoning) with model: {model_name}")
            self.perception_module = PerceptionModule(model_name=model_name)
            self.memory_module = MemoryModule(model_name=model_name)
            self.reasoning_module = ReasoningModule(model_name=model_name)
        
    async def get_action(self, observation, reward):
        """
        Process observation to get game action using either full pipeline or Base_module.
        
        Args:
            observation: The game observation (RGB image)
            reward: The current reward from the environment
            
        Returns:
            dict: A dictionary containing move and thought
        """
        try:
            if self.use_base_module:
                # Simplified direct approach
                print("\n" + "="*80)
                action_plan = self.base_module.process_observation(observation, self.img_path)
                print("\nACTION PLAN:")
                print(f"Action: {action_plan['move']}")
                print(f"Thought: {action_plan['thought']}")
                print("="*80 + "\n")
                self.last_action = action_plan["move"]
                return action_plan
            else:
                # Full pipeline approach
                # Step 1: Perception - Analyze the frame
                perception_data = self.perception_module.analyze_frame(observation, self.img_path)
                
                # Print perception data in a readable format
                print("\n" + "="*80)
                print("PERCEPTION DATA:")
                print(f"Mario position: ({perception_data['mario']['x']}, {perception_data['mario']['y']})")
                print(f"Game state: {perception_data['game_state']}")
                print("Environment:")
                for key, value in perception_data['environment'].items():
                    if value and value != []:
                        print(f"  - {key}: {value}")
                
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
                action_plan = await self.reasoning_module.plan_action(
                    current_perception=perception_data,
                    memory_summary=memory_summary,
                    img_path=self.img_path
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
            # Fallback to a safe default action
            default_action = ("[NOOP]", 1)
            self.last_action = default_action
            return {
                "move": default_action,
                "thought": f"Error occurred: {str(e)}"
            }
        
    def parse_agent_response(self, response):
        """Legacy method for backward compatibility"""
        if self.use_base_module:
            return self.base_module._parse_response(response)
        else:
            return self.reasoning_module._parse_response(response)

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

def get_player_state(env):
    """
    Return the current player state.
    
    Args:
        env: The game environment
        
    Returns:
        int: Player state value
        
    Note:
        0x00 : Leftmost of screen
        0x01 : Climbing vine
        0x02 : Entering reversed-L pipe
        0x03 : Going down a pipe
        0x04 : Auto-walk
        0x05 : Auto-walk
        0x06 : Dead
        0x07 : Entering area
        0x08 : Normal
        0x09 : Cannot move
        0x0B : Dying
        0x0C : Palette cycling, can't move
    """
    ram = env.get_ram()
    return ram[0x000e]

async def run_actions(env, actions, fps=30):
    """
    Run a sequence of actions on the given env at the specified frames per second.
    """
    sleep_time = 1.0 / fps
    for idx, action in enumerate(actions):
        observation, reward, terminated, truncated, info = env.step(action)
        # Add x_position and player_state to info
        info['x_pos'] = get_mario_position(env)
        info['player_state'] = get_player_state(env)
        # log each step's data
        log_to_jsonl(info, idx, reward, terminated, truncated)
        env.render()
        await asyncio.sleep(sleep_time)
        
        # Handle dying state (0x0B) by continuing to use NOOP until state changes
        player_state = info['player_state']
        if player_state == 11 or player_state == 0:  # Dying state (0x0B = 11 in decimal)
            print("Mario is dying! Using NOOP actions until state changes...")
            noop_action = all_actions["[NOOP]"]
            while player_state == 11 or player_state == 0:
                observation, reward, terminated, truncated, info = env.step(noop_action)
                env.render()
                await asyncio.sleep(sleep_time)
                player_state = get_player_state(env)
                info['player_state'] = player_state
                log_to_jsonl(info, idx, reward, terminated, truncated)
                if terminated or truncated:
                    break
                
        if terminated or truncated:
            return observation, reward, terminated, truncated, info
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

def log_to_jsonl(info, count, reward, terminated, truncated, json_file=None):
    """
    Append observation, info, reward, and termination flags to a JSON Lines log file.
    """
    if json_file is None:
        json_file = os.path.join(CACHE_DIR, 'data_log.jsonl')
    
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

def store_experiment_info(model_name, game_name="SuperMarioBros-Nes"):
    """
    Store information about the current experiment in a JSON file.
    
    Args:
        model_name (str): Name of the model being used
        game_name (str): Name of the game being played
    """
    experiment_info = {
        "timestamp": datetime.now().isoformat(),
        "model_name": model_name,
        "game_name": game_name,
        "cache_directory": CACHE_DIR,
        "system_info": {
            "platform": os.name,
            "python_version": os.sys.version
        },
        "configuration": {
            "max_short_side": MAX_SHORT_SIDE,
            "max_long_side": MAX_LONG_SIDE
        }
    }
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(EXPERIMENT_INFO_FILE), exist_ok=True)
    
    # Write to JSON file
    with open(EXPERIMENT_INFO_FILE, 'w') as f:
        json.dump(experiment_info, f, indent=2)
    
    print(f"Experiment info saved to {EXPERIMENT_INFO_FILE}")

def log_agent_data(perception_data, memory_summary, action_plan, run_count, step_count):
    """
    Log agent's perception, memory, and action data to separate files.
    
    Args:
        perception_data: The agent's perception data
        memory_summary: The agent's memory summary
        action_plan: The agent's action plan (containing move and thought)
        run_count: Current run number
        step_count: Current step number
    """
    # Create log directory if it doesn't exist
    log_dir = os.path.join(CACHE_DIR, f"run_{run_count}_logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Log perception data
    if perception_data:
        perception_file = os.path.join(log_dir, f"perception_step_{step_count}.json")
        with open(perception_file, 'w') as f:
            json.dump(perception_data, f, indent=2)
    
    # Log memory summary
    if memory_summary:
        memory_file = os.path.join(log_dir, f"memory_step_{step_count}.json")
        with open(memory_file, 'w') as f:
            json.dump(memory_summary, f, indent=2)
    
    # Log action plan (thought and move)
    if action_plan:
        action_file = os.path.join(log_dir, f"action_step_{step_count}.json")
        with open(action_file, 'w') as f:
            # Convert tuple to list for JSON serialization
            if 'move' in action_plan and isinstance(action_plan['move'], tuple):
                action_plan_copy = action_plan.copy()
                action_plan_copy['move'] = list(action_plan_copy['move'])
                json.dump(action_plan_copy, f, indent=2)
            else:
                json.dump(action_plan, f, indent=2)

def generate_overall_statistics(cache_dir, all_individual_death_x_coords, averages_for_each_run, args):
    """
    Generate and save overall statistics file combining results from all runs.
    
    Args:
        cache_dir (str): Directory to save statistics
        all_individual_death_x_coords (list): List of all x-positions at deaths
        averages_for_each_run (list): List of average x-positions per run
        args: Command line arguments
    """
    try:
        print("\n========== GENERATING OVERALL STATISTICS ==========")
        
        # Convert NumPy values to native Python types
        all_death_positions = [int(pos) if hasattr(pos, 'item') else pos for pos in all_individual_death_x_coords]
        run_averages = [float(avg) if hasattr(avg, 'item') else avg for avg in averages_for_each_run]
        
        # Prepare the overall statistics data
        overall_stats = {
            "experiment_timestamp": datetime.now().isoformat(),
            "num_runs": args.num_runs,
            "model_name": args.model,
            "module_type": "random" if args.random else ("base" if args.base else "full"),
            "individual_run_statistics": [],
            "all_death_x_positions": all_death_positions,  # Use converted values
            "per_run_averages": run_averages,  # Use converted values
            "overall_average": None
        }
        
        # Add each run's statistics
        for run_num in range(1, args.num_runs + 1):
            run_stats_file = os.path.join(cache_dir, f"run_{run_num}_stats.json")
            try:
                if os.path.exists(run_stats_file):
                    with open(run_stats_file, 'r') as f:
                        run_stats = json.load(f)
                        overall_stats["individual_run_statistics"].append(run_stats)
                        
                        # Add the average to per_run_averages if it exists and not already added
                        if run_stats.get("average_death_x_position") is not None and len(run_averages) < len(averages_for_each_run):
                            # This is redundant now as we've already converted averages
                            pass
            except Exception as e:
                print(f"Error reading run {run_num} statistics: {e}")
        
        # Calculate the final overall average
        if run_averages:  # Use converted values
            overall_stats["overall_average"] = sum(run_averages) / len(run_averages)
        
        # Print statistics to console
        if all_death_positions:  # Use converted values
            print(f"All individual x-positions at death across all runs: {all_death_positions}")
        else:
            print("No deaths recorded across any runs.")

        if run_averages:  # Use converted values
            print(f"Average x-position for each run (based on deaths in that run): {[f'{avg:.2f}' for avg in run_averages]}")
            final_overall_average = sum(run_averages) / len(run_averages)
            print(f"Final average of per-run averages: {final_overall_average:.2f}")
        else:
            print("No run averages to calculate final overall average (no runs had deaths).")
        
        # First ensure all values are JSON serializable
        def ensure_serializable(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: ensure_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [ensure_serializable(item) for item in obj]
            return obj
            
        # Apply the serialization fix to the entire stats object
        overall_stats = ensure_serializable(overall_stats)
        
        # Save the overall statistics to a single file
        overall_stats_file = os.path.join(cache_dir, "overall_statistics.json")
        with open(overall_stats_file, 'w') as f:
            json.dump(overall_stats, f, indent=2)
        print(f"\nOverall statistics for all runs saved to: {overall_stats_file}")
    
    except Exception as e:
        print(f"Error generating overall statistics: {e}")
        # Try a simpler fallback
        try:
            # Convert values to ensure they're serializable
            simple_death_positions = [int(pos) if hasattr(pos, 'item') else pos for pos in all_individual_death_x_coords]
            simple_averages = [float(avg) if hasattr(avg, 'item') else avg for avg in averages_for_each_run]
            
            fallback_stats = {
                "timestamp": datetime.now().isoformat(),
                "all_death_x_positions": simple_death_positions,
                "overall_average": sum(simple_averages) / len(simple_averages) if simple_averages else None
            }
            fallback_file = os.path.join(cache_dir, "fallback_overall_stats.json")
            with open(fallback_file, 'w') as f:
                json.dump(fallback_stats, f, indent=2)
            print(f"Saved fallback statistics to: {fallback_file}")
        except Exception as e2:
            print(f"Critical error: Could not save even fallback statistics: {e2}")

async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Super Mario Bros AI agent')
    parser.add_argument('--model', type=str, default="claude-3-5-sonnet-latest", 
                        help='Model name to use for inference (default: claude-3-5-sonnet-latest)')
    parser.add_argument('--base', action='store_true', 
                        help='Use simplified Base_module instead of full pipeline')
    parser.add_argument('--random', action='store_true',
                        help='Use random actions (only forward and jump) instead of AI agent')
    parser.add_argument('--num_runs', type=int, default=1,
                        help='Number of game runs to perform (default: 1)')
    args = parser.parse_args()
    
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # Initialize structures for statistics
    all_individual_death_x_coords = []
    averages_for_each_run = []
    
    # Register the statistics generation function to run at exit
    atexit.register(
        generate_overall_statistics, 
        CACHE_DIR, 
        all_individual_death_x_coords,
        averages_for_each_run,
        args
    )
    
    # Store experiment info with added info about module choice
    experiment_info = {
        "timestamp": datetime.now().isoformat(),
        "model_name": args.model,
        "game_name": "SuperMarioBros-Nes",
        "module_type": "random" if args.random else ("base" if args.base else "full"),
        "num_runs": args.num_runs,
        "cache_directory": CACHE_DIR,
        "system_info": {
            "platform": os.name,
            "python_version": os.sys.version
        },
        "configuration": {
            "max_short_side": MAX_SHORT_SIDE,
            "max_long_side": MAX_LONG_SIDE
        }
    }
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(EXPERIMENT_INFO_FILE), exist_ok=True)
    
    # Write to JSON file
    with open(EXPERIMENT_INFO_FILE, 'w') as f:
        json.dump(experiment_info, f, indent=2)
    
    print(f"Experiment info saved to {EXPERIMENT_INFO_FILE}")
    
    env = retro.make(
        game="SuperMarioBros-Nes",
        obs_type=retro.Observations.IMAGE,
        render_mode='human',
        record=CACHE_DIR
    )
    
    # Print controller button mapping
    print("Buttons:", env.buttons)
    
    # Initialize the agent only if not using random mode
    agent = None
    if not args.random:
        agent = SuperMarioBrosAgent(
            img_path=GRID_IMG_PATH, 
            model_name=args.model,
            use_base_module=args.base
        )
        
        if args.base:
            print(f"Using Base_module with model: {args.model}")
        else:
            print(f"Using full pipeline with model: {args.model}")
    else:
        print("\n" + "*" * 80)
        print("RUNNING WITH RANDOM ACTIONS - Only forward and jump moves")
        print("*" * 80 + "\n")
    
    # Define random actions (only forward and jump)
    random_actions = [
        all_actions["[right]"],        # Move right
        all_actions["[right,A]"],      # Move right and jump
        all_actions["[right,A,B]"],    # Move right, jump, and run
        all_actions["[right,B]"],      # Move right and run
    ]
    
    # Run the game for the specified number of times
    for run_count in range(1, args.num_runs + 1):
        print(f"\n========== STARTING RUN {run_count}/{args.num_runs} ==========\n")
        
        current_run_death_x_coords = []
        
        # Reset the environment at the start of each run
        observation = env.reset() # Capture initial observation
        
        count = 0
        sleep_time = 1.0 / 30
        default_action = all_actions["[NOOP]"]
        
        # Perform an initial step to get initial info, including lives
        observation, reward, terminated, truncated, info = env.step(default_action)
        previous_lives = info.get('lives', 3) # Default to 3 if not available, though it should be
        initial_lives_for_run = previous_lives
        print(f"Run {run_count}: Starting with {initial_lives_for_run} lives. Initial x_pos: {info.get('x_pos', get_mario_position(env))}")

        # Add player state to initial info
        info['player_state'] = get_player_state(env)
        info['x_pos'] = get_mario_position(env) # Ensure x_pos is in initial log
        
        # Save initial observation with grid
        log_to_jsonl(info, count, reward, terminated, truncated)
        
        count += 1
        
        while True:
            print(f"\n--- RUN {run_count} - STEP {count} ---")
            
            current_x_pos_before_actions = get_mario_position(env) # X-pos before any action in this step

            # Check if Mario is dying and handle with NOOPs if needed
            player_state = get_player_state(env)
            if player_state == 11 or player_state == 0:  # Dying state (0x0B = 11 in decimal)
                print("Mario is dying! Using NOOP actions until state changes...")
                # This dying loop itself will use NOOPs.
                # The death would have been registered based on lives count drop *before* entering this.
                # Or, if lives drop inside this, run_actions modification would be needed.
                # For now, assume life drop is detected based on `previous_lives` against `info` from `run_actions` or the NOOP.
                
                # We need to ensure 'previous_lives' is updated correctly even through this NOOP sequence
                temp_obs, temp_rew, temp_term, temp_trunc, temp_info = observation, reward, terminated, truncated, info
                while player_state == 11 or player_state == 0:
                    temp_obs, temp_rew, temp_term, temp_trunc, temp_info = env.step(default_action)
                    temp_info['player_state'] = get_player_state(env) # Update player_state
                    temp_info['x_pos'] = get_mario_position(env)
                    current_lives_in_dying_loop = temp_info.get('lives', previous_lives)

                    if current_lives_in_dying_loop < previous_lives:
                        num_lost = previous_lives - current_lives_in_dying_loop
                        print(f"  Run {run_count}: Life lost during dying sequence. x_pos: {temp_info['x_pos']}. Prev_lives: {previous_lives}, Curr_lives: {current_lives_in_dying_loop}")
                        for _ in range(num_lost):
                            # Use x_pos from *this* step where death confirmed, or `current_x_pos_before_actions` if more appropriate
                            death_x_pos_in_dying = temp_info['x_pos'] 
                            current_run_death_x_coords.append(death_x_pos_in_dying)
                            all_individual_death_x_coords.append(death_x_pos_in_dying)
                    previous_lives = current_lives_in_dying_loop
                    
                    env.render()
                    await asyncio.sleep(sleep_time)
                    player_state = get_player_state(env) # Update for loop condition
                    log_to_jsonl(temp_info, count, temp_rew, temp_term, temp_trunc) # Log these NOOP steps
                    if temp_term or temp_trunc:
                        break
                
                observation, reward, terminated, truncated, info = temp_obs, temp_rew, temp_term, temp_trunc, temp_info
                if terminated or truncated:
                    print(f"Run {run_count} over! Environment terminated or truncated after dying sequence.")
                    break # Break from the main game loop for the run
            
            actions_for_sequence = []
            llm_action_response = None

            if args.random:
                # Random mode: Choose from forward and jump actions
                chosen_action_arr = random.choice(random_actions)
                # Random frame count between 5 and 25
                frame_count = random.randint(5, 25)
                actions_for_sequence = [chosen_action_arr] * frame_count
                
                action_name = next((k for k, v in all_actions.items() if v == chosen_action_arr), "[right]")
                llm_action_response = { # For logging consistency
                    "move": (action_name, frame_count),
                    "thought": "Random action (forward and jump only)"
                }
            else:
                # Get agent's action plan
                llm_action_response = await agent.get_action(observation, reward)
                
                # Log agent data (perception, memory, action)
                perception_data_for_log = None
                memory_summary_for_log = None
                
                if not agent.use_base_module:
                    # Re-analyze frame for current observation to get latest perception for logging
                    # This might be slightly redundant if analyze_frame was called inside get_action,
                    # but ensures we log perception corresponding to the state *before* the action.
                    # To avoid re-calling, ensure get_action returns what's needed or store it.
                    # For now, assuming agent.perception_module.analyze_frame can be called if needed,
                    # or that data is fresh enough from agent's internal call.
                    # Let's rely on agent.get_action to have done its perception.
                    # The current log_agent_data takes perception_data, memory_summary, which are not directly available here
                    # unless get_action returns them or they are stored in the agent.
                    # This part might need adjustment depending on how SuperMarioBrosAgent makes data available.
                    # Simplified: pass None if not readily available post get_action
                    pass

                log_agent_data(
                    agent.perception_module.analyze_frame(observation, GRID_IMG_PATH) if not agent.use_base_module and hasattr(agent, 'perception_module') else None, 
                    agent.memory_module.get_memory_summary() if not agent.use_base_module and hasattr(agent, 'memory_module') else None, 
                    llm_action_response, 
                    run_count, 
                    count
                )
                
                action_name, frame_count = llm_action_response['move']
                actual_action_arr = all_actions[action_name]
                actions_for_sequence = [actual_action_arr] * frame_count
            
            # State *before* this main action sequence.
            # `current_x_pos_before_actions` was captured at the start of the while loop.
            # `previous_lives` is the lives count before `run_actions`.

            # Execute the determined action sequence
            # The original script had a single NOOP step here, then run_actions.
            # observation, reward, terminated, truncated, info_after_noop = env.step(default_action)
            # await asyncio.sleep(sleep_time)
            # x_pos_before_run_actions = info_after_noop.get('x_pos', current_x_pos_before_actions)
            # lives_before_run_actions = info_after_noop.get('lives', previous_lives)
            # if lives_before_run_actions < previous_lives: # Death in the NOOP
            #     num_lost = previous_lives - lives_before_run_actions
            #     print(f"  Run {run_count}: Life lost in pre-sequence NOOP. x_pos: {x_pos_before_run_actions}")
            #     for _ in range(num_lost):
            #         current_run_death_x_coords.append(x_pos_before_run_actions)
            #         all_individual_death_x_coords.append(x_pos_before_run_actions)
            # previous_lives = lives_before_run_actions # Update for run_actions check

            # Let's use current_x_pos_before_actions as the reference if a death occurs in `run_actions`.
            # `previous_lives` should be the count *before* `run_actions`.

            if not (terminated or truncated): # Only run actions if game not already over
                observation, reward, terminated, truncated, info_after_sequence = await run_actions(env, actions_for_sequence, fps=30)
                
                current_lives_after_sequence = info_after_sequence.get('lives', previous_lives)
                
                if current_lives_after_sequence < previous_lives:
                    num_lost = previous_lives - current_lives_after_sequence
                    print(f"  Run {run_count}: {num_lost} life/lives lost during action sequence. x_pos before sequence: {current_x_pos_before_actions}. Prev_lives: {previous_lives}, Curr_lives: {current_lives_after_sequence}")
                    for _ in range(num_lost):
                        current_run_death_x_coords.append(current_x_pos_before_actions)
                        all_individual_death_x_coords.append(current_x_pos_before_actions)
                
                previous_lives = current_lives_after_sequence
                info = info_after_sequence # Update main info for logging
            else: # Game ended from dying loop or other condition before run_actions
                pass


            # Add x_position to info for the main log_to_jsonl
            # This info is from the end of run_actions or the dying loop
            info['x_pos'] = get_mario_position(env) # get latest x_pos
            log_to_jsonl(info, count, reward, terminated, truncated)
            env.render()
            count += 1
            
            if terminated or truncated:
                print(f"Run {run_count} over! Environment terminated or truncated. Final lives: {previous_lives}")
                
                # Initialize run_stats with complete, valid JSON structure
                run_stats = {
                    "run_number": run_count,
                    "death_x_positions": [],  # Default empty list that will always be valid JSON
                    "average_death_x_position": None,
                    "completed_without_deaths": True
                }

                if current_run_death_x_coords:
                    # Convert any numpy values to native Python types to ensure JSON serialization works
                    death_positions = [int(pos) if hasattr(pos, 'item') else pos for pos in current_run_death_x_coords]
                    run_average = sum(death_positions) / len(death_positions)
                    averages_for_each_run.append(run_average)
                    print(f"  Deaths in this run at x-positions: {death_positions}")
                    print(f"  Average x-position for deaths in this run: {run_average:.2f}")
                    run_stats["death_x_positions"] = death_positions
                    run_stats["average_death_x_position"] = float(run_average)  # Ensure it's a regular float
                    run_stats["completed_without_deaths"] = False
                else:
                    print("  No deaths recorded in this run for averaging (e.g., level completed or error).")

                # Write run-specific stats to a JSON file
                run_stats_file_path = os.path.join(CACHE_DIR, f"run_{run_count}_stats.json")
                try:
                    # Ensure we write a complete, valid JSON structure by using dumps first (for validation)
                    json_data = json.dumps(run_stats, indent=2)
                    with open(run_stats_file_path, 'w') as f:
                        f.write(json_data)
                    print(f"  Run statistics saved to: {run_stats_file_path}")
                except Exception as e:
                    print(f"  Error saving run statistics to {run_stats_file_path}: {e}")
                    # Try a more robust fallback with just the essential information
                    try:
                        minimal_stats = {
                            "run_number": run_count,
                            "completed": True,
                            "had_deaths": len(current_run_death_x_coords) > 0
                        }
                        with open(run_stats_file_path, 'w') as f:
                            json.dump(minimal_stats, f)
                        print(f"  Saved minimal fallback statistics instead.")
                    except:
                        print(f"  Critical failure: Could not save even minimal statistics.")
                    
                break # Exit while True loop for this run
        
        print(f"\n========== COMPLETED RUN {run_count}/{args.num_runs} ==========\n")
    
    # Close the environment after all runs are completed
    env.close()
    print("All runs completed. Environment closed.")

    # Generate overall statistics immediately (in addition to atexit handler)
    # This will run if the script completes normally
    generate_overall_statistics(CACHE_DIR, all_individual_death_x_coords, averages_for_each_run, args)
    
    # Remove the atexit handler since we've already run it
    atexit.unregister(generate_overall_statistics)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Error in main execution: {e}")
        # The atexit handler will still run to save statistics