# filename: gamingagent/agents/sokoban_agent.py
import numpy as np
import time
import os
import re
import json
import sys
from collections import deque # Added for MemoryModule
from gamingagent.envs.sokoban_env import CustomSokobanEnv
from tools.serving.api_providers import (
    anthropic_completion, anthropic_text_completion,
    openai_completion, openai_text_reasoning_completion,
    gemini_completion, gemini_text_completion,
    deepseek_text_reasoning_completion,
    together_ai_completion,
    xai_grok_completion
)
from gamingagent.utils.utils import convert_to_json_serializable # Added for JSONL logging
import argparse # Added for command-line arguments
import io
import base64
# NOTE: Pillow (PIL) is required for image conversion. Please install it: pip install Pillow
from PIL import Image

CACHE_DIR = "cache/sokoban"
# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

# --- Helper functions (can be kept separate or moved into relevant modules) ---

def matrix_to_text_table(matrix):
    """Convert a 2D list matrix into a structured text table."""
    if matrix is None:
        return "No board matrix available."

    header = "ID  | Item Type    | Position (col, row)" # Clarify position order
    line_separator = "-" * len(header)

    # Ensure the item map covers all characters used in the env
    item_map = {
        '#': 'Wall',
        '@': 'Worker',
        '$': 'Box',
        '?': 'Dock',
        '*': 'Box on Dock',
        ' ': 'Floor',
        '+': 'Worker on Dock'
    }

    table_rows = [header, line_separator]
    item_id = 1

    # Check if matrix is valid before iterating
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        print(f"Warning: Invalid matrix format received in matrix_to_text_table: {matrix}")
        return "Invalid board matrix format."


    for row_idx, row in enumerate(matrix):
        for col_idx, cell in enumerate(row):
            cell_char = str(cell)
            item_type = item_map.get(cell_char, f'Unknown ({cell_char})')
            table_rows.append(f"{item_id:<3} | {item_type:<12} | ({col_idx}, {row_idx})")
            item_id += 1

    return "\n".join(table_rows)

# Updated logging function to accept full path
def log_move_and_thought(move: str, thought: str, latency: float, log_file_path: str):
    """
    Logs the move and thought process into the specified log file.
    """
    log_entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Move: {move}, Thought: {thought}, Latency: {latency:.2f} sec\n"
    try:
        with open(log_file_path, "a") as log_file:
            log_file.write(log_entry)
    except Exception as e:
        print(f"[ERROR] Failed to write log entry to {log_file_path}: {e}")

# Updated JSONL logging function to accept full path
def log_step_data_to_jsonl(step_data: dict, log_file_path: str):
    """
    Append step data to the specified JSON Lines log file.

    Args:
        step_data (dict): Dictionary containing data for the current step.
        log_file_path (str): Path to the JSONL file.
    """
    if not log_file_path:
        print("[ERROR] No log file path provided for JSONL logging.")
        return

    try:
        # Convert NumPy types etc. to be JSON serializable
        record = convert_to_json_serializable(step_data)

        with open(log_file_path, 'a') as f:
            json.dump(record, f)
            f.write('\n')
    except Exception as e:
        print(f"[ERROR] Failed to write JSONL log entry to {log_file_path}: {e}")
        # Avoid printing potentially huge data structures
        # print(f"Data that failed: {record}")

# --- Image Conversion Helper ---
def numpy_array_to_base64(image_array):
    """Converts a NumPy array (RGB) to a base64 encoded PNG image string."""
    if image_array is None:
        return None
    try:
        pil_image = Image.fromarray(image_array.astype('uint8'), 'RGB')
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str
    except Exception as e:
        print(f"[ERROR] Failed to convert NumPy array to base64: {e}", file=sys.stderr)
        return None

# --- Perception Module ---

class PerceptionModule:
    """Handles retrieving and formatting the game state."""
    def __init__(self, env: CustomSokobanEnv):
        self.env = env

    def get_current_state(self) -> dict:
        """
        Gets the current game state as a character matrix and text table.

        Returns:
            dict: {'matrix': list[list[str]], 'text_table': str} or None on error.
        """
        char_matrix = self.env.get_char_matrix()
        if char_matrix is None:
            print("Error: PerceptionModule could not get character matrix.", file=sys.stderr)
            return None

        board_table = matrix_to_text_table(char_matrix)

        return {
            "matrix": char_matrix,
            "text_table": board_table
        }

# --- Memory Module ---

class MemoryModule:
    """Stores and retrieves history of game states, actions, and thoughts."""
    def __init__(self, max_history=5):
        self.max_history = max_history
        # Stores tuples of (perception_dict, action_data)
        # action_data = {'move': str, 'thought': str}
        self.history = deque(maxlen=max_history)

    def add_entry(self, perception_data: dict, action_data: dict):
        """Adds a new state-action entry to memory."""
        if perception_data and action_data:
            timestamp = time.time()
            self.history.append({
                "timestamp": timestamp,
                "perception": perception_data,
                "action_data": action_data
            })

    def get_memory_summary(self) -> str:
        """Provides a summary of the recent history for the prompt."""
        if not self.history:
            return "No previous actions or thoughts available."

        # Get the most recent entry (the one just before the current decision)
        last_entry = self.history[-1]
        last_action_data = last_entry.get("action_data", {"move": "N/A", "thought": "N/A"})
        move = last_action_data.get('move', 'N/A')
        thought = last_action_data.get('thought', 'N/A')

        # You could expand this to include more history if needed
        summary = f"Previous Action: {move}, Previous Thought: {thought}"
        return summary

# --- Reasoning Module ---

class ReasoningModule:
    """Uses an LLM to decide the next action based on perception and memory."""
    def __init__(self, api_provider: str, model_name: str, system_prompt: str, thinking: bool):
        self.api_provider = api_provider
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.thinking = thinking
        self.last_api_latency = 0.0

    def plan_action(self, current_perception: dict, memory_summary: str) -> dict:
        """
        Calls the LLM API to get the next move and thought.

        Args:
            current_perception (dict): Output from PerceptionModule.
            memory_summary (str): Output from MemoryModule.

        Returns:
            dict: {'move': str, 'thought': str, 'latency': float}
        """
        move = "noop" # Default move
        thought = "Error during API call or parsing."
        start_time = time.time()
        latency = 0.0

        if not current_perception or 'text_table' not in current_perception:
             thought = "Error: Invalid perception data received."
             print(f"ReasoningModule Error: {thought}", file=sys.stderr)
             return {"move": move, "thought": thought, "latency": latency}

        board_table = current_perception['text_table']

        # Construct the prompt
        prompt = (
            f"## Previous Action/Thought\\n{memory_summary}\\n\\n" # Use summary from memory module
            f"## Current Sokoban Board State\\n{board_table}\\n\\n"
            "## Task\\nAnalyze the current board state AND your previous action/thought. Decide the single best action for the worker (@ or +). "
            "Your goal is to push all boxes ($) onto the designated dock locations (?).\n\n"
            "**Legend:**\n- `@`: Worker\n- `+`: Worker on dock\n- `$`: Box\n- `*`: Box on dock\n- `?`: Empty dock\n- `#`: Wall\n- ` `: Floor\n\n"
            "**Rules:**\n- You can **move** Up, Down, Left, Right onto empty floor (` `) or docks (`?`).\n- You can **push** a box ($ or *) Up, Down, Left, Right if the space beyond it is empty (` ` or `?`).\n- Avoid deadlocks (pushing boxes into corners unnecessarily).\n\n"
            "**Instructions:**\n1. Review the current board and your last action/thought.\n"
            "2. Determine the next best action: `up`, `down`, `left`, `right` to **move**, OR `push up`, `push down`, `push left`, `push right` to **push** a box.\n"
            "3. Briefly explain your reasoning.\n\n"
            "## Output Format\\nReturn ONLY the next action and a brief thought process in the specified format:\\n"
            "move: <action>, thought: <brief_reasoning>\\n\\n"
            "**Examples:**\nmove: right, thought: Moving right to get behind the box.\nmove: push up, thought: Pushing the box at (2,3) upwards onto the dock.\nmove: left, thought: Repositioning to avoid blocking the path."

        )

        # Call the LLM API
        response = ""
        base64_image = None # Not using vision

        try:
            print(f"Calling {self.model_name} via {self.api_provider} API...")
            # --- Select API Function --- #
            if self.api_provider == "anthropic":
                response = anthropic_text_completion(self.system_prompt, self.model_name, prompt, self.thinking)
            elif self.api_provider == "openai":
                response = openai_text_reasoning_completion(self.system_prompt, self.model_name, prompt)
            elif self.api_provider == "gemini":
                response = gemini_text_completion(self.system_prompt, self.model_name, prompt)
            elif self.api_provider == "deepseek":
                response = deepseek_text_reasoning_completion(self.system_prompt, self.model_name, prompt)
            elif self.api_provider == "together_ai":
                response = together_ai_completion(self.system_prompt, self.model_name, prompt)
            elif self.api_provider == "xai":
                response = xai_grok_completion(self.system_prompt, self.model_name, prompt)
            else:
                raise NotImplementedError(f"API provider: {self.api_provider} is not supported.")
            # --- API Call End --- #
            latency = time.time() - start_time
            self.last_api_latency = latency # Store latency
            print(f"API call latency: {latency:.2f} sec")

            # Parse the response
            pattern = r'move:\s*([\w\s]+),\s*thought:\s*(.*)'
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)

            if match:
                move = match.group(1).strip().lower()
                thought = match.group(2).strip()
                print(f"LLM proposed move: {move}, thought: {thought}")
            else:
                thought = f"Could not parse move/thought from LLM response: {response}"
                print(f"Warning: {thought}")
                move = "parse_error" # Indicate parsing failure

        except Exception as e:
            latency = time.time() - start_time
            self.last_api_latency = latency
            thought = f"Error during LLM API call or processing: {e}"
            print(f"Error: {thought}", file=sys.stderr)
            move = "api_error" # Indicate API failure

        return {"move": move, "thought": thought, "latency": latency}


# --- Refactored Agent Class (With Memory) ---

class SokobanAgent:
    """
    An agent for interacting with the CustomSokobanEnv using modular components including memory.
    """
    def __init__(self, env: CustomSokobanEnv = None,
                 render_mode: str = None, api_provider: str = 'openai', model_name: str = 'gpt-4-turbo',
                 system_prompt: str = "You are an expert Sokoban player.", thinking: bool = True,
                 max_memory_history: int = 5, text_log_path: str = None, jsonl_log_path: str = None):
        """
        Initializes the modular agent with memory.
        Args:
            text_log_path (str): Path to the text log file for the run.
            jsonl_log_path (str): Path to the JSONL log file for the run.
        """
        if env:
            self.env = env
        else:
            self.env = CustomSokobanEnv(render_mode=render_mode)

        self.action_space = self.env.action_space

        # -- Use Provided Log Filenames --
        if not text_log_path or not jsonl_log_path:
            # Fallback if paths not provided (shouldn't happen with main script update)
            print("Warning: Log paths not provided to SokobanAgent. Generating default names.", file=sys.stderr)
            run_timestamp = time.strftime('%Y%m%d_%H%M%S')
            safe_model_name = re.sub(r'[^a-zA-Z0-9_-]', '_', model_name)
            base_filename = f"{run_timestamp}_{safe_model_name}_sokoban_fallback"
            self.text_log_file = os.path.join(CACHE_DIR, f"{base_filename}.log")
            self.jsonl_log_file = os.path.join(CACHE_DIR, f"{base_filename}.jsonl")
        else:
            self.text_log_file = text_log_path
            self.jsonl_log_file = jsonl_log_path
        # Print only once if possible (handled in main script now)
        # print(f"Logging text to: {self.text_log_file}")
        # print(f"Logging JSONL to: {self.jsonl_log_file}")
        # -- End Log Filename Handling --

        # Initialize Modules
        self.perception_module = PerceptionModule(self.env)
        self.memory_module = MemoryModule(max_history=max_memory_history)
        self.reasoning_module = ReasoningModule(
            api_provider=api_provider,
            model_name=model_name,
            system_prompt=system_prompt,
            thinking=thinking
        )

        # Mapping from textual moves (lowercase) to environment action IDs
        self.move_to_action_id = {
            'up': 5, 'down': 6, 'left': 7, 'right': 8,
            'push up': 1, 'push down': 2, 'push left': 3, 'push right': 4,
            'noop': 0,
            'parse_error': 0,
            'api_error': 0
        }

    def select_action(self, observation):
        """
        Selects an action by orchestrating the perception, memory, and reasoning modules.
        """
        # 1. Perception: Get current state
        current_perception = self.perception_module.get_current_state()
        if current_perception is None:
             print("Error: Agent failed to get perception. Taking NoOp action.", file=sys.stderr)
             return self.move_to_action_id['noop']

        # 2. Memory: Get summary of past actions/thoughts
        memory_summary = self.memory_module.get_memory_summary()

        # 3. Reasoning: Plan the next action based on current perception and memory
        action_plan = self.reasoning_module.plan_action(current_perception, memory_summary)
        current_move = action_plan.get('move', 'noop')
        current_thought = action_plan.get('thought', 'Reasoning failed.')
        latency = action_plan.get('latency', 0.0)

        # 4. Log the planned move and thought (using unique filename)
        log_move_and_thought(current_move, current_thought, latency, self.text_log_file)

        # 5. Memory Update: Add the *current* perception and the *planned* action/thought to memory
        self.memory_module.add_entry(current_perception, action_plan)

        # 6. Map move string to action ID
        action_id = self.move_to_action_id.get(current_move, 0)
        if current_move not in self.move_to_action_id:
             print(f"Warning: Reasoning module move '{current_move}' not in action map. Using NoOp.")

        # 7. Return action ID
        return action_id

    def run_episode(self, max_steps=200, level_index=None, render_delay=2):
        """
        Runs a single episode in the environment using the agent with memory.
        """
        total_reward = 0
        steps = 0
        options = {'level_index': level_index} if level_index is not None else None
        try:
            observation, info = self.env.reset(options=options)
            terminated = False
            truncated = False

            # Reset memory for the new episode
            self.memory_module.history.clear()

            # Log initial state (using unique filename)
            initial_perception = self.perception_module.get_current_state()
            log_step_data_to_jsonl({
                'step': 0,
                'perception': initial_perception,
                'action_plan': None,
                'reward': 0.0,
                'terminated': terminated,
                'truncated': truncated,
                'info': info,
                'api_latency': 0.0
            }, self.jsonl_log_file)

            while not terminated and not truncated and steps < max_steps:
                current_perception_for_decision = self.perception_module.get_current_state()

                if self.env.render_mode == 'human':
                    self.env.render()
                    time.sleep(render_delay)

                action = self.select_action(observation)

                last_mem_entry = self.memory_module.history[-1] if self.memory_module.history else None
                action_plan_for_log = None
                api_latency_for_log = 0.0
                if last_mem_entry:
                    action_plan_for_log = last_mem_entry.get('action_data')
                    if action_plan_for_log:
                        api_latency_for_log = action_plan_for_log.get('latency', 0.0)

                observation, reward, terminated, truncated, info = self.env.step(action)
                steps += 1
                total_reward += reward

                # Log step data (using unique filename)
                log_step_data_to_jsonl({
                    'step': steps,
                    'perception': current_perception_for_decision,
                    'action_plan': action_plan_for_log,
                    'reward': reward,
                    'terminated': terminated,
                    'truncated': truncated,
                    'info': info,
                    'api_latency': api_latency_for_log
                }, self.jsonl_log_file)

                move_for_print = action_plan_for_log.get('move', 'N/A') if action_plan_for_log else "N/A"
                thought_for_print = action_plan_for_log.get('thought', 'N/A') if action_plan_for_log else "N/A"
                print(f"Step {steps}: Action ID: {action} (Move: {move_for_print}), Thought: {thought_for_print}, Reward: {reward}")

                if terminated:
                    print(f"Episode finished after {steps} steps (Terminated). Total reward: {total_reward}")
                elif truncated:
                    print(f"Episode finished after {steps} steps (Truncated). Total reward: {total_reward}")

            if self.env.render_mode == 'human':
                 self.env.render()
                 time.sleep(1)

        except Exception as e:
            print(f"An error occurred during the episode: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return total_reward, steps

        return total_reward, steps

    def close_env(self):
        """Closes the environment."""
        self.env.close()

# --- Agent without Perception Module (Basic Agent) ---

class BasicAgent: # Renamed from DirectStateAgent
    """
    A basic agent that uses image input and a reasoning module.
    It does not instantiate PerceptionModule or MemoryModule.
    """
    def __init__(self, env: CustomSokobanEnv = None,
                 render_mode: str = None, api_provider: str = 'openai', model_name: str = 'gpt-4-turbo',
                 system_prompt: str = "You are an expert Sokoban player tasked with solving the puzzle by analyzing the provided image.", thinking: bool = True,
                 text_log_path: str = None, jsonl_log_path: str = None, image_dir_path: str = None):
        """
        Initializes the basic agent.
        Args:
            text_log_path (str): Path to the text log file for the run.
            jsonl_log_path (str): Path to the JSONL log file for the run.
            image_dir_path (str): Path to the run directory where step images will be saved.
        """
        if env:
            self.env = env
            # Ensure the provided env supports RGB rendering if needed by select_action
            if 'rgb_array' not in self.env.metadata.get('render_modes', []):
                 # If running locally, it might work anyway, but warn the user.
                 print(f"Warning: Provided environment for BasicAgent might not support 'rgb_array' render mode needed for image input.", file=sys.stderr)
        else:
            # If creating env, ensure it supports rgb_array. Force it if necessary.
            effective_render_mode = render_mode if render_mode else 'rgb_array' # Default to rgb_array if none provided
            self.env = CustomSokobanEnv(render_mode=effective_render_mode)
            if 'rgb_array' not in self.env.metadata.get('render_modes', []):
                 # This shouldn't happen with CustomSokobanEnv if mode is rgb_array, but good practice.
                 raise ValueError("BasicAgent requires an environment capable of 'rgb_array' rendering.")


        self.action_space = self.env.action_space

        # -- Use Provided Log Filenames --
        if not text_log_path or not jsonl_log_path:
             # Fallback if paths not provided (shouldn't happen with main script update)
            print("Warning: Log paths not provided to BasicAgent. Generating default names.", file=sys.stderr)
            run_timestamp = time.strftime('%Y%m%d_%H%M%S')
            safe_model_name = re.sub(r'[^a-zA-Z0-9_-]', '_', model_name)
            base_filename = f"{run_timestamp}_{safe_model_name}_sokoban_basic_vision_fallback"
            self.text_log_file = os.path.join(CACHE_DIR, f"{base_filename}.log")
            self.jsonl_log_file = os.path.join(CACHE_DIR, f"{base_filename}.jsonl")
        else:
            self.text_log_file = text_log_path
            self.jsonl_log_file = jsonl_log_path
        # Print only once if possible (handled in main script now)
        # print(f"Logging text to: {self.text_log_file}")
        # print(f"Logging JSONL to: {self.jsonl_log_file}")
        # -- End Log Filename Handling --

        self.image_dir_path = image_dir_path # Store image directory path

        if self.image_dir_path and not os.path.exists(self.image_dir_path):
             print(f"Warning: Provided image directory does not exist: {self.image_dir_path}. Image saving might fail.", file=sys.stderr)

        # Initialize Only Reasoning Module
        self.reasoning_module = ReasoningModule(
            api_provider=api_provider,
            model_name=model_name,
            system_prompt=system_prompt, # System prompt updated slightly
            thinking=thinking
        )

        # Mapping from textual moves (lowercase) to environment action IDs (remains the same)
        self.move_to_action_id = {
            'up': 5, 'down': 6, 'left': 7, 'right': 8,
            'push up': 1, 'push down': 2, 'push left': 3, 'push right': 4,
            'noop': 0,
            'parse_error': 0,
            'api_error': 0
        }

        self.last_action_data = {"move": "N/A", "thought": "Initial state"}

    def select_action(self, observation):
        """
        Selects an action based on the rendered image of the game state,
        saves the image (overwriting the previous one), and returns the action ID and image path.
        Args:
            observation: Current environment observation (not directly used for vision).

        Returns:
            tuple[int, str | None]: (action_id, path_to_saved_image or None)
        """
        # 1. Perception: Get game state as an image
        image_array = None
        saved_image_path = None # Initialize path to None
        try:
            image_array = self.env.render(mode='rgb_array')
        except Exception as e:
            print(f"Error: Failed to render environment to RGB array: {e}", file=sys.stderr)
            self.last_action_data = {"move": "noop", "thought": f"Failed to render image: {e}", "latency": 0.0}
            log_move_and_thought("noop", f"Failed to render image: {e}", 0.0, self.text_log_file)
            return self.move_to_action_id['noop'], None # Return None for path

        if image_array is None:
            print("Error: Rendered image is None. Taking NoOp action.", file=sys.stderr)
            self.last_action_data = {"move": "noop", "thought": "Rendered image was None.", "latency": 0.0}
            log_move_and_thought("noop", "Rendered image was None.", 0.0, self.text_log_file)
            return self.move_to_action_id['noop'], None # Return None for path

        # --- Save the rendered image (overwrite) --- #
        if self.image_dir_path:
            try:
                image_filename = "current_step.png" # Constant filename
                saved_image_path = os.path.join(self.image_dir_path, image_filename)
                pil_image = Image.fromarray(image_array.astype('uint8'), 'RGB')
                pil_image.save(saved_image_path, format="PNG")
                # print(f"Saved image to {saved_image_path}") # Optional: uncomment for verbose logging
            except Exception as e:
                print(f"[ERROR] Failed to save step image to {saved_image_path}: {e}", file=sys.stderr)
                saved_image_path = None # Ensure path is None if saving failed
        # --- End image saving --- #

        # 2. Convert image to base64
        base64_image = numpy_array_to_base64(image_array)
        if base64_image is None:
            print("Error: Failed to convert image to base64. Taking NoOp action.", file=sys.stderr)
            self.last_action_data = {"move": "noop", "thought": "Failed image base64 conversion.", "latency": 0.0}
            log_move_and_thought("noop", "Failed image base64 conversion.", 0.0, self.text_log_file)
            # Return the path even if conversion failed, as image might have been saved
            return self.move_to_action_id['noop'], saved_image_path

        # 3. Define the NEW full prompt for the Reasoning Module (using image)
        full_prompt = f"""
## Sokoban Game Task (Image Input)

You are playing Sokoban. Your goal is to push all boxes onto the designated dock locations. Analyze the provided image of the current game state.

**Image Legend:**
- Human Figure (Blue Shirt, Jeans): Worker (You)
- Brown Wooden Crates: Boxes
- Dashed Square with 'x': Dock locations (Targets)
- Gray Brick Blocks: Walls (Impassable)
- Sandy/Beige Floor: Empty space

**Rules:**
- You can **move** the worker Up, Down, Left, Right onto empty floor spaces or docks.
- You can **push** a single box Up, Down, Left, or Right if the space beyond the box in the push direction is empty.
- You cannot push boxes into walls or other boxes.
- You win when all boxes are on docks.

**Task:**
Based ONLY on the provided image, decide the single best action for the worker.
- If you want to **move** into an empty space, specify the direction: `up`, `down`, `left`, or `right`.
- If you want to **push** a box, specify the action: `push up`, `push down`, `push left`, or `push right`.
Prioritize actions that make progress towards pushing boxes to docks. Avoid actions that could lead to deadlocks (e.g., pushing boxes into corners unless it's onto a dock).

**Output Format:**
Return ONLY the next action and a brief thought process in the specified format:
move: <action>, thought: <brief_reasoning>

**Examples:**
move: right, thought: Moving the worker right to get behind the red box.
move: push down, thought: Pushing the box below the worker downwards towards the green dock.
move: push left, thought: Pushing the box to the left onto the target square.
"""

        # 4. Reasoning: Call vision-capable API function.
        start_time = time.time()
        response = ""
        move = "noop"
        thought = "Error during API call or parsing."
        latency = 0.0

        try:
            print(f"Calling {self.reasoning_module.model_name} via {self.reasoning_module.api_provider} API with image... (BasicAgent)")

            # --- Select VISION API Function --- #
            # NOTE: Ensure the model_name used actually supports vision for the provider!
            if self.reasoning_module.api_provider == "anthropic":
                 # Assuming anthropic_completion can handle image/text prompts
                 response = anthropic_completion(self.reasoning_module.system_prompt, self.reasoning_module.model_name, base64_image, full_prompt, self.reasoning_module.thinking)
            elif self.reasoning_module.api_provider == "openai":
                 # Assuming openai_completion can handle image/text prompts
                 response = openai_completion(self.reasoning_module.system_prompt, self.reasoning_module.model_name, base64_image, full_prompt)
            elif self.reasoning_module.api_provider == "gemini":
                 # Assuming gemini_completion can handle image/text prompts
                 response = gemini_completion(self.reasoning_module.system_prompt, self.reasoning_module.model_name, base64_image, full_prompt)
            # Add other providers here if they have vision capabilities AND you have corresponding functions
            # elif self.reasoning_module.api_provider == "deepseek":
            #     # Assuming deepseek has a vision-specific function or completion handles it
            #     response = deepseek_???_completion(...)
            # elif self.reasoning_module.api_provider == "together_ai":
            #     # Assuming together_ai_completion handles vision or has a specific function
            #     response = together_ai_completion(...)
            # elif self.reasoning_module.api_provider == "xai":
            #      # Assuming xai_grok_completion handles vision or has a specific function
            #      response = xai_grok_completion(...)
        

            latency = time.time() - start_time
            print(f"API call latency: {latency:.2f} sec")

            # Parse the response (same parsing logic)
            pattern = r'move:\s*([\w\s]+),\s*thought:\s*(.*)'
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                move = match.group(1).strip().lower()
                thought = match.group(2).strip()
                print(f"LLM proposed move: {move}, thought: {thought}")
            else:
                thought = f"Could not parse move/thought from LLM response: {response}"
                print(f"Warning: {thought}")
                move = "parse_error"

        except Exception as e:
            latency = time.time() - start_time
            thought = f"Error during LLM API call or processing: {e}"
            print(f"Error: {thought}", file=sys.stderr)
            move = "api_error"

        action_plan = {"move": move, "thought": thought, "latency": latency}

        # 5. Log the planned move and thought
        log_move_and_thought(move, thought, latency, self.text_log_file)

        # 6. Store action data for logging in run_episode
        self.last_action_data = action_plan

        # 7. Map move string to action ID
        action_id = self.move_to_action_id.get(move, 0)
        if move not in self.move_to_action_id:
             print(f"Warning: LLM move '{move}' not in action map. Using NoOp.")

        # 8. Return action ID and saved image path
        return action_id, saved_image_path

    def run_episode(self, max_steps=200, level_index=None, render_delay=2):
        """
        Runs a single episode in the environment using the basic agent (vision based).
        NOTE: JSONL logging still uses text-based perception for context, but now includes image path.
        """
        total_reward = 0
        steps = 0
        options = {'level_index': level_index} if level_index is not None else None
        last_saved_image_path = None # Track the path saved in the *previous* step for logging
        try:
            observation, info = self.env.reset(options=options)
            terminated = False
            truncated = False

            # Log initial state (using text representation for log context)
            initial_matrix = self.env.get_char_matrix()
            initial_perception_for_log = {
                'matrix': initial_matrix,
                'text_table': matrix_to_text_table(initial_matrix)
            }
            log_step_data_to_jsonl({
                'step': 0,
                'perception': initial_perception_for_log, # Log text state
                'action_plan': None,
                'image_path': None, # No image for initial state
                'reward': 0.0,
                'terminated': terminated,
                'truncated': truncated,
                'info': info,
                'api_latency': 0.0
            }, self.jsonl_log_file)

            while not terminated and not truncated and steps < max_steps:
                # Get text state for logging *before* taking the action
                current_matrix_for_log = self.env.get_char_matrix()
                current_perception_for_log = {
                     'matrix': current_matrix_for_log,
                     'text_table': matrix_to_text_table(current_matrix_for_log)
                 }

                if self.env.render_mode == 'human':
                    # If human rendering is enabled, render *before* the action is selected/taken
                    self.env.render()
                    time.sleep(render_delay)

                # Action selection now uses the image internally, saves it, and returns path
                action, current_saved_image_path = self.select_action(observation)

                # Get the action plan decided in select_action
                action_plan_for_log = self.last_action_data
                api_latency_for_log = action_plan_for_log.get('latency', 0.0) if action_plan_for_log else 0.0

                # --- Log step data --- #
                # Log the perception state *before* the action, and the image path that was saved *during* this step's decision
                log_step_data_to_jsonl({
                    'step': steps + 1, # Log step is 1-indexed for consistency with printed output
                    'perception': current_perception_for_log, # Log text state from before action
                    'action_plan': action_plan_for_log, # Log action/thought/latency from select_action
                    'image_path': current_saved_image_path, # Log path saved by select_action
                    'reward': None, # Reward will be available after env.step
                    'terminated': None,
                    'truncated': None,
                    'info': None,
                    'api_latency': api_latency_for_log
                }, self.jsonl_log_file)
                # --- End logging --- #

                # --- Take action in environment --- #
                observation, reward, terminated, truncated, info = self.env.step(action)
                steps += 1
                total_reward += reward
                # --- End take action --- #

                # --- Update last log entry with outcome --- #
                # For simplicity, we'll keep it as is: the log entry for step N contains the state before action N
                # and the action plan for N. The reward/termination for action N is implicitly shown in the next step's entry.

                move_for_print = action_plan_for_log.get('move', 'N/A') if action_plan_for_log else "N/A"
                thought_for_print = action_plan_for_log.get('thought', 'N/A') if action_plan_for_log else "N/A"
                print(f"Step {steps}: Action ID: {action} (Move: {move_for_print}), Thought: {thought_for_print}, Reward: {reward}")

                if terminated:
                    print(f"Episode finished after {steps} steps (Terminated). Total reward: {total_reward}")
                elif truncated:
                    print(f"Episode finished after {steps} steps (Truncated). Total reward: {total_reward}")

            # Render final state if human mode enabled
            if self.env.render_mode == 'human':
                 self.env.render()
                 time.sleep(1)

        except Exception as e:
            print(f"An error occurred during the episode: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return total_reward, steps

        return total_reward, steps

    def close_env(self):
        """Closes the environment."""
        self.env.close()


# Example usage (updated for agent selection and level looping)
if __name__ == "__main__":
    # --- Configuration & Argument Parsing --- #
    parser = argparse.ArgumentParser(description="Run Sokoban LLM Agent")
    parser.add_argument("--api_provider", type=str, default="openai",
                        choices=["openai", "anthropic", "gemini", "deepseek", "together_ai", "xai"],
                        help="API provider to use.")
    parser.add_argument("--model_name", type=str, default="gpt-4o",
                        help="Specific model name for the chosen provider.")
    parser.add_argument("--level", type=int, default=1, # Restored level argument
                        help="Starting level index to test.")
    parser.add_argument("--steps", type=int, default=100,
                        help="Maximum steps per episode.")
    parser.add_argument("--render_mode", type=str, default="human",
                        choices=["human", "rgb_array", "tiny_human", "tiny_rgb_array", "raw", "none"],
                        help="Rendering mode (or 'none' to disable).")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Delay in seconds between steps when rendering.")
    parser.add_argument("--memory", type=int, default=5,
                        help="Number of past steps for memory (used by 'full' agent type).")
    parser.add_argument("--agent_type", type=str, default="full",
                        choices=["full", "basic"],
                        help="Type of agent to run ('full' includes memory/perception modules, 'basic' uses direct state/vision).")

    args = parser.parse_args()

    # Use parsed arguments
    API_PROVIDER = args.api_provider
    MODEL_NAME = args.model_name
    STARTING_LEVEL = args.level # Use the provided level as starting point
    MAX_STEPS_PER_EPISODE = args.steps
    RENDER_MODE = args.render_mode if args.render_mode != "none" else None
    RENDER_DELAY = args.delay
    MAX_MEMORY = args.memory
    AGENT_TYPE = args.agent_type
    # --- End Configuration --- #

    # --- Generate Base Run Name & Create Run Directory ---
    run_timestamp = time.strftime('%Y%m%d_%H%M%S')
    safe_model_name = re.sub(r'[^a-zA-Z0-9_-]', '_', MODEL_NAME)
    agent_suffix = "basic_vision" if AGENT_TYPE == "basic" else "full"
    # Base name used for the run directory and log files inside it
    base_run_name = f"{run_timestamp}_{safe_model_name}_sokoban_{agent_suffix}"

    # Create the main directory for this run
    run_directory = os.path.join(CACHE_DIR, base_run_name)
    os.makedirs(run_directory, exist_ok=True)
    print(f"Created run directory: {run_directory}")
    # --- End Directory Creation ---

    # --- Define Log Paths and Image Path (within the run directory) ---
    text_log_file = os.path.join(run_directory, f"{base_run_name}.log")
    jsonl_log_file = os.path.join(run_directory, f"{base_run_name}.jsonl")
    print(f"Logging text for this run to: {text_log_file}")
    print(f"Logging JSONL for this run to: {jsonl_log_file}")

    # Image directory path is the run directory itself (only relevant for basic/vision agent)
    image_storage_path = None
    if AGENT_TYPE == "basic":
        image_storage_path = run_directory # Images saved directly into the run directory
        print(f"Saving step images for this run to: {image_storage_path}")
    # --- End Path Definitions ---

    current_level_index = STARTING_LEVEL # Start from the specified level
    while True:
        print("\n" + "="*30 + f" Attempting Level {current_level_index} " + "="*30 + "\n")

        agent = None # Initialize agent to None for this level attempt
        try:
            # Create the selected agent type for the current level
            if AGENT_TYPE == "full":
                print(f"Initializing SokobanAgent for Level {current_level_index}...")
                agent = SokobanAgent(
                    render_mode=RENDER_MODE,
                    api_provider=API_PROVIDER,
                    model_name=MODEL_NAME,
                    max_memory_history=MAX_MEMORY,
                    text_log_path=text_log_file,     # Pass fixed path
                    jsonl_log_path=jsonl_log_file    # Pass fixed path
                )
            elif AGENT_TYPE == "basic":
                print(f"Initializing BasicAgent for Level {current_level_index}...")
                agent = BasicAgent(
                    render_mode=RENDER_MODE,
                    api_provider=API_PROVIDER,
                    model_name=MODEL_NAME,
                    text_log_path=text_log_file,    # Pass log path
                    jsonl_log_path=jsonl_log_file,  # Pass log path
                    image_dir_path=image_storage_path # Pass run directory as image path
                )

            if agent: # Only run if agent was successfully initialized
                print(f"Running episode for Level {current_level_index} using {API_PROVIDER}/{MODEL_NAME} with {AGENT_TYPE} agent...")
                reward, steps_taken = agent.run_episode(
                    level_index=current_level_index, # Pass level to run
                    max_steps=MAX_STEPS_PER_EPISODE,
                    render_delay=RENDER_DELAY
                )
                print(f"Level {current_level_index} finished. Total Reward: {reward}, Steps: {steps_taken}")

                # Check if the episode ended because the step limit was reached
                if steps_taken >= MAX_STEPS_PER_EPISODE:
                    print(f"Level {current_level_index} failed by exceeding the step limit ({MAX_STEPS_PER_EPISODE}). Terminating.")
                    agent.close_env() # Ensure environment is closed before breaking
                    break # Exit the while True loop

                agent.close_env() # Close env specific to this agent instance
                # Increment level ONLY after successful completion or non-step-limit failure
                current_level_index += 1
            else:
                print(f"Error: Could not initialize agent type '{AGENT_TYPE}' for Level {current_level_index}", file=sys.stderr)
                break

        except (ValueError, FileNotFoundError, RuntimeError) as e:
            # Catch errors likely coming from _load_level_from_file during env.reset()
            print(f"\nCould not load or run Level {current_level_index}. Assuming it's the last level or an error occurred.")
            print(f"Error details: {e}")
            if agent:
                agent.close_env()
            break # Exit the loop
        except Exception as e:
            # Catch any other unexpected errors
            print(f"\nAn unexpected error occurred during Level {current_level_index}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            if agent:
                agent.close_env()
            break # Exit the loop