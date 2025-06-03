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
from PIL import Image
import traceback

CACHE_DIR = "cache/sokoban"
os.makedirs(CACHE_DIR, exist_ok=True)

# --- Helper functions ---
def matrix_to_text_table(matrix):
    if matrix is None: return "No board matrix available."
    item_map = {'#': 'Wall', '@': 'Worker', '$': 'Box', '?': 'Dock', '*': 'Box on Dock', ' ': 'Floor', '+': 'Worker on Dock'}
    header = "ID  | Item Type    | Position (col, row)"
    rows = [header, "-" * len(header)]
    item_id = 1
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        print(f"Warning: Invalid matrix format: {matrix}", file=sys.stderr)
        return "Invalid board matrix format."
    for r, row_data in enumerate(matrix):
        for c, cell in enumerate(row_data):
            item_type = item_map.get(str(cell), f'Unknown ({cell})')
            rows.append(f"{item_id:<3} | {item_type:<12} | ({c}, {r})")
            item_id += 1
    return "\\n".join(rows)

def log_move_and_thought(
    log_file_path: str, level_index: int, step_num: int, action_id: int, move: str, thought: str, 
    latency: float, reward: float, boxes_on_target: int, total_boxes: int, 
    terminated: bool, truncated: bool
):
    """Logs a comprehensive single line for the current step, including level index."""
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    term_str = "T" if terminated else "F"
    trunc_str = "T" if truncated else "F"
    lvl_str = f"L{level_index}" if level_index is not None else "L?"
    
    # Corrected format string to include all parts
    log_entry = (
        f"[{ts}] {lvl_str} Step: {step_num:<3} | Act: {action_id} ({move}) | "
        f"Rw: {reward:<+5.1f} | Box: {boxes_on_target}/{total_boxes} | Term: {term_str} | Trunc: {trunc_str} | "
        f"Lat: {latency:<4.2f}s | Th: {thought.strip()}\n"
    )
    try:
        with open(log_file_path, "a") as log_file: log_file.write(log_entry)
    except Exception as e:
        print(f"[ERROR] Failed to write log entry to {log_file_path}: {e}")

# --- Image Conversion Helper ---
def numpy_array_to_base64(image_array):
    """Converts a NumPy array (RGB) to a base64 encoded PNG image string."""
    if image_array is None: return None
    try:
        pil_img = Image.fromarray(image_array.astype('uint8'), 'RGB')
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e: print(f"[ERROR] Numpy to base64 failed: {e}", file=sys.stderr); return None

# --- Perception Module ---
class PerceptionModule:
    def __init__(self, env: CustomSokobanEnv): self.env = env
    def get_current_state(self) -> dict | None:
        char_matrix = self.env.get_char_matrix()
        if char_matrix is None:
            print("Error: PerceptionModule could not get char matrix.", file=sys.stderr)
            return None
        return {"matrix": char_matrix, "text_table": matrix_to_text_table(char_matrix)}

# --- Memory Module ---
class MemoryModule:
    def __init__(self, max_history=5):
        self.history = deque(maxlen=max_history)
    def add_entry(self, perception_data: dict, action_data: dict):
        # perception_data for SokobanAgent is {'matrix': ..., 'text_table': ...}
        # perception_data for MemoryOnlyAgent could be {'image_base64': ...}
        if perception_data and action_data:
            self.history.append({"timestamp": time.time(), "perception": perception_data, "action_data": action_data})
    def get_memory_summary(self) -> str:
        if not self.history: return "No previous actions or thoughts available."
        last_entry = self.history[-1]
        action_data = last_entry.get("action_data", {"move": "N/A", "thought": "N/A"})
        return f"Previous Action: {action_data.get('move', 'N/A')}, Previous Thought: {action_data.get('thought', 'N/A')}"
    def clear(self): self.history.clear()

# --- Reasoning Module (Refactored) ---
class ReasoningModule:
    def __init__(self, api_provider: str, model_name: str, system_prompt: str, thinking: bool):
        self.api_provider = api_provider
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.thinking = thinking

    def _call_llm_api(self, prompt: str, base64_image: str | None = None) -> str:
        response_text = ""
        # print(f"DEBUG: ReasoningModule calling {self.model_name} via {self.api_provider}. Image: {bool(base64_image)}")
        # print(f"DEBUG: Prompt (first 200 chars): {prompt[:200]}")
        if self.model_name == "o1-mini":
            base64_image = None
        if base64_image:
            if self.api_provider == "anthropic": response_text = anthropic_completion(self.system_prompt, self.model_name, base64_image, prompt, self.thinking)
            elif self.api_provider == "openai": response_text = openai_completion(self.system_prompt, self.model_name, base64_image, prompt)
            elif self.api_provider == "gemini": response_text = gemini_completion(self.system_prompt, self.model_name, base64_image, prompt)
            elif self.api_provider == "together_ai": response_text = together_ai_completion(self.system_prompt, self.model_name, prompt, base64_image)
            # Add other vision-capable providers here
            # elif self.api_provider == "deepseek" and hasattr(globals().get('deepseek_vision_completion', None), '__call__'):
            # response_text = deepseek_vision_completion(self.system_prompt, self.model_name, base64_image, prompt)
            else: raise NotImplementedError(f"Vision API provider: {self.api_provider} not supported for vision.")
        else:
            if self.api_provider == "anthropic": response_text = anthropic_text_completion(self.system_prompt, self.model_name, prompt, self.thinking)
            elif self.api_provider == "openai": response_text = openai_text_reasoning_completion(self.system_prompt, self.model_name, prompt)
            elif self.api_provider == "gemini": response_text = gemini_text_completion(self.system_prompt, self.model_name, prompt)
            elif self.api_provider == "deepseek": response_text = deepseek_text_reasoning_completion(self.system_prompt, self.model_name, prompt)
            elif self.api_provider == "together_ai": response_text = together_ai_completion(self.system_prompt, self.model_name, prompt)
            elif self.api_provider == "xai": response_text = xai_grok_completion(self.system_prompt, self.model_name, prompt)
            else: raise NotImplementedError(f"Text API provider: {self.api_provider} not supported.")
        return response_text

    def _parse_response(self, response_text: str) -> tuple[str, str]:
        pattern = r'^move:\s*([\w\s]+),\s*thought:\s*(.*)' # Anchor to start of line
        matches = list(re.finditer(pattern, response_text, re.IGNORECASE | re.MULTILINE))
        
        if matches:
            # Use the last match found
            last_match = matches[-1]
            move = last_match.group(1).strip().lower()
            thought = last_match.group(2).strip()
            # Clean up potential unwanted prefixes from the thought itself
            thought = re.sub(r'(?i)^\s*the final answer is:\s*', '', thought).strip()
            return move, thought
        else:
            # Fallback if the specific pattern is not found at all
            return "parse_error", f"Could not parse move/thought pattern from LLM response: {response_text}"

    def plan_action_with_text(self, current_perception_text_table: str, memory_summary: str | None) -> dict:
        start_time = time.time()
        mem_summary = memory_summary if memory_summary else "No previous actions or thoughts available."
        # Restoring detailed Rules, Instructions, Output Format but NO Legend
        prompt = (
            f"## Previous Action/Thought\n{mem_summary}\n\n"
            f"## Current Sokoban Board State\n{current_perception_text_table}\n\n"
            "## Task\nAnalyze the current board state (provided above) AND your previous action/thought. "
            "Decide the single best action for the worker. "
            "Your goal is to push all boxes onto the designated dock locations.\n\n"
            "**Rules:**\n- You can **move** Up, Down, Left, Right onto empty floor (` `) or docks (`?`).\n"
            "- You can **push** a box ($ or *) Up, Down, Left, Right if the space beyond it is empty (` ` or `?`).\n"
            "- Avoid deadlocks (pushing boxes into corners unnecessarily).\n\n"
            "**Instructions:**\n1. Review the current board and your last action/thought.\n"
            "2. Determine the next best action: `up`, `down`, `left`, `right` to **move**, OR `push up`, `push down`, `push left`, `push right` to **push** a box.\n"
            "3. Briefly explain your reasoning.\n\n"
            "## Output Format\nReturn ONLY the next action and a brief thought process in the specified format:\n"
            "move: <action>, thought: <brief_reasoning>\n\n"
            "**Examples:**\nmove: right, thought: Moving right to get behind the box.\nmove: push up, thought: Pushing the box at (2,3) upwards onto the dock." # Added examples back
        )
        try:
            # print(f"[{self.__class__.__name__}] Calling LLM for text-based plan...")
            raw_response = self._call_llm_api(prompt)
            latency = time.time() - start_time
            move, thought = self._parse_response(raw_response)
            print(f"Text API latency: {latency:.2f} sec. Move: {move}. Thought: {thought}")
            return {"move": move, "thought": thought, "latency": latency}
        except Exception as e:
            latency = time.time() - start_time
            thought = f"Error in text LLM call: {e}"
            print(f"Error: {thought}", file=sys.stderr); traceback.print_exc()
            return {"move": "api_error", "thought": thought, "latency": latency}

    def plan_action_with_vision(self, base64_image: str, memory_summary: str | None) -> dict:
        start_time = time.time()
        mem_summary = memory_summary if memory_summary else "No previous actions or thoughts available."
        # Restoring detailed sections, including Image Legend
        prompt = (
            f"## Previous Action/Thought\n{mem_summary}\n\n"
            "## Sokoban Game Task (Image Input)\n"
            "You are playing Sokoban. Analyze the provided image and your previous action/thought (if any) "
            "to push all boxes to docks.\n\n"
            "**Image Legend:**\n- Human Figure (Blue Shirt, Jeans): Worker (You)\n- Brown Wooden Crates: Boxes\n- Dashed Square with 'x': Dock locations (Targets)\n- Gray Brick Blocks: Walls (Impassable)\n- Sandy/Beige Floor: Empty space\n\n" # Detailed legend
            "**Rules:**\n- You can **move** the worker Up, Down, Left, Right onto empty floor spaces or docks.\n"
            "- You can **push** a single box Up, Down, Left, or Right if the space beyond the box in the push direction is empty.\n"
            "- You cannot push boxes into walls or other boxes.\n"
            "- You win when all boxes are on docks.\n\n" # Detailed rules
            "**Task:**\nBased on the provided image AND your memory of the last action (if available), decide the single best action.\n"
            "- If you want to **move**, specify: `up`, `down`, `left`, or `right`.\n"
            "- If you want to **push** a box, specify: `push up`, `push down`, `push left`, or `push right`.\n"
            "Prioritize progress. Avoid deadlocks.\n\n" # Detailed task
            "**Output Format:**\nReturn ONLY the next action and a brief thought process:\n"
            "move: <action>, thought: <brief_reasoning>\n\n" # Detailed format
            "**Examples:**\nmove: right, thought: Moving the worker right to get behind the red box.\nmove: push down, thought: Pushing the box below the worker downwards towards the green dock." # Detailed examples
        )
        try:
            # print(f"[{self.__class__.__name__}] Calling LLM for vision-based plan...")
            raw_response = self._call_llm_api(prompt, base64_image=base64_image)
            latency = time.time() - start_time
            move, thought = self._parse_response(raw_response)
            print(f"Vision API latency: {latency:.2f} sec. Move: {move}. Thought: {thought}")
            return {"move": move, "thought": thought, "latency": latency}
        except Exception as e:
            latency = time.time() - start_time
            thought = f"Error in vision LLM call: {e}"
            print(f"Error: {thought}", file=sys.stderr); traceback.print_exc()
            return {"move": "api_error", "thought": thought, "latency": latency}

# --- Base Agent Class ---
class BaseSokobanAgent:
    DEFAULT_MOVE_MAP = {
            'up': 5, 'down': 6, 'left': 7, 'right': 8,
            'push up': 1, 'push down': 2, 'push left': 3, 'push right': 4,
        'noop': 0, 'parse_error': 0, 'api_error': 0
    }
    def __init__(self, env_render_mode: str | None, api_provider: str, model_name: str,
                 system_prompt: str, thinking: bool,
                 text_log_path: str,
                 agent_type_suffix: str, env: CustomSokobanEnv | None = None):
        self.env = env if env else CustomSokobanEnv(render_mode=env_render_mode)
        self.text_log_file = text_log_path
        self.agent_type_suffix = agent_type_suffix # For logging/identification
        self.last_action_plan = {"move": "noop", "thought": "Initial state", "latency": 0.0}

        self.reasoning_module = ReasoningModule(
            api_provider=api_provider, model_name=model_name,
            system_prompt=system_prompt, thinking=thinking
        )
        # Ensure action space is available if env is created here
        self.action_space = self.env.action_space 

    def _get_action_id(self, move_str: str) -> int:
        action_id = self.DEFAULT_MOVE_MAP.get(move_str, 0)
        if move_str not in self.DEFAULT_MOVE_MAP:
            print(f"Warning: LLM move '{move_str}' not in action map. Using NoOp.", file=sys.stderr)
        return action_id

    def run_episode(self, max_steps=200, level_index=None, render_delay=0.1):
        total_reward = 0
        steps = 0
        options = {'level_index': level_index} if level_index is not None else None
        saved_image_path_for_log = None # For vision agents

        try:
            observation, info = self.env.reset(options=options)
            terminated, truncated = False, False

            # Reset memory for the new episode if agent uses it
            print(f"[DEBUG] Agent {type(self).__name__}: hasattr(self, 'memory_module') = {hasattr(self, 'memory_module')}")
            if hasattr(self, 'memory_module') and self.memory_module:
                print(f"[DEBUG] Agent {type(self).__name__}: self.memory_module is {self.memory_module}")
                self.memory_module.clear()

            # For vision agents, the initial text perception is still useful for context before first image.
            initial_text_perception_for_log = PerceptionModule(self.env).get_current_state()

            while not terminated and not truncated and steps < max_steps:
                if self.env.render_mode == 'human': self.env.render(); time.sleep(render_delay)

                action_result = self.select_action(observation) # Returns action_id or (action_id, image_path)
                
                if isinstance(action_result, tuple): # Vision agents return image path
                    action_id, saved_image_path_for_log = action_result
                else: # Text agents return only action_id
                    action_id = action_result
                    saved_image_path_for_log = None

                action_plan_for_log = self.last_action_plan # From agent's select_action

                observation, reward, terminated, truncated, info = self.env.step(action_id)
                steps += 1
                total_reward += reward
                print(f"[DEBUG] Step {steps} Outcome: Reward={reward}, InfoBoxesOnTarget={info.get('boxes_on_target', 'N/A')}") # Debug Print

                # Log the comprehensive step information to the text log
                log_move_and_thought(
                    log_file_path=self.text_log_file,
                    level_index=level_index,
                    step_num=steps,
                    action_id=action_id,
                    move=action_plan_for_log.get('move', 'N/A'),
                    thought=action_plan_for_log.get('thought', 'N/A'),
                    latency=action_plan_for_log.get('latency', 0.0),
                    reward=reward,
                    boxes_on_target=info.get('boxes_on_target', 0),
                    total_boxes=self.env.num_boxes, # Assuming env has num_boxes attribute
                    terminated=terminated,
                    truncated=truncated
                )

                move_p = action_plan_for_log.get('move', 'N/A')
                thought_p = action_plan_for_log.get('thought', 'N/A')
                print(f"Step {steps}: ActionID={action_id} (Move: {move_p}), Thought: {thought_p}, Reward: {reward}")

                if terminated or truncated: break
            
            status = "Terminated" if terminated else "Truncated" if truncated else "MaxSteps"
            print(f"Level {level_index if level_index else 'random'} finished: {status} in {steps} steps. Reward: {total_reward}")
            if self.env.render_mode == 'human': self.env.render(); time.sleep(1)

        except Exception as e:
            print(f"Error in {self.__class__.__name__} run_episode: {e}", file=sys.stderr)
            traceback.print_exc()
        return total_reward, steps

    def select_action(self, observation):
        raise NotImplementedError("Subclasses must implement select_action")

    def close_env(self): self.env.close()

# --- SokobanAgent (Full: Perception + Memory + Text Reasoning) ---
class SokobanAgent(BaseSokobanAgent):
    def __init__(self, env: CustomSokobanEnv = None, render_mode: str = None,
                 api_provider: str = 'openai', model_name: str = 'gpt-4-turbo',
                 system_prompt: str = "You are an expert Sokoban player.", thinking: bool = True,
                 max_memory_history: int = 5, text_log_path: str = None):
        super().__init__(render_mode, api_provider, model_name, system_prompt, thinking,
                         text_log_path, "full", env)
        self.perception_module = PerceptionModule(self.env)
        self.memory_module = MemoryModule(max_history=max_memory_history)

    def select_action(self, observation):
        current_perception = self.perception_module.get_current_state()
        if not current_perception:
            self.last_action_plan = {"move": "noop", "thought": "Perception failed", "latency": 0.0}
            return self._get_action_id("noop"), None

        memory_summary = self.memory_module.get_memory_summary()
        self.last_action_plan = self.reasoning_module.plan_action_with_text(
            current_perception['text_table'], memory_summary
        )
        self.memory_module.add_entry(current_perception, self.last_action_plan)
        return self._get_action_id(self.last_action_plan['move']), None

# --- BasicAgent (Vision Reasoning, No Perception Module, No Memory) ---
class BasicAgent(BaseSokobanAgent):
    def __init__(self, env: CustomSokobanEnv = None, render_mode: str = None,
                 api_provider: str = 'openai', model_name: str = 'gpt-4-turbo', # Ensure this model supports vision
                 system_prompt: str = "You are a Sokoban expert analyzing an image.", thinking: bool = True,
                 text_log_path: str = None, image_dir_path: str = None):
        # BasicAgent needs rgb_array for its own perception if env is created by it
        effective_render_mode = render_mode if render_mode else 'rgb_array'
        super().__init__(effective_render_mode, api_provider, model_name, system_prompt, thinking,
                         text_log_path, "basic_vision", env)
        if 'rgb_array' not in self.env.metadata.get('render_modes', []):
            # This might be too strict if an env is passed in that CAN render rgb_array but doesn't list it
            # However, it's safer to ensure compatibility.
            # Consider a more nuanced check or relying on render() to fail if not possible.
             print(f"Warning: BasicAgent created with env not explicitly listing 'rgb_array' in render_modes. Effective mode: {self.env.render_mode}", file=sys.stderr)

        self.image_dir_path = image_dir_path # For saving images, used in select_action

    def select_action(self, observation):
        saved_image_path = None
        try:
            image_array = self.env.render(mode='rgb_array')
            if image_array is None: raise ValueError("Rendered image is None")

            base64_image = numpy_array_to_base64(image_array)
            if base64_image is None: raise ValueError("Base64 conversion failed")

            if self.image_dir_path:
                try:
                    fn = "current_step_visual.png" # Constant filename
                    saved_image_path = os.path.join(self.image_dir_path, fn)
                    Image.fromarray(image_array.astype('uint8'), 'RGB').save(saved_image_path)
                except Exception as img_e: print(f"[ERROR] Failed to save image to {saved_image_path}: {img_e}", file=sys.stderr)
            
            # BasicAgent does not use memory, so memory_summary is None
            self.last_action_plan = self.reasoning_module.plan_action_with_vision(base64_image, None)

        except Exception as e:
            thought = f"Error in BasicAgent image processing or vision call: {e}"
            print(f"Error: {thought}", file=sys.stderr); traceback.print_exc()
            self.last_action_plan = {"move": "api_error", "thought": thought, "latency": 0.0}
        
        action_id = self._get_action_id(self.last_action_plan['move'])
        return action_id, saved_image_path # Return image path for logging

# --- PerceptionOnlyAgent (Perception Module + Text Reasoning, No Memory) ---
class PerceptionOnlyAgent(BaseSokobanAgent):
    def __init__(self, env: CustomSokobanEnv = None, render_mode: str = None,
                 api_provider: str = 'openai', model_name: str = 'gpt-4-turbo',
                 system_prompt: str = "You are an expert Sokoban player analyzing text.", thinking: bool = True,
                 text_log_path: str = None):
        super().__init__(render_mode, api_provider, model_name, system_prompt, thinking,
                         text_log_path, "perception_only", env)
        self.perception_module = PerceptionModule(self.env)
        # No memory_module for this agent

    def select_action(self, observation):
        current_perception = self.perception_module.get_current_state()
        if not current_perception:
            self.last_action_plan = {"move": "noop", "thought": "Perception failed", "latency": 0.0}
            return self._get_action_id("noop"), None

        self.last_action_plan = self.reasoning_module.plan_action_with_text(
            current_perception['text_table'], None
        )
        return self._get_action_id(self.last_action_plan['move']), None

# --- MemoryOnlyAgent (Vision Reasoning + Memory, No Text Perception Module) ---
class MemoryOnlyAgent(BaseSokobanAgent):
    def __init__(self, env: CustomSokobanEnv = None, render_mode: str = None,
                 api_provider: str = 'openai', model_name: str = 'gpt-4-turbo',
                 system_prompt: str = "You are a Sokoban expert using images and memory.", thinking: bool = True,
                 max_memory_history: int = 5,
                 text_log_path: str = None, image_dir_path: str = None):
        effective_render_mode = render_mode if render_mode else 'rgb_array'
        super().__init__(effective_render_mode, api_provider, model_name, system_prompt, thinking,
                         text_log_path, "memory_vision", env)
        if 'rgb_array' not in self.env.metadata.get('render_modes', []):
             print(f"Warning: MemoryOnlyAgent created with env not explicitly listing 'rgb_array'. Effective mode: {self.env.render_mode}", file=sys.stderr)
        
        self.memory_module = MemoryModule(max_history=max_memory_history)
        self.image_dir_path = image_dir_path

    def select_action(self, observation):
        saved_image_path = None
        base64_image_for_memory = None # For storing in memory
        try:
            image_array = self.env.render(mode='rgb_array')
            if image_array is None: raise ValueError("Rendered image is None")

            base64_image_for_reasoning = numpy_array_to_base64(image_array)
            if base64_image_for_reasoning is None: raise ValueError("Base64 conversion failed")
            base64_image_for_memory = base64_image_for_reasoning # Save for memory entry

            if self.image_dir_path:
                try:
                    fn = "current_step_visual.png" # Constant filename
                    saved_image_path = os.path.join(self.image_dir_path, fn)
                    Image.fromarray(image_array.astype('uint8'), 'RGB').save(saved_image_path)
                except Exception as img_e: print(f"[ERROR] Failed to save image to {saved_image_path}: {img_e}", file=sys.stderr)

            memory_summary = self.memory_module.get_memory_summary()
            self.last_action_plan = self.reasoning_module.plan_action_with_vision(
                base64_image_for_reasoning, memory_summary
            )
        except Exception as e:
            thought = f"Error in MemoryOnlyAgent image processing or vision call: {e}"
            print(f"Error: {thought}", file=sys.stderr); traceback.print_exc()
            self.last_action_plan = {"move": "api_error", "thought": thought, "latency": 0.0}

        perception_for_memory = {"image_base64_hash": hash(base64_image_for_memory)} if base64_image_for_memory else {"image_base64_hash": None}
        self.memory_module.add_entry(perception_for_memory, self.last_action_plan)
        
        action_id = self._get_action_id(self.last_action_plan['move'])
        return action_id, saved_image_path

# --- Random Agent (Chooses random basic move) ---
class RandomAgent(BaseSokobanAgent):
    def __init__(self, env: CustomSokobanEnv = None, render_mode: str = None,
                 api_provider: str = 'openai', model_name: str = 'gpt-4-turbo', # Not used by random
                 system_prompt: str = "This agent chooses actions randomly.", thinking: bool = False, # Not used
                 text_log_path: str = None):
        super().__init__(render_mode, api_provider, model_name, system_prompt, thinking,
                         text_log_path, "random", env)
        # No perception, memory, or reasoning module needed

    def select_action(self, observation):
        import random
        possible_moves = ['up', 'down', 'left', 'right', 'push up', 'push down', 'push left', 'push right']
        chosen_move = random.choice(possible_moves)
        
        # Set last_action_plan for logging purposes, latency is negligible
        self.last_action_plan = {"move": chosen_move, "thought": "Randomly selected.", "latency": 0.0}
        
        action_id = self._get_action_id(chosen_move)
        # Random agent does not produce images, so second part of tuple is None
        return action_id, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Sokoban LLM Agent")
    parser.add_argument("--api_provider", type=str, default="openai",
                        choices=["openai", "anthropic", "gemini", "deepseek", "together_ai", "xai"],
                        help="API provider to use.")
    parser.add_argument("--model_name", type=str, default="gpt-4o",
                        help="Specific model name for the chosen provider.")
    parser.add_argument("--level", type=int, default=1, # Restored level argument
                        help="Starting level index to test.")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--render_mode", type=str, default="human", choices=["human", "rgb_array", "tiny_human", "tiny_rgb_array", "raw", "none"])
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument("--memory", type=int, default=10, help="Max history for agents with memory.")
    parser.add_argument("--agent_type", type=str, default="full", choices=["full", "basic", "perception_only", "memory_only", "random"])
    args = parser.parse_args()

    run_ts = time.strftime('%Y%m%d_%H%M%S')
    safe_model_name = re.sub(r'[^a-zA-Z0-9_-]', '_', args.model_name)
    
    agent_configs = {
        "full": {"suffix": "full", "class": SokobanAgent, "needs_image_dir": False, "uses_memory": True},
        "basic": {"suffix": "basic_vision", "class": BasicAgent, "needs_image_dir": True, "uses_memory": False},
        "perception_only": {"suffix": "perception_only", "class": PerceptionOnlyAgent, "needs_image_dir": False, "uses_memory": False},
        "memory_only": {"suffix": "memory_vision", "class": MemoryOnlyAgent, "needs_image_dir": True, "uses_memory": True},
        "random": {"suffix": "random", "class": RandomAgent, "needs_image_dir": False, "uses_memory": False}
    }
    config = agent_configs[args.agent_type]
    base_run_name = f"{run_ts}_{safe_model_name}_sokoban_{config['suffix']}"
    run_dir = os.path.join(CACHE_DIR, base_run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run directory: {run_dir}")

    text_log = os.path.join(run_dir, f"{base_run_name}.log")
    print(f"Text log: {text_log}")

    image_dir = run_dir if config["needs_image_dir"] else None
    if image_dir: print(f"Image save dir: {image_dir}")

    current_level = args.level
    env_render_mode = args.render_mode if args.render_mode != "none" else None

    while True:
        print("\\n" + "="*30 + f" Level {current_level} (Agent: {args.agent_type}) " + "="*30 + "\\n")
        agent_instance = None
        try:
            agent_params = {
                "render_mode": env_render_mode,
                "api_provider": args.api_provider,
                "model_name": args.model_name,
                "text_log_path": text_log,
            }
            if config["uses_memory"]: agent_params["max_memory_history"] = args.memory
            if config["needs_image_dir"]: agent_params["image_dir_path"] = image_dir
            
            agent_class = config["class"]
            # Specific system prompts per agent for clarity
            if agent_class == SokobanAgent: agent_params["system_prompt"] = "You are an expert Sokoban player using text and memory."
            elif agent_class == BasicAgent: agent_params["system_prompt"] = "You are a Sokoban expert analyzing an image."
            elif agent_class == PerceptionOnlyAgent: agent_params["system_prompt"] = "You are an expert Sokoban player analyzing text."
            elif agent_class == MemoryOnlyAgent: agent_params["system_prompt"] = "You are a Sokoban expert using images and memory."
            elif agent_class == RandomAgent: agent_params["system_prompt"] = "This agent chooses actions randomly."


            print(f"Initializing {agent_class.__name__}...")
            agent_instance = agent_class(**agent_params)

            reward, steps = agent_instance.run_episode(
                level_index=current_level,
                max_steps=args.steps,
                render_delay=args.delay
            )
            if steps >= args.steps and reward < 10: # Assuming 10 is solve reward
                print(f"Level {current_level} failed (max steps). Terminating.")
                break

            current_level += 1 # Move to next level

        except (ValueError, FileNotFoundError, RuntimeError) as e:
            if "Level" in str(e) and "not found" in str(e) or "out of range" in str(e) : # Check specific error messages
                print(f"Could not load Level {current_level}. Likely end of levels. Details: {e}")
            else:
                print(f"Error during Level {current_level} run: {e}")
                traceback.print_exc()
            break
        except Exception as e:
            print(f"Unexpected error during Level {current_level}: {e}", file=sys.stderr)
            traceback.print_exc()
            break
        finally:
            if agent_instance: agent_instance.close_env()
            # Check if it was the last attempt or if loop should continue
            # This logic might need refinement based on how max levels are determined
            if 'e' in locals() and ("not found" in str(e) or "out of range" in str(e)): # if level loading error caused break
                 pass # End of levels reached
            elif 'steps' in locals() and steps >= args.steps: # if max steps caused break
                 pass # stop after max steps failure

    print("\\n" + "="*30 + " Run Finished " + "="*30 + "\\n")