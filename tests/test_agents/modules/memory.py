import os
import json
import time
import numpy as np
from collections import deque
from datetime import datetime
from tools.serving import APIManager

# Cache directories and file paths
CACHE_DIR = os.path.join("cache", "2048_experiments", datetime.now().strftime("%Y%m%d_%H%M%S"))
MEMORY_FILE = os.path.join(CACHE_DIR, "memory.json")
os.makedirs(CACHE_DIR, exist_ok=True)

class MemoryModule:
    def __init__(self, memory_file=MEMORY_FILE, max_memory=10, model_name="claude-3-7-sonnet-latest"):
        """
        Initialize the Memory Module for tracking 2048 game state history.
        
        Args:
            memory_file (str): Path to the memory JSON file
            max_memory (int): Maximum number of game states to remember
            model_name (str): Name of the model to use for reflections
        """
        self.memory_file = memory_file
        self.max_memory = max_memory
        self.memory = deque(maxlen=max_memory)
        self.model_name = model_name
        self.api_manager = APIManager(game_name="2048")
        
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
            # Convert memory to JSON-serializable format
            serializable_memory = self._convert_numpy_types(list(self.memory))
            
            with open(self.memory_file, 'w') as f:
                json.dump(serializable_memory, f, indent=2)
        except Exception as e:
            print(f"Error saving memory: {e}")
            
    def generate_reflection(self, current_perception, last_action):
        """
        Generate a reflection on the current state by comparing it with previous states.
        
        Args:
            current_perception (dict): The current perceived game state
            last_action: The previous action taken
            
        Returns:
            str: A reflection on the current state and how it relates to previous states and actions
        """
        try:
            # If there are not enough previous states, return a default reflection
            if len(self.memory) < 1:
                return "Not enough history to generate a meaningful reflection."
            
            # Get the previous state and ensure it's JSON serializable
            previous_state = self._convert_numpy_types(self.memory[-1]["game_state"])
            previous_action = self.memory[-1].get("last_action", None)
            
            # Convert current perception to JSON serializable format
            current_perception = self._convert_numpy_types(current_perception)
            
            system_prompt = """You are an analytical assistant for a 2048 AI agent. Your task is to generate a brief, insightful reflection on the game state changes and the effectiveness of recent actions.
Focus on strategic insights and patterns that would help the agent make better decisions.
Keep your reflections short, precise, and actionable.
"""
            
            user_prompt = f"""Please analyze the following 2048 game states and actions to generate a brief reflection:

Previous Game State:
{json.dumps(previous_state, indent=2)}

Previous Action: {previous_action}

Current Game State:
{json.dumps(current_perception, indent=2)}

Last Action: {last_action}

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
            action: Action taken in the previous state
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
        
    def _convert_numpy_types(self, obj):
        """Convert NumPy types to Python native types to avoid JSON serialization errors."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return self._convert_numpy_types(obj.tolist())
        elif isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(i) for i in obj]
        else:
            return obj 