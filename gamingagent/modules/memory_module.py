import os
import json
import time
import datetime
from abc import abstractmethod
from .core_module import CoreModule
import re

class MemoryModule(CoreModule):
    """
    Memory module that tracks game state history.
    
    Game-specific implementations should inherit from this class and implement
    the required abstract methods.
    """
    
    def __init__(self, model_name="claude-3-7-sonnet-latest", cache_dir="cache", 
                 system_prompt="", prompt="", max_memory=10):
        """
        Initialize the memory module.
        
        Args:
            model_name (str): The name of the model to use for inference.
            cache_dir (str): Directory for storing logs and cache files.
            system_prompt (str): System prompt for LLM calls.
            prompt (str): Default user prompt for LLM calls.
            max_memory (int): Maximum number of memory entries to maintain.
        """
        super().__init__(
            module_name="memory_module",
            model_name=model_name,
            system_prompt=system_prompt,
            prompt=prompt,
            cache_dir=cache_dir
        )
        self.max_memory = max_memory
        self.memory = []
        
        # Load existing memory if available
        self.load_memory()
    
    def load_memory(self):
        """Load memory entries from the module log file if it exists."""
        try:
            if os.path.exists(self.module_file):
                with open(self.module_file, 'r') as f:
                    log_entries = json.load(f)
                    
                    # Extract memory entries from log entries
                    memory_entries = []
                    for entry in log_entries:
                        if all(key in entry for key in ["timestamp", "prev_game_state", "last_action", "thought", "reflection"]):
                            memory_entries.append({
                                "timestamp": entry.get("timestamp"),
                                "game_state": entry.get("prev_game_state"),
                                "last_action": entry.get("last_action"),
                                "thought": entry.get("thought"),
                                "reflection": entry.get("reflection")
                            })
                    
                    # Limit to max_memory entries
                    self.memory = memory_entries[-self.max_memory:] if len(memory_entries) > self.max_memory else memory_entries
                print(f"Memory loaded successfully: {len(self.memory)} entries from {self.module_file}")
            else:
                print(f"Memory file does not exist: {self.module_file}")
                self.memory = []
        except Exception as e:
            print(f"Error loading memory: {e}")
            self.memory = []
    
    def generate_reflection(self, prev_perception, current_perception, last_action, last_thought):
        """
        Generate a reflection on the current game state based on the memory.
        
        Args:
            prev_perception: Previous perception data (before action was taken)
            current_perception: Current perception data (after action was taken)
            last_action: The action taken that led to current perception
            last_thought: The reasoning behind the action
            
        Returns:
            str: A reflection on the current game state
        """
        # Build previous game states section (previous observation + action + thought)
        prev_game_states = ""
        
        # Add previous perception
        if prev_perception:
            prev_game_states += "Previous Observation:\n"
            prev_game_states += f"{str(prev_perception)}\n\n"
        
        # Add action and thought that led to current state
        if last_action or last_thought:
            prev_game_states += "Action and Thought:\n"
            if last_action:
                prev_game_states += f"- Action: {str(last_action)}\n"
            if last_thought:
                prev_game_states += f"- Thought: {str(last_thought)}\n"
            prev_game_states += "\n"
        
        # If no previous data at all
        if not prev_game_states:
            prev_game_states = "No previous game states available.\n"
        
        # Build current observation section
        current_observation = f"{str(current_perception)}"
        
        # Format the prompt template with the sections
        formatted_prompt = self.prompt.format(
            prev_game_states=prev_game_states,
            current_observation=current_observation
        )
        
        # Call the API for text-only completion
        response = self.api_manager.text_only_completion(
            model_name=self.model_name,
            system_prompt=self.system_prompt,
            prompt=formatted_prompt,
            thinking=False,
            reasoning_effort=self.reasoning_effort,
            token_limit=self.token_limit
        )
        
        # Return the parsed response
        parsed_response = self._parse_response(response)
        return parsed_response["reflection"]
    
    def _parse_response(self, response):
        """
        Parse the reflection response from the LLM.
        
        Args:
            response (str): The raw response from the LLM
            
        Returns:
            dict: Parsed reflection data
        """
        
        if not response:
            return {"reflection": "No reflection generated."}
        
        # Try to extract reflection from structured format first
        reflection_match = re.search(r'(?:^|\n)(?:#\s*)?reflection:(.+?)(?=(?:\n(?:#\s*)?[a-zA-Z]+:)|$)', 
                                    response, re.DOTALL | re.IGNORECASE)
        
        if reflection_match:
            # Extract the reflection content from the pattern match
            reflection = reflection_match.group(1).strip()
        else:
            # If no structured format found, use the entire response
            reflection = response.strip()
            
        return {
            "reflection": reflection
        }
    
    def update_memory(self, game_state, action=None, thought=None):
        """
        Add a new game state to memory and generate reflection.
        This is the primary method for updating the agent's memory with new experiences.
        
        Args:
            game_state: The current perceived game state to add
            action: Action taken that led to this game state
            thought: Reasoning behind the action
        
        Returns:
            str: Generated reflection
        """
        # Get timestamp for this entry
        timestamp = time.time()
        
        # Get previous perception (if any)
        prev_perception = None
        if self.memory:
            prev_perception = self.memory[-1].get('game_state')
        
        # Generate reflection comparing previous state to current state
        reflection = self.generate_reflection(prev_perception, game_state, action, thought)
        
        # Create memory entry
        memory_entry = {
            "timestamp": timestamp,
            "game_state": game_state,
            "last_action": action,
            "thought": thought,
            "reflection": reflection
        }
        
        # Add to memory
        self.memory.append(memory_entry)
        if len(self.memory) > self.max_memory:
            self.memory.pop(0)  # Remove the oldest entry
        
        # Log the memory update with only essential information
        self.log({
            "current_perception": str(game_state),
            "last_action": action,
            "last_thought": str(thought) if thought else None,
            "reflection": reflection
        })
        
        return reflection
    
    def get_memory_summary(self):
        """
        Get a summary of the memory for the reasoning module.
        
        Returns:
            dict: Dictionary containing previous game states, current perception, and reflection
                  as three simple string values
        """
        if not self.memory:
            return {
                "prev_game_states": "No memory entries available.",
                "current_perception": "",
                "reflection": ""
            }
        
        # Process previous states into a single formatted string
        prev_game_states_str = ""
        
        # Add all previous states except the latest one
        if len(self.memory) > 1:
            # Only include the last 3 entries (excluding the latest) to keep it concise
            previous_entries = self.memory[:-1]
            if len(previous_entries) > 3:
                previous_entries = previous_entries[-3:]
                
            for i, entry in enumerate(previous_entries):
                prev_game_states_str += f"State {i+1}:\n"
                
                # Order: thought, action -> observation -> reflection
                if entry.get('thought'):
                    prev_game_states_str += f"Thought: {str(entry.get('thought'))}\n"
                if entry.get('last_action'):
                    prev_game_states_str += f"Action: {str(entry.get('last_action'))}\n"
                if entry.get('game_state'):
                    prev_game_states_str += f"Observation: {str(entry.get('game_state'))}\n"
                if entry.get('reflection'):
                    prev_game_states_str += f"Reflection: {str(entry.get('reflection'))}\n"
                prev_game_states_str += "\n"
        
        if not prev_game_states_str:
            prev_game_states_str = "No previous game states available."
        
        # Get the latest state information
        latest = self.memory[-1] if self.memory else {}
        
        # Extract current perception and reflection from latest entry
        current_perception = str(latest.get('game_state', "")) if latest else ""
        reflection = str(latest.get('reflection', "")) if latest else ""
        
        # Return structured dictionary with string values
        return {
            "prev_game_states": prev_game_states_str,
            "current_perception": current_perception,
            "reflection": reflection
        }
