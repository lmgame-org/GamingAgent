import numpy as np
import logging
from tools.serving import APIManager

def load_module_prompts():
    """Load module prompts from config file."""
    import os
    import json
    
    config_paths = [
        os.path.join("configs", "custom_05_doom", "module_prompts.json"),
        os.path.join("GamingAgent", "configs", "custom_05_doom", "module_prompts.json"),
        os.path.join(os.path.dirname(__file__), "..", "..", "configs", "custom_05_doom", "module_prompts.json")
    ]
    
    for config_path in config_paths:
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load config from {config_path}: {e}")
    
    raise ValueError("No module prompts configuration file found.")

class DoomBaseModule:
    """
    A simplified module that directly processes observation images and returns actions for Doom.
    This module skips separate perception and memory stages used in the full pipeline.
    """
    def __init__(self, model_name="gpt-4o"):
        """Initialize the base module for Doom."""
        self.available_actions = ["move_left", "move_right", "attack"]
        self.model_name = model_name
        
        # Set up logging
        self.logger = logging.getLogger('doom_base_module')
        self.logger.setLevel(logging.INFO)
        
        # Initialize API manager
        self.api_manager = APIManager(
            game_name="doom",
            base_cache_dir="cache/doom"
        )
        
    def process_observation(self, observation, info=None):
        """
        Process the observation and return a random action.
        
        Args:
            observation: The current game observation
            info: Additional information about the game state (not used)
            
        Returns:
            int: Action index (0: move_left, 1: move_right, 2: attack)
        """
        # Return a random action index (0, 1, or 2)
        return np.random.randint(0, 3)

    async def get_action(self, observation, info):
        """
        Get action from LLM based on current state.
        
        Args:
            observation: The current game observation
            info: Additional information about the game state
            
        Returns:
            dict: Action and thought process
        """
        try:
            # Get current game state
            game_state = {
                "health": info.get("health", 0),
                "ammo": info.get("ammo", 0),
                "kills": info.get("kills", 0),
                "position_x": info.get("position_x", 0),
                "position_y": info.get("position_y", 0),
                "angle": info.get("angle", 0),
                "is_episode_finished": info.get("is_episode_finished", False),
                "is_player_dead": info.get("is_player_dead", False)
            }
            
            # Load module prompts
            module_prompts = load_module_prompts()
            base_module = module_prompts.get("base_module", {})
            
            # Get system message and prompt template
            system_message = base_module.get("system_prompt", "")
            prompt_template = base_module.get("prompt", "")
            
            # Format user message with game state
            user_message = prompt_template.format(
                textual_representation=f"""Health: {game_state['health']}
Ammo: {game_state['ammo']}
Kills: {game_state['kills']}
Position: ({game_state['position_x']}, {game_state['position_y']})
Angle: {game_state['angle']}
Status: {'Finished' if game_state['is_episode_finished'] else 'In Progress'}"""
            )
            
            # Get action from LLM
            response = self.api_manager.text_only_completion(
                model_name=self.model_name,
                system_prompt=system_message,
                prompt=user_message,
                temperature=0.7
            )
            
            # Parse response to get action and thought
            try:
                # Split response into lines and find action line
                lines = response.strip().split('\n')
                action_line = next((line for line in lines if line.startswith('action:')), None)
                thought_line = next((line for line in lines if line.startswith('thought:')), None)
                
                if action_line:
                    action = action_line.split('action:')[1].strip().lower()
                else:
                    self.logger.warning("No action found in response, defaulting to attack")
                    action = "attack"
                
                if thought_line:
                    thought = thought_line.split('thought:')[1].strip()
                else:
                    thought = "No reasoning provided"
                
            except Exception as e:
                self.logger.error(f"Error parsing LLM response: {e}")
                action = "attack"
                thought = f"Error parsing response: {str(e)}"
            
            # Validate action
            valid_actions = ["move_left", "move_right", "attack"]
            if action not in valid_actions:
                self.logger.warning(f"Invalid action received: {action}. Defaulting to attack.")
                action = "attack"
                thought = f"Invalid action received: {action}. Defaulting to attack."
            
            # Log action details
            self.logger.info(f"Action selected: {action}")
            self.logger.info(f"Thought: {thought}")
            self.logger.info(f"Game state at action: {game_state}")
            
            return {
                "action": action,
                "thought": thought
            }
            
        except Exception as e:
            self.logger.error(f"Error in get_action: {str(e)}")
            return {
                "action": "attack",
                "thought": f"Error occurred: {str(e)}. Defaulting to attack."
            }
