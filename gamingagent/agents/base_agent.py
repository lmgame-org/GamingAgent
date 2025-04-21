import os
import json
from typing import Any, Dict, Optional, Union, Tuple
import numpy as np
from datetime import datetime

class BaseAgent:
    """Base class for all agents."""
    
    def __init__(
        self,
        env: Any,
        game_name: str,
        api_provider: str,
        model_name: str
    ):
        """Initialize the base agent.
        
        Args:
            env: The environment to interact with
            game_name: Name of the game
            api_provider: Name of the API provider
            model_name: Name of the model to use
        """
        self.env = env
        self.game_name = game_name
        self.api_provider = api_provider
        self.model_name = model_name
        
        # Set up cache directory
        self._setup_cache_directory()
        
    def _setup_cache_directory(self) -> None:
        """Set up the cache directory structure."""
        # Create base cache directory
        self.cache_dir = os.path.join("cache", self.game_name)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Create subdirectories
        self.logs_dir = os.path.join(self.cache_dir, "logs")
        self.actions_dir = os.path.join(self.cache_dir, "actions")
        self.states_dir = os.path.join(self.cache_dir, "states")
        
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.actions_dir, exist_ok=True)
        os.makedirs(self.states_dir, exist_ok=True)
        
        # Save agent configuration
        self._save_config()
        
    def _save_config(self) -> None:
        """Save agent configuration to cache directory."""
        config = {
            "game_name": self.game_name,
            "api_provider": self.api_provider,
            "model_name": self.model_name,
            "timestamp": datetime.now().isoformat()
        }
        
        config_path = os.path.join(self.cache_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
            
    def log_action(self, action: np.ndarray, step: int) -> None:
        """Log an action taken by the agent.
        
        Args:
            action: The action taken
            step: The current step number
        """
        action_data = {
            "step": step,
            "action": action.tolist(),
            "timestamp": datetime.now().isoformat()
        }
        
        action_path = os.path.join(self.actions_dir, f"action_{step:06d}.json")
        with open(action_path, "w") as f:
            json.dump(action_data, f, indent=4)
            
    def log_state(self, observation: np.ndarray, step: int) -> None:
        """Log the current state/observation.
        
        Args:
            observation: The current observation
            step: The current step number
        """
        state_data = {
            "step": step,
            "observation_shape": observation.shape,
            "timestamp": datetime.now().isoformat()
        }
        
        state_path = os.path.join(self.states_dir, f"state_{step:06d}.json")
        with open(state_path, "w") as f:
            json.dump(state_data, f, indent=4)
            
    def select_action(self, observation: Union[np.ndarray, Tuple[np.ndarray, Dict]]) -> np.ndarray:
        """Select an action based on the current observation.
        
        Args:
            observation: The current observation
            
        Returns:
            The selected action
        """
        raise NotImplementedError("Subclasses must implement select_action")
        
    def reset(self) -> None:
        """Reset the agent's state."""
        pass
        
    def close(self) -> None:
        """Clean up resources."""
        pass