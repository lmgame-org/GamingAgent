import os
import json
from typing import Any, Dict, Optional, Union, Tuple
import numpy as np
from datetime import datetime
import logging
from gamingagent.utils.logger import Logger

class BaseAgent:
    """Base class for all agents."""
    
    def __init__(
        self,
        env: Any,
        game_name: str,
        api_provider: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        """Initialize the base agent.
        
        Args:
            env: The environment to interact with
            game_name: Name of the game
            api_provider: Optional name of the API provider
            model_name: Optional name of the model
        """
        self.env = env
        self.game_name = game_name
        self.api_provider = api_provider
        self.model_name = model_name
        
        # Get current datetime for subdirectory
        current_time = datetime.now()
        datetime_str = current_time.strftime("%Y%m%d_%H%M%S")
        
        # Set up cache directory
        if api_provider is None or model_name is None:
            self.cache_dir = os.path.join("cache", game_name, "random_play", datetime_str)
        else:
            self.cache_dir = os.path.join("cache", game_name, api_provider, model_name, datetime_str)
            
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Set up subdirectories
        self.actions_dir = os.path.join(self.cache_dir, "actions")
        self.states_dir = os.path.join(self.cache_dir, "states")
        self.logs_dir = os.path.join(self.cache_dir, "logs")
        
        # Create all directories
        os.makedirs(self.actions_dir, exist_ok=True)
        os.makedirs(self.states_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Initialize actions log file
        self.actions_log_file = os.path.join(self.actions_dir, "actions.json")
        self.actions_log = []
        
        # Initialize logging
        self.logger = Logger(
            name=f"{game_name}_agent",
            log_dir=self.logs_dir,
            level=logging.INFO
        )
        
        # Save agent configuration
        self._save_config()
        
    def _save_config(self) -> None:
        """Save agent configuration to cache directory."""
        config = {
            "game_name": self.game_name,
            "api_provider": self.api_provider,
            "model_name": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "cache_dir": self.cache_dir
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
        
        # Append to actions log
        self.actions_log.append(action_data)
        
        # Write entire log to file
        with open(self.actions_log_file, 'w') as f:
            json.dump(self.actions_log, f, indent=4)
            
    def log_state(self, observation: Union[np.ndarray, Tuple[np.ndarray, dict]], step: int) -> None:
        """Log the current state/observation.
        
        Args:
            observation: The current observation (can be numpy array or tuple)
            step: The current step number
        """
        # Handle both numpy array and tuple observations
        if isinstance(observation, tuple):
            # For tuple observations (RAM), log the shape of the first element
            obs_shape = observation[0].shape
            obs_type = "ram"
        else:
            # For numpy array observations (image)
            obs_shape = observation.shape
            obs_type = "image"
            
        state_data = {
            "step": step,
            "observation_shape": obs_shape,
            "observation_type": obs_type,
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