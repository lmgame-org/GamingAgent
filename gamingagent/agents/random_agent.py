import numpy as np
from typing import Any, Union, Tuple
from .base_agent import BaseAgent

class RandomAgent(BaseAgent):
    """Agent that selects random actions from the action space."""
    
    def __init__(
        self,
        env: Any,
        game_name: str
    ):
        """Initialize random agent.
        
        Args:
            env: The environment to interact with
            game_name: Name of the game
        """
        super().__init__(
            env=env,
            game_name=game_name,
            api_provider=None,
            model_name=None
        )
        
    def select_action(self, observation: Union[np.ndarray, Tuple[np.ndarray, dict]]) -> np.ndarray:
        """Select a random action from the action space.
        
        Args:
            observation: The current observation from the environment
            
        Returns:
            A random action from the action space
        """
        # Get action space from environment
        action_space = self.env.action_space
        
        # Generate random action
        action = action_space.sample()
        
        # Log the action
        self.log_action(action, 0)
        self.log_state(observation, 0)
        
        return action 