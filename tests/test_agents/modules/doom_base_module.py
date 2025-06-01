import numpy as np

class DoomBaseModule:
    """
    A simplified module that directly processes observation images and returns actions for Doom.
    This module skips separate perception and memory stages used in the full pipeline.
    """
    def __init__(self):
        """Initialize the base module for Doom."""
        self.available_actions = ["move_left", "move_right", "attack"]
        
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
