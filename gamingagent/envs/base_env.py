from gymnasium import Env
from typing import Any, Dict, Optional, Tuple
import numpy as np

class BaseEnv(Env):
    """Base environment class that all game environments must inherit from."""
    
    def __init__(self):
        """Initialize the base environment."""
        super().__init__()
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to its initial state."""
        raise NotImplementedError
        
    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one time step within the environment."""
        raise NotImplementedError
        
    def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        raise NotImplementedError
        
    def close(self) -> None:
        """Close the environment."""
        raise NotImplementedError