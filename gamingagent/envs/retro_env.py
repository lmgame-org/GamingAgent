import os
import retro
import time
from typing import Any, Dict, Optional, Tuple
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, Discrete

class ClassicVideoGameEnv(Env):
    """Classic video game environment using Gymnasium."""
    
    # ROM file extensions for different systems
    ROM_EXTENSIONS = {
        'Genesis': '.md',
        'SNES': '.sfc',
        'NES': '.nes',
        'Atari2600': '.a26',
        'GameBoy': '.gb',
        'GameBoyAdvance': '.gba',
        'GameBoyColor': '.gbc',
        'GameGear': '.gg',
        'TurboGrafx16': '.pce',
        'MasterSystem': '.sms'
    }
    
    def __init__(
        self,
        game: str,
        state: str = retro.State.DEFAULT,
        scenario: str = "scenario",
        record: bool = False,
        render_mode: Optional[str] = None,
        frame_delay: float = 0.016  # 60 FPS
    ):
        """Initialize the environment."""
        super().__init__()
        
        # Add custom integration path
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
        CUSTOM_INTEGRATIONS_PATH = os.path.join(PROJECT_ROOT, "custom_integrations", "roms")
        
        # Clear any existing custom paths and add our custom path
        retro.data.Integrations.clear_custom_paths()
        retro.data.add_custom_integration(CUSTOM_INTEGRATIONS_PATH)
        
        # Set up the environment
        self.env = retro.make(
            game=game,
            state=state,
            scenario=scenario,
            record=record,
            render_mode=render_mode
        )
        
        # Set up action and observation spaces
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        
        # Store frame delay
        self.frame_delay = frame_delay
        self.last_frame_time = time.time()
        
    def step(self, action):
        """Take a step in the environment."""
        # Add frame delay to control speed
        current_time = time.time()
        elapsed = current_time - self.last_frame_time
        if elapsed < self.frame_delay:
            time.sleep(self.frame_delay - elapsed)
        self.last_frame_time = time.time()
        
        # Take the step
        return self.env.step(action)
        
    def reset(self, **kwargs):
        """Reset the environment."""
        self.last_frame_time = time.time()
        return self.env.reset(**kwargs)
        
    def render(self):
        """Render the environment."""
        return self.env.render()
        
    def close(self):
        """Close the environment."""
        return self.env.close()
        
    def seed(self, seed=None):
        """Set the random seed."""
        return self.env.seed(seed)
        
    def record_movie(self, path: str) -> None:
        """Start recording a movie."""
        self.env.record_movie(path)
        
    def stop_record(self) -> None:
        """Stop recording the movie."""
        self.env.stop_record()
        
    @property
    def unwrapped(self):
        """Return the base environment."""
        return self