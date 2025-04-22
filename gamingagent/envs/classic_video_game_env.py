import os
import retro
import time
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
import threading

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
    
    metadata = {"render_modes": ["human", "rgb_array"], "video.frames_per_second": 60.0}
    
    def __init__(
        self,
        game: str,
        state: str = retro.State.DEFAULT,
        scenario: str = "scenario",
        record: bool = False,
        render_mode: Optional[str] = None,
        frame_delay: float = 0.016,  # 60 FPS
        **kwargs
    ):
        """Initialize the environment.
        
        Args:
            game: The name or path for the game to run
            state: The initial state file to load, minus the extension
            scenario: The scenario file to load, minus the extension
            record: Whether to record gameplay
            render_mode: The render mode to use
            frame_delay: The delay between frames
            **kwargs: Additional arguments to pass to the environment
        """
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
            render_mode=render_mode,
            **kwargs
        )
        
        # Set up action and observation spaces
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        
        # Store frame delay
        self.frame_delay = frame_delay
        self.last_frame_time = time.time()
        
        # Get button information from the environment
        self.buttons = self.env.buttons
        self.num_buttons = len(self.buttons)
        
        # Initialize state
        self._state = None
        self._info = None
        self._terminated = False
        self._truncated = False
        
        # Initialize step count
        self.step_count = 0
        
    def render(self, mode: Optional[str] = None) -> Union[np.ndarray, None]:
        """Render the environment.
        
        Args:
            mode: The render mode to use. If None, uses the default mode.
            
        Returns:
            If mode is "rgb_array", returns the current frame as a numpy array.
            Otherwise, returns None and renders to the screen.
        """
        if mode == "rgb_array":
            return self.env.get_screen()
        return self.env.render()
        
    def get_observation(self) -> np.ndarray:
        """Get the current observation as an RGB frame.
        
        Returns:
            The current frame as a numpy array
        """
        return self.render(mode="rgb_array")
        
    def step(self, action):
        """Take a step in the environment."""
        # Add frame delay to control speed
        current_time = time.time()
        elapsed = current_time - self.last_frame_time
        if elapsed < self.frame_delay:
            time.sleep(self.frame_delay - elapsed)
        self.last_frame_time = time.time()
        
        # Take the step and increment count
        self.step_count += 1
        return self.env.step(action)
        
    def reset(self, **kwargs):
        """Reset the environment."""
        self.last_frame_time = time.time()
        self.step_count = 0  # Reset step count
        return self.env.reset(**kwargs)
        
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
