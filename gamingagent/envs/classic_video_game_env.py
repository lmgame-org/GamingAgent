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

class RealTimeClassicVideoGameEnv(ClassicVideoGameEnv):
    """A real-time version of the classic video game environment with frame skipping and FPS control.
    
    This environment extends ClassicVideoGameEnv to provide better real-time control
    by implementing frame skipping and FPS management. It ensures the game runs at
    a consistent speed regardless of the agent's processing time.
    """
    
    def __init__(
        self,
        game: str,
        state: str = retro.State.DEFAULT,
        scenario: str = "scenario",
        record: bool = False,
        render_mode: Optional[str] = None,
        target_fps: float = 60.0,
        frame_skip: int = 1,
        **kwargs
    ):
        """Initialize the real-time environment.
        
        Args:
            game: The name or path for the game to run
            state: The initial state file to load, minus the extension
            scenario: The scenario file to load, minus the extension
            record: Whether to record gameplay
            render_mode: The render mode to use
            target_fps: Target frames per second for the game
            frame_skip: Number of frames to skip between actions
            **kwargs: Additional arguments to pass to the environment
        """
        super().__init__(
            game=game,
            state=state,
            scenario=scenario,
            record=record,
            render_mode=render_mode,
            frame_delay=1.0/target_fps,
            **kwargs
        )
        
        self.target_fps = target_fps
        self.frame_skip = frame_skip
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.fps_start_time = time.time()
        
        # Thread-safe buffers for latest frame and step result
        self._latest_frame = None
        self._step_result = None
        self._frame_lock = threading.Lock()
        self._step_lock = threading.Lock()
        self._action_lock = threading.Lock()
        
        # Simulation thread control
        self._stop_event = threading.Event()
        self._sim_thread = None
        self.current_action = np.zeros(self.num_buttons, dtype=np.uint8)
        self._initialized = False
        self._last_step_time = time.time()
        
    def _sim_loop(self):
        """Main simulation loop that runs in a separate thread."""
        dt = 1.0 / self.target_fps
        
        # Initial reset
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
            
        with self._frame_lock:
            self._latest_frame = obs
        with self._step_lock:
            self._step_result = (obs, 0.0, False, False, {})
            
        self._initialized = True
        
        while not self._stop_event.is_set():
            start_time = time.time()
            
            # Get current action safely
            with self._action_lock:
                current_action = self.current_action.copy()
            
            # Execute action for frame_skip frames
            for _ in range(self.frame_skip):
                if self._stop_event.is_set():
                    break
                    
                # Execute the action directly
                obs, reward, terminated, truncated, info = self.env.step(current_action)
                self.step_count += 1
                
                # Get RGB frame
                frame = self.get_observation()
                
                # Update frame buffer
                with self._frame_lock:
                    self._latest_frame = frame
                    
                # Update step result
                with self._step_lock:
                    self._step_result = (obs, reward, terminated, truncated, info)
                    
                # Update frame count and FPS
                self.frame_count += 1
                
                if terminated or truncated:
                    obs = self.env.reset()
                    if isinstance(obs, tuple):
                        obs = obs[0]
                    with self._frame_lock:
                        self._latest_frame = obs
                    with self._step_lock:
                        self._step_result = (obs, 0.0, False, False, {})
                    break
                
                # Control timing within frame skip
                elapsed = time.time() - start_time
                if elapsed < dt:
                    time.sleep(dt - elapsed)
            
            # Update FPS counter every second
            current_time = time.time()
            if current_time - self.fps_start_time >= 1.0:
                self.last_frame_time = current_time
                self.fps_start_time = current_time
                self.frame_count = 0
                
    def start(self):
        """Start the simulation thread."""
        if self._sim_thread is None:
            self._sim_thread = threading.Thread(target=self._sim_loop, daemon=True)
            self._sim_thread.start()
            
    def step(self, action):
        """Take a step in the environment."""
        # Store the action for the simulation thread
        with self._action_lock:
            self.current_action = action.copy()
        
        # Wait for initialization
        while not self._initialized:
            time.sleep(0.1)
            
        # Get the latest step result
        step_result = self.get_step_result()
        if step_result is None:
            return None
            
        # Add frame delay to control speed
        current_time = time.time()
        elapsed = current_time - self.last_frame_time
        if elapsed < self.frame_delay:
            time.sleep(self.frame_delay - elapsed)
        self.last_frame_time = time.time()
        
        return step_result
        
    def reset(self, **kwargs):
        """Reset the environment."""
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.step_count = 0
        self._initialized = False
        
        # Reset the environment
        obs = super().reset(**kwargs)
        if isinstance(obs, tuple):
            obs = obs[0]
            
        # Start simulation thread if not already running
        if self._sim_thread is None:
            self.start()
            
        # Wait for initialization
        while not self._initialized:
            time.sleep(0.1)
            
        return obs
        
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame from the environment.
        
        Returns:
            The latest frame as a numpy array, or None if no frame is available
        """
        with self._frame_lock:
            return self._latest_frame
            
    def get_step_result(self) -> Optional[Tuple]:
        """Get the latest step result from the environment.
        
        Returns:
            Tuple of (observation, reward, terminated, truncated, info), or None if no result is available
        """
        with self._step_lock:
            return self._step_result
            
    def close(self):
        """Clean up resources."""
        if self._stop_event is not None:
            self._stop_event.set()
        if self._sim_thread is not None:
            self._sim_thread.join(timeout=1.0)
        super().close()
        
    def get_fps(self) -> float:
        """Get the current actual FPS of the environment.
        
        Returns:
            The current frames per second
        """
        current_time = time.time()
        elapsed = current_time - self.fps_start_time
        if elapsed > 0:
            return self.frame_count / elapsed
        return 0.0