import time
import threading
import numpy as np
from typing import Optional, Tuple
import retro
from gamingagent.envs.classic_video_game_env import ClassicVideoGameEnv

class RealTimeVideoGameEnv(ClassicVideoGameEnv):
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
        target_fps: float = 30.0,  # Default to 30 FPS
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
            target_fps: Target frames per second for the game (default: 30)
            frame_skip: Number of frames to skip between actions
            **kwargs: Additional arguments to pass to the environment
        """
        super().__init__(
            game=game,
            state=state,
            scenario=scenario,
            record=record,
            render_mode=render_mode,
            **kwargs
        )
        
        self.target_fps = target_fps
        self.frame_skip = frame_skip
        self.frame_delay = 1.0 / target_fps  # Time between frames in seconds
        
        # FPS tracking
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.last_frame_time = time.time()
        
        # Thread-safe buffers for latest frame and step result
        self._latest_frame = None
        self._step_result = None
        self._frame_lock = threading.Lock()
        self._step_lock = threading.Lock()
        self._action_lock = threading.Lock()
        
        # Current action state
        self.current_action = np.zeros(self.num_buttons, dtype=np.uint8)
            
    def step(self, action):
        """Take a step in the environment."""
        # Store the action for the simulation thread
        with self._action_lock:
            self.current_action = action.copy()
        
        # Execute action for frame_skip frames
        for _ in range(self.frame_skip):
            # Execute the action directly
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.step_count += 1
            self.frame_count += 1
            
            # Get RGB frame
            frame = self.get_observation()
            
            # Update frame buffer
            with self._frame_lock:
                self._latest_frame = frame
                
            # Update step result
            with self._step_lock:
                self._step_result = (obs, reward, terminated, truncated, info)
            
            if terminated or truncated:
                break
                
            # Enforce frame timing
            current_time = time.time()
            elapsed = current_time - self.last_frame_time
            if elapsed < self.frame_delay:
                time.sleep(self.frame_delay - elapsed)
            self.last_frame_time = time.time()
                
        # Update FPS counter every second
        current_time = time.time()
        if current_time - self.fps_start_time >= 1.0:
            self.fps_start_time = current_time
            self.frame_count = 0
                
        return self._step_result
        
    def reset(self, **kwargs):
        """Reset the environment."""
        self.frame_count = 0
        self.step_count = 0
        self.fps_start_time = time.time()
        self.last_frame_time = time.time()
        
        # Reset the environment
        obs = super().reset(**kwargs)
        if isinstance(obs, tuple):
            obs = obs[0]
            
        # Update frame buffer
        with self._frame_lock:
            self._latest_frame = obs
            
        # Update step result
        with self._step_lock:
            self._step_result = (obs, 0.0, False, False, {})
            
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
            
    def close(self):
        """Clean up resources."""
        super().close()
