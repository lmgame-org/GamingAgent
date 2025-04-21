import time
import threading
import numpy as np
from typing import Optional, List, Tuple, Any, Dict, Union
from queue import Queue, Empty
from PIL import Image
from io import BytesIO
import base64
from gamingagent.agents.base_agent import BaseAgent
from gamingagent.providers.api_provider_manager import APIProviderManager
from gamingagent.utils.utils import encode_image

class RealTimeMarioAgent(BaseAgent):
    """A real-time version of the Mario agent that uses concurrent short and long workers.
    
    This agent extends MarioAgent to provide better real-time control by using two types
    of workers that run concurrently:
    - Short worker: Plans 2 frames ahead with frame_skip=2
    - Long worker: Plans 4 frames ahead with frame_skip=4
    
    The workers alternate execution with a 0.5s interval between them.
    """
    
    def __init__(
        self,
        env: Any,
        game_name: str,
        api_provider: str = "anthropic",
        model_name: str = "claude-3-opus-20240229",
        short_worker_frame_skip: int = 2,
        long_worker_frame_skip: int = 4
    ):
        """Initialize the real-time Mario agent.
        
        Args:
            env: The environment to interact with
            game_name: Name of the game
            api_provider: Name of the API provider
            model_name: Name of the model to use
            short_worker_frame_skip: Frame skip for short worker
            long_worker_frame_skip: Frame skip for long worker
        """
        super().__init__(env, game_name, api_provider, model_name)
        
        # Verify logging setup
        self.logger.info(f"Initializing RealTimeMarioAgent with game: {game_name}")
        self.logger.info(f"Cache directory: {self.cache_dir}")
        self.logger.info(f"Logs directory: {self.logs_dir}")
        self.logger.info(f"Actions directory: {self.actions_dir}")
        self.logger.info(f"States directory: {self.states_dir}")
        
        self.api_manager = APIProviderManager()
        self.api_manager.initialize_providers(anthropic_model=model_name)
        self.provider = self.api_manager.get_provider(api_provider)
        
        # Worker control
        self._stop_event = threading.Event()
        self._short_worker_thread = None
        self._long_worker_thread = None
        
        # Frame skip settings
        self.short_worker_frame_skip = short_worker_frame_skip
        self.long_worker_frame_skip = long_worker_frame_skip
        
        # Action queue
        self._action_queue = []
        self._action_lock = threading.Lock()
        
        self.logger.info("RealTimeMarioAgent initialization complete")
        
    def _encode_frame(self, frame: np.ndarray) -> str:
        """Encode a frame as a base64 PNG string.
        
        Args:
            frame: The frame to encode as a numpy array
            
        Returns:
            The base64-encoded PNG string
        """
        # Convert to PIL Image
        img = Image.fromarray(frame)
        
        # Save to in-memory buffer
        buf = BytesIO()
        img.save(buf, format="PNG")
        
        # Base64 encode
        return base64.b64encode(buf.getvalue()).decode("ascii")
        
    def _short_worker_loop(self):
        """Short-term worker that plans 2 frames ahead."""
        while not self._stop_event.is_set():
            # Get current frame
            frame = self.env.get_latest_frame()
            if frame is None:
                time.sleep(0.1)
                continue
                
            # Encode frame
            b64_image = self._encode_frame(frame)
            
            # Generate prompt
            prompt = f"""You are playing Super Mario Bros. Based on the current game screen, 
            you need to decide what action to take for the next {self.short_worker_frame_skip} frames.
            The available actions are:
            - RIGHT: Move right
            - LEFT: Move left
            - UP: Jump
            - DOWN: Crouch
            - A: Run (B button)
            - B: Jump (A button)
            
            You can combine these actions. For example, "RIGHT + A" means run right.
            
            Look at the game screen and respond with ONLY the action(s) you want to take.
            For example: "RIGHT + A" or "UP + B" or "LEFT".
            
            Current game screen:"""
            
            try:
                # Call LLM
                response = self.provider.generate_with_images(
                    prompt=prompt,
                    images=[b64_image],
                    max_tokens=50
                )
                
                # Parse response into action
                action = self._parse_response(response)
                
                # Add to action queue
                with self._action_lock:
                    self._action_queue.append((action, self.short_worker_frame_skip))
                    
            except Exception as e:
                self.logger.error(f"Error in short worker: {str(e)}")
                time.sleep(0.1)
                
    def _long_worker_loop(self):
        """Long-term worker that plans 4 frames ahead."""
        while not self._stop_event.is_set():
            # Get current frame
            frame = self.env.get_latest_frame()
            if frame is None:
                time.sleep(0.1)
                continue
                
            # Encode frame
            b64_image = self._encode_frame(frame)
            
            # Generate prompt
            prompt = f"""You are playing Super Mario Bros. Based on the current game screen, 
            you need to decide what action to take for the next {self.long_worker_frame_skip} frames.
            The available actions are:
            - RIGHT: Move right
            - LEFT: Move left
            - UP: Jump
            - DOWN: Crouch
            - A: Run (B button)
            - B: Jump (A button)
            
            You can combine these actions. For example, "RIGHT + A" means run right.
            
            Look at the game screen and respond with ONLY the action(s) you want to take.
            For example: "RIGHT + A" or "UP + B" or "LEFT".
            
            Current game screen:"""
            
            try:
                # Call LLM
                response = self.provider.generate_with_images(
                    prompt=prompt,
                    images=[b64_image],
                    max_tokens=50
                )
                
                # Parse response into action
                action = self._parse_response(response)
                
                # Add to action queue
                with self._action_lock:
                    self._action_queue.append((action, self.long_worker_frame_skip))
                    
            except Exception as e:
                self.logger.error(f"Error in long worker: {str(e)}")
                time.sleep(0.1)
                
    def start(self):
        """Start the worker threads."""
        if self._short_worker_thread is None:
            self._short_worker_thread = threading.Thread(target=self._short_worker_loop, daemon=True)
            self._short_worker_thread.start()
            
        if self._long_worker_thread is None:
            self._long_worker_thread = threading.Thread(target=self._long_worker_loop, daemon=True)
            self._long_worker_thread.start()
            
    def select_action(self, observation: Union[np.ndarray, Tuple[np.ndarray, Dict]]) -> np.ndarray:
        """Select an action based on the current observation.
        
        Args:
            observation: The current observation
            
        Returns:
            The selected action
        """
        # Get next action from queue
        with self._action_lock:
            if self._action_queue:
                action, _ = self._action_queue.pop(0)
                return action
                
        # Default action if queue is empty
        return np.zeros(self.env.num_buttons, dtype=np.uint8)
        
    def close(self):
        """Clean up resources."""
        self._stop_event.set()
        if self._short_worker_thread is not None:
            self._short_worker_thread.join(timeout=1.0)
        if self._long_worker_thread is not None:
            self._long_worker_thread.join(timeout=1.0)
        super().close() 