import time
import threading
import numpy as np
from typing import Optional, Tuple, Any, Dict, Union
from PIL import Image
from io import BytesIO
import base64
from gamingagent.agents.base_agent import BaseAgent
from gamingagent.providers.api_provider_manager import APIProviderManager
import os
from datetime import datetime

class RealTimeMarioAgent(BaseAgent):
    """A real-time version of the Mario agent that uses two workers to execute actions.
    
    This agent uses two workers that run every 0.5 seconds:
    - Short worker: Executes actions for 15 frames (0.5s at 30fps)
    - Long worker: Executes actions for 30 frames (1s at 30fps)
    """
    
    def __init__(
        self,
        env: Any,
        game_name: str,
        api_provider: str = "anthropic",
        model_name: str = "claude-3-opus-20240229",
        short_worker_frame_skip: int = 15,  # 0.5s at 30fps
        long_worker_frame_skip: int = 30    # 1s at 30fps
    ):
        """Initialize the real-time Mario agent.
        
        Args:
            env: The environment to interact with
            game_name: Name of the game
            api_provider: Name of the API provider
            model_name: Name of the model to use
            short_worker_frame_skip: Frame skip for short worker (15 frames = 0.5s at 30fps)
            long_worker_frame_skip: Frame skip for long worker (30 frames = 1s at 30fps)
        """
        super().__init__(env, game_name, api_provider, model_name)
        
        # Verify logging setup
        self.logger.info(f"Initializing RealTimeMarioAgent with game: {game_name}")
        
        # Initialize API provider
        try:
            self.logger.info(f"Initializing API provider: {api_provider} with model: {model_name}")
            self.api_manager = APIProviderManager()
            self.api_manager.initialize_providers(anthropic_model=model_name)
            self.provider = self.api_manager.get_provider(api_provider)
            self.logger.info("API provider initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize API provider: {str(e)}")
            raise
        
        # Worker control
        self._stop_event = threading.Event()
        self._short_worker_thread = None
        self._long_worker_thread = None
        self._worker_interval = 0.5  # 0.5 seconds between worker runs
        self._frame_delay = 1.0 / 30.0  # 30 FPS
        
        # Frame skip settings
        self.short_worker_frame_skip = short_worker_frame_skip  # 15 frames = 0.5s at 30fps
        self.long_worker_frame_skip = long_worker_frame_skip    # 30 frames = 1s at 30fps
        
        # Action state
        self._action_lock = threading.Lock()
        self._last_short_worker_time = 0
        self._last_long_worker_time = 0
        
        # Action queue
        self._action_queue = []
        self._action_queue_lock = threading.Lock()
        
        # Create observations directory
        self.observations_dir = os.path.join(self.cache_dir, "observations")
        os.makedirs(self.observations_dir, exist_ok=True)
        self.logger.info(f"Created observations directory at {self.observations_dir}")
        
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
        
    def _get_action_from_frame(self, frame: np.ndarray, frame_skip: int) -> Optional[np.ndarray]:
        """Get an action from the API based on the current frame.
        
        Args:
            frame: The current frame
            frame_skip: Number of frames to plan for
            
        Returns:
            The action array, or None if the API call fails
        """
        try:
            # Encode frame
            b64_image = self._encode_frame(frame)
            
            # Generate prompt
            prompt = f"""You are playing Super Mario Bros. Based on the current game screen, 
            you need to decide what action to take for the next {frame_skip} frames.
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
            
            IMPORTANT: Respond with ONLY the action(s), nothing else. No explanations or descriptions.
            
            Current game screen:"""
            
            # Call LLM
            self.logger.info(f"Making API call to get action for {frame_skip} frames...")
            response = self.provider.generate_with_images(
                prompt=prompt,
                images=[b64_image],
                max_tokens=50
            )
            
            if not response:
                self.logger.error("Empty response from API")
                return None
                
            self.logger.info(f"API response received: {response}")
            
            # Extract just the action from the response
            # Look for the first line that contains only action words
            action_line = None
            for line in response.split('\n'):
                line = line.strip().upper()
                if all(word in ['RIGHT', 'LEFT', 'UP', 'DOWN', 'A', 'B', '+'] for word in line.split()):
                    action_line = line
                    break
                    
            if not action_line:
                self.logger.error(f"Could not find action in response: {response}")
                return None
                
            # Parse response into action
            action = self._parse_response(action_line)
            if action is None:
                self.logger.error(f"Failed to parse response: {action_line}")
                return None
                
            self.logger.info(f"Parsed action: {action}")
            return action
            
        except Exception as e:
            self.logger.error(f"Error getting action: {str(e)}")
            return None
            
    def _parse_response(self, response: str) -> Optional[np.ndarray]:
        """Parse the API response into an action array.
        
        Args:
            response: The API response string
            
        Returns:
            The parsed action array, or None if parsing fails
        """
        try:
            # Default action (no buttons pressed)
            action = np.zeros(self.env.num_buttons, dtype=np.uint8)
            
            # Get the button indices from the environment
            buttons = self.env.buttons
            self.logger.info(f"Available buttons: {buttons}")
            
            # Map of action names to button indices
            # Using the same mapping as RetroInteractive
            action_map = {
                "RIGHT": buttons.index("RIGHT"),
                "LEFT": buttons.index("LEFT"),
                "UP": buttons.index("UP"),
                "DOWN": buttons.index("DOWN"),
                "A": buttons.index("A"),  # Jump button (Z key)
                "B": buttons.index("B")   # Run button (X key)
            }
            
            # Clean up the response
            response = response.strip().upper()
            self.logger.info(f"Parsing response: {response}")
            
            # Split response into individual actions
            actions = response.split("+")
            actions = [a.strip() for a in actions]
            
            # Set corresponding indices to 1
            for a in actions:
                if a in action_map:
                    idx = action_map[a]
                    action[idx] = 1
                    self.logger.info(f"Setting button {a} at index {idx}")
                else:
                    self.logger.warning(f"Unknown action: {a}")
                    
            self.logger.info(f"Final action array: {action}")
            return action
            
        except Exception as e:
            self.logger.error(f"Error parsing response: {str(e)}")
            return None
            
    def _save_observation(self, frame: np.ndarray, worker_type: str) -> None:
        """Save an observation frame to the cache directory.
        
        Args:
            frame: The frame to save
            worker_type: Type of worker that generated the frame (short/long)
        """
        try:
            # Create timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            
            # Create filename
            filename = f"{worker_type}_worker_{timestamp}.png"
            filepath = os.path.join(self.observations_dir, filename)
            
            # Convert frame to PIL Image and save
            img = Image.fromarray(frame)
            img.save(filepath)
            
            self.logger.debug(f"Saved observation to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving observation: {str(e)}")
            
    def _worker_loop(self, is_short: bool):
        """Worker loop that executes actions directly.
        
        Args:
            is_short: Whether this is the short-term worker
        """
        worker_name = "Short" if is_short else "Long"
        frame_skip = self.short_worker_frame_skip if is_short else self.long_worker_frame_skip
        last_time_key = "_last_short_worker_time" if is_short else "_last_long_worker_time"
        
        while not self._stop_event.is_set():
            current_time = time.time()
            last_time = getattr(self, last_time_key)
            
            # Check if enough time has passed since last run
            if current_time - last_time < self._worker_interval:
                time.sleep(0.1)  # Small sleep to prevent busy waiting
                continue
                
            # Get current frame
            frame = self.env.get_latest_frame()
            if frame is None:
                self.logger.debug(f"{worker_name} worker: No frame available, waiting...")
                time.sleep(0.1)
                continue
                
            # Save the observation
            self._save_observation(frame, worker_name.lower())
                
            # Get action from API
            action = self._get_action_from_frame(frame, frame_skip)
            if action is None:
                self.logger.error(f"{worker_name} worker: Failed to get action")
                time.sleep(0.1)
                continue
                
            # Add action to queue for frame_skip frames
            with self._action_queue_lock:
                for _ in range(frame_skip):
                    self._action_queue.append(action)
                    
            setattr(self, last_time_key, current_time)
            self.logger.info(f"{worker_name} worker: Added action for {frame_skip} frames to queue")
            
    def start(self):
        """Start the worker threads."""
        if self._short_worker_thread is None:
            self._short_worker_thread = threading.Thread(
                target=self._worker_loop,
                args=(True,),
                daemon=True
            )
            self._short_worker_thread.start()
            self.logger.info("Short worker thread started")
            
        if self._long_worker_thread is None:
            self._long_worker_thread = threading.Thread(
                target=self._worker_loop,
                args=(False,),
                daemon=True
            )
            self._long_worker_thread.start()
            self.logger.info("Long worker thread started")
            
    def close(self):
        """Clean up resources."""
        self._stop_event.set()
        if self._short_worker_thread is not None:
            self._short_worker_thread.join(timeout=1.0)
        if self._long_worker_thread is not None:
            self._long_worker_thread.join(timeout=1.0)
        super().close()

    def select_action(self, observation: Union[np.ndarray, Tuple[np.ndarray, Dict]]) -> np.ndarray:
        """Select an action based on the current observation.
        
        Args:
            observation: The current observation from the environment
            
        Returns:
            The selected action as a numpy array
        """
        # Get the next action from the queue
        with self._action_queue_lock:
            if not self._action_queue:
                # If no actions in queue, return no buttons pressed
                return np.zeros(self.env.num_buttons, dtype=np.uint8)
            
            # Get the next action from the queue
            action = self._action_queue.pop(0)
            
        # Log the selected action
        self.logger.info(f"Selected action: {action}")
        
        return action 