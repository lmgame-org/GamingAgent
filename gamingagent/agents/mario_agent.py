import os
import time
import json
import base64
import numpy as np
import cv2
from datetime import datetime
from io import BytesIO
from PIL import Image
from typing import Any, List, Dict, Optional, Tuple

from .base_agent import LLMAgent
from gamingagent.utils.logger import Logger

class MarioAgent(LLMAgent):
    """LLM-powered agent for Super Mario Bros with action timing and caching."""
    
    def __init__(
        self,
        env: Any,
        provider: Any,  # BaseProvider from providers
        logger: Logger,
        cache_dir: str = "cache/mario",
        max_tokens: int = 1000,
        temperature: float = 0.7,
        top_p: float = 0.9,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        action_hold_time: int = 4,  # Number of frames to hold an action
        test_pattern: Optional[List[str]] = None,
        use_test_pattern: bool = True,  # Default to test pattern for smooth gameplay
        action_hold_times: Optional[Dict[str, float]] = None,
        save_interval: int = 100,  # Save less frequently
        frame_skip: int = 2,      # Skip 2 frames between actions
        target_fps: int = 30,     # Target 30 FPS
        api_call_interval: int = 60  # Call API less frequently
    ):
        """
        Initialize Mario agent.
        
        Args:
            env: Game environment
            provider: LLM provider instance
            logger: Logger instance
            cache_dir: Directory for caching observations and actions
            max_tokens: Maximum tokens for LLM response
            temperature: LLM temperature
            top_p: LLM top_p parameter
            frequency_penalty: LLM frequency penalty
            presence_penalty: LLM presence penalty
            action_hold_time: Frames to hold each action
            test_pattern: Optional test pattern for actions
            use_test_pattern: Whether to use test pattern instead of LLM
            action_hold_times: Dictionary of action-specific hold times
            save_interval: Save every 30 frames instead of every frame
            frame_skip: Skip 2 frames between actions
            target_fps: Target 30 FPS
            api_call_interval: Call API every 60 frames
        """
        super().__init__(env, provider)
        self.logger = logger
        self.cache_dir = cache_dir
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.action_hold_time = action_hold_time
        self.test_pattern = test_pattern or []
        self.use_test_pattern = use_test_pattern
        self.action_hold_times = action_hold_times or {}
        self.test_pattern_index = 0
        self.current_action = None
        self.action_counter = 0
        self.total_actions = 0
        self.save_interval = save_interval
        self.frame_skip = frame_skip
        self.target_fps = target_fps
        self.render_interval = 1.0 / target_fps
        self.last_render_time = 0
        self.api_call_interval = api_call_interval
        self.last_api_call_time = 0
        
        # Create cache directories
        self.setup_cache_dirs()
        
    def setup_cache_dirs(self):
        """Setup cache directory structure"""
        # Create timestamp-based run directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = os.path.join(self.cache_dir, f'run_{timestamp}')
        
        # Create subdirectories
        self.obs_dir = os.path.join(self.run_dir, 'observations')
        self.state_dir = os.path.join(self.run_dir, 'states')
        self.metadata_dir = os.path.join(self.run_dir, 'metadata')
        
        # Create all directories
        for directory in [self.obs_dir, self.state_dir, self.metadata_dir]:
            os.makedirs(directory, exist_ok=True)
            
        self.logger.info(f"Cache directories created at: {self.run_dir}")
        
    def reset(self) -> None:
        """Reset agent's internal state."""
        super().reset()  # Reset conversation history from LLMAgent
        self.test_pattern_index = 0
        self.current_action = None
        self.action_counter = 0
        self.total_actions = 0
        
    def _extract_game_state(self, observation: Tuple[np.ndarray, Dict]) -> Dict:
        """Extract game state from observation tuple."""
        # Get the game state from the observation tuple
        game_state = observation[1]  # The second element is the info dict
        
        # Extract relevant information
        state = {
            "xscrollLo": game_state.get("xscrollLo", 0),
            "yscrollLo": game_state.get("yscrollLo", 0),
            "enemies": game_state.get("enemies", []),
            "coins": game_state.get("coins", []),
            "score": game_state.get("score", 0),
            "lives": game_state.get("lives", 3)
        }
        
        return state
        
    def _image_to_base64(self, image: np.ndarray) -> str:
        """Convert numpy array image to base64 string efficiently."""
        try:
            # Handle nested tuple observation format
            if isinstance(image, tuple):
                # The observation tuple contains (screen, info)
                # screen might be a tuple itself, so we need to get the actual numpy array
                screen = image[0]
                while isinstance(screen, tuple):
                    screen = screen[0]
                
                if not isinstance(screen, np.ndarray):
                    raise ValueError(f"Could not find numpy array in observation. Final type: {type(screen)}")
                image = screen
            
            # Ensure image is a numpy array
            if not isinstance(image, np.ndarray):
                raise ValueError(f"Image is not a numpy array. Type: {type(image)}")
            
            # Convert to PIL Image
            img = Image.fromarray(image)
            
            # Convert to PNG bytes
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            base64_image = base64.b64encode(buffered.getvalue()).decode()
            
            return base64_image
            
        except Exception as e:
            self.logger.error(f"Error converting image to base64: {e}")
            self.logger.error(f"Image type: {type(image)}")
            if isinstance(image, tuple):
                self.logger.error(f"Tuple length: {len(image)}")
                self.logger.error(f"First element type: {type(image[0])}")
                if isinstance(image[0], tuple):
                    self.logger.error(f"Nested tuple length: {len(image[0])}")
                    self.logger.error(f"Nested first element type: {type(image[0][0])}")
            return ""
        
    def format_prompt(self, observation: np.ndarray) -> str:
        """Format the observation into a prompt for the LLM."""
        prompt = """You are controlling Mario in Super Mario Bros. Analyze the game state image and generate the next action.
The action should be a list of 9 boolean values representing button states: [B, null, SELECT, START, UP, DOWN, LEFT, RIGHT, A]

CRITICAL CONTROLS:
- To JUMP: Set index 8 (A button) to True
- To move RIGHT: Set index 7 (RIGHT) to True
- To RUN: Set index 0 (B button) to True

Action Guidelines:
1. HOLD actions for sufficient duration:
   - Running (B): Hold for at least 1 second
   - Direction (LEFT/RIGHT): Hold for at least 0.5 seconds
   - Jumping (A): Hold for about 0.3 seconds for small jumps, 0.5 seconds for big jumps

2. Common Action Patterns:
   - Basic jump right: [False, False, False, False, False, False, False, True, True]
   - Running jump right: [True, False, False, False, False, False, False, True, True]
   - Full speed run right: [True, False, False, False, False, False, False, True, False]
   - Jump in place: [False, False, False, False, False, False, False, False, True]

JUMPING RULES:
- Jump BEFORE reaching gaps or obstacles
- Hold jump (A) longer for higher jumps
- Combine run (B) + jump (A) to jump further
- Jump on enemies to defeat them

Strategy tips:
- Progress requires moving RIGHT and JUMPING
- Jump early rather than late
- Use running jumps for long gaps

Output ONLY the action list in valid Python syntax, e.g.:
[True, False, False, False, False, False, False, True, True]"""
        return prompt
        
    def parse_response(self, response: str) -> List[bool]:
        """Parse the LLM's response into a valid Mario action array."""
        try:
            # Clean up response to get just the action list
            start_idx = response.find("[")
            end_idx = response.find("]") + 1
            if start_idx >= 0 and end_idx > start_idx:
                action_str = response[start_idx:end_idx]
                action_list = eval(action_str)
                
                # Validate action list
                if isinstance(action_list, list) and len(action_list) == 9 and all(isinstance(x, bool) for x in action_list):
                    # Prevent LEFT and RIGHT being pressed simultaneously
                    if action_list[6] and action_list[7]:  # If both LEFT and RIGHT are True
                        self.logger.warning("Both LEFT and RIGHT pressed, defaulting to RIGHT")
                        action_list[6] = False  # Disable LEFT
                    
                    return action_list
                    
        except Exception as e:
            self.logger.error(f"Error parsing response: {e}")
            
        # Default to moving right if parsing fails
        return [False, False, False, False, False, False, False, True, False]
        
    def _should_change_action(self, proposed_action: List[bool]) -> bool:
        """Determine if we should change the current action based on timing."""
        if self.current_action is None:
            return True
            
        # Get action-specific hold time
        hold_time = self.action_hold_time
        
        # Check if this is a jump action (A button pressed)
        if proposed_action[8]:  # A button (jump)
            hold_time = int(self.action_hold_times.get("jump", 0.3) * 60)  # Convert seconds to frames
        # Check if this is a run action (B button pressed)
        elif proposed_action[0]:  # B button (run)
            hold_time = int(self.action_hold_times.get("run", 0.5) * 60)
        # Check if this is a direction action (LEFT or RIGHT pressed)
        elif proposed_action[6] or proposed_action[7]:  # LEFT or RIGHT
            hold_time = int(self.action_hold_times.get("direction", 0.4) * 60)
            
        # Change action if:
        # 1. We've held the current action long enough
        # 2. The proposed action is different from current action
        if self.action_counter >= hold_time or proposed_action != self.current_action:
            return True
            
        return False
        
    def get_test_pattern(self) -> List[bool]:
        """Get the next action from the test pattern."""
        if not self.test_pattern:
            return [False, False, False, False, False, False, False, True, False]
            
        action = self.test_pattern[self.test_pattern_index]
        self.test_pattern_index = (self.test_pattern_index + 1) % len(self.test_pattern)
        return action
        
    def select_action(self, observation: np.ndarray) -> List[bool]:
        """Select the next action based on the observation."""
        # For smooth gameplay, use test pattern by default
        if self.use_test_pattern:
            proposed_action = self.get_test_pattern()
        else:
            try:
                # Only call API every api_call_interval frames
                if self.total_actions % self.api_call_interval == 0:
                    # Convert image to base64
                    image_base64 = self._image_to_base64(observation)
                    if not image_base64:
                        raise ValueError("Failed to convert image to base64")
                    
                    # Generate action using provider with both image and text
                    prompt = self.format_prompt(observation)
                    response = self.provider.generate_with_images(prompt, [image_base64])
                    
                    # Log API response
                    self.logger.info(f"\nAPI Response (Step {self.total_actions}):")
                    self.logger.info(f"Response: {response}")
                    
                    proposed_action = self.parse_response(response)
                else:
                    # Reuse last action if not time to update
                    proposed_action = self.current_action or [False, False, False, False, False, False, False, True, False]
                
            except Exception as e:
                self.logger.error(f"Error processing observation: {e}")
                # Fall back to test pattern
                proposed_action = self.get_test_pattern()
            
        # Check if we should change the current action based on timing
        if self._should_change_action(proposed_action):
            self.current_action = proposed_action
            self.action_counter = 0
        else:
            self.action_counter += 1
            
        self.total_actions += 1
        return self.current_action
        
    def _log_action_stats(self, observation: Tuple[np.ndarray, Dict], action: List[bool]) -> None:
        """Log statistics about the current action."""
        state = self._extract_game_state(observation)
        stats = {
            "action": action,
            "position": {
                "x": state["xscrollLo"],
                "y": state["yscrollLo"]
            },
            "score": state["score"],
            "lives": state["lives"]
        }
        self.logger.info(f"Action stats: {json.dumps(stats)}")
        
    def save_observation(
        self,
        observation: np.ndarray,
        step: int,
        reward: float,
        info: Dict,
        action: List[bool],
        env: Any = None,
        episode_start_time: Optional[str] = None
    ) -> None:
        """Save observation image, metadata, and game state."""
        # Only save every save_interval frames
        if step % self.save_interval != 0:
            return
            
        try:
            # Ensure observation is in correct format
            if isinstance(observation, tuple):
                observation = observation[0]
            
            # Save observation image
            obs_filename = os.path.join(self.obs_dir, f'obs_{step:06d}.png')
            cv2.imwrite(obs_filename, observation)
            
            # Prepare metadata
            metadata = {
                'step': step,
                'reward': reward,
                'action': action,
                'timestamp': datetime.now().isoformat()
            }
            
            # Add info if it's a dictionary
            if isinstance(info, dict):
                metadata['info'] = info
            else:
                metadata['info'] = str(info)
            
            # Save metadata
            meta_filename = os.path.join(self.metadata_dir, f'meta_{step:06d}.json')
            with open(meta_filename, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            # Save state if environment is provided
            if env is not None and episode_start_time is not None:
                try:
                    state = env.em.get_state()
                    state_path = os.path.join(self.state_dir, f'state_{episode_start_time}_step_{step:06d}.state')
                    with open(state_path, 'wb') as f:
                        f.write(state)
                except Exception as e:
                    self.logger.error(f"Could not save state at step {step}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error saving observation at step {step}: {e}")
            self.logger.error(f"Observation type: {type(observation)}")
            self.logger.error(f"Info type: {type(info)}")
        
    def close(self) -> None:
        """Clean up resources."""
        pass  # No cleanup needed for now

    def step(self, observation: np.ndarray) -> List[bool]:
        """Execute action for multiple frames with frame skip."""
        # Get action from agent without frame timing control
        # Let the environment handle frame timing
        action = self.select_action(observation)
        return action