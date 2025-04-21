import numpy as np
from typing import Union, Tuple, Dict, Any
from gamingagent.agents.base_agent import BaseAgent
from gamingagent.providers.api_provider_manager import APIProviderManager
from gamingagent.utils.utils import encode_image

class MarioAgent(BaseAgent):
    """Agent that uses Claude 3 to play Mario games."""
    
    def __init__(
        self,
        env: Any,
        game_name: str,
        api_provider: str = "anthropic",
        model_name: str = "claude-3-opus-20240229"
    ):
        """Initialize the Mario agent."""
        super().__init__(env, game_name, api_provider, model_name)
        self.api_manager = APIProviderManager()
        self.api_manager.initialize_providers(anthropic_model=model_name)
        self.provider = self.api_manager.get_provider(api_provider)
        self.logger.info(f"Initialized with {api_provider} and {model_name}")
        
    def _worker(self, observation: np.ndarray) -> str:
        """Make API call to Claude 3 with the game observation."""
        prompt = """You are playing Super Mario Bros. Based on the current game screen, 
        you need to decide what action to take. The available actions are:
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
            # Encode image using utility function
            img_base64 = encode_image(observation)
            
            # Make API call with image
            response = self.provider.generate_with_images(
                prompt=prompt,
                images=[img_base64],
                max_tokens=50
            )
            
            # Log the API call
            self.logger.info(f"API Response: {response}")
            
            if not response:
                self.logger.warning("Empty response from API, using default action")
                return "RIGHT"  # Default action if API fails
                
            return response
            
        except Exception as e:
            self.logger.error(f"Error in API call: {str(e)}")
            return "RIGHT"  # Default action on error
        
    def _parse_response(self, response: str) -> np.ndarray:
        """Parse Claude 3's response into an action array."""
        # Default action (no buttons pressed)
        action = np.zeros(self.env.num_buttons, dtype=np.uint8)
        
        # Map of action names to button indices
        # Using the same mapping as RetroInteractive
        action_map = {
            "RIGHT": self.env.buttons.index("RIGHT"),
            "LEFT": self.env.buttons.index("LEFT"),
            "UP": self.env.buttons.index("UP"),
            "DOWN": self.env.buttons.index("DOWN"),
            "A": self.env.buttons.index("A"),  # Jump button
            "B": self.env.buttons.index("B")   # Run button
        }
        
        try:
            # Split response into individual actions
            actions = response.strip().upper().split("+")
            actions = [a.strip() for a in actions]
            
            # Set corresponding indices to 1
            for a in actions:
                if a in action_map:
                    action[action_map[a]] = 1
                    
            return action
            
        except Exception as e:
            self.logger.error(f"Error parsing response: {str(e)}")
            # Return default action (move right) on error
            action[action_map["RIGHT"]] = 1
            return action
        
    def select_action(self, observation: Union[np.ndarray, Tuple[np.ndarray, Dict]]) -> np.ndarray:
        """Select an action based on the current observation."""
        # Handle tuple observations (RAM)
        if isinstance(observation, tuple):
            observation = observation[0]
            
        # Get response from Claude 3
        response = self._worker(observation)
        
        # Parse response into action
        action = self._parse_response(response)
        
        # Log the action
        self.log_action(action, self.env.step_count)
        
        return action
