import numpy as np
from abc import abstractmethod
from .core_module import CoreModule
from tools.utils import scale_image_up
import re

# TODO: 1. with visual state (vision only) 2. without visual state (text only)

class BaseModule(CoreModule):
    """
    Base module that directly processes observations and returns actions.
    This is a simplified module that bypasses the perception-memory-reasoning pipeline.
    
    Game-specific implementations should inherit from this class and implement
    the required abstract methods.
    """
    
    def __init__(self, model_name="claude-3-7-sonnet-latest", cache_dir="cache",
                 system_prompt="", prompt="", token_limit=100000, reasoning_effort="high"):
        """
        Initialize the base module.
        
        Args:
            model_name (str): The name of the model to use for inference.
            cache_dir (str): Directory for storing logs and cache files.
            system_prompt (str): System prompt for LLM calls.
            prompt (str): Default user prompt for LLM calls.
            token_limit (int): Maximum number of tokens for API calls.
            reasoning_effort (str): Reasoning effort for API calls (low, medium, high).
        """
        super().__init__(
            module_name="base_module",
            model_name=model_name,
            system_prompt=system_prompt,
            prompt=prompt,
            cache_dir=cache_dir,
            token_limit=token_limit,
            reasoning_effort=reasoning_effort
        )
        self.last_action = None

    
    def process_observation(self, observation:str):
        """
        Process the observation directly to plan the next action.
        Default implementation treats the observation as an image path and uses vision API.
        
        Args:
            observation: The game observation (typically an image path)
            
        Returns:
            dict: A dictionary containing 'action' and 'thought' keys
        """
        # Scale up image if needed
        scale_image_up(observation)
        
        # Call the vision API
        response = self.api_manager.vision_text_completion(
            model_name=self.model_name,
            system_prompt=self.system_prompt,
            prompt=self.prompt,
            image_path=observation,
            thinking=True,
            reasoning_effort=self.reasoning_effort,
            token_limit=self.token_limit
        )
        
        # Parse and log the response
        parsed_response = self._parse_response(response)
        self.log({
            "response": response,
            "thought": parsed_response.get("thought"),
            "action": parsed_response.get("action")
        })
        
        return parsed_response
    
    def _parse_response(self, response):
        """
        Parse the response to extract thought and action.
        
        Args:
            response (str): The raw response from the LLM
            
        Returns:
            dict: A dictionary containing action and thought
        """
        if not response:
            return {"action": None, "thought": "No response received"}
        
        # Initialize result with defaults
        result = {
            "action": None,
            "thought": ""
        }
        
        # Use regex to find thought and action sections
        # Match patterns like "thought:", "# thought:", "Thought:", etc.
        thought_pattern = r'(?:^|\n)(?:#\s*)?thought:(.+?)(?=(?:\n(?:#\s*)?(?:action|move):)|$)'
        action_pattern = r'(?:^|\n)(?:#\s*)?(?:action|move):(.+?)(?=(?:\n(?:#\s*)?thought:)|$)'
        
        # Find thought section using regex (case insensitive)
        thought_match = re.search(thought_pattern, response, re.DOTALL | re.IGNORECASE)
        if thought_match:
            result["thought"] = thought_match.group(1).strip()
        
        # Find action section using regex (case insensitive)
        action_match = re.search(action_pattern, response, re.DOTALL | re.IGNORECASE)
        if action_match:
            result["action"] = action_match.group(1).strip()
        
        # If no structured format was found, treat the whole response as thought
        if not result["thought"] and not result["action"]:
            result["thought"] = response.strip()
        elif not result["thought"]:  # If only action was found
            # Look for any text before the action as thought
            pre_action = re.split(r'(?:^|\n)(?:#\s*)?(?:action|move):', response, flags=re.IGNORECASE)[0]
            if pre_action and pre_action.strip():
                result["thought"] = pre_action.strip()
        
        # Normalize action format if needed
        if result["action"]:
            # Handle specific action formats if needed
            pass
        
        return result
