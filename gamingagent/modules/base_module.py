import numpy as np
from abc import abstractmethod
from .core_module import CoreModule, Observation
from tools.utils import scale_image_up
import re
import os

# TODO: 
# 1. with visual state (vision only) 
# 2. without visual state (text only) 
# 3. with visual state + text state (both)

class BaseModule(CoreModule):
    """
    Base module that directly processes visual/textual observations and returns actions.
    This is a simplified module that leverages gaming harness (in replacement of the agentic perception-memory-reasoning workflow).
    
    Game-specific implementations should inherit from this class and implement
    the required abstract methods.
    """
    
    def __init__(self, 
                model_name="claude-3-7-sonnet-latest", 
                observation_mode="vision",
                cache_dir="cache",
                system_prompt="", 
                prompt="", 
                token_limit=100000, 
                reasoning_effort="high"
        ):
        """
        Initialize the base module.
        
        Args:
            model_name (str): The name of the model to use for inference.
            observation_mode (str): Mode for processing observations:
                - "vision": Uses image path as input
                - "text": Uses textual representation as input
                - "both": Uses both image path and textual representation as inputs
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
        self.observation_mode = observation_mode
        self.observation = Observation()  # Observation data class
            
    def plan_action(self, 
            observation
        ):
        """
        Process the observation to plan the next action based on the observation_mode.
        If no observations are provided, uses previously set observations via set_perception_observation().
        
        Args:
            observation (Observation): A complete Observation instance
            
        Returns:
            dict: A dictionary containing 'action' and 'thought' keys
        """
        # Update observation
        if observation:
            self.observation.set_perception_observation(observation)
        
        # Validate observation based on mode
        if self.observation_mode in ["vision", "both"]:
            assert self.observation.img_path is not None, "No vision observation available"
        if self.observation_mode in ["text", "both"]: 
            assert (self.observation.textual_representation is not None) or (self.observation.processed_visual_description is not None), "No textual representation available"
        
        response = None
        if self.observation_mode == "vision":
            # Vision-based processing: observation is the image path
            # Scale up image if needed
            new_img_path = scale_image_up(self.observation.get_img_path())
            
            print(f"""
------------------------ BASE MODULE VISION API — SYSTEM PROMPT ------------------------
{self.system_prompt}
------------------------ END SYSTEM PROMPT ------------------------
""")
            print(f"""
------------------------ BASE MODULE VISION API — USER PROMPT ------------------------
{self.prompt}
------------------------ END USER PROMPT ------------------------
""")

            # Call the vision API
            response = self.api_manager.vision_text_completion(
                model_name=self.model_name,
                system_prompt=self.system_prompt,
                prompt=self.prompt,
                image_path=new_img_path,
                thinking=True,
                reasoning_effort=self.reasoning_effort,
                token_limit=self.token_limit
            )
        
        elif self.observation_mode == "text":
            # Create the full prompt with the text-based game state
            full_prompt = observation.get_complete_prompt(observation_mode=self.observation_mode, prompt_template=self.prompt)
            
            # Call the text API with the textual representation in the prompt
            response = self.api_manager.text_only_completion(
                model_name=self.model_name,
                system_prompt=self.system_prompt,
                prompt=full_prompt,
                thinking=True,
                reasoning_effort=self.reasoning_effort,
                token_limit=self.token_limit
            )    
        
        elif self.observation_mode == "both":
            # Both vision and text processing                
            # Scale up image if needed
            new_img_path = scale_image_up(self.observation.get_img_path())
            
            # Create the full prompt with the text-based game state
            full_prompt = observation.get_complete_prompt(observation_mode=self.observation_mode, prompt_template=self.prompt)
            
            # Call the vision API with both the image and textual representation
            response = self.api_manager.vision_text_completion(
                model_name=self.model_name,
                system_prompt=self.system_prompt,
                prompt=full_prompt,
                image_path=new_img_path,
                thinking=True,
                reasoning_effort=self.reasoning_effort,
                token_limit=self.token_limit
            )
        
        else:
            raise NotImplementedError(f"observation mode: {self.observation_mode} not supported.")
        
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
            "thought": None
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
            # action is left as none
        
        # If only thought is found, action is left as none
        
        # Normalize action format if needed
        if result["action"]:
            # Process specific action formats if needed
            pass
        
        return result