import numpy as np
from abc import abstractmethod
from .core_module import CoreModule, Observation
from tools.utils import scale_image_up
import re
import os
import time

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
                reasoning_effort="high",
                vllm_url=None,
                modal_url=None
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
            reasoning_effort=reasoning_effort,
            vllm_url=vllm_url,
            modal_url=modal_url
        )
        self.observation_mode = observation_mode
            
    def plan_action(self, observation, custom_prompt=None):
        """
        Process the observation to plan the next action based on the observation_mode.
        If no observations are provided, uses previously set observations via set_perception_observation().
        
        Args:
            observation (Observation): A complete Observation instance
            
        Returns:
            dict: A dictionary containing 'action' and 'thought' keys
        """
        # Validate observation based on mode
        if self.observation_mode in ["vision", "both"]:
            assert observation.img_path is not None, "No vision observation available"
        if self.observation_mode in ["text", "both"]: 
            assert (observation.textual_representation is not None) or (observation.processed_visual_description is not None), "No textual representation available"
        
        # Create the full prompt with the text-based game state
        full_context = observation.get_complete_prompt(observation_mode=self.observation_mode, prompt_template=self.prompt)

        response = None
        if self.observation_mode in ["vision", "both"]:
            image_path = scale_image_up(observation.get_img_path())
            if not image_path:
                print("Warning: No image path provided for vision API call. Using text-only API.")
            response = self._call_vision_api(full_context, image_path, custom_prompt)
        else:
            response = self._call_text_api(full_context, custom_prompt)
        
        # returned API response should be a tuple
        response_string = response[0]
        
        # Parse and log the response
        parsed_response = self._parse_response(response_string)
        if parsed_response is None:
            parsed_response = {}
        parsed_response["raw_response_str"] = response_string


        self.log({
            "response": response_string,
            "thought": parsed_response.get("thought"),
            "action": parsed_response.get("action")
        })
        
        return parsed_response
    
    def _call_vision_api(self, context, img_path, custom_prompt=None):
        """
        Call the vision API with text context and image.
        
        Args:
            context (str): Formatted context with perception and memory
            img_path (str): Path to the current game image
            custom_prompt (str, optional): Custom prompt to use
            
        Returns:
            str: Raw response from the API
        """
        # Create user prompt with context
        if custom_prompt:
            user_prompt = context + "\n\n" + custom_prompt
        else:
            user_prompt = context

        print(f"""
------------------------ VISION API — FINAL USER PROMPT ------------------------
{user_prompt}
------------------------ END FINAL USER PROMPT ------------------------
""")
        
        # Call the vision-text API
        response = self.api_manager.vision_text_completion(
            model_name=self.model_name,
            system_prompt=self.system_prompt,
            prompt=user_prompt,
            image_path=img_path,
            thinking=True,
            reasoning_effort=self.reasoning_effort,
            token_limit=self.token_limit
        )
        
        return response
    
    def _call_text_api(self, context, custom_prompt=None):
        """
        Call the text-only API with context.
        
        Args:
            context (str): Formatted context with perception and memory data
            custom_prompt (str, optional): Custom prompt to use
            
        Returns:
            str: Raw response from the API
        """
        # Create user prompt
        if custom_prompt:
            user_prompt = context + "\n\n" + custom_prompt
        else:
            user_prompt = context
        
        print(f"""
------------------------ TEXT API - FINAL USER PROMPT ------------------------
{user_prompt}
------------------------ END TEXT API PROMPT ------------------------
""")
        # Call the API
        response = self.api_manager.text_only_completion(
            model_name=self.model_name,
            system_prompt=self.system_prompt,
            prompt=user_prompt,
            thinking=True,
            reasoning_effort=self.reasoning_effort,
            token_limit=self.token_limit
        )
        
        return response
    
    def _parse_response(self, response):
        """
        Parse the response to extract thought and action.
        
        Args:
            response (str): The raw response from the LLM
            
        Returns:
            dict: A dictionary containing action and thought
        """


        print(f"response: {response}")
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