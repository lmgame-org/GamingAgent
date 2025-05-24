import numpy as np
from abc import abstractmethod
from .core_module import CoreModule, Observation
from tools.utils import scale_image_up
import re
import os
import json

# TODO: 1. with visual state (vision only) 2. without visual state (text only) 3 with visual state + text state (both)

class BaseModule(CoreModule):
    """
    Base module that directly processes observations and returns actions.
    This is a simplified module that bypasses the perception-memory-reasoning pipeline.
    
    Game-specific implementations should inherit from this class and implement
    the required abstract methods.
    """
    
    def __init__(self, model_name="claude-3-7-sonnet-latest", cache_dir="cache",
                 system_prompt="", prompt="", token_limit=100000, reasoning_effort="high",
                 observation_mode="vision"):
        """
        Initialize the base module.
        
        Args:
            model_name (str): The name of the model to use for inference.
            cache_dir (str): Directory for storing logs and cache files.
            system_prompt (str): System prompt for LLM calls.
            prompt (str): Default user prompt for LLM calls.
            token_limit (int): Maximum number of tokens for API calls.
            reasoning_effort (str): Reasoning effort for API calls (low, medium, high).
            observation_mode (str): Mode for processing observations:
                - "vision": Uses image path as input
                - "text": Uses symbolic representation as input
                - "both": Uses both image path and text representation as inputs
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
        self.observation_mode = observation_mode
        self.observation = Observation()  # Using the Observation data class

    def set_observation(self, observation=None, img_path=None, symbolic_representation=None):
        """
        Set the current observation for later processing.
        
        Args:
            observation (Observation, optional): A complete Observation instance
            img_path (str, optional): For "vision" or "both" modes: image path
            symbolic_representation (str, optional): For "text" or "both" modes: symbolic representation of game board
        """
        # If an Observation is directly provided, use it
        if observation is not None:
            self.observation = observation
        # Otherwise, update the existing observation with provided data
        else:
            if self.observation is None:
                self.observation = Observation()
                
            if img_path is not None and self.observation_mode in ["vision", "both"]:
                self.observation.img_path = img_path
                
            if symbolic_representation is not None and self.observation_mode in ["text", "both"]:
                self.observation.symbolic_representation = str(symbolic_representation) if symbolic_representation is not None else None
            
    def process_observation(self, observation=None, img_path=None, symbolic_representation=None):
        """
        Process the observation to plan the next action based on the observation_mode.
        If no observations are provided, uses previously set observations via set_observation().
        
        Args:
            observation (Observation, optional): A complete Observation instance
            img_path (str, optional): For "vision" or "both" mode: image path
            symbolic_representation (str, optional): For "text" or "both" mode: symbolic representation of game board
            
        Returns:
            dict: A dictionary containing 'action' and 'thought' keys
        """
        # Update stored observations if provided
        if observation is not None or img_path is not None or symbolic_representation is not None:
            self.set_observation(observation, img_path, symbolic_representation)
            
        # Validate required observations based on mode
        if self.observation_mode in ["vision", "both"] and self.observation.img_path is None:
            return {"action": None, "thought": "No vision observation available"}
                
        if self.observation_mode in ["text", "both"] and self.observation.symbolic_representation is None:
            return {"action": None, "thought": "No symbolic observation available"}
        
        response = None
        
        if self.observation_mode == "vision":
            # Vision-based processing: observation is the image path
            # Scale up image if needed
            new_img_path = scale_image_up(self.observation.get_img_path())
            
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
            # Text-based processing: observation is symbolic representation only
            text_repr = f"Symbolic Representation:\n{self.observation.get_symbolic_representation()}"
                
            # Create the full prompt with the symbolic representation
            full_prompt = f"Text Observation:\n{text_repr}\n\n{self.prompt}"
            
            # Call the text API with the symbolic representation in the prompt
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
            
            # Use only symbolic representation
            text_repr = f"Symbolic Representation:\n{self.observation.get_symbolic_representation()}"
            
            # Create a prompt that includes the symbolic representation
            full_prompt = f"Text Observation:\n{text_repr}\n\n{self.prompt}"
            
            # Call the vision API with both the image and symbolic representation
            response = self.api_manager.vision_text_completion(
                model_name=self.model_name,
                system_prompt=self.system_prompt,
                prompt=full_prompt,
                image_path=new_img_path,
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
        Prioritizes parsing the entire response as JSON if possible (for structured outputs like Candy Crush).
        Falls back to regex search for 'thought:' and 'action:' tags for other games.
        """
        if not response:
            return {"action": None, "thought": "No response received"}
        
        result = {"action": None, "thought": ""}

        try:
            # Attempt to parse the entire response as JSON first
            # This is useful for models that directly output JSON as requested by some prompts (e.g., Candy Crush)
            json_match = re.search(r'\s*(\{.*?\})\s*$', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                data = json.loads(json_str)
                result["thought"] = data.get("thought", "")
                move_data = data.get("move")

                if isinstance(move_data, list) and len(move_data) == 2 and \
                   all(isinstance(coord, (list, tuple)) and len(coord) == 2 and \
                       all(isinstance(val, int) for val in coord) for coord in move_data):
                    # Convert to tuple of tuples for canonical representation
                    coord1 = tuple(move_data[0])
                    coord2 = tuple(move_data[1])
                    # Ensure consistent order for the string key: (min_coord, max_coord)
                    # This matches the format in CandyCrushEnvWrapper's dynamic action mapping
                    result["action"] = f"({min(coord1, coord2)},{max(coord1, coord2)})"
                    return result # Successfully parsed JSON with valid move
                elif move_data is not None: # Move data present but not in expected format
                    result["thought"] = result.get("thought", "") + f" (Note: LLM move data '{move_data}' was not in the expected list-of-lists-of-ints format)"
                    result["action"] = None # Action is invalid
                    # Still return here as JSON was found, even if move part was malformed
                    return result 
            # If JSON parsing of the whole response failed or didn't yield a move, fall through to regex (original method)

        except json.JSONDecodeError:
            # Not a valid JSON object as the whole response, proceed to regex parsing
            pass 
        except Exception as e:
            # Other unexpected error during JSON parsing attempt
            print(f"[BaseModule] Unexpected error during JSON parsing attempt in _parse_response: {e}. Falling back to regex.")
            result["thought"] = f"Error during JSON parsing attempt: {e}. Response: {response[:100]}..." # Log error in thought
            # Fall through to regex parsing for safety

        # Regex parsing (original method as fallback)
        thought_pattern = r'(?:^|\n)(?:#\s*)?thought:(.+?)(?=(?:\n(?:#\s*)?(?:action|move):)|$)'
        action_pattern = r'(?:^|\n)(?:#\s*)?(?:action|move):(.+?)(?=(?:\n(?:#\s*)?thought:)|$)'
        
        thought_match = re.search(thought_pattern, response, re.DOTALL | re.IGNORECASE)
        if thought_match:
            result["thought"] = thought_match.group(1).strip()
        
        action_match = re.search(action_pattern, response, re.DOTALL | re.IGNORECASE)
        if action_match:
            result["action"] = action_match.group(1).strip()
        
        if not result["thought"] and not result["action"]:
            result["thought"] = response.strip()
        elif not result["thought"] and result["action"]:
            pre_action_match = re.match(r'(.*?)(?:^|\n)(?:#\s*)?(?:action|move):', response, re.DOTALL | re.IGNORECASE)
            if pre_action_match and pre_action_match.group(1):
                 result["thought"] = pre_action_match.group(1).strip()
        
        return result
