import numpy as np
import os
import json
import datetime
from abc import ABC, abstractmethod
from PIL import Image
from .core_module import CoreModule, Observation

# TODO: customize perception data analysis
# TODO: texual vs system two track

class PerceptionModule(CoreModule):
    """
    Perception module that analyzes game state to extract relevant features.
    
    Game-specific implementations should inherit from this class and implement
    the required abstract methods.
    """
    
    def __init__(self, model_name="claude-3-7-sonnet-latest", observation=None, 
                 cache_dir="cache", system_prompt="", prompt="",
                 token_limit=100000, reasoning_effort="high"):
        """
        Initialize the perception module.
        
        Args:
            model_name (str): The name of the model to use for inference.
            observation: The initial game state observation (Observation dataclass).
            cache_dir (str): Directory for storing logs and cache files.
            system_prompt (str): System prompt for LLM calls.
            prompt (str): Default user prompt for LLM calls.
            token_limit (int): Maximum number of tokens for API calls.
            reasoning_effort (str): Reasoning effort for API calls (low, medium, high).
        """
        super().__init__(
            module_name="perception_module",
            model_name=model_name,
            system_prompt=system_prompt,
            prompt=prompt,
            cache_dir=cache_dir,
            token_limit=token_limit,
            reasoning_effort=reasoning_effort
        )
        
        # Initialize observation
        self.observation = observation if observation is not None else Observation()
        self.new_observation = observation if observation is not None else Observation()
        
        # Create observations directory for storing game state images
        self.obs_dir = os.path.join(cache_dir, "observations")
        os.makedirs(self.obs_dir, exist_ok=True)
        
    @abstractmethod
    def process_observation(self, observation):
        """
        Process a new observation to update the internal state.
        This method should be implemented by game-specific subclasses.
        
        There are two processing tracks:
        1. Image processing: If observation.img_path is not None, implement image processing
           (scaling, grid drawing, etc.)
        2. Symbolic processing: If observation.symbolic_representation is not None,
           perform game state analysis based on the symbolic representation
        
        The processed symbolic representation should be a dictionary with string keys
        representing different aspects of the game state.
        
        Args:
            observation: The new game observation (Observation dataclass)
            
        Returns:
            Observation: A new or updated Observation with processed data
        """
        # This is an abstract method that must be implemented by subclasses
        # The implementation should follow this pattern:
        
        # Set the observation
        self.observation = observation
        self.new_observation = observation  # Update both observation instances
        
        # Sync instance variables with observation properties
        self.img_path = self.observation.img_path
        self.symbolic_representation = self.observation.symbolic_representation
        
        # Process based on available data:
        
        # 1. If img_path is available, process the image
        # if self.img_path is not None:
        #     # Implement image processing (scaling, grid detection, etc.)
        #     pass
        
        # 2. If symbolic_representation is available, analyze the game state
        # if self.symbolic_representation is not None:
        #     # Implement game state analysis and create a dictionary
        #     analyzed_state = {
        #         "key1": "value1",
        #         "key2": "value2",
        #         # ... other game state information
        #     }
        #     processed_representation = analyzed_state
        # else:
        #     processed_representation = {}
        
        # Return a new or updated Observation
        # return Observation(
        #     symbolic_representation=processed_representation,
        #     img_path=self.img_path
        # )
    
    def get_perception_summary(self):
        """
        Get a summary of the current perception.
        Uses Observation.get_parsed_representation() to parse the symbolic representation.
        
        Returns:
            dict: A dictionary containing img_path and symbolic_representation
        """
        result = {
            "img_path": self.new_observation.img_path if self.new_observation else None,
            "symbolic_representation": self.new_observation.get_parsed_representation() if self.new_observation else ""
        }
        return result
    
    def load_obs(self, img_path):
        """
        Load an observation image from disk.
        
        Args:
            img_path (str): Path to the image file
            
        Returns:
            Observation: An Observation dataclass containing the loaded image
        """
        try:
            img = Image.open(img_path)
            img_array = np.array(img)
            
            # Create and return Observation dataclass
            return Observation(
                symbolic_representation=img_array,
                img_path=img_path
            )
        except Exception as e:
            print(f"Error loading observation from {img_path}: {e}")
            return None

    def _parse_response(self, response):
        """
        Parse LLM response to extract structured perception data.
        
        Args:
            response (str): The raw response from the LLM
            
        Returns:
            dict: Structured perception data
        """
        import re
        
        if not response:
            return {"symbolic_representation": "", "game_state_details": ""}
        
        # Initialize result dictionary
        result = {
            "symbolic_representation": "",
            "game_state_details": ""
        }
        
        # Pattern to match symbolic representation and game state details sections
        symbolic_pattern = r'(?:^|\n)(?:#\s*)?(?:symbolic[_ ]?representation):(.+?)(?=(?:\n(?:#\s*)?(?:game[_ ]?state[_ ]?details|observations|environment):)|$)'
        details_pattern = r'(?:^|\n)(?:#\s*)?(?:game[_ ]?state[_ ]?details|observations|environment):(.+?)(?=(?:\n(?:#\s*)?symbolic[_ ]?representation:)|$)'
        
        # Find symbolic representation section
        symbolic_match = re.search(symbolic_pattern, response, re.DOTALL | re.IGNORECASE)
        if symbolic_match:
            result["symbolic_representation"] = symbolic_match.group(1).strip()
        
        # Find game state details section
        details_match = re.search(details_pattern, response, re.DOTALL | re.IGNORECASE)
        if details_match:
            result["game_state_details"] = details_match.group(1).strip()
        
        # If no structured format was found, try to intelligently parse the content
        if not result["symbolic_representation"] and not result["game_state_details"]:
            # Try to detect if the response looks like a structured game state
            if ":" in response and ("{" in response or "[" in response):
                # Looks like structured data, put in symbolic representation
                result["symbolic_representation"] = response.strip()
            else:
                # Treat as general observation
                result["game_state_details"] = response.strip()
        
        # Try to parse symbolic_representation as JSON if it looks like JSON
        try:
            import json
            if result["symbolic_representation"] and (
                result["symbolic_representation"].strip().startswith("{") or 
                result["symbolic_representation"].strip().startswith("[")
            ):
                json_data = json.loads(result["symbolic_representation"])
                return {
                    "symbolic_representation": json_data,
                    "game_state_details": result["game_state_details"]
                }
        except json.JSONDecodeError:
            # Not valid JSON, keep as string
            pass
        
        return result
