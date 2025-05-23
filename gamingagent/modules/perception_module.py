import numpy as np
import os
import json
import datetime
from abc import ABC, abstractmethod
from PIL import Image
from .core_module import CoreModule, Observation

import copy

class PerceptionModule(CoreModule):
    """
    Perception module that analyzes game state to extract relevant features.
    
    Game-specific implementations should inherit from this class and implement
    the required abstract methods.
    """
    
    def __init__(self, 
                model_name="claude-3-7-sonnet-latest", 
                observation=None,
                observation_mode="vision",
                cache_dir="cache", 
                system_prompt="", 
                prompt="",
                token_limit=100000, 
                reasoning_effort="high"
        ):
        """
        Initialize the perception module.
        
        Args:
            model_name (str): The name of the model to use for inference.
            observation: The initial game state observation (Observation dataclass).
            observation_mode (str): Mode for processing observations:
                - "vision": Uses image path as input
                - "text": Uses symbolic representation/textual description as input
                - "both": Uses both image path and text representation as inputs
            cache_dir (str): Directory for storing logs and cache files.
            system_prompt (str): System prompt for perception module VLM calls.
            prompt (str): Default user prompt for perception module VLM calls.
            token_limit (int): Maximum number of tokens for VLM calls.
            reasoning_effort (str): Reasoning effort for reasoning VLM calls (low, medium, high).
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

        valid_observation_modes = ["vision", "text", "both"]
        assert observation_mode in valid_observation_modes, f"Invalid observation_mode: {observation_mode}, choose only from: {valid_observation_modes}"
        self.observation_mode = observation_mode
        
        # Initialize observation
        self.observation = observation if observation is not None else Observation()
        self.processed_observation = copy.deepcopy(observation) if observation is not None else Observation()
        
        # Create observations directory for storing game state images
        self.obs_dir = os.path.join(cache_dir, "observations")
        os.makedirs(self.obs_dir, exist_ok=True)
        
    def process_observation(self, observation):
        """
        Process a new observation to update the internal state.
        This method should be implemented by game-specific subclasses.
        
        There are two processing tracks:
        1. With graphics (with image): reads from observation.img_path
            a. perform image editing (scaling, grid drawing, etc.) --> new_img_path
            b. perform image visual element extraction --> processed_visual_description
        2. Without graphics (without image): reads from observation.textual_representation and observation.processed_visual_description
            a. perform game state analysis based on the textual representation
        
        Args:
            observation: The new game observation
            
        Returns:
            processed_observation: An updated observation with processed data
        """
        # Set the observation
        self.observation = observation
        self.processed_observation = copy.deepcopy(observation)
        
        # read variables from observation
        img_path = self.observation.img_path
        textual_representation = self.observation.textual_representation

        '''
        `-->` represents conversion performed by perception module
        observation |-- img  |--> processed_img
                    |        |--> processed_visual_description 
                    |
                    |-- textual_representation  |-- symbolic
                                                |-- descriptive (e.g. story adventure)
        '''
        
        # Process based on observation source
        if self.observation_mode in ["text"]:
            assert self.observation.textual_representation is not None, "to proceed with the game, at very least textual representations should be provided in observation."

            # TODO: add textual representation processing logic
            self.processed_observation.textual_representation = self.observation.textual_representation

            return self.processed_observation
        elif self.observation_mode in ["vision", "both"]:
            assert self.observation.img_path is not None, "to process from graphic representation, image should have been prepared and path should exist in observation."
            new_img_path = scale_image_up(self.observation.get_img_path())
            
            processed_visual_description = self.api_manager.vision_text_completion(
                model_name=self.model_name,
                system_prompt=self.system_prompt,
                prompt=self.prompt,
                image_path=new_img_path,
                thinking=True,
                reasoning_effort=self.reasoning_effort,
                token_limit=self.token_limit
            )

            self.processed_observation.processed_visual_description = processed_visual_description
            self.processed_observation.image_path = new_img_path

            return self.processed_observation
        else:
            raise NotImplementedError(f"observation mode: {observation_mode} not supported.")
    
    def get_perception_summary(self):
        """
        Get a summary of the current perception.
        Uses Observation.get_parsed_representation() to parse the symbolic representation.
        
        Returns:
            dict: A dictionary containing 
                1) img_path
                2) textual_representation
                3) processed_visual_description
        """
        result = {
            "img_path": self.processed_observation.img_path,
            "textual_representation": self.processed_observation.get_parsed_representation(),
            "processed_visual_description": self.process_observation.processed_visual_description
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
                textual_representation=img_array,
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
            return {"textual_representation": "", "game_state_details": ""}
        
        # Initialize result dictionary
        result = {
            "textual_representation": "",
            "game_state_details": ""
        }
        
        # Pattern to match symbolic representation and game state details sections
        symbolic_pattern = r'(?:^|\n)(?:#\s*)?(?:symbolic[_ ]?representation):(.+?)(?=(?:\n(?:#\s*)?(?:game[_ ]?state[_ ]?details|observations|environment):)|$)'
        details_pattern = r'(?:^|\n)(?:#\s*)?(?:game[_ ]?state[_ ]?details|observations|environment):(.+?)(?=(?:\n(?:#\s*)?symbolic[_ ]?representation:)|$)'
        
        # Find symbolic representation section
        symbolic_match = re.search(symbolic_pattern, response, re.DOTALL | re.IGNORECASE)
        if symbolic_match:
            result["textual_representation"] = symbolic_match.group(1).strip()
        
        # Find game state details section
        details_match = re.search(details_pattern, response, re.DOTALL | re.IGNORECASE)
        if details_match:
            result["game_state_details"] = details_match.group(1).strip()
        
        # If no structured format was found, try to intelligently parse the content
        if not result["textual_representation"] and not result["game_state_details"]:
            # Try to detect if the response looks like a structured game state
            if ":" in response and ("{" in response or "[" in response):
                # Looks like structured data, put in symbolic representation
                result["textual_representation"] = response.strip()
            else:
                # Treat as general observation
                result["game_state_details"] = response.strip()
        
        # Try to parse textual_representation as JSON if it looks like JSON
        try:
            import json
            if result["textual_representation"] and (
                result["textual_representation"].strip().startswith("{") or 
                result["textual_representation"].strip().startswith("[")
            ):
                json_data = json.loads(result["textual_representation"])
                return {
                    "textual_representation": json_data,
                    "game_state_details": result["game_state_details"]
                }
        except json.JSONDecodeError:
            # Not valid JSON, keep as string
            pass
        
        return result
