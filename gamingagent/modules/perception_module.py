import numpy as np
import os
import json
import datetime
from abc import ABC, abstractmethod
from PIL import Image
from .core_module import CoreModule, Observation

import copy

from tools.utils import scale_image_up

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
                reasoning_effort="high",
                scaffolding=None,
                use_perception=True,
                vllm_url=None,
                modal_url=None
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
            scaffolding (dict, optional): Scaffolding configuration dictionary with function and arguments.
                                     Default is None (no scaffolding). The function should take an observation
                                     and return a modified observation.
                                     Example: {"func": draw_grid_on_image, "funcArgs": {"grid_dim": [5, 5]}}
            use_perception (bool): Whether to use perception or not.
        """
        super().__init__(
            module_name="perception_module",
            model_name=model_name,
            system_prompt=system_prompt,
            prompt=prompt,
            cache_dir=cache_dir,
            token_limit=token_limit,
            reasoning_effort=reasoning_effort,
            vllm_url=vllm_url,
            modal_url=modal_url
        )

        valid_observation_modes = ["vision", "text", "both"]
        assert observation_mode in valid_observation_modes, f"Invalid observation_mode: {observation_mode}, choose only from: {valid_observation_modes}"
        self.observation_mode = observation_mode
        self.scaffolding = scaffolding
        self.use_perception = use_perception
        
        # Initialize observation
        self.observation = observation if observation is not None else Observation()
        self.processed_observation = copy.deepcopy(observation) if observation is not None else Observation()
        
        # Create observations directory for storing game state images
        self.obs_dir = os.path.join(cache_dir, "observations")
        os.makedirs(self.obs_dir, exist_ok=True)
        
    def _apply_scaffolding(self, observation):
        """
        Apply scaffolding function to the observation if specified.
        
        Args:
            observation: The observation to process
            
        Returns:
            observation: The potentially modified observation
        """
        if self.scaffolding is not None:
            scaffolding_func = self.scaffolding.get('func')
            scaffolding_args = self.scaffolding.get('funcArgs', {})
            if scaffolding_func and callable(scaffolding_func):
                try:
                    # Pass the observation to the scaffolding function and get back a modified observation
                    return scaffolding_func(observation, **scaffolding_args)
                except Exception as e:
                    print(f"Warning: Scaffolding function failed: {e}. Using original observation.")
                    return observation
            else:
                print("Warning: Invalid scaffolding configuration. Using original observation.")
                return observation
        return observation

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

            # Apply scaffolding function if specified
            self.processed_observation = self._apply_scaffolding(self.processed_observation)

            return self.processed_observation
        elif self.observation_mode in ["vision", "both"]:
            assert self.observation.img_path is not None, "to process from graphic representation, image should have been prepared and path should exist in observation."
            
            # First scale up the image
            new_img_path = scale_image_up(self.observation.get_img_path())
            self.processed_observation.img_path = new_img_path
            
            # Apply scaffolding function if specified
            self.processed_observation = self._apply_scaffolding(self.processed_observation)

            if self.use_perception:
                processed_visual_description = self.api_manager.vision_text_completion(
                    model_name=self.model_name,
                    system_prompt=self.system_prompt,
                    prompt=self.prompt,
                    image_path=self.processed_observation.img_path,
                    thinking=True,
                    reasoning_effort=self.reasoning_effort,
                    token_limit=self.token_limit
                )
                # returned API response should be a tuple
                actual_processed_visual_description = processed_visual_description[0]
                self.processed_observation.processed_visual_description = actual_processed_visual_description
            else:
                # Skip perception API call - set to None or a default message
                self.processed_observation.processed_visual_description = None

            return self.processed_observation
        else:
            raise NotImplementedError(f"observation mode: {self.observation_mode} not supported.")
    
    def get_perception_summary(self, observation):
        """
        Get a summary of the current perception.
        Uses Observation.get_textual_representation() to retrieve the symbolic representation.
        
        Returns:
            dict: A dictionary containing 
                1) img_path
                2) textual_representation
                3) visual_description
        """
        result = {
            "img_path": observation.img_path,
            "textual_representation": observation.get_textual_representation(),
            "processed_visual_description": observation.processed_visual_description
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
        # TODO: no specific response parsing need for perception module as of 05/27/2025
        pass