import numpy as np
from abc import abstractmethod
from .core_module import CoreModule, Observation
from tools.utils import scale_image_up
import re
import os
from typing import Optional

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
        # --- DEBUG PRINT --- 
        print(f"DEBUG (BaseModule.__init__): Received system_prompt: '{system_prompt[:100]}...'")
        print(f"DEBUG (BaseModule.__init__): Received prompt: '{prompt[:100]}...'")
        # --- END DEBUG PRINT ---
        self.observation_mode = observation_mode
        self.observation = Observation()  # Observation data class
            
    def plan_action(self, 
            observation=None, 
            img_path=None, 
            textual_representation=None,
            additional_prompt_context: Optional[str] = None
        ):
        """
        Process the observation to plan the next action based on the observation_mode.
        If no observations are provided, uses previously set observations via set_perception_observation().
        
        Args:
            observation (Observation, optional): A complete Observation instance
            img_path (str, optional): For "vision" or "both" mode: image path
            textual_representation (str, optional): For "text" or "both" mode: textual representation of game board
            additional_prompt_context (str, optional): Additional text to prepend to the main prompt for the LLM.
            
        Returns:
            dict: A dictionary containing 'action' and 'thought' keys
        """
        # Update observation
        if observation or img_path or textual_representation:
            self.observation.set_perception_observation(observation, img_path, textual_representation)
        
        # Validate observation based on mode
        if self.observation_mode in ["vision", "both"]:
            assert self.observation.img_path is not None, "No vision observation available"
        if self.observation_mode in ["text", "both"]: 
            assert (self.observation.textual_representation is not None) or (self.observation.processed_visual_description is not None), "No textual representation available"
    
        def prepare_text_based_game_state():
            textual_representation = self.observation.get_textual_representation()
            processed_visual_description = self.observation.get_processed_visual_description()
            if textual_representation and processed_visual_description:
                text_repr = f"Game Textual Representation:\n{textual_representation}\n\nGame Visual Elements Description:\n{processed_visual_description}\n\n"
            elif textual_representation:
                text_repr = f"Game Textual Representation:\n{textual_representation}\n\n"
            elif processed_visual_description:
                text_repr = f"Game Visual Elements Description:\n{processed_visual_description}\n\n"
            else:
                text_repr = "No Text-Based Game State Provided."
            
            return text_repr
        
        response = None
        if self.observation_mode == "vision":
            # Vision-based processing: observation is the image path
            # Scale up image if needed
            new_img_path = scale_image_up(self.observation.get_img_path())
            
            # --- DEBUG PRINT ---
            print(f"DEBUG (BaseModule.plan_action for VISION): self.system_prompt: '{self.system_prompt[:100]}...'")
            # --- MODIFIED prompt logging ---
            current_prompt_for_llm = self.prompt
            if additional_prompt_context:
                current_prompt_for_llm = additional_prompt_context + self.prompt
            print(f"DEBUG (BaseModule.plan_action for VISION): current_prompt_for_llm: '{current_prompt_for_llm[:150]}...'")
            # --- END DEBUG PRINT ---
            
            # Call the vision API
            response = self.api_manager.vision_text_completion(
                model_name=self.model_name,
                system_prompt=self.system_prompt,
                prompt=current_prompt_for_llm,
                image_path=new_img_path,
                thinking=True,
                reasoning_effort=self.reasoning_effort,
                token_limit=self.token_limit
            )
        
        elif self.observation_mode == "text":
            # Create the full prompt with the text-based game state
            # TODO (lanxiang): replace with Observation.get_complete_prompt
            text_repr = prepare_text_based_game_state()
            full_prompt = f"{self.prompt}\n\n{text_repr}"
            if additional_prompt_context:
                full_prompt = additional_prompt_context + full_prompt
            
            # --- DEBUG PRINT ---
            print(f"DEBUG (BaseModule.plan_action for TEXT): self.system_prompt: '{self.system_prompt[:100]}...'")
            print(f"DEBUG (BaseModule.plan_action for TEXT): full_prompt: '{full_prompt[:100]}...'") # Check combined prompt
            # --- END DEBUG PRINT ---
            
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
            # TODO (lanxiang): replace with Observation.get_complete_prompt
            text_repr = prepare_text_based_game_state()
            full_prompt = f"{self.prompt}\n\n{text_repr}"
            if additional_prompt_context:
                full_prompt = additional_prompt_context + full_prompt
            
            # --- DEBUG PRINT ---
            print(f"DEBUG (BaseModule.plan_action for BOTH): self.system_prompt: '{self.system_prompt[:100]}...'")
            print(f"DEBUG (BaseModule.plan_action for BOTH): full_prompt: '{full_prompt[:100]}...'") # Check combined prompt
            # --- END DEBUG PRINT ---
            
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
        
        print(f"[BaseModule DEBUG _parse_response] Raw LLM response:\n---\n{response}\n---")

        # Initialize result with defaults
        result = {
            "action": None,
            "thought": None,
            "parsed_dialogue": None
        }
        
        # Use regex to find thought and action sections
        # Match patterns like \"thought:\", \"# thought:\", \"Thought:\", etc.
        
        # REMOVED (?:^|\n) anchor from start of patterns for diagnostic purposes
        thought_pattern = r'(?:#\s*)?thought:\s*(.+)(?=(?:$|\n(?:action|move|dialog):))'
        action_pattern = r'(?:#\s*)?(?:action|move):\s*([^\n]+)'
        dialogue_pattern = r'(?:#\s*)?dialog(?:ue)?[:-]\s*([^:]+?):\s*(.+?)(?=(?:$|\n\s*(?:action|move|thought|Options|Evidence|Scene):))'
        
        # Find thought section using regex (case insensitive)
        thought_match = re.search(thought_pattern, response, re.DOTALL | re.IGNORECASE)
        print(f"[BaseModule DEBUG _parse_response] thought_match (pattern: {thought_pattern}): {thought_match}")
        if thought_match:
            result["thought"] = thought_match.group(1).strip()
        
        # Find action section using regex (case insensitive)
        action_match = re.search(action_pattern, response, re.DOTALL | re.IGNORECASE)
        print(f"[BaseModule DEBUG _parse_response] action_match (pattern: {action_pattern}): {action_match}")
        if action_match:
            result["action"] = action_match.group(1).strip()
        
        # Find dialogue section using regex (case insensitive)
        dialogue_match = re.search(dialogue_pattern, response, re.DOTALL | re.IGNORECASE)
        print(f"[BaseModule DEBUG _parse_response] dialogue_match (pattern: {dialogue_pattern}): {dialogue_match}")
        if dialogue_match:
            speaker = dialogue_match.group(1).strip()
            text = dialogue_match.group(2).strip()
            result["parsed_dialogue"] = {"speaker": speaker, "text": text}
        
        # If no structured format was found, treat the whole response as thought
        if not result["thought"] and not result["action"] and not result["parsed_dialogue"]:
            result["thought"] = response.strip()
        elif not result["thought"] and not result["parsed_dialogue"]:  # If only action was found
            # Look for any text before the action as thought
            pre_action = re.split(r'(?:^|\n)(?:#\\s*)?(?:action|move):\s*', response, flags=re.IGNORECASE)[0]
            if pre_action and pre_action.strip():
                result["thought"] = pre_action.strip()
            # action is left as none
        
        # If only thought is found, action is left as none
        
        # Normalize action format if needed
        if result["action"]:
            # Process specific action formats if needed
            pass
        
        print(f"[BaseModule DEBUG _parse_response] Final parsed result: {result}")
        return result
