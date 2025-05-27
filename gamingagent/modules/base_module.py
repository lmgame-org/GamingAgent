import numpy as np
from abc import abstractmethod
from .core_module import CoreModule, Observation
from tools.utils import scale_image_up
import re
import os
from typing import Optional
import json

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
        
        print(f"""
------------------------ BASE MODULE API — SYSTEM PROMPT ------------------------
{self.system_prompt}
------------------------ END SYSTEM PROMPT ------------------------
""")
        print(f"""
------------------------ BASE MODULE API — USER PROMPT ------------------------
{self.prompt}
------------------------ END USER PROMPT ------------------------
""")

        response = None
        # Create the full prompt with the text-based game state
        full_prompt = observation.get_complete_prompt(observation_mode=self.observation_mode, prompt_template=self.prompt)
        
        if self.observation_mode == "vision":
            # Vision-based processing: observation is the image path
            # Scale up image if needed
            new_img_path = scale_image_up(self.observation.get_img_path())

            # Call the vision API
            response = self.api_manager.vision_text_completion(
                model_name=self.model_name,
                system_prompt=self.system_prompt,
                prompt=full_prompt,
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
        #print("============================================")
        #print(f"base module generated response:\n{response}\n")
        #print("============================================")

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
        Handles both plain text with keywords and JSON formatted responses.
        
        Args:
            response (str): The raw response from the LLM
            
        Returns:
            dict: A dictionary containing action, thought, and parsed_dialogue
        """
        if not response:
            return {"action": None, "thought": "No response received", "parsed_dialogue": None}

        # print(f"[BaseModule DEBUG _parse_response] Raw LLM response:\n---\n{response}\n---")

        result = {
            "action": None,
            "thought": None,
            "parsed_dialogue": None
        }

        def _strip_markdown_fences(text):
            # Remove ```json ... ``` or ``` ... ```
            match = re.match(r'^```(?:json)?\n(.*?)\n```$', text, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
            return text.strip()

        stripped_response = _strip_markdown_fences(response)

        # Attempt JSON parsing first
        try:
            json_data = json.loads(stripped_response)
            if isinstance(json_data, dict):
                # print(f"[BaseModule DEBUG _parse_response] Successfully parsed as JSON: {json_data}")
                result["action"] = json_data.get("move", json_data.get("action"))
                result["thought"] = json_data.get("thought")
                
                dialogue_str = json_data.get("Dialog", json_data.get("dialog"))
                if dialogue_str and isinstance(dialogue_str, str):
                    parts = dialogue_str.split(":", 1)
                    if len(parts) == 2:
                        speaker = parts[0].strip()
                        text = parts[1].strip()
                        result["parsed_dialogue"] = {"speaker": speaker, "text": text}
                    else: # Could be just text without a speaker
                        result["parsed_dialogue"] = {"speaker": "Unknown", "text": dialogue_str.strip()}
                
                # If we got at least action or thought or dialogue from JSON, consider it a success
                if result["action"] or result["thought"] or result["parsed_dialogue"]:
                    # print(f"[BaseModule DEBUG _parse_response] Parsed from JSON: {result}")
                    return result
        except json.JSONDecodeError:
            # print(f"[BaseModule DEBUG _parse_response] Not a valid JSON response or markdown stripping failed. Falling back to regex.")
            pass # Not JSON, or malformed, proceed to regex parsing
        
        # Fallback to Regex parsing if JSON parsing fails or doesn't yield useful data
        # print(f"[BaseModule DEBUG _parse_response] Attempting regex parsing on: {stripped_response}")
        
        # Use regex to find thought and action sections
        # Match patterns like \"thought:\", \"# thought:\", \"Thought:\", etc.
        
        # REMOVED (?:^|\n) anchor from start of patterns for diagnostic purposes
        thought_pattern = r'(?:#\s*)?thought:\s*(.+)(?=(?:$|\n(?:action|move|dialog):))'
        action_pattern = r'(?:#\s*)?(?:action|move):\s*([^\n]+)'
        # Refined dialogue pattern to be non-greedy and have a more robust lookahead
        dialogue_pattern = r'(?:#\s*)?dialog(?:ue)?[:-]\s*([^:]+?):\s*(.+?)(?=(?:$|\n\s*(?:action|move|thought|Options|Evidence|Scene):))'
        
        # Find thought section using regex (case insensitive)
        thought_match = re.search(thought_pattern, stripped_response, re.DOTALL | re.IGNORECASE) # Use stripped_response
        # print(f"[BaseModule DEBUG _parse_response] thought_match (pattern: {thought_pattern}): {thought_match}")
        if thought_match:
            result["thought"] = thought_match.group(1).strip()
        
        # Find action section using regex (case insensitive)
        action_match = re.search(action_pattern, stripped_response, re.DOTALL | re.IGNORECASE) # Use stripped_response
        # print(f"[BaseModule DEBUG _parse_response] action_match (pattern: {action_pattern}): {action_match}")
        if action_match:
            result["action"] = action_match.group(1).strip()
        
        # Find dialogue section using regex (case insensitive)
        dialogue_match = re.search(dialogue_pattern, stripped_response, re.DOTALL | re.IGNORECASE) # Use stripped_response
        # print(f"[BaseModule DEBUG _parse_response] dialogue_match (pattern: {dialogue_pattern}): {dialogue_match}")
        if dialogue_match:
            speaker = dialogue_match.group(1).strip()
            text = dialogue_match.group(2).strip()
            result["parsed_dialogue"] = {"speaker": speaker, "text": text}
        
        # If no structured format was found by either JSON or Regex, treat the whole (stripped) response as thought
        if not result["thought"] and not result["action"] and not result["parsed_dialogue"]:
            result["thought"] = stripped_response # Use stripped_response
        elif not result["thought"] and not result["parsed_dialogue"]:  # If only action was found by regex
            # Look for any text before the action as thought
            pre_action_parts = re.split(r'(?:^|\n)(?:#\s*)?(?:action|move):\s*', stripped_response, flags=re.IGNORECASE) # Use stripped_response
            if pre_action_parts and pre_action_parts[0] and pre_action_parts[0].strip():
                result["thought"] = pre_action_parts[0].strip()
            # action is already set from regex
        
        # If only thought is found by regex, action is left as none (default)
        # If only dialogue is found by regex, action/thought are none (default)
        
        # print(f"[BaseModule DEBUG _parse_response] Final parsed result (after fallbacks): {result}")
        return result
