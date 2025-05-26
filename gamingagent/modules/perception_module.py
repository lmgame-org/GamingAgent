import numpy as np
import os
import json
import datetime
from abc import ABC, abstractmethod
from PIL import Image
from .core_module import CoreModule, Observation
import re
from typing import Optional, Dict, Any

import copy

from tools.utils import scale_image_up

class PerceptionModule(CoreModule):
    """
    Perception module that analyzes game state to extract relevant features.
    
    Game-specific implementations should inherit from this class and implement
    the required abstract methods.
    """
    
    def __init__(self, 
                game_name: Optional[str] = None,
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
            game_name (str, optional): Name of the game, for game-specific logic.
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
        self.game_name = game_name
        
        # Initialize observation
        self.observation = observation if observation is not None else Observation()
        self.processed_observation = copy.deepcopy(observation) if observation is not None else Observation()
        
        # Create observations directory for storing game state images
        self.obs_dir = os.path.join(cache_dir, "observations")
        os.makedirs(self.obs_dir, exist_ok=True)
        
    def _extract_dialogue_from_description(self, description_text: str) -> Optional[Dict[str, str]]:
        """
        Extracts structured dialogue (speaker: text) from a given text.
        Tries to find patterns like "Dialog: Speaker: Text content".
        """
        if not description_text:
            return None
        dialogue_pattern = r'(?:#\s*)?dialog(?:ue)?[:-]\s*([^:]+?):\s*(.+?)(?=(?:$|\n\s*\n|\n[A-Z0-9_][a-zA-Z0-9_ ]*:))'
        
        dialogue_match = re.search(dialogue_pattern, description_text, re.DOTALL | re.IGNORECASE)
        
        if dialogue_match:
            speaker = dialogue_match.group(1).strip()
            text = dialogue_match.group(2).strip()
            # print(f"[PerceptionModule DEBUG] Extracted dialogue: Speaker='{speaker}', Text='{text[:50]}...'")
            return {"speaker": speaker, "text": text}
        
        # print(f"[PerceptionModule DEBUG] No dialogue pattern matched in description: '{description_text[:100]}...'")
        return None

    def _parse_detailed_perception(self, llm_output: str) -> Dict[str, Any]:
        """
        Parses the detailed perception output from the LLM based on the
        Ace Attorney perception prompt format.
        """
        if not llm_output:
            return {}

        # Initialize default values
        parsed_data = {
            "game_state_from_perception": None,
            "parsed_dialogue": None, # Will be {"speaker": ..., "text": ...}
            "dialogue_continuation_from_perception": None,
            "options_from_perception": None,
            "selected_evidence_from_perception": None,
            "scene_description_from_perception": None
        }

        # Regex patterns based on the specified output format
        # Using re.IGNORECASE | re.DOTALL for flexibility
        patterns = {
            "game_state_from_perception": r"Game State:\\s*(.+?)(?:\\nDialog:|\\nDialogue Continuation:|\\nOptions:|\\nEvidence:|\\nScene:|$)",
            "dialogue_full": r"Dialog:\\s*([^:]+?):\\s*(.+?)(?:\\nDialogue Continuation:|\\nOptions:|\\nEvidence:|\\nScene:|$)",
            "dialogue_continuation_from_perception": r"Dialogue Continuation:\\s*(.+?)(?:\\nOptions:|\\nEvidence:|\\nScene:|$)",
            "options_from_perception": r"Options:\\s*(.+?)(?:\\nEvidence:|\\nScene:|$)",
            "selected_evidence_from_perception": r"Evidence:\\s*(.+?)(?:\\nScene:|$)",
            "scene_description_from_perception": r"Scene:\\s*(.+?$)"
        }

        # Extract Game State
        match = re.search(patterns["game_state_from_perception"], llm_output, re.IGNORECASE | re.DOTALL)
        if match:
            parsed_data["game_state_from_perception"] = match.group(1).strip()

        # Extract Dialogue (speaker and text)
        match = re.search(patterns["dialogue_full"], llm_output, re.IGNORECASE | re.DOTALL)
        if match:
            speaker = match.group(1).strip()
            text = match.group(2).strip()
            if speaker.lower() != "none" and text.lower() != "none":
                 parsed_data["parsed_dialogue"] = {"speaker": speaker, "text": text}

        # Extract Dialogue Continuation
        match = re.search(patterns["dialogue_continuation_from_perception"], llm_output, re.IGNORECASE | re.DOTALL)
        if match:
            parsed_data["dialogue_continuation_from_perception"] = match.group(1).strip()
            if parsed_data["dialogue_continuation_from_perception"].lower() == "none":
                 parsed_data["dialogue_continuation_from_perception"] = None


        # Extract Options
        match = re.search(patterns["options_from_perception"], llm_output, re.IGNORECASE | re.DOTALL)
        if match:
            options_text = match.group(1).strip()
            parsed_data["options_from_perception"] = options_text if options_text.lower() != "none" else None
            
        # Extract Evidence
        match = re.search(patterns["selected_evidence_from_perception"], llm_output, re.IGNORECASE | re.DOTALL)
        if match:
            evidence_text = match.group(1).strip()
            parsed_data["selected_evidence_from_perception"] = evidence_text if evidence_text.lower() != "none" else None

        # Extract Scene Description
        match = re.search(patterns["scene_description_from_perception"], llm_output, re.IGNORECASE | re.DOTALL)
        if match:
            parsed_data["scene_description_from_perception"] = match.group(1).strip()
        
        # print(f"[PerceptionModule DEBUG _parse_detailed_perception] Parsed data: {json.dumps(parsed_data, indent=2)}")
        return parsed_data

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

            processed_visual_description_text = self.api_manager.vision_text_completion(
                model_name=self.model_name,
                system_prompt=self.system_prompt,
                prompt=self.prompt,
                image_path=new_img_path,
                thinking=True,
                reasoning_effort=self.reasoning_effort,
                token_limit=self.token_limit
            )

            self.processed_observation.processed_visual_description = processed_visual_description_text
            self.processed_observation.image_path = new_img_path

            # Parse the detailed perception output
            # For now, we assume this module instance is for Ace Attorney if this prompt is used.
            # A more robust way would be to pass game_name or have game-specific perception modules.
            if self.processed_observation.processed_visual_description:
                if self.game_name == "ace_attorney":
                    detailed_data = self._parse_detailed_perception(self.processed_observation.processed_visual_description)
                    
                    self.processed_observation.game_state_from_perception = detailed_data.get("game_state_from_perception")
                    self.processed_observation.parsed_dialogue = detailed_data.get("parsed_dialogue")
                    self.processed_observation.dialogue_continuation_from_perception = detailed_data.get("dialogue_continuation_from_perception")
                    self.processed_observation.options_from_perception = detailed_data.get("options_from_perception")
                    self.processed_observation.selected_evidence_from_perception = detailed_data.get("selected_evidence_from_perception")
                    self.processed_observation.scene_description_from_perception = detailed_data.get("scene_description_from_perception")
                else:
                    # For non-Ace Attorney games, ensure these specific fields are None
                    # And attempt generic dialogue extraction if needed (or remove if not)
                    # For now, removing generic dialogue extraction from here, assuming BaseModule or other
                    # game-specific modules handle it if necessary for other games.
                    self.processed_observation.game_state_from_perception = None
                    self.processed_observation.parsed_dialogue = None
                    self.processed_observation.dialogue_continuation_from_perception = None
                    self.processed_observation.options_from_perception = None
                    self.processed_observation.selected_evidence_from_perception = None
                    self.processed_observation.scene_description_from_perception = None
            else:
                # Ensure fields are None if no description text was processed
                self.processed_observation.game_state_from_perception = None
                
            return self.processed_observation
        else:
            raise NotImplementedError(f"observation mode: {self.observation_mode} not supported.")
    
    def get_perception_summary(self):
        """
        Get a summary of the current perception.
        Includes detailed parsed fields if available.
        """
        # Base result
        result = {
            "img_path": self.processed_observation.img_path,
            "textual_representation": self.processed_observation.get_textual_representation(),
            "processed_visual_description": self.processed_observation.processed_visual_description, # Raw LLM output
             # Fields parsed by _parse_detailed_perception
            "game_state_from_perception": getattr(self.processed_observation, 'game_state_from_perception', None),
            "parsed_dialogue": getattr(self.processed_observation, 'parsed_dialogue', None),
            "dialogue_continuation_from_perception": getattr(self.processed_observation, 'dialogue_continuation_from_perception', None),
            "options_from_perception": getattr(self.processed_observation, 'options_from_perception', None),
            "selected_evidence_from_perception": getattr(self.processed_observation, 'selected_evidence_from_perception', None),
            "scene_description_from_perception": getattr(self.processed_observation, 'scene_description_from_perception', None)
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
