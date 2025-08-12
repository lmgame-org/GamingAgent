import json
import os
import datetime
from abc import ABC, abstractmethod
from tools.serving import APIManager
from dataclasses import dataclass
from typing import Optional, Dict, Any

from collections import deque

import string
########################################################################################
#TODO: Add grid_size to observation for perception module to draw the grid on the image#
########################################################################################
@dataclass
class GameTrajectory:
    def __init__(self, max_length: int = 10, need_background: bool = False, background_prefix_str: Optional[str] = "Game Background:"):
        self.max_length = max_length
        self.trajectory = deque(maxlen=max_length)
        self.need_background = need_background
        self.background: Optional[str] = None
        self.background_prefix_string: Optional[str] = background_prefix_str

    def add(self, entry: str):
        self.trajectory.append(entry)
    
    def set_background(self, background_content: str):
        """Sets the background content for the trajectory, intended to be called once."""
        if self.background is None: # Only set if not already set
            self.background = background_content

    def get(self) -> Optional[str]:
        if not self.trajectory:
            history_text_repr = ""
        else:
            history_text_repr = f"Past {self.max_length} turn(s) game trajectory (each turn an unique hash)\n" + "\n".join(self.trajectory)
        
        if self.need_background and self.background is not None:
            if history_text_repr: # If there is history, add background before it
                return f"{self.background_prefix_string}\n{self.background}\n\n{history_text_repr}"
            else: # If no history, just return the background
                return f"{self.background_prefix_string}\n{self.background}"
        elif not self.trajectory: # No background needed and no trajectory
             return None
        else: # Background not needed or not set, but trajectory exists
            return history_text_repr

@dataclass
class Observation:
    """
    Dataclass representing a game observation.
    Can contain multiple types of observations:
    - img_path: Path to the image file for visual observations.
    - game_trajectory: Memory module — past N turns in the game trajectory, each turn contains (state, action, reward).
    - reflection: Memory module — Textual reflection from the game trajectory.
    - textual_representation: Perception module — Textual representation of the game state (read from game)
    - processed_visual_description: Perception module — Textual description of the image (extracted and processed from image)
    """

    BASE_ATTR = {
        "textual_representation",
    }

    PERCEPTION_ATTR = {
        "processed_visual_description",
    }

    MEMORY_ATTR = {
        "game_trajectory",
        "reflection",
    }

    EPISODICAL_ATTR = {
        "background",
    }

    def __init__(
        self,
        img_path: Optional[str] = None,
        game_trajectory: Optional[GameTrajectory] = None,
        reflection: Optional[str] = None,
        processed_visual_description: Optional[str] = None,
        textual_representation: Optional[str] = None,
        background: Optional[str] = None,
        trajectory_includes_background: Optional[bool] = True,
        max_memory: Optional[int] = 10,
    ):
        """
        Initialize an Observation instance.
        """
        self.max_memory = max_memory
        self.game_trajectory = game_trajectory or GameTrajectory(max_length=self.max_memory, need_background=trajectory_includes_background)
        self.img_path = img_path
        self.reflection = reflection
        self.processed_visual_description = processed_visual_description
        self.textual_representation = textual_representation
        self.background = background
        self.trajectory_includes_background = trajectory_includes_background
    
    def set_perception_observation(self, observation=None, img_path=None, textual_representation=None, processed_visual_description=None):
        """
        Set the current observation from raw game states.
        
        Args:
            observation (Observation, optional): An Observation instance. If provided, its attributes are copied.
            img_path (str, optional): Overrides or sets img_path. For "vision" or "both" modes.
            textual_representation (str, optional): Overrides or sets textual_representation. For "text" or "both" modes.
            processed_visual_description (str, optional): Overrides or sets processed_visual_description. For "text" or "both" modes.
        """
        # If an Observation object is directly provided, copy its relevant attributes to self
        if observation is not None:
            if hasattr(observation, 'img_path') and observation.img_path is not None:
                self.img_path = observation.img_path
            if hasattr(observation, 'textual_representation') and observation.textual_representation is not None:
                self.textual_representation = observation.textual_representation
            if hasattr(observation, 'processed_visual_description') and observation.processed_visual_description is not None:
                self.processed_visual_description = observation.processed_visual_description
            # If the passed 'observation' object also carries memory attributes, copy them too.
            if hasattr(observation, 'game_trajectory') and observation.game_trajectory is not None:
                 self.game_trajectory = observation.game_trajectory
            if hasattr(observation, 'reflection') and observation.reflection is not None:
                 self.reflection = observation.reflection

        # Update/override with individual arguments if they are provided.
        if img_path is not None:
            self.img_path = img_path
                
        if textual_representation is not None:
            self.textual_representation = textual_representation
        
        if processed_visual_description is not None:
            self.processed_visual_description = processed_visual_description
    
    def set_memory_observation(self, observation=None, game_trajectory=None, reflection=None):
        """
        Set the current memory context.
        
        Args:
            observation (Observation, optional): A complete Observation instance. If provided, its attributes are copied.
            game_trajectory (GameTrajectory, optional): past N game states.
            reflection (str, optional): latest reflection synthesized from memory module.
        """
        # If an Observation object is directly provided, copy its relevant attributes to self
        if observation is not None:
            if hasattr(observation, 'game_trajectory') and observation.game_trajectory is not None:
                self.game_trajectory = observation.game_trajectory
            if hasattr(observation, 'reflection') and observation.reflection is not None:
                self.reflection = observation.reflection
            # If the passed 'observation' object also carries perception attributes, copy them too.
            if hasattr(observation, 'img_path') and observation.img_path is not None:
                self.img_path = observation.img_path
            if hasattr(observation, 'textual_representation') and observation.textual_representation is not None:
                self.textual_representation = observation.textual_representation
            if hasattr(observation, 'processed_visual_description') and observation.processed_visual_description is not None:
                self.processed_visual_description = observation.processed_visual_description

        # Update/override with individual arguments if they are provided
        if game_trajectory is not None:
            self.game_trajectory = game_trajectory
                
        if reflection is not None:
            self.reflection = reflection
    
    def get_img_path(self) -> str:
        """
        Get the image path as a string.
        
        Returns:
            str: The image path or empty string if None. None is only used, when no visual observations used.
        """
        return self.img_path if self.img_path is not None else ""

    def get_game_trajectory(self) -> str:
        return self.game_trajectory.get()

    def get_reflection(self) -> str:
        return self.reflection if self.reflection is not None else ""
    
    def get_processed_visual_description(self) -> str:
        """
        Get the description of visual lements in the game state, processed from the game state image (as a string).
        
        Returns:
            str: The visual description or empty string if None
        """
        return self.processed_visual_description if self.processed_visual_description is not None else ""
    
    def get_textual_representation(self) -> str:
        """
        Get the textual representation of the game state (as a string).
        
        Returns:
            str: The textual representation or empty string if None
        """
        return self.textual_representation if self.textual_representation is not None else ""

    def get_background(self) -> str:
        """
        Get the static background information for the episode (as a string).
        
        Returns:
            str: The background information or empty string if None
        """
        return self.background if self.background is not None else ""
    
    def get_complete_prompt(
        self,
        observation_mode,
        prompt_template,
        use_memory_module: bool = False,
        use_perception_module: bool = False,
    ) -> str:
        """
        Always allowed  → BASE_ATTR  
        +Perception     → PERCEPTION_ATTR (if ``use_perception_module``)  
        +Memory         → MEMORY_ATTR (if ``use_memory_module``)

        Any variable referenced in the template NOT in the allowed‑set raises a ValueError.
        Any variable used in the template is not found in harness, insert "N/A".
        """
        formatter = string.Formatter()
        var_names = [fld for _, fld, _, _ in formatter.parse(prompt_template) if fld]
        assert var_names, "Expected at least one variable in prompt_template."

        # Collect values for referenced attributes (initialize with "N/A")
        harness_content_map = {name: "N/A" for name in var_names}
        # Fill in existing values
        for name in var_names:
            if name == "game_trajectory":
                gt_instance = getattr(self, name, None)
                harness_content_map[name] = gt_instance.get() if gt_instance else "N/A"
            elif name == "background":
                # If 'background' is explicitly requested by the template,
                # provide it only if trajectory_includes_background is true and background has content.
                if self.trajectory_includes_background and self.background is not None:
                    harness_content_map[name] = self.background
                else:
                    harness_content_map[name] = "N/A" # Explicitly N/A if not applicable or not set
            else:
                # For other attributes like textual_representation, reflection, processed_visual_description
                attr_val = getattr(self, name, None)
                harness_content_map[name] = attr_val if attr_val is not None else "N/A"
        
        # Determine allowed variables
        # TODO: make the code segment debug-use only
        allowed_vars = set()
        # textual_representation is always a possibility
        if "textual_representation" in self.BASE_ATTR: # Check if it's defined in BASE_ATTR
            allowed_vars.add("textual_representation")

        if self.trajectory_includes_background: # The flag on Observation determines if background is "allowed"
            allowed_vars |= self.EPISODICAL_ATTR 

        if use_perception_module:
            allowed_vars |= self.PERCEPTION_ATTR
        if use_memory_module:
            allowed_vars |= self.MEMORY_ATTR

        # print("allowed variables:")
        # print(allowed_vars)

        return prompt_template.format(**harness_content_map)

    def get_memory_summary(self) -> dict[str, str]:
        """
        Provide the reasoning module with:
          • up‑to‑N past lines (already formatted by GameTrajectory)
          • no extra metadata dance
        """
        past = self.game_trajectory.get() or "No previous game states available."
        latest = self.game_trajectory.trajectory[-1] if self.game_trajectory.trajectory else "N/A"

        result = {
            "game_trajectory": past,
            "current_state": latest,   # includes (obs, action, thought)
            "reflection": latest.split("Reflection:", 1)[-1].strip()
                         if "Reflection:" in latest else "N/A",
        }
        return result
    
    def get_perception_summary(self):
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
            "img_path": self.img_path,
            "textual_representation": self.get_textual_representation(),
            "processed_visual_description": self.processed_visual_description
        }
        return result
    
    def to_json_string(self) -> str:
        """
        Get a JSON string representation of the observation data.

        Returns:
            str: A JSON string containing all observation attributes.
        """
        data = {
            "img_path": self.img_path,
            "game_trajectory": self.game_trajectory.get() if self.game_trajectory else None,
            "reflection": self.reflection,
            "processed_visual_description": self.processed_visual_description,
            "textual_representation": self.textual_representation,
            "background": self.background if self.trajectory_includes_background else None,
        }
        return json.dumps(data)

    def __str__(self) -> str:
        """
        Return the JSON string representation of the observation when str() is called or when printed.
        """
        return self.to_json_string()


class CoreModule(ABC):
    """
    Core module that serves as the foundation for all other modules.
    Provides common functionality for API calls, logging, and response parsing.
    """
    
    def __init__(self, 
                module_name, 
                model_name="claude-3-7-sonnet-latest", 
                system_prompt="", 
                prompt="", 
                cache_dir="cache",
                token_limit=100000, 
                reasoning_effort="high",
                vllm_url=None,
                modal_url=None
        ):
        """
        Initialize the core module with basic parameters.
        
        Args:
            module_name (str): Name of the module.
            model_name (str): The name of the model to use for inference.
            system_prompt (str): Default system prompt for LLM calls.
            prompt (str): Default user prompt for LLM calls.
            cache_dir (str): Directory for storing logs and cache files.
            token_limit (int): Maximum number of tokens for API calls.
            reasoning_effort (str): Reasoning effort for API calls (low, medium, high).
        """

        print(f"core module token limit: {token_limit}")
        self.module_name = module_name
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.prompt = prompt
        self.cache_dir = cache_dir
        self.token_limit = token_limit
        self.reasoning_effort = reasoning_effort
        
        # Initialize API manager
        self.api_manager = APIManager(
            game_name=module_name.replace("_module", ""), 
            vllm_url=vllm_url,
            modal_url=modal_url
        )
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize logger file path
        self.module_file = os.path.join(cache_dir, f"{module_name}.json")
        
    def log(self, data):
        """
        Log module data to the module file.
        
        Args:
            data (dict): Data to be logged.
        """
        try:
            # Add timestamp to log entry
            log_entry = {
                "datetime": datetime.datetime.now().isoformat(),
                **data
            }
            
            # Create or append to log file
            existing_logs = []
            if os.path.exists(self.module_file):
                try:
                    with open(self.module_file, 'r') as f:
                        existing_logs = json.load(f)
                except json.JSONDecodeError:
                    existing_logs = []
            
            # Ensure existing_logs is a list
            if not isinstance(existing_logs, list):
                existing_logs = []
            
            existing_logs.append(log_entry)
            
            # Write updated logs back to file
            with open(self.module_file, 'w') as f:
                json.dump(existing_logs, f, indent=2)
                
        except Exception as e:
            print(f"Error logging to {self.module_file}: {e}")
    
    @abstractmethod
    def _parse_response(self, response):
        """
        Parse LLM response to extract structured information.
        
        Args:
            response (str): The raw response from the LLM
            
        Returns:
            dict: Structured information extracted from the response
        """
        pass
