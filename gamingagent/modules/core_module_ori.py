import json
import os
import datetime
from abc import ABC, abstractmethod
from tools.serving import APIManager
from dataclasses import dataclass
from typing import Optional, Any

from collections import deque

import string
########################################################################################
#TODO: Add grid_size to observation for perception module to draw the grid on the image#
########################################################################################
@dataclass
class GameTrajectory:
    def __init__(self, max_length: int = 10):
        self.history_length = max_length
        self.trajectory = deque(maxlen=max_length)

    def add(self, entry: str):
        self.trajectory.append(entry)

    def get(self) -> Optional[str]:
        if not self.trajectory:
            return None
        return f"Past {self.history_length} turn(s) game trajectory (each turn an unique hash)\n" + "\n".join(self.trajectory)


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

    def __init__(
        self,
        img_path: Optional[str] = None,
        game_trajectory: Optional[GameTrajectory] = None,
        reflection: Optional[str] = None,
        processed_visual_description: Optional[str] = None,
        textual_representation: Optional[str] = None
    ):
        """
        Initialize an Observation instance.
        """
        self.game_trajectory = game_trajectory or GameTrajectory(max_length=10)
        self.img_path = img_path
        self.reflection = reflection
        self.processed_visual_description = processed_visual_description
        self.textual_representation = textual_representation
    
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
        # These apply regardless of whether 'observation' (object) was passed, allowing specific overrides.
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
            attr = getattr(self, name, None)
            if name == "game_trajectory":
                harness_content_map[name] = attr.get() if attr else "N/A"
            else:
                harness_content_map[name] = attr if attr is not None else "N/A"
        
        # Determine allowed variables
        # TODO: make the code segment debug-use only
        allowed_vars = set()
        if observation_mode in ["text", "both"]:
            allowed_vars |= self.BASE_ATTR
        if use_perception_module:
            allowed_vars |= self.PERCEPTION_ATTR
        if use_memory_module:
            allowed_vars |= self.MEMORY_ATTR

        print("allowed variables:")
        print(allowed_vars)

        return prompt_template.format(**harness_content_map)

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
            "textual_representation": self.textual_representation
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
                reasoning_effort="medium"
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
        self.module_name = module_name
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.prompt = prompt
        self.cache_dir = cache_dir
        self.token_limit = token_limit
        self.reasoning_effort = reasoning_effort
        
        # Initialize API manager
        self.api_manager = APIManager(game_name=module_name.replace("_module", ""))
        
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
