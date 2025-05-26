import os
import json
import datetime
from abc import ABC, abstractmethod
from PIL import Image
import numpy as np
import importlib
import inspect
from typing import Optional, Any

# Import modules
from gamingagent.modules import BaseModule, PerceptionModule, MemoryModule, ReasoningModule, Observation

class BaseAgent(ABC):
    """
    Base agent class that provides the foundation for game-specific agents.
    Implements common functionality like module management, caching, and workflow.
    
    Game-specific agents should inherit from this class and implement
    the required abstract methods.
    """
    
    def __init__(self,
            game_name,
            model_name,
            config_path=None,
            harness=True,
            max_memory=10,
            cache_dir=None,
            custom_modules=None, 
            observation_mode="vision",    # change the abstraction to with or without image
            env: Optional[Any] = None # ADDED: Environment instance
        ):
        """
        Initialize the agent with base parameters and modules.
        
        Args:
            game_name (str): Name of the game (used for organizing cache)
            model_name (str): Name of the model to use for inference
            config_path (str, optional): Path to config file with prompts
            harness (bool): If True, uses perception-memory-reasoning pipeline;
                           If False, uses base module only
            max_memory (int): Maximum number of memory entries to store
            cache_dir (str, optional): Custom cache directory path
            custom_modules (dict, optional): Custom module classes to use
            observation_mode (str): Mode for processing observations ("vision", "text", or "both")
            env (Optional[Any]): The game environment instance, used for specific interactions like dialogue handling.
        """
        self.game_name = game_name
        self.model_name = model_name
        self.harness = harness
        self.max_memory = max_memory
        self.observation_mode = observation_mode
        self.env = env # ADDED: Store environment instance
        
        # Set up cache directory following the specified pattern
        if cache_dir is None:
            # Use first 10 chars of model name
            model_name_short = model_name[:15].replace("-", "_")
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            cache_dir = os.path.join("cache", game_name, model_name_short, timestamp)
        
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Create observations directory
        self.observations_dir = os.path.join(self.cache_dir, "observations")
        os.makedirs(self.observations_dir, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize modules
        self.modules = self._initialize_modules(custom_modules)
        
        # Print diagnostic information about loaded modules
        print(f"Agent for '{self.game_name}' initialized with model '{self.model_name}'.")
        if self.harness:
            print("  Agent is in HARNESS mode (Perception-Memory-Reasoning pipeline).")
            
            # TODO (lanxiang): make expected modules to use configurable
            expected_modules = ["perception_module", "memory_module", "reasoning_module"]
            
            for module_name in expected_modules:
                if self.modules.get(module_name) and self.modules[module_name] is not None:
                    print(f"    -> Using {module_name.replace('_', ' ').title()}: {self.modules[module_name].__class__.__name__}")
                else:
                    print(f"    -> WARNING: {module_name.replace('_', ' ').title()} not loaded correctly for harness mode.")
        else:
            print("  Agent is in NON-HARNESS mode (BaseModule direct pipeline).")
            if self.modules.get("base_module") and self.modules["base_module"] is not None:
                 print(f"    -> Using Base Module: {self.modules['base_module'].__class__.__name__}")
            else:
                 print("    -> WARNING: Base Module not loaded correctly.")
        
        # Save agent configuration
        self._save_agent_config()
    
    def _load_config(self, config_path):
        """
        Load configuration from file or use defaults.
        
        Args:
            config_path (str): Path to config file
            
        Returns:
            dict: Configuration dictionary
        """
        config = {
            "base_module": {
                "system_prompt": "",
                "prompt": ""
            },
            "perception_module": {
                "system_prompt": "",
                "prompt": ""
            },
            "memory_module": {
                "system_prompt": "",
                "prompt": ""
            },
            "reasoning_module": {
                "system_prompt": "",
                "prompt": ""
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    file_config = json.load(f)
                    
                # Update config with file values
                for module in config:
                    if module in file_config:
                        config[module].update(file_config[module])
                        
                print(f"Loaded configuration from {config_path}")
            except Exception as e:
                print(f"Error loading config from {config_path}: {e}")
        
        return config
    
    def _initialize_modules(self, custom_modules=None):
        """
        Initialize the required modules based on agent configuration.
        
        Args:
            custom_modules (dict): Dictionary mapping module names to custom module classes
            
        Returns:
            dict: Dictionary of initialized modules
        """
        modules = {}
        
        # Always initialize base module
        # to support without-harness decision-making
        
        # TODO: make arguments configurable
        modules["base_module"] = BaseModule(
            model_name=self.model_name,
            cache_dir=self.cache_dir,
            system_prompt=self.config["base_module"]["system_prompt"],
            prompt=self.config["base_module"]["prompt"],
            observation_mode=self.observation_mode,
            token_limit=100000,
            reasoning_effort="high"
        )

        # Initialize perception, memory, and reasoning modules if using harness
        if self.harness:
            # Perception module
            if custom_modules and "perception_module" in custom_modules:
                perception_cls = custom_modules["perception_module"]
                modules["perception_module"] = perception_cls(
                    model_name=self.model_name,
                    observation_mode=self.observation_mode,
                    cache_dir=self.cache_dir,
                    system_prompt=self.config["perception_module"]["system_prompt"],
                    prompt=self.config["perception_module"]["prompt"]
                )
            else:
                # Can't use default PerceptionModule as it's abstract
                # Subclasses must provide this
                modules["perception_module"] = None
            
            # Memory module
            if custom_modules and "memory_module" in custom_modules:
                memory_cls = custom_modules["memory_module"]
                modules["memory_module"] = memory_cls(
                    model_name=self.model_name,
                    cache_dir=self.cache_dir,
                    system_prompt=self.config["memory_module"]["system_prompt"],
                    prompt=self.config["memory_module"]["prompt"],
                    max_memory=self.max_memory
                )
            else:
                modules["memory_module"] = MemoryModule(
                    model_name=self.model_name,
                    cache_dir=self.cache_dir,
                    system_prompt=self.config["memory_module"]["system_prompt"],
                    prompt=self.config["memory_module"]["prompt"],
                    max_memory=self.max_memory
                )
            
            # TODO (lanxiang): make reasoning efforts configurable
            # Reasoning module
            if custom_modules and "reasoning_module" in custom_modules:
                reasoning_cls = custom_modules["reasoning_module"]
                modules["reasoning_module"] = reasoning_cls(
                    model_name=self.model_name,
                    cache_dir=self.cache_dir,
                    system_prompt=self.config["reasoning_module"]["system_prompt"],
                    prompt=self.config["reasoning_module"]["prompt"]
                )
            else:
                # Can't use default ReasoningModule as it's abstract
                # Subclasses must provide this
                modules["reasoning_module"] = None
        
        return modules
    
    def _save_agent_config(self):
        """Save agent configuration for reference."""
        config_file = os.path.join(self.cache_dir, "agent_config.json")
        
        config_data = {
            "game_name": self.game_name,
            "model_name": self.model_name,
            "observation_mode": self.observation_mode,
            "harness": self.harness,
            "max_memory": self.max_memory,
            "cache_dir": self.cache_dir,
            "modules": {
                module: module_instance.__class__.__name__ 
                for module, module_instance in self.modules.items() 
                if module_instance is not None
            },
            "config": self.config
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
            
        print(f"Saved agent configuration to {config_file}")
    
    def save_obs(self, observation, filename=None):
        """
        Save observation as PNG image.
        
        Args:
            observation: Numpy array representing the observation
            filename: Optional custom filename (without extension)
            
        Returns:
            str: Path to the saved image
        """
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            counter = len([f for f in os.listdir(self.observations_dir) if f.endswith('.png')])
            filename = f"obs_img_{counter:04d}_{timestamp}"
        
        # Ensure filename doesn't have extension
        if filename.endswith('.png'):
            filename = filename[:-4]
            
        # Create full path
        filepath = os.path.join(self.observations_dir, f"{filename}.png")
        
        # Convert observation to image and save
        if isinstance(observation, np.ndarray):
            if len(observation.shape) == 3 and observation.shape[2] in [1, 3, 4]:
                # RGB/RGBA image
                img = Image.fromarray(observation)
                img.save(filepath)
                print(f"Saved observation image to {filepath}")
                return filepath
            else:
                # Non-image array
                print(f"Warning: Observation shape {observation.shape} doesn't match image format.")
                return None
        else:
            print(f"Warning: Observation is not a numpy array, cannot save as image.")
            return None
    
    def get_action(self, observation):
        """
        Get the next action based on the current observation.
        
        Args:
            observation: Current observation from the environment.
                If it's an Observation object, it will be used directly.
                Otherwise, it will be converted to an Observation object based on observation_mode.
            
        Returns:
            Action to take in the environment
        """
        # Ensure observation is an Observation object
        if not isinstance(observation, Observation):
            img_path_for_observation = None
            symbolic_representation_for_observation = None

            if isinstance(observation, str) and os.path.exists(observation):
                # It's an image path
                img_path_for_observation = observation
            elif isinstance(observation, np.ndarray):
                # It's a numpy array, save it as an image
                saved_img_path = self.save_obs(observation)
                if saved_img_path:
                    img_path_for_observation = saved_img_path
                elif self.observation_mode == "vision" or self.observation_mode == "both":
                    # Critical: In vision/both mode, if image saving fails, we cannot proceed without an image.
                    # Log this critical failure
                    print("CRITICAL: Failed to save numpy array as image in vision/both mode.")
                    # Return a failure/error or raise an exception
                    raise ValueError("Failed to process visual observation: Cannot save image.")
            else:
                # Input is not an image path and not a numpy array.
                # It's likely symbolic data.
                symbolic_representation_for_observation = str(observation)
            
            # Now, create the Observation object based on what we've gathered and the mode
            if self.observation_mode == "vision":
                if img_path_for_observation:
                    observation = Observation(img_path=img_path_for_observation)
                else:
                    # Critical: In vision mode, but no image path was derived from input.
                    print(f"CRITICAL: Vision mode selected, but input '{observation}' could not be resolved to an image.")
                    raise ValueError("Vision mode requires a valid image path or image data.")
            elif self.observation_mode == "text":
                observation = Observation(symbolic_representation=symbolic_representation_for_observation)
            elif self.observation_mode == "both":
                observation = Observation(img_path=img_path_for_observation, symbolic_representation=symbolic_representation_for_observation)
            else: # Should not happen if modes are validated
                raise ValueError(f"Unsupported observation_mode: {self.observation_mode}")
        
        if not self.harness:
            # Unharness mode: Use base module directly with the Observation object
            print("Invoking WITHOUT HARNESS mode.")

            base_mod = self.modules["base_module"]
            additional_prompt_context = None
            if self.game_name == "ace_attorney" and self.env and hasattr(self.env, "get_mapped_dialogue_event_for_prompt"):
                # print("[BaseAgent get_action] Ace Attorney non-harness mode: Getting mapped dialogue event.")
                dialogue_event_keyword = self.env.get_mapped_dialogue_event_for_prompt()
                if dialogue_event_keyword:
                    additional_prompt_context = f"Previous Dialogue Event Context: {dialogue_event_keyword}\n\n"
                    # print(f"[BaseAgent get_action] Using additional context: {additional_prompt_context}")
                # else:
                    # print("[BaseAgent get_action] No dialogue event keyword returned from env.")

            result = base_mod.plan_action(
                observation=observation,
                additional_prompt_context=additional_prompt_context # Pass to BaseModule
            )

            print(f"[BaseAgent DEBUG] Result received from base_mod.plan_action(): {result}")

            print(f"[BaseAgent DEBUG] Type of self.env: {type(self.env)}")
            print(f"[BaseAgent DEBUG] hasattr(self.env, 'store_llm_extracted_dialogue'): {hasattr(self.env, 'store_llm_extracted_dialogue')}")

            if self.game_name == "ace_attorney" and self.env and hasattr(self.env, "store_llm_extracted_dialogue"):
                print(f"[BaseAgent DEBUG] In Ace Attorney non-harness mode. Checking for dialogue to store.")
                if result and "parsed_dialogue" in result and result["parsed_dialogue"]:
                    print(f"[BaseAgent DEBUG] Parsed dialogue from BaseModule: {result['parsed_dialogue']}")
                    print(f"[BaseAgent DEBUG] Attempting to call store_llm_extracted_dialogue on env.")
                    self.env.store_llm_extracted_dialogue(result["parsed_dialogue"])
                else:
                    print(f"[BaseAgent DEBUG] No parsed dialogue from BaseModule to store. Result (inside 'if' check): {result}")
            return result
        
        else:
            # Harness mode: Perception -> Memory -> Reasoning
            print("Invoking WITH HARNESS mode.")

            perception_module = self.modules.get("perception_module")
            memory_module = self.modules.get("memory_module")
            reasoning_module = self.modules.get("reasoning_module")
            
            if perception_module is None:
                raise ValueError("Perception module is required for harness mode")
            if reasoning_module is None:
                raise ValueError("Reasoning module is required for harness mode")
            
            # 1. Process observation with perception module (already an Observation)
            processed_observation = perception_module.process_observation(observation)
            perception_data = perception_module.get_perception_summary()

            print("perception data:")
            print(perception_data)
            
            # 2. Update memory with perception data
            memory_summary = None
            if memory_module:
                processed_observation = memory_module.update_memory(processed_observation, perception_data)
                memory_summary = memory_module.get_memory_summary()
            
            print("memory data:")
            print(memory_summary)
            
            # 3. Plan action with reasoning module
            action_plan = reasoning_module.plan_action(
                observation=processed_observation,
                perception_data=perception_data,
                memory_summary=memory_summary
            )
            
            print("action plan:")
            print(action_plan)
            
            return action_plan
