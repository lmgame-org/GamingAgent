import os
import json
import datetime
from abc import ABC, abstractmethod
from PIL import Image
import numpy as np
import importlib
import inspect

# Import modules
from gamingagent.modules import BaseModule, PerceptionModule, MemoryModule, ReasoningModule, Observation

class BaseAgent(ABC):
    """
    Base agent class that provides the foundation for game-specific agents.
    Implements common functionality like module management, caching, and workflow.
    
    Game-specific agents should inherit from this class and implement
    the required abstract methods.
    """
    
    def __init__(self, game_name, model_name, config_path=None, harness=True, 
                 max_memory=10, cache_dir=None, custom_modules=None, observation_mode="vision"):
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
        """
        self.game_name = game_name
        self.model_name = model_name
        self.harness = harness
        self.max_memory = max_memory
        self.observation_mode = observation_mode
        
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
        if custom_modules and "base_module" in custom_modules:
            # Use custom base module class
            base_cls = custom_modules["base_module"]
            modules["base_module"] = base_cls(
                model_name=self.model_name,
                cache_dir=self.cache_dir,
                system_prompt=self.config["base_module"]["system_prompt"],
                prompt=self.config["base_module"]["prompt"],
                observation_mode=self.observation_mode,
                token_limit=100000,
                reasoning_effort="high"
            )
        else:
            # Use default BaseModule
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
            # Convert to Observation based on observation type and mode
            if isinstance(observation, str) and os.path.exists(observation):
                # It's an image path
                observation = Observation(img_path=observation)
            elif isinstance(observation, np.ndarray):
                # It's a numpy array, save it as an image
                img_path = self.save_obs(observation)
                if img_path:
                    observation = Observation(img_path=img_path)
                else:
                    return {"action": None, "thought": "Failed to process image observation"}
            else:
                # It's likely symbolic data based on observation_mode
                if self.observation_mode == "vision":
                    # Unexpected input for vision mode, but try our best
                    observation = Observation(symbolic_representation=str(observation))
                elif self.observation_mode == "text":
                    # Text mode expects symbolic representation
                    observation = Observation(symbolic_representation=str(observation))
                elif self.observation_mode == "both":
                    # Both mode, try to interpret as symbolic if not an image
                    observation = Observation(symbolic_representation=str(observation))
        
        if not self.harness:
            # Unharness mode: Use base module directly with the Observation object
            result = self.modules["base_module"].process_observation(observation=observation)
            return result
        else:
            # Harness mode: Perception -> Memory -> Reasoning
            perception_module = self.modules.get("perception_module")
            memory_module = self.modules.get("memory_module")
            reasoning_module = self.modules.get("reasoning_module")
            
            if perception_module is None:
                raise ValueError("Perception module is required for harness mode")
            if reasoning_module is None:
                raise ValueError("Reasoning module is required for harness mode")
            
            # 1. Process observation with perception module (already an Observation)
            processed_obs = perception_module.process_observation(observation)
            perception_data = perception_module.get_perception_summary()
            
            # 2. Update memory with perception data
            memory_summary = None
            if memory_module:
                memory_module.update_memory(perception_data)
                memory_summary = memory_module.get_memory_summary()
            
            # 3. Plan action with reasoning module
            action_plan = reasoning_module.plan_action(
                perception_data=perception_data,
                memory_summary=memory_summary
            )
            
            return action_plan
