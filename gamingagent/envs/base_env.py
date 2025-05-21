import gymnasium as gym
import yaml
import os
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Any, Optional, Dict, List

from gamingagent.modules.core_module import Observation # Assuming Observation is in core_module

class BaseGameEnv(ABC):
    """
    Abstract Base Class for game environment wrappers.
    Manages a Gymnasium or Retro environment and provides a standardized interface.
    """
    def __init__(self, 
                 game_name: str, 
                 observation_mode: str,
                 agent_obs_path_creator: callable, # e.g., lambda ep, step: os.path.join(base_dir, f"obs_e{ep}_s{step}.png")
                 env_type: str = "custom",
                 config_root_dir: str = "configs"):
        """
        Initializes the base game environment wrapper.

        Args:
            game_name (str): Name of the game, corresponding to a config folder.
            observation_mode (str): Agent's expected observation mode ("vision", "text", "both").
            agent_obs_path_creator (callable): A function that takes (episode_id, step_num) 
                                             and returns a unique path for saving vision observations.
            env_type (str): Type of environment ("gymnasium" or "retro").
            config_root_dir (str): Root directory for game configurations.
        """
        self.game_name = game_name
        self.observation_mode = observation_mode
        self.env_type = env_type.lower()
        self.config_root_dir = config_root_dir
        self.config_path = os.path.join(self.config_root_dir, self.game_name, "config.yaml")
        self.agent_obs_path_creator = agent_obs_path_creator

        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found for game '{self.game_name}' at {self.config_path}")

        self._load_config()

        self.env: Optional[gym.Env] = None
        self.current_raw_observation: Any = None
        self.current_info: Dict[str, Any] = {}
        
        # These should be updated by the runner or within the episode loop
        self.current_episode_id = 0
        self.current_step_num = 0

    def _load_config(self):
        """Loads game-specific configuration from the YAML file."""
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.env_id = self.config.get("env_id", None) # Allow env_id to be None initially
        # The check for env_id will be done by the concrete class before gym.make
        # if not self.env_id:
        #     raise ValueError(f"\'env_id\' not specified in config {self.config_path}")
        
        self.env_init_kwargs = self.config.get("env_init_kwargs", {})
        self.action_mapping_config = self.config.get("action_mapping", {})
        self.move_to_action_idx = {str(k).lower(): v for k, v in self.action_mapping_config.items()}
        self.action_idx_to_move = {v: str(k).lower() for k, v in self.action_mapping_config.items()}

    @abstractmethod
    def _initialize_env(self) -> None:
        """Initializes the specific game environment (gym.make or retro.make)."""
        pass

    @abstractmethod
    def get_board_state(self, raw_observation: Any, info: Dict[str, Any]) -> Any:
        """
        Extracts or derives the canonical board state from raw observation and info.
        This state is used for image creation or text representation.
        """
        pass

    @abstractmethod
    def extract_observation(self, raw_observation: Any, info: Dict[str, Any]) -> Observation:
        """
        Processes raw environment output into an Observation object for the agent.
        This might involve creating and saving images for vision mode.
        """
        pass

    def reset(self, seed: Optional[int] = None, episode_id: int = 1) -> Tuple[Observation, Dict[str, Any]]:
        """Resets the environment and returns the initial observation for the agent."""
        if not self.env:
            self._initialize_env()
            if not self.env: # Double check after initialization attempt
                 raise ConnectionError("Environment could not be initialized.")

        self.current_raw_observation, self.current_info = self.env.reset(seed=seed)
        self.current_episode_id = episode_id
        self.current_step_num = 0
        
        agent_observation = self.extract_observation(self.current_raw_observation, self.current_info)
        return agent_observation, self.current_info

    def step(self, agent_action_str: Optional[str]) -> Tuple[Observation, float, bool, bool, Dict[str, Any], int]:
        """
        Takes a step in the environment using the agent's string action.
        Handles invalid actions by sampling a random action.

        Returns:
            Tuple: (agent_observation, reward, terminated, truncated, info, executed_env_action_idx)
        """
        if not self.env:
            raise ConnectionError("Environment not initialized. Call reset() first.")

        self.current_step_num += 1
        env_action_idx: Optional[int] = None
        
        is_skip_action = agent_action_str is None or not agent_action_str.strip() or agent_action_str.lower() == "skip"

        if is_skip_action:
            # If agent explicitly wants to skip or provides no valid action string,
            # we need to decide how the environment handles a "no-op".
            # For many envs, there isn't a true no-op. Taking a random action might be disruptive.
            # Best approach: If an env has a defined NOOP action, use that. Otherwise, this is tricky.
            # For now, if agent says "skip", we will not step the env, and return current state.
            # This assumes "skip" means the agent doesn't want to change the env state.
            # This behavior might need to be game-specific.
            # Let's assume for now that a "skip" means we take no action and return current obs.
            # However, the problem states to take a random action if invalid.
            # Let's refine: if action_str is None/Empty -> random. if "skip" -> also random for now as per general instruction.
            # This can be refined later if a true "skip" without env interaction is needed.
            print(f"Agent action '{agent_action_str}' implies skip or is invalid. Taking random action.")
            env_action_idx = self.env.action_space.sample()
        else:
            env_action_idx = self.move_to_action_idx.get(agent_action_str.lower())
            if env_action_idx is None:
                print(f"Agent action '{agent_action_str}' not in mapping. Taking random action.")
                env_action_idx = self.env.action_space.sample()
        
        self.current_raw_observation, reward, terminated, truncated, self.current_info = self.env.step(env_action_idx)
        agent_observation = self.extract_observation(self.current_raw_observation, self.current_info)
        
        return agent_observation, float(reward), terminated, truncated, self.current_info, env_action_idx

    def render_human(self) -> None:
        """Renders the environment for human viewing if supported."""
        if self.env and hasattr(self.env, 'render') and self.env.render_mode == "human":
            self.env.render()

    def close(self) -> None:
        """Closes the environment."""
        if self.env:
            self.env.close()
            self.env = None
            print(f"Environment {self.game_name} closed.")

    def map_env_action_to_agent_action(self, env_action_idx: int) -> str:
        """Maps an environment's integer action index to an agent's string action."""
        return self.action_idx_to_move.get(env_action_idx, f"unknown_idx_{env_action_idx}")

    @property
    def action_space(self) -> Optional[gym.Space]:
        return self.env.action_space if self.env else None

    @property
    def observation_space(self) -> Optional[gym.Space]:
        # This refers to the raw observation space of the underlying environment.
        return self.env.observation_space if self.env else None

    def get_current_episode_step_num(self) -> Tuple[int, int]:
        return self.current_episode_id, self.current_step_num
