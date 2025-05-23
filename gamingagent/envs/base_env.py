import gymnasium as gym
import yaml
import os
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Any, Optional, Dict, List, Callable
import json
import retro # Import retro here, as it might be used in _initialize_env

from gamingagent.modules.core_module import Observation
from tools.utils import convert_numpy_to_python

# Define a constant for skip action index, if preferred, though -1 is also clear.
SKIP_ACTION_IDX = -1 

class BaseGameEnv(ABC):
    """
    Abstract Base Class for game environment wrappers.
    Manages a Gymnasium or Retro environment and provides a standardized interface
    for agent interaction, observation processing, and logging.
    """
    def __init__(self, 
                 game_name: str, 
                 observation_mode: str,
                 agent_observations_base_dir: str,
                 env_type: str = "custom",
                 config_root_dir: str = "configs",
                 log_root_dir: str = "runs_output"):
        self.game_name = game_name
        self.observation_mode = observation_mode
        self.env_type = env_type.lower()
        self.config_root_dir = config_root_dir
        self.config_path = os.path.join(self.config_root_dir, self.game_name, "config.yaml")
        self.agent_observations_base_dir = agent_observations_base_dir
        os.makedirs(self.agent_observations_base_dir, exist_ok=True)
        
        self.log_root_dir = log_root_dir
        os.makedirs(self.log_root_dir, exist_ok=True)

        self.episode_log_file_path: Optional[str] = None
        self.episode_log_file_handle: Optional[Any] = None

        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"[BaseGameEnv] Configuration file not found for game '{self.game_name}' at {self.config_path}")

        self._load_config()

        self.env: Optional[gym.Env] = None
        self.current_raw_observation: Any = None
        self.current_info: Dict[str, Any] = {}
        
        self.current_episode_id = 0
        self.current_step_num = 0

    def _load_config(self):
        """Loads game-specific configuration from the YAML file."""
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.env_id = self.config.get("env_id", None)
        self.env_init_kwargs = self.config.get("env_init_kwargs", {})
        self.action_mapping_config = self.config.get("action_mapping", {})
        self.move_to_action_idx = {str(k).lower(): v for k, v in self.action_mapping_config.items()}
        self.action_idx_to_move = {v: str(k).lower() for k, v in self.action_mapping_config.items()}
        self.render_mode_for_make = self.config.get("render_mode_for_make", "human")

    def _initialize_env(self) -> None:
        """
        Initializes the game environment (e.g., using gym.make or retro.make).
        Uses `self.render_mode_for_make` for the render_mode during environment creation.
        Concrete environment wrappers can override `self.env_id` and `self.render_mode_for_make`
        in their `__init__` methods before this method is called (typically during `reset`).
        """
        if not self.env_id:
            raise ValueError(f"[BaseGameEnv] 'env_id' not specified. It must be set by the concrete class or in config.yaml for game: {self.game_name}")

        print(f"[BaseGameEnv] Initializing environment: game='{self.game_name}', env_id='{self.env_id}', type='{self.env_type}', render_mode='{self.render_mode_for_make}'")

        kwargs = self.env_init_kwargs.copy()
        if 'render_mode' in kwargs:
            print(f"[BaseGameEnv] Warning: 'render_mode' found in 'env_init_kwargs' ({kwargs['render_mode']}). It will be overridden by 'render_mode_for_make' ({self.render_mode_for_make}).")
            del kwargs['render_mode']

        if self.env_type == "custom":
            print(f"[BaseGameEnv] Using gym.make(id='{self.env_id}', render_mode='{self.render_mode_for_make}', **{kwargs})")
            self.env = gym.make(self.env_id, render_mode=self.render_mode_for_make, **kwargs)
        elif self.env_type == "retro":
            try:
                print(f"[BaseGameEnv] Using retro.make(game='{self.env_id}', render_mode='{self.render_mode_for_make}', **{kwargs})")
                self.env = retro.make(game=self.env_id, render_mode=self.render_mode_for_make, **kwargs) 
            except ImportError:
                print("[BaseGameEnv] ERROR: The 'retro' library is not installed. Please install it to use retro environments.")
                raise
            except Exception as e:
                print(f"[BaseGameEnv] ERROR: Failed to create retro environment '{self.env_id}' with retro.make. Error: {e}")
                raise
        else:
            raise ValueError(f"[BaseGameEnv] Unsupported env_type: '{self.env_type}'. Must be 'custom' or 'retro'.")

        if not self.env:
            raise ConnectionError(f"[BaseGameEnv] Failed to initialize environment '{self.env_id}' of type '{self.env_type}'.")
        
        print(f"[BaseGameEnv] Environment '{self.env_id}' initialized successfully.")

    @abstractmethod
    def extract_observation(self, raw_observation: Any, info: Dict[str, Any]) -> Observation:
        """
        Processes raw environment output into an Observation object for the agent.
        This might involve creating and saving images for vision mode, and/or
        extracting textual/symbolic information.
        This method is game-specific and must be implemented by concrete subclasses.
        """
        pass

    def reset(self, seed: Optional[int] = None, episode_id: int = 1) -> Tuple[Observation, Dict[str, Any]]:
        """Resets the environment and returns the initial observation for the agent."""
        if not self.env:
            self._initialize_env()
            if not self.env: # Double check after initialization attempt
                 raise ConnectionError("[BaseGameEnv] Environment could not be initialized.")

        self.current_raw_observation, self.current_info = self.env.reset(seed=seed)
        self.current_episode_id = episode_id
        self.current_step_num = 0 # Reset step number for the new episode

        # Close and reopen log file for the new episode
        if self.episode_log_file_handle is not None:
            try:
                self.episode_log_file_handle.close()
            except Exception as e:
                print(f"[BaseGameEnv] Warning: Error closing previous episode log file: {e}")
            self.episode_log_file_handle = None
        
        self.episode_log_file_path = os.path.join(self.log_root_dir, f"episode_{self.current_episode_id:03d}_log.jsonl")
        try:
            self.episode_log_file_handle = open(self.episode_log_file_path, 'a')
            print(f"[BaseGameEnv] Logging episode {self.current_episode_id} data to: {self.episode_log_file_path}")
        except Exception as e:
            print(f"[BaseGameEnv] ERROR: Could not open episode log file {self.episode_log_file_path}: {e}")
            self.episode_log_file_handle = None
        
        # Extract the initial observation for the agent
        agent_observation = self.extract_observation(self.current_raw_observation, self.current_info)
        return agent_observation, self.current_info

    def _log_and_print_step_data(self, agent_action_str: Optional[str], thought_process: str, reward: float, info: Dict[str, Any], terminated: bool, truncated: bool, time_taken_s: float, perf_score: float, agent_observation: Observation):
        """Helper function to log step data to file and print a summary to console."""
        # executed_action_str = self.map_env_action_to_agent_action(executed_env_action_idx) # Removed
        
        # Standardized console printout - ExecAct removed, added PerfScore
        print(f"[BaseGameEnv] E{self.current_episode_id} S{self.current_step_num}: AgentAct='{agent_action_str}', R={reward:.2f}, Perf={perf_score:.2f}, Term={terminated}, Trunc={truncated}, T={time_taken_s:.2f}s")

        log_entry = {
            "episode_id": self.current_episode_id,
            "step": self.current_step_num,
            "agent_action": agent_action_str,
            "thought": thought_process,
            "reward": float(reward),
            "perf_score": float(perf_score),
            "info": info, 
            "agent_observation": str(agent_observation),
            "terminated": terminated,
            "truncated": truncated,
            # "executed_env_action_idx": int(executed_env_action_idx), # Removed
            "time_taken_s": float(time_taken_s)
        }
        
        if self.episode_log_file_handle:
            try:
                serializable_log_entry = convert_numpy_to_python(log_entry)
                self.episode_log_file_handle.write(json.dumps(serializable_log_entry) + '\n')
                self.episode_log_file_handle.flush()
            except TypeError as te:
                print(f"[BaseGameEnv] CRITICAL ERROR (Log Serialization): Failed to serialize log_entry. Details: {te}")
                print(f"[BaseGameEnv] Problematic Log entry (some items might be stringified): {json.dumps(convert_numpy_to_python(log_entry), default=str)}")
            except Exception as e:
                print(f"[BaseGameEnv] CRITICAL ERROR (Log Write): Failed to write log_entry. Details: {e}")
                print(f"[BaseGameEnv] Problematic Log entry (some items might be stringified): {json.dumps(convert_numpy_to_python(log_entry), default=str)}")
        else:
            print(f"[BaseGameEnv] Warning: Episode log file handle is None. Cannot write log for E{self.current_episode_id} S{self.current_step_num}. Path intended: {self.episode_log_file_path}")

    def step(self, agent_action_str: Optional[str], thought_process: str, time_taken_s: float) -> Tuple[Observation, float, bool, bool, Dict[str, Any], float]:
        """
        Takes a step in the environment using the agent's string action.
        Handles invalid or "skip" actions by not stepping the underlying environment
        and returning a neutral outcome (0 reward, no termination).
        Returns observation, reward, terminated, truncated, info, and perf_score.
        """
        if not self.env:
            raise ConnectionError("[BaseGameEnv] Environment not initialized. Call reset() first.")

        self.current_step_num += 1
        
        is_explicit_skip = agent_action_str is None or not agent_action_str.strip() or agent_action_str.lower() == "skip"
        env_action_idx: Optional[int] = None
        action_is_valid_move = False

        if not is_explicit_skip:
            env_action_idx = self.move_to_action_idx.get(agent_action_str.lower())
            if env_action_idx is not None:
                action_is_valid_move = True
        
        should_skip_turn = is_explicit_skip or not action_is_valid_move

        if should_skip_turn:
            action_to_log = agent_action_str if agent_action_str else "None"
            print(f"[BaseGameEnv] Agent action '{action_to_log}' implies skip or is invalid. Doing nothing for this step.")
            
            reward = 0.0
            terminated = False
            truncated = False
            # executed_env_action_idx_for_log = SKIP_ACTION_IDX # Removed
            current_agent_observation = self.extract_observation(self.current_raw_observation, self.current_info) # Re-extract from current state

            # Verify termination status after extracting observation
            terminated, truncated = self.verify_termination(current_agent_observation, terminated, truncated)

            current_step_perf_score = self.perf_score(reward, self.current_info) # Calculate perf_score for skipped step

            self._log_and_print_step_data(agent_action_str, thought_process, reward, self.current_info, terminated, truncated, time_taken_s, current_step_perf_score, current_agent_observation) # Pass current_agent_observation
            return current_agent_observation, reward, terminated, truncated, self.current_info, current_step_perf_score # Return signature changed
        else:
            # This is a valid, non-skip action; env_action_idx must be valid here
            self.current_raw_observation, reward, terminated, truncated, self.current_info = self.env.step(env_action_idx)
            new_agent_observation = self.extract_observation(self.current_raw_observation, self.current_info)

            # Verify termination status after extracting observation
            terminated, truncated = self.verify_termination(new_agent_observation, terminated, truncated)
            
            current_step_perf_score = self.perf_score(float(reward), self.current_info) # Calculate perf_score

            self._log_and_print_step_data(agent_action_str, thought_process, float(reward), self.current_info, terminated, truncated, time_taken_s, current_step_perf_score, new_agent_observation) # Pass new_agent_observation
            return new_agent_observation, float(reward), terminated, truncated, self.current_info, current_step_perf_score # Return signature changed

    def verify_termination(self, observation: Observation, current_terminated: bool, current_truncated: bool) -> Tuple[bool, bool]:
        """
        Verifies or overrides the termination status based on game-specific logic.
        By default, it returns the existing termination and truncation status.
        Subclasses can override this to implement custom logic (e.g., detect stuck states).

        Args:
            observation (Observation): The current agent observation after the step.
            current_terminated (bool): The termination status from the environment.
            current_truncated (bool): The truncation status from the environment.

        Returns:
            Tuple[bool, bool]: The potentially modified (terminated, truncated) status.
        """
        return current_terminated, current_truncated

    def render_human(self) -> None:
        """Renders the environment for human viewing if supported and mode is 'human'."""
        if self.env and hasattr(self.env, 'render') and self.render_mode_for_make == "human": # Check against render_mode_for_make
            self.env.render()
        elif self.render_mode_for_make != "human":
             print(f"[BaseGameEnv] Render mode is '{self.render_mode_for_make}', not 'human'. Skipping human rendering.")

    def close(self) -> None:
        """Closes the environment and any open log files."""
        if self.env:
            self.env.close()
            self.env = None
            print(f"[BaseGameEnv] Environment '{self.game_name}' closed.")
        
        if self.episode_log_file_handle is not None:
            try:
                self.episode_log_file_handle.close()
                print(f"[BaseGameEnv] Episode log file closed: {self.episode_log_file_path}")
            except Exception as e:
                print(f"[BaseGameEnv] Warning: Error closing episode log file during env close: {e}")
            self.episode_log_file_handle = None
            self.episode_log_file_path = None

    def map_env_action_to_agent_action(self, env_action_idx: int) -> str:
        """Maps an environment's integer action index to an agent's string action representation."""
        return self.action_idx_to_move.get(env_action_idx, f"unknown_idx_{env_action_idx}")

    @property
    def action_space(self) -> Optional[gym.Space]:
        return self.env.action_space if self.env else None

    @property
    def observation_space(self) -> Optional[gym.Space]:
        return self.env.observation_space if self.env else None

    def get_current_episode_step_num(self) -> Tuple[int, int]:
        return self.current_episode_id, self.current_step_num

    def _create_agent_observation_path(self, episode_id: int, step_num: int) -> str:
        """
        Creates a unique path for saving an agent's visual observation (if any).
        The path is structured as: {base_dir}/env_obs_e{episode_id}_s{step_num + 1}.png
        `step_num` is assumed to be the 0-indexed current step number.
        """
        return os.path.join(self.agent_observations_base_dir, f"env_obs_e{episode_id:03d}_s{step_num + 1:04d}.png")

    @abstractmethod
    def game_replay(self, trajectory_data: List[Dict[str, Any]]) -> None:
        """
        Replays a game episode from the log file.
        """
        pass


    def perf_score(self, reward: float, info: Dict[str, Any]) -> float:
        """
        Calculates the performance score for a game episode.
        """
        return reward