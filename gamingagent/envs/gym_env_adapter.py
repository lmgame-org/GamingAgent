import os
import json
import datetime
import hashlib
from typing import Optional, Dict, Any, Tuple

from gamingagent.modules.core_module import Observation
from gamingagent.envs.env_utils import create_board_image_2048 # Import the image creation function
from tools.utils import convert_numpy_to_python

SKIP_ACTION_IDX = -1 # Consistent with BaseGameEnv

class GymEnvAdapter:
    def __init__(self,
                 game_name: str,
                 observation_mode: str, # "vision", "text", "both"
                 agent_cache_dir: str, # Used for logs and observations
                 game_specific_config_path: str, # Path to game_env_config.json
                 max_steps_for_stuck: Optional[int] = None):
        self.game_name = game_name
        self.observation_mode = observation_mode
        self.agent_cache_dir = agent_cache_dir
        self.agent_observations_dir = os.path.join(self.agent_cache_dir, "observations")
        os.makedirs(self.agent_observations_dir, exist_ok=True)

        self.current_episode_id = 0
        self.current_step_num = 0
        self.episode_log_file_path: Optional[str] = None
        self.episode_log_file_handle: Optional[Any] = None

        # For stuck detection (from TwentyFortyEightEnvWrapper)
        self._last_observation_hash: Optional[str] = None
        self._unchanged_obs_count: int = 0
        self._max_unchanged_steps: int = max_steps_for_stuck if max_steps_for_stuck is not None else 10 # Default from wrapper

        # Load game-specific config (action mapping, etc.)
        self.action_mapping_config: Dict[str, int] = {}
        self.move_to_action_idx: Dict[str, int] = {}
        self.action_idx_to_move: Dict[int, str] = {}
        self._load_game_specific_config(game_specific_config_path)

    def _load_game_specific_config(self, config_path: str):
        print(f"[GymEnvAdapter] Loading game-specific config from: {config_path}")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                action_mapping = config.get("action_mapping")
                if isinstance(action_mapping, dict):
                    self.action_mapping_config = action_mapping
                    self.move_to_action_idx = {str(k).lower(): int(v) for k, v in action_mapping.items()}
                    self.action_idx_to_move = {int(v): str(k).lower() for k, v in action_mapping.items()}
                    print(f"[GymEnvAdapter] Loaded action mapping: {self.move_to_action_idx}")
                else:
                    print(f"[GymEnvAdapter] Warning: 'action_mapping' in {config_path} is not a dict or is missing.")

                max_unchanged = config.get("max_unchanged_steps_for_termination")
                if isinstance(max_unchanged, int) and max_unchanged > 0:
                    self._max_unchanged_steps = max_unchanged
                    print(f"[GymEnvAdapter] Loaded max_unchanged_steps_for_termination: {self._max_unchanged_steps}")
                elif self._max_unchanged_steps is None: # Only if not set by constructor
                    self._max_unchanged_steps = 10 # Default if not in config and not in constructor
                    print(f"[GymEnvAdapter] Using default max_unchanged_steps_for_termination: {self._max_unchanged_steps}")
                
            except json.JSONDecodeError as e:
                print(f"[GymEnvAdapter] Error decoding JSON from {config_path}: {e}. Using defaults for action mapping and stuck detection.")
            except Exception as e:
                print(f"[GymEnvAdapter] Error loading config from {config_path}: {e}. Using defaults.")
        else:
            print(f"[GymEnvAdapter] Warning: Game-specific config {config_path} not found. Action mapping will be empty, default stuck detection.")

    def reset_episode(self, episode_id: int):
        self.current_episode_id = episode_id
        self.current_step_num = 0
        self._last_observation_hash = None
        self._unchanged_obs_count = 0

        if self.episode_log_file_handle is not None:
            try:
                self.episode_log_file_handle.close()
            except Exception as e:
                print(f"[GymEnvAdapter] Warning: Error closing previous episode log file: {e}")
            self.episode_log_file_handle = None
        
        self.episode_log_file_path = os.path.join(self.agent_cache_dir, f"episode_{self.current_episode_id:03d}_log.jsonl")
        try:
            self.episode_log_file_handle = open(self.episode_log_file_path, 'a')
            print(f"[GymEnvAdapter] Logging episode {self.current_episode_id} data to: {self.episode_log_file_path}")
        except Exception as e:
            print(f"[GymEnvAdapter] ERROR: Could not open episode log file {self.episode_log_file_path}: {e}")
            self.episode_log_file_handle = None

    def create_agent_observation(self, board_powers: Any, perf_score_for_image: Optional[float] = None) -> Observation:
        img_path_for_agent = None
        text_representation_for_agent = None

        if self.observation_mode in ["vision", "both"]:
            img_path_for_agent = self._create_agent_observation_path(self.current_episode_id, self.current_step_num)
            create_board_image_2048(board_powers, img_path_for_agent, perf_score=perf_score_for_image)
        
        if self.observation_mode in ["text", "both"]:
            if isinstance(board_powers, list):
                 text_representation_for_agent = str(board_powers)
            elif hasattr(board_powers, 'tolist'): # For numpy arrays
                 text_representation_for_agent = str(board_powers.tolist())
            else:
                 text_representation_for_agent = str(board_powers)
            
        return Observation(
            img_path=img_path_for_agent, 
            textual_representation=text_representation_for_agent
        )

    def _create_agent_observation_path(self, episode_id: int, step_num: int) -> str:
        return os.path.join(self.agent_observations_dir, f"env_obs_e{episode_id:03d}_s{step_num:04d}.png")

    def log_step_data(self, agent_action_str: Optional[str], thought_process: str, reward: float, info: Dict[str, Any], terminated: bool, truncated: bool, time_taken_s: float, perf_score: float, agent_observation: Observation):
        print(f"[GymEnvAdapter] E{self.current_episode_id} S{self.current_step_num}: AgentAct='{agent_action_str}', R={reward:.2f}, Perf={perf_score:.2f}, Term={terminated}, Trunc={truncated}, T={time_taken_s:.2f}s")

        log_entry = {
            "episode_id": self.current_episode_id,
            "step": self.current_step_num,
            "agent_action": agent_action_str,
            "thought": thought_process,
            "reward": float(reward),
            "perf_score": float(perf_score),
            "info": info, 
            "agent_observation": str(agent_observation), # Uses Observation.to_json_string()
            "terminated": terminated,
            "truncated": truncated,
            "time_taken_s": float(time_taken_s)
        }
        
        if self.episode_log_file_handle:
            try:
                serializable_log_entry = convert_numpy_to_python(log_entry)
                self.episode_log_file_handle.write(json.dumps(serializable_log_entry) + '\n')
                self.episode_log_file_handle.flush()
            except Exception as e:
                print(f"[GymEnvAdapter] CRITICAL ERROR (Log Write): Failed to write log_entry. Details: {e}")
        else:
            print(f"[GymEnvAdapter] Warning: Episode log file handle is None. Cannot write log.")

    def verify_termination(self, agent_observation: Observation, current_terminated: bool, current_truncated: bool) -> Tuple[bool, bool]:
        if current_terminated or current_truncated:
            return current_terminated, current_truncated

        current_obs_hash: Optional[str] = None
        if self.observation_mode == "text" or self.observation_mode == "both":
            if agent_observation.textual_representation:
                current_obs_hash = hashlib.md5(agent_observation.textual_representation.encode()).hexdigest()
        elif self.observation_mode == "vision": # Only vision, rely on image hash
            if agent_observation.img_path and os.path.exists(agent_observation.img_path):
                try:
                    with open(agent_observation.img_path, 'rb') as f_img:
                        current_obs_hash = hashlib.md5(f_img.read()).hexdigest()
                except Exception as e:
                    print(f"[GymEnvAdapter] Warning: Could not hash image {agent_observation.img_path} for stuck detection: {e}")
                    return current_terminated, current_truncated # Cannot determine if stuck
            else:
                return current_terminated, current_truncated # No image to hash
        else: # Should not happen
            return current_terminated, current_truncated

        if current_obs_hash is None: # Fallback if no hash could be generated (e.g. vision mode but no image path)
             return current_terminated, current_truncated

        if self._last_observation_hash == current_obs_hash:
            self._unchanged_obs_count += 1
        else:
            self._unchanged_obs_count = 0
        
        self._last_observation_hash = current_obs_hash

        if self._unchanged_obs_count >= self._max_unchanged_steps:
            print(f"[GymEnvAdapter] Terminating episode due to unchanged observation for {self._max_unchanged_steps} steps.")
            return True, current_truncated # Set terminated to True
        
        return current_terminated, current_truncated

    def calculate_perf_score(self, reward: float, info: Dict[str, Any]) -> float:
        # For 2048, the reward from the environment is usually the score obtained in that step.
        # The performance score can be the same as the reward, or can be customized.
        # For simplicity, let's use the reward as the performance score here.
        # This can be easily extended; for example, info.get('total_score') could be used,
        # or a score based on the highest tile achieved.
        return float(reward) 

    def map_agent_action_to_env_action(self, agent_action_str: Optional[str]) -> Optional[int]:
        if agent_action_str is None or not agent_action_str.strip():
            return None # Represents a skip or no action
        
        action_str_lower = agent_action_str.lower()
        if action_str_lower == "skip":
             return None # Explicit skip

        return self.move_to_action_idx.get(action_str_lower)

    def close_log_file(self):
        if self.episode_log_file_handle:
            try:
                self.episode_log_file_handle.close()
                print(f"[GymEnvAdapter] Episode log file closed: {self.episode_log_file_path}")
            except Exception as e:
                print(f"[GymEnvAdapter] Warning: Error closing log file: {e}")
            self.episode_log_file_handle = None
            self.episode_log_file_path = None

    def increment_step(self):
        self.current_step_num +=1 