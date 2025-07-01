import os
import json
import datetime
import hashlib
from typing import Optional, Dict, Any, Tuple, List
import numpy as np

from gamingagent.modules.core_module import Observation
from tools.utils import convert_numpy_to_python

SKIP_ACTION_IDX = -1 # Consistent with BaseGameEnv

class GymEnvAdapter:
    """
    Adapts a standard Gymnasium environment for use with the GamingAgent framework.

    The adapter handles several key responsibilities:
    1.  Loading game-specific configurations (e.g., action mappings, stuck detection parameters)
        from a JSON file (`game_env_config.json`).
    2.  Managing episode lifecycle: resetting state, incrementing steps.
    3.  Creating agent-facing `Observation` objects (potentially including image rendering
        via `env_utils.create_board_image_2048` and textual representations) based on the
        raw environment state and specified observation mode.
    4.  Logging detailed step-by-step data for each episode to a JSONL file.
    5.  Mapping agent action strings (e.g., "up", "left") to environment-specific action indices.
    6.  Performing stuck detection by hashing observations and terminating if the observation
        remains unchanged for a configured number of steps.
    7.  Calculating a performance score for each step (currently mirrors reward, but extensible).
    8.  Collecting results from all episodes within a run.
    9.  Generating a final JSON summary (`gym_run_summary.json`) for the entire run, including
        run settings, individual episode results, and overall statistics (mean, std, min, max).

    Key Parameters:
        game_name (str): The internal name of the game (e.g., "twenty_forty_eight").
        observation_mode (str): How the agent observes the environment. 
                                Options: "vision", "text", "both".
        agent_cache_dir (str): Directory to store episode logs and generated observation images.
        game_specific_config_path (Optional[str]): Path to the game's JSON configuration file 
                                         (e.g., `game_env_config.json`) which contains action
                                         mappings and other game-specific settings for the adapter.
        max_steps_for_stuck (Optional[int]): Number of consecutive unchanged observations before
                                           the episode is terminated due to being stuck. Overrides
                                           value from `game_specific_config_path` if provided.
    """
    def __init__(self,
                 game_name: str,
                 observation_mode: str, # "vision", "text", "both"
                 agent_cache_dir: str, # Used for logs and observations
                 game_specific_config_path: Optional[str] = None, # Path to game_env_config.json, now optional
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
        if game_specific_config_path: # This check will now work as intended
            self._load_game_specific_config(game_specific_config_path)
        
        self.all_episode_results: List[Dict] = [] # To store results of each episode

    def save_frame_and_get_path(self, frame: "np.ndarray") -> str:
        """
        Convenience for wrappers: save a NumPy frame to the canonical
        observations folder and return the path.  All path formatting is
        centralised here.
        """
        path = self._create_agent_observation_path(
            self.current_episode_id, self.current_step_num
        )
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            from PIL import Image  # local import to avoid hard dep for text‑only games
            Image.fromarray(frame).save(path)
        except Exception as e:
            print(f"[GymEnvAdapter] Warning: could not save frame to {path}: {e}")
        return path
    
    def _load_game_specific_config(self, config_path: str):
        """Loads game-specific settings like action mapping and stuck detection params from JSON."""
        print(f"[GymEnvAdapter DEBUG _load_game_specific_config] Called with config_path: {config_path}")
        print(f"[GymEnvAdapter] Loading game-specific config from: {config_path}")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                action_mapping = config.get("action_mapping")
                if isinstance(action_mapping, dict):
                    self.action_mapping_config = action_mapping
                    # Values can be int or list, store them as is
                    self.move_to_action_idx = {str(k).lower(): v for k, v in action_mapping.items()}
                    
                    # action_idx_to_move might be less straightforward if v is a list.
                    # For now, we only map if v is an int. This part might need rethinking
                    # if bi-directional mapping is critical for list-based actions.
                    self.action_idx_to_move = {}
                    for k, v in action_mapping.items():
                        if isinstance(v, int):
                            self.action_idx_to_move[v] = str(k).lower()
                        # If v is a list, creating a reverse mapping is ambiguous without a unique key for the list itself.
                        # For now, we skip adding list actions to action_idx_to_move.
                        # If needed, the list could be converted to a tuple to be a dict key, but utility is questionable here.

                    print(f"[GymEnvAdapter] Loaded action mapping (move_to_action_idx): {self.move_to_action_idx}")
                    print(f"[GymEnvAdapter] Loaded reverse action mapping (action_idx_to_move, int actions only): {self.action_idx_to_move}")
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
        """
        Resets the adapter's state for a new episode.

        Args:
            episode_id (int): The identifier for the new episode.
        """
        self.current_episode_id = episode_id
        self.current_step_num = 0
        self._last_observation_hash = None
        self._unchanged_obs_count = 0
  
        # Clear results from previous set of runs if adapter is reused.
        if self.current_episode_id == 1 or not self.all_episode_results: # A simple check, or clear if episode_id resets to 1
            self.all_episode_results = []

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

    def create_agent_observation(self, 
                                 img_path: Optional[str] = None, 
                                 text_representation: Optional[str] = None, 
                                 background_info: Optional[str] = None,
                                 max_memory: Optional[int] = 10) -> Observation:
        """
        Creates an agent-facing Observation object from pre-defined image path, text representation, and background info.
        The environment is responsible for generating these components based on its state and
        the adapter's observation_mode.

        Args:
            img_path (Optional[str]): Path to the observation image file.
            text_representation (Optional[str]): Textual representation of the observation.
            background_info (Optional[str]): Static background information for the episode.

        Returns:
            Observation: An Observation object suitable for the agent.
        """
        # Determine if background information should be included in the trajectory
        trajectory_includes_bg = bool(background_info)

        agent_observation = Observation(
            img_path=img_path,
            textual_representation=text_representation,
            background=background_info,
            trajectory_includes_background=trajectory_includes_bg,
            max_memory=max_memory
        )
        
        # Now, set perception-specific parts if any, without passing background again
        agent_observation.set_perception_observation(
            img_path=img_path, # Pass img_path again as set_perception_observation expects it
            textual_representation=text_representation # Pass text_representation again as set_perception_observation expects it
            # processed_visual_description is not handled here, assumed to be set by PerceptionModule if used
        )
        return agent_observation

    def _create_agent_observation_path(self, episode_id: int, step_num: int) -> str:
        """Generates a unique file path for an observation image."""
        return os.path.join(self.agent_observations_dir, f"env_obs_e{episode_id:03d}_s{step_num:04d}.png")

    def log_step_data(self, agent_action_str: Optional[str], thought_process: str, reward: float, info: Dict[str, Any], terminated: bool, truncated: bool, time_taken_s: float, perf_score: float, agent_observation: Observation):
        """
        Logs all relevant data for a single step to the current episode's JSONL file.

        Args:
            agent_action_str (Optional[str]): The action string from the agent.
            thought_process (str): The agent's thought process for the action.
            reward (float): Reward received from the environment for this step.
            info (Dict[str, Any]): Additional information from the environment's step function.
            terminated (bool): Whether the episode terminated this step.
            truncated (bool): Whether the episode truncated this step (e.g., time limit).
            time_taken_s (float): Time taken by the agent to decide on the action.
            perf_score (float): Performance score calculated for this step.
            agent_observation (Observation): The agent-facing observation for this step.
        """
        # Conditional print: Only print if not an uneventful AUTO_NO_OP, or if it's an interesting event, or a milestone AUTO_NO_OP step.
        if agent_action_str != "<AUTO_NO_OP>" or reward != 0 or perf_score != 0 or terminated or truncated or (agent_action_str == "<AUTO_NO_OP>" and self.current_step_num > 0 and self.current_step_num < 10):
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
        """
        Checks for episode termination due to stuck state (unchanged observations).

        Args:
            agent_observation (Observation): The current agent-facing observation.
            current_terminated (bool): The termination status from the environment.
            current_truncated (bool): The truncation status from the environment.

        Returns:
            Tuple[bool, bool]: Updated (terminated, truncated) status.
        """
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
        """
        Calculates a performance score for the current step.
        This base implementation simply returns the reward.
        Game-specific environments can override this method for custom logic.

        Args:
            reward (float): The reward received from the environment for the step.
            info (Dict[str, Any]): Additional information from the environment.

        Returns:
            float: The performance score for this step.
        """

        return reward # Now returns reward directly

    def map_agent_action_to_env_action(self, agent_action_str: Optional[str]) -> Optional[Any]:
        """
        Maps an agent's action string (e.g., "left") to an environment-specific action.
        The action can be an integer index or a list/array representing button states.

        Args:
            agent_action_str (Optional[str]): The action string from the agent.

        Returns:
            Optional[Any]: The corresponding action for the Gymnasium environment (int or List/np.array),
                           or None if the action is invalid, None, or "skip".
        """
        if agent_action_str is None or not agent_action_str.strip():
            return None # Represents a skip or no action
        
        action_str_lower = agent_action_str.lower()
        if action_str_lower == "skip":
             return None # Explicit skip

        # self.move_to_action_idx now holds the direct value (int or list)
        action_val = self.move_to_action_idx.get(action_str_lower)
        
        if isinstance(action_val, list):
            return np.array(action_val, dtype=bool) # Convert to numpy array for retro env
        elif isinstance(action_val, int):
            return action_val
        else:
            # If action_val is None (not found) or some other unexpected type
            if action_val is None:
                 print(f"[GymEnvAdapter] Warning: Action '{agent_action_str}' not found in move_to_action_idx.")
            else:
                 print(f"[GymEnvAdapter] Warning: Action '{agent_action_str}' mapped to unexpected type: {type(action_val)}.")
            return None

    def close_log_file(self):
        """Closes the current episode's log file if it's open."""
        if self.episode_log_file_handle:
            try:
                self.episode_log_file_handle.close()
                print(f"[GymEnvAdapter] Episode log file closed: {self.episode_log_file_path}")
            except Exception as e:
                print(f"[GymEnvAdapter] Warning: Error closing log file: {e}")
            self.episode_log_file_handle = None
            self.episode_log_file_path = None

    def increment_step(self):
        """Increments the current step number for the episode."""
        self.current_step_num +=1 

    def record_episode_result(self, episode_id: int, score: float, steps: int, total_reward: float, total_perf_score: float):
        """
        Records the final results of a single game episode.

        Args:
            episode_id (int): Identifier for the episode.
            score (float): Final score achieved in the episode (e.g., from env info).
            steps (int): Total number of steps taken in the episode.
            total_reward (float): Sum of all rewards received during the episode.
            total_perf_score (float): Sum of all performance scores during the episode.
        """
        self.all_episode_results.append({
            "run_id": episode_id,
            "score": score,
            "steps": steps,
            "total_reward_for_episode": total_reward,
            "total_perf_score_for_episode": total_perf_score
        })

    def finalize_and_save_summary(self, run_settings: Dict) -> Dict:
        """
        Calculates summary statistics from all recorded episodes, saves the full summary
        (settings, individual results, stats) to `gym_run_summary.json` in the `agent_cache_dir`,
        and returns the overall statistics dictionary.

        Args:
            run_settings (Dict): Dictionary of run settings (e.g., from parsed CLI arguments)
                                 to be included in the summary file.

        Returns:
            Dict: A dictionary containing the overall statistics (mean, std, min, max for relevant metrics).
        """
        summary_data = {
            "settings": run_settings,
            "individual_run_results": self.all_episode_results,
            "overall_stat_summary": {}
        }
        
        num_runs = len(self.all_episode_results)

        if num_runs > 0:
            scores = [r['score'] for r in self.all_episode_results]
            steps_list = [r['steps'] for r in self.all_episode_results]
            total_rewards_list = [r['total_reward_for_episode'] for r in self.all_episode_results]
            total_perf_scores_list = [r['total_perf_score_for_episode'] for r in self.all_episode_results]

            stats_map = {
                "Final Env Scores": scores,
                "Steps Taken": steps_list,
                "Total Rewards": total_rewards_list,
                "Total Performance Scores": total_perf_scores_list
            }

            for key, values in stats_map.items():
                # Ensure values is a list and not empty before processing
                if isinstance(values, list) and values:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    min_val = np.min(values)
                    max_val = np.max(values)
                    
                    summary_data["overall_stat_summary"][key.lower().replace(" ", "_")] = {
                        "mean": float(mean_val),
                        "std": float(std_val),
                        "min": int(min_val) if isinstance(min_val, (np.integer, int)) else float(min_val),
                        "max": int(max_val) if isinstance(max_val, (np.integer, int)) else float(max_val),
                        "values": [int(v) if isinstance(v, (np.integer, int)) else float(v) for v in values]
                    }
                else:
                    summary_data["overall_stat_summary"][key.lower().replace(" ", "_")] = {
                        "mean": None, "std": None, "min": None, "max": None, "values": []
                    }
        
        summary_file_path = os.path.join(self.agent_cache_dir, "gym_run_summary.json")
        try:
            serializable_summary_data = convert_numpy_to_python(summary_data)
            with open(summary_file_path, 'w') as f:
                json.dump(serializable_summary_data, f, indent=2)
            print(f"[GymEnvAdapter] Run summary saved to: {summary_file_path}")
        except Exception as e:
            print(f"[GymEnvAdapter] CRITICAL ERROR: Failed to save run summary to {summary_file_path}. Details: {e}")
            if 'serializable_summary_data' in locals():
                 print(f"Problematic data (first 200 chars): {str(serializable_summary_data)[:200]}")


        return summary_data["overall_stat_summary"] 