import os
import json
import glob
from collections import defaultdict
import numpy as np
import math # Added for log2

class GameLogProcessor:
    def __init__(self, game_name, model_name_prefix_for_search, authoritative_model_name, score_transformation_rule: str | None = None):
        self.game_name = game_name
        # Used for finding directories, replace hyphens and then take prefix
        self.model_name_prefix_for_search = model_name_prefix_for_search.replace("-", "_")[:15]
        # Used as the actual key for reporting, taken directly from eval.py model_list
        self.authoritative_model_name = authoritative_model_name 
        self.score_transformation_rule = score_transformation_rule
        self.raw_data = defaultdict(lambda: defaultdict(list)) # authoritative_model_name -> harness_status -> list of episode_data
        self._collect_data()

    def _apply_score_transformation(self, score: float, rule: str | None) -> float:
        if rule == "log2_times_10":
            if score <= 0: # log2 is undefined for 0 or negative numbers
                return 0.0 # Or handle as an error/warning, returning 0 for now
            return math.log2(score) * 10
        # Add other rules here if needed
        # elif rule == "another_rule":
        #     return ...
        return score # Return original score if no rule matches or rule is None

    def _convert_numpy_to_python(self, item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        elif isinstance(item, dict):
            return {k: self._convert_numpy_to_python(v) for k, v in item.items()}
        elif isinstance(item, list):
            return [self._convert_numpy_to_python(i) for i in item]
        elif isinstance(item, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                            np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(item)
        elif isinstance(item, (np.float_, np.float16, np.float32, np.float64)):
            return float(item)
        elif isinstance(item, np.bool_):
            return bool(item)
        return item

    def _collect_data(self):
        model_parent_dir_pattern = os.path.join("cache", self.game_name, self.model_name_prefix_for_search, "*")
        # print(f"[DEBUG] Stage 1: Searching for model parent directories matching: {model_parent_dir_pattern}")
        potential_model_parent_dirs = glob.glob(model_parent_dir_pattern)
        # print(f"[DEBUG] Stage 1: Found potential model parent directories (experiment run directories): {potential_model_parent_dirs}")

        actual_experiment_dirs = []
        if not potential_model_parent_dirs:
            # print(f"[DEBUG] Stage 1: No potential model parent directories (experiment run directories) found.") # Adjusted message
            pass # Keep silent if no dirs found initially, warning will be printed later if still no actual_experiment_dirs
        
        for exp_run_dir in potential_model_parent_dirs: # p_dir is actually the experiment run directory
            # print(f"[DEBUG] Stage 2: Processing potential experiment run directory: {exp_run_dir}")
            if os.path.isdir(exp_run_dir):
                agent_config_target_path = os.path.join(exp_run_dir, "agent_config.json")
                # print(f"[DEBUG] Stage 2a: Checking for agent_config.json at: {agent_config_target_path}")
                if os.path.exists(agent_config_target_path):
                    # print(f"[DEBUG] Stage 2a: Found agent_config.json. Adding {exp_run_dir} to actual_experiment_dirs.")
                    actual_experiment_dirs.append(exp_run_dir)
                # else:
                    # print(f"[DEBUG] Stage 2a: agent_config.json NOT found at {agent_config_target_path}.")
            # else:
                # print(f"[DEBUG] Stage 2: {exp_run_dir} is not a directory.")

        if not actual_experiment_dirs:
            print(f"No experiment directories containing agent_config.json found under parent(s) matching: {model_parent_dir_pattern}")
            return
        
        # print(f"[DEBUG] Proceeding with actual experiment directories: {actual_experiment_dirs}")

        for exp_dir in actual_experiment_dirs:
            agent_config_path = os.path.join(exp_dir, "agent_config.json")
            # agent_config.json existence is already verified above for adding to actual_experiment_dirs

            try:
                with open(agent_config_path, 'r') as f:
                    agent_config = json.load(f)
                
                harness_status_bool = agent_config.get("harness", False)
                harness_status = "harness_true" if harness_status_bool else "harness_false"
                
            except json.JSONDecodeError:
                print(f"Warning: Could not decode agent_config.json in {exp_dir}. Skipping this experiment dir.")
                continue
            except Exception as e:
                print(f"Warning: Error reading agent_config.json in {exp_dir}: {e}. Skipping this experiment dir.")
                continue

            episode_log_pattern = os.path.join(exp_dir, "episode_*_log.jsonl")
            episode_logs = glob.glob(episode_log_pattern)

            if not episode_logs:
                print(f"Warning: No episode logs (.jsonl) found in {exp_dir} for model {self.authoritative_model_name}. Pattern: {episode_log_pattern}")
                continue

            for log_file_path in episode_logs:
                episode_steps_data = []
                try:
                    with open(log_file_path, 'r') as f:
                        for line_number, line in enumerate(f):
                            try:
                                step_detail = json.loads(line)
                                episode_steps_data.append(step_detail)
                            except json.JSONDecodeError:
                                print(f"Warning: Could not decode JSON from line {line_number + 1} in {log_file_path}. Skipping line: {line.strip()}")
                                continue
                    
                    if not episode_steps_data:
                        print(f"Warning: Episode log {log_file_path} was empty or all lines failed to parse. Skipping.")
                        continue

                    episode_id_str = os.path.basename(log_file_path).replace("episode_", "").replace("_log.jsonl", "")
                    
                    total_reward_for_episode = 0
                    raw_total_episode_perf_score = 0.0 # This will be the raw score
                    num_steps_in_episode = len(episode_steps_data)
                    time_taken_for_episode = None 
                    
                    last_step_detail = episode_steps_data[-1]
                    if isinstance(last_step_detail, dict):
                        terminated = last_step_detail.get("terminated", False)
                        truncated = last_step_detail.get("truncated", False)
                        info_field = last_step_detail.get("info", {})
                        if isinstance(info_field, dict) and "total_time_taken" in info_field and (terminated or truncated):
                            time_taken_for_episode = info_field["total_time_taken"]                            
                        elif isinstance(info_field, str) and (terminated or truncated): 
                            try:
                                parsed_info_field = json.loads(info_field.replace("'", "\"")) 
                                if isinstance(parsed_info_field, dict) and "total_time_taken" in parsed_info_field:
                                     time_taken_for_episode = parsed_info_field["total_time_taken"]
                            except json.JSONDecodeError:
                                pass 
                    
                    if time_taken_for_episode is None:
                        sum_step_times = 0
                        valid_step_times_found = False
                        for step_detail_for_time in episode_steps_data:
                            if isinstance(step_detail_for_time, dict) and "time_taken_s" in step_detail_for_time:
                                try:
                                    sum_step_times += float(step_detail_for_time["time_taken_s"])
                                    valid_step_times_found = True
                                except (ValueError, TypeError):
                                    print(f"Warning: Invalid value for time_taken_s in {log_file_path}: {step_detail_for_time['time_taken_s']}")
                        if valid_step_times_found:
                            time_taken_for_episode = sum_step_times

                    # Calculate total reward and total perf_score from all steps
                    for step_detail in episode_steps_data:
                        if isinstance(step_detail, dict):
                            total_reward_for_episode += step_detail.get("reward", 0)
                            raw_total_episode_perf_score += step_detail.get("perf_score", 0.0) # Sum perf_score

                    # Apply transformation only if a rule is provided and store it under a different key
                    if self.score_transformation_rule:
                        transformed_score = self._apply_score_transformation(
                            raw_total_episode_perf_score, 
                            self.score_transformation_rule
                        )
                        current_episode_summary = {
                            "episode_id": episode_id_str,
                            "steps": num_steps_in_episode,
                            "total_reward": self._convert_numpy_to_python(total_reward_for_episode),
                            "total_episode_perf_score": self._convert_numpy_to_python(raw_total_episode_perf_score), # Store raw score here
                            "final_score_for_ranking": self._convert_numpy_to_python(transformed_score)
                        }
                    else:
                        current_episode_summary = {
                            "episode_id": episode_id_str,
                            "steps": num_steps_in_episode,
                            "total_reward": self._convert_numpy_to_python(total_reward_for_episode),
                            "total_episode_perf_score": self._convert_numpy_to_python(raw_total_episode_perf_score), # Store raw score here
                        }

                    # Collect all agent observations for the episode
                    step_infos_for_episode = []
                    for step_detail in episode_steps_data:
                        if isinstance(step_detail, dict) and "info" in step_detail:
                            step_infos_for_episode.append(self._convert_numpy_to_python(step_detail["info"]))

                    # Collect all agent observations for the episode
                    step_observations_for_episode = []
                    for step_detail in episode_steps_data:
                        if isinstance(step_detail, dict) and "agent_observation" in step_detail:
                            obs_data = step_detail.get("agent_observation")
                            if isinstance(obs_data, str): # It should be a JSON string
                                try:
                                    parsed_obs = json.loads(obs_data)
                                    step_observations_for_episode.append(self._convert_numpy_to_python(parsed_obs))
                                except json.JSONDecodeError:
                                    print(f"Warning: Could not decode agent_observation JSON string in {log_file_path} for step. Storing as raw string.")
                                    step_observations_for_episode.append(obs_data) # Fallback to raw string
                            else:
                                # If it's not a string, store it as is (after numpy conversion)
                                step_observations_for_episode.append(self._convert_numpy_to_python(obs_data))
                        # else: Optionally append a placeholder like None if "agent_observation" is missing

                    current_episode_summary["step_infos"] = step_infos_for_episode
                    current_episode_summary["step_observations"] = step_observations_for_episode
                    if time_taken_for_episode is not None:
                        current_episode_summary["total_time_taken"] = self._convert_numpy_to_python(time_taken_for_episode)
                    
                    self.raw_data[self.authoritative_model_name][harness_status].append(current_episode_summary)

                except FileNotFoundError:
                    print(f"Error: Log file {log_file_path} not found during processing. This shouldn't happen if glob found it.")
                except Exception as e:
                    print(f"Warning: Error processing log file {log_file_path}: {e}. Skipping file.")
        
        if not self.raw_data:
            print(f"No data successfully collected for game: {self.game_name}, model prefix search: {self.model_name_prefix_for_search}")

    def generate_model_perf_update(self):
        update_data = defaultdict(lambda: defaultdict(dict))
        for model_name, harness_data in self.raw_data.items():
            for harness_status, episodes in harness_data.items():
                if episodes:
                    # Use 'final_score_for_ranking' if available, else 'total_episode_perf_score' (raw)
                    scores = [
                        ep.get('final_score_for_ranking', ep['total_episode_perf_score']) 
                        for ep in episodes
                    ]
                    if model_name not in update_data:
                        update_data[model_name] = defaultdict(dict)
                    update_data[model_name][harness_status][self.game_name] = scores
        return dict(update_data)

    def generate_game_perf_update(self):
        game_perf_data = defaultdict(lambda: defaultdict(dict))
        for model_name, harness_data in self.raw_data.items():
            for harness_status, episodes in harness_data.items():
                if not episodes:
                    continue
                
                num_all_episodes = len(episodes)
                
                # Always use the raw 'total_episode_perf_score' for game_perf.json
                all_total_rewards = [ep['total_reward'] for ep in episodes]
                all_raw_total_perf_scores = [ep['total_episode_perf_score'] for ep in episodes]

                # Prepare episodes_data for game_perf.json, ensuring no transformed scores are included
                episodes_data_for_json = []
                for ep_original in episodes:
                    ep_copy = ep_original.copy() # Create a copy to modify
                    ep_copy.pop('final_score_for_ranking', None) # Remove transformed score if it exists
                    episodes_data_for_json.append(ep_copy)

                current_harness_perf_data = {
                    "num_episodes_processed": num_all_episodes,
                    "total_reward_values": all_total_rewards,
                    "total_perf_score_values": all_raw_total_perf_scores, # List of raw scores
                    "episodes_data": episodes_data_for_json # List of episode dicts with raw scores
                }
                
                if model_name not in game_perf_data: 
                    game_perf_data[model_name] = defaultdict(dict)
                game_perf_data[model_name][harness_status] = current_harness_perf_data
        
        return dict(game_perf_data)

def load_json_file(file_path, default_data=None):
    if default_data is None:
        default_data = {}
    if not os.path.exists(file_path):
        return default_data
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from {file_path}. Returning default data.")
        return default_data
    except Exception as e:
        print(f"Warning: An unexpected error occurred while loading {file_path}: {e}. Returning default data.")
        return default_data

def save_json_file(data, file_path):
    try:
        file_dir = os.path.dirname(file_path)
        if file_dir and not os.path.exists(file_dir):
            os.makedirs(file_dir, exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Data saved to {file_path}")
    except Exception as e:
        print(f"Error: Could not save data to {file_path}: {e}")

def update_model_perf_rank(existing_data, new_data_from_processor, game_name_processed, force: bool = False):
    if not isinstance(existing_data, dict):
        print(f"Warning: existing_data for model_perf_rank is not a dict. Initializing to empty dict. Type was: {type(existing_data)}")
        existing_data = {}

    for model_name, harness_dict_new in new_data_from_processor.items():
        if model_name not in existing_data:
            existing_data[model_name] = {}
        elif not isinstance(existing_data[model_name], dict):
            print(f"Warning: existing_data[{model_name}] for model_perf_rank is not a dict. Re-initializing. Type was: {type(existing_data[model_name])}")
            existing_data[model_name] = {}
        
        for harness_status, game_scores_dict_new in harness_dict_new.items():
            if harness_status not in existing_data[model_name]:
                existing_data[model_name][harness_status] = {}
            elif not isinstance(existing_data[model_name][harness_status], dict):
                print(f"Warning: existing_data[{model_name}][{harness_status}] for model_perf_rank is not a dict. Re-initializing. Type was: {type(existing_data[model_name][harness_status])}")
                existing_data[model_name][harness_status] = {}
            
            for game_key_new, scores_new in game_scores_dict_new.items():
                if game_key_new == game_name_processed:
                    if game_key_new not in existing_data[model_name][harness_status] or force:
                        existing_data[model_name][harness_status][game_key_new] = scores_new
                        update_type = "Forced update to" if force and game_key_new in existing_data[model_name][harness_status] else "Added"
                        print(f"{update_type} model_perf_rank: Scores for {model_name}/{harness_status}/{game_key_new}")
                    else:
                        print(f"Skipped model_perf_rank update for {model_name}/{harness_status}/{game_key_new}: Scores already exist (force=False).")
    return existing_data

def update_game_perf_data(existing_data, new_data_from_processor, force: bool = False):
    if not isinstance(existing_data, dict):
        print(f"Warning: existing_data for game_perf is not a dict. Initializing to empty dict. Type was: {type(existing_data)}")
        existing_data = {}

    for model_name, harness_dict_new in new_data_from_processor.items():
        if model_name not in existing_data:
            existing_data[model_name] = {}
        elif not isinstance(existing_data[model_name], dict):
            print(f"Warning: existing_data[{model_name}] for game_perf is not a dict. Re-initializing. Type was: {type(existing_data[model_name])}")
            existing_data[model_name] = {}
            
        for harness_status, perf_metrics_new in harness_dict_new.items():
            if harness_status not in existing_data[model_name] or force:
                existing_data[model_name][harness_status] = perf_metrics_new # This now includes the new lists
                update_type = "Forced update to" if force and harness_status in existing_data[model_name] else "Added"
                print(f"{update_type} game_perf: Data for {model_name}/{harness_status}")
            else:
                # If you want to merge or update specific fields if harness_status exists,
                print(f"Skipped game_perf update for {model_name}/{harness_status}: Data already exists (force=False).")
    return existing_data
