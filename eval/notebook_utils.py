import os
import glob
import zipfile 
import requests
import json
import pandas as pd
import shutil
import matplotlib.pyplot as plt
import numpy as np
import random
import cvxpy as cp
from scipy.optimize import minimize, LinearConstraint

def check_evaluation_files(download_cache_sample: bool, download_url):
    report = {"cache_directory": None}  # Initialize with None
    cache_dir_path = "cache"
    cache_zip_filename = "cache.zip"  # Standard name for the zip file (local or downloaded)

    if download_cache_sample:
        print("Mode: Download sample cache.")
        # Google Drive direct download link from ID: 1uCen9DfMcKuHOyc-Lhrxa_ZeEOuWJfXM

        # 1. Clean up existing cache directory and/or target zip file before download
        if os.path.isdir(cache_dir_path):
            print(f"Removing existing cache directory: '{cache_dir_path}'")
            try:
                shutil.rmtree(cache_dir_path)
            except OSError as e:
                print(f"Error removing directory {cache_dir_path}: {e}. Aborting download process.")
                return report  # report["cache_directory"] is still None
        
        if os.path.isfile(cache_zip_filename):
            print(f"Removing existing local zip file named '{cache_zip_filename}' before download.")
            try:
                os.remove(cache_zip_filename)
            except OSError as e:
                print(f"Error removing existing zip file {cache_zip_filename}: {e}. Aborting download process.")
                return report

        # 2. Download the cache zip file
        print(f"Downloading cache from Google Drive to '{cache_zip_filename}'...")
        downloaded_file_path = None
        try:
            import gdown  # Attempt to import gdown here
            # Using fuzzy=True can help with some Google Drive links if the direct one has issues.
            downloaded_file_path = gdown.download(download_url, cache_zip_filename, quiet=False, fuzzy=True)
            if downloaded_file_path is None or not os.path.isfile(cache_zip_filename):
                print(f"Error: Failed to download cache. 'gdown.download' did not confirm success or file '{cache_zip_filename}' not found post-download.")
                # Clean up if a partial/empty file was created
                if os.path.isfile(cache_zip_filename):
                    try: os.remove(cache_zip_filename)
                    except OSError: pass
                return report
            print(f"Successfully downloaded to '{cache_zip_filename}'.")
        except ImportError:
            print("Error: The 'gdown' library is not installed. Please install it (e.g., 'pip install gdown') to enable downloading the cache.")
            return report
        except Exception as e:
            print(f"An error occurred during download: {e}")
            if os.path.isfile(cache_zip_filename): # Clean up if download failed and left a file
                 try: os.remove(cache_zip_filename)
                 except OSError: pass
            return report

        # 3. Unzip the downloaded file
        print(f"Attempting to unzip '{cache_zip_filename}' to '{cache_dir_path}'.")
        try:
            with zipfile.ZipFile(cache_zip_filename, 'r') as zip_ref:
                zip_ref.extractall(cache_dir_path)  # Extracts to 'cache/' directory
            if os.path.isdir(cache_dir_path):
                report["cache_directory"] = cache_dir_path
                print(f"Successfully unzipped '{cache_zip_filename}' to '{cache_dir_path}'.")
            else:
                print(f"Error: Directory '{cache_dir_path}' not found after unzipping.")
        except zipfile.BadZipFile:
            print(f"Error: Downloaded file '{cache_zip_filename}' is not a valid zip file or is corrupted.")
        except Exception as e:
            print(f"An error occurred during unzipping: {e}")
        finally:
            # Clean up the downloaded zip file after attempting to extract
            if os.path.isfile(cache_zip_filename):
                print(f"Removing downloaded zip file: '{cache_zip_filename}'")
                try:
                    os.remove(cache_zip_filename)
                except OSError as e:
                    print(f"Error removing downloaded zip file '{cache_zip_filename}': {e}")
    
    else:  # download_cache_sample is False - check local files
        print("Mode: Check local cache.")
        if os.path.isdir(cache_dir_path):
            report["cache_directory"] = cache_dir_path
            print(f"Using existing local cache directory: '{cache_dir_path}'")
        elif os.path.isfile(cache_zip_filename):
            print(f"Found local cache zip file: '{cache_zip_filename}'. Attempting to unzip.")
            
            # If 'cache' directory exists (e.g. from partial previous attempt), remove it before unzipping.
            if os.path.isdir(cache_dir_path):
                print(f"Removing existing directory '{cache_dir_path}' before unzipping from local '{cache_zip_filename}'.")
                try:
                    shutil.rmtree(cache_dir_path)
                except OSError as e:
                    print(f"Error removing directory {cache_dir_path}: {e}. Cannot proceed with unzipping local zip.")
                    return report 
            # Handle if cache_dir_path is a file (conflicting name)
            elif os.path.exists(cache_dir_path) and not os.path.isdir(cache_dir_path):
                 print(f"Found a file/link at '{cache_dir_path}' which conflicts with the target directory name. Removing it.")
                 try:
                     os.remove(cache_dir_path)
                 except OSError as e:
                     print(f"Error removing conflicting file/link {cache_dir_path}: {e}. Cannot unzip local zip.")
                     return report

            try:
                with zipfile.ZipFile(cache_zip_filename, 'r') as zip_ref:
                    zip_ref.extractall(cache_dir_path)  # Extracts to 'cache/'
                if os.path.isdir(cache_dir_path):
                    report["cache_directory"] = cache_dir_path
                    print(f"Successfully unzipped local '{cache_zip_filename}' to '{cache_dir_path}'.")
                else:
                    print(f"Error: Failed to create directory '{cache_dir_path}' after unzipping local zip.")
            except zipfile.BadZipFile:
                print(f"Error: Local '{cache_zip_filename}' is not a valid zip file or is corrupted.")
            except Exception as e:
                print(f"An error occurred during unzipping local zip: {e}")
        else:
            print(f"Info: No local cache directory '{cache_dir_path}' or local cache zip file '{cache_zip_filename}' found.")

    # Final status message based on whether 'cache_dir_path' ("cache") is set
    if report["cache_directory"] == cache_dir_path:
        print(f"Final Result: Cache is available at '{report['cache_directory']}'.")
    else:
        # Ensure report["cache_directory"] is None if not successfully set to cache_dir_path
        report["cache_directory"] = None 
        print("Final Result: No cache directory named 'cache' is available.")

    return report

def generate_evaluation_map(file_info):
    """
    Generates a map from agent_config.json paths to a list of corresponding episode_*.jsonl paths,
    handling nested subdirectories in cache. Now only processes cache.

    Args:
        file_info (dict): The report from check_evaluation_files.

    Returns:
        dict: A map where keys are paths to agent_config.json files and
              values are lists of paths to corresponding episode_*.jsonl files.
    """
    run_map = {}

    cache_location = file_info.get("cache_directory")

    if cache_location:
        print(f"Info: Attempting to generate map from cache location: {cache_location}")

        if os.path.isdir(cache_location):
            print(f"Info: Processing cache directory: {cache_location}")
            for dirpath, dirnames, filenames in os.walk(cache_location):
                if "agent_config.json" in filenames:
                    current_agent_config = os.path.join(dirpath, "agent_config.json")
                    episode_log_pattern = os.path.join(dirpath, "episode_*.jsonl")
                    episode_logs = sorted(glob.glob(episode_log_pattern))
                    
                    if episode_logs:
                        run_map[current_agent_config] = episode_logs
                        print(f"  Mapped in dir '{dirpath}': '{current_agent_config}' to {len(episode_logs)} log(s)")
        
        elif os.path.isfile(cache_location) and cache_location.endswith(".zip"):
            # This part of the logic might need review if cache_location from file_info
            # is always expected to be a directory after check_evaluation_files runs.
            # For now, keeping the zip processing logic if a .zip path is somehow provided.
            print(f"Info: Processing cache zip file: {cache_location}")
            files_by_internal_dir = {}
            try:
                with zipfile.ZipFile(cache_location, 'r') as zip_f:
                    for item_name_in_zip in zip_f.namelist():
                        normalized_item_path = item_name_in_zip.replace('\\\\', '/')
                        if normalized_item_path.endswith('/'): 
                            continue

                        internal_dir_path = os.path.dirname(normalized_item_path)
                        internal_filename = os.path.basename(normalized_item_path)
                        
                        if internal_dir_path not in files_by_internal_dir:
                            files_by_internal_dir[internal_dir_path] = []
                        files_by_internal_dir[internal_dir_path].append(internal_filename)
                
                for internal_dir, filenames_in_dir in files_by_internal_dir.items():
                    if "agent_config.json" in filenames_in_dir:
                        agent_config_zip_path = f"{internal_dir}/agent_config.json" if internal_dir else "agent_config.json"
                        episode_logs_zip_paths = []
                        for fname in filenames_in_dir:
                            if fname.startswith("episode_") and fname.endswith(".jsonl"):
                                log_path_in_zip = f"{internal_dir}/{fname}" if internal_dir else fname
                                episode_logs_zip_paths.append(log_path_in_zip)
                        
                        if episode_logs_zip_paths:
                            run_map[agent_config_zip_path] = sorted(episode_logs_zip_paths)
                            print(f"  Mapped in zip (dir '{internal_dir}'): '{agent_config_zip_path}' to {len(episode_logs_zip_paths)} log(s)")

            except zipfile.BadZipFile:
                print(f"Error: Could not read zip file {cache_location}. It may be corrupted.")
            except Exception as e:
                print(f"An error occurred while processing zip file {cache_location}: {e}")
        
        if run_map:
             print(f"Info: Map generation from cache completed.")
        else:
            # This warning will trigger if cache_location was valid but no mappings were found.
            print(f"Warning: Cache location '{cache_location}' was processed, but no valid agent_config to episode_log mappings were found.")
    else:
        # This implies file_info did not contain a "cache_directory"
        print("Info: No cache directory provided in file_info. No map will be generated from cache.")


    if not run_map:
        print("Warning: No evaluation map could be generated with the current settings and files.")
        
    return run_map

def _try_parse_json_string_for_function(value, field_name_for_error, log_file_basename, line_num_str):
    """
    Helper to parse a field that might be a JSON string.
    Used by the process_evaluation_run_map function.
    """
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value # Return original string if parsing fails
    return value # Return as is if not a string

def process_evaluation_run_map(evaluation_run_map):
    """
    Processes an evaluation_run_map to extract and structure data from
    agent_config.json and episode_*.jsonl files.

    Args:
        evaluation_run_map (dict): A map where keys are paths to agent_config.json
                                   files and values are lists of paths to
                                   corresponding episode_*.jsonl files.

    Returns:
        dict: A dictionary where keys are tuples (game_name, model_name,
              observation_mode, harness_bool) and values are lists of
              dictionaries, each representing a processed episode.
              Each episode dictionary contains:
              'episode_id', 'total_steps', 'total_reward',
              'total_perf_score', 'total_time_taken',
              'agent_observations' (list), 'infos' (list).
    """
    grouped_results = {}

    if not evaluation_run_map:
        print("Warning: The provided evaluation_run_map is empty. No processing will occur.")
        return grouped_results

    for agent_config_filepath, episode_log_filepaths in evaluation_run_map.items():
        # 1. Read and parse agent_config.json
        game_name = None
        model_name = None
        observation_mode = None
        harness_bool = None # Store harness as boolean

        if not os.path.isfile(agent_config_filepath):
            print(f"Error: Agent config file not found at '{agent_config_filepath}'. Skipping associated logs.")
            continue
        
        try:
            with open(agent_config_filepath, 'r') as f_agent_config:
                agent_config_data = json.load(f_agent_config)
            
            game_name = agent_config_data.get("game_name")
            model_name = agent_config_data.get("model_name")
            observation_mode = agent_config_data.get("observation_mode")
            harness_bool = agent_config_data.get("harness") # This should ideally be boolean (True/False)
            
            if harness_bool is None: # Handle if 'harness' key is missing
                print(f"Warning: 'harness' key missing in {agent_config_filepath}. Assuming False.")
                harness_bool = False

        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from agent config: '{agent_config_filepath}'. Skipping.")
            continue
        except Exception as e:
            print(f"Error reading agent config '{agent_config_filepath}': {e}. Skipping.")
            continue

        # Create the grouping key
        # Ensure all parts of the key are actual values to avoid issues with None in dict keys
        if game_name is None or model_name is None or observation_mode is None:
            print(f"Warning: Missing one or more key identifiers (game_name, model_name, observation_mode) in '{agent_config_filepath}'. Skipping this config and its logs.")
            continue
            
        group_key = (game_name, model_name, observation_mode, harness_bool)

        if group_key not in grouped_results:
            grouped_results[group_key] = []

        # 2. Process corresponding episode logs
        if not episode_log_filepaths:
            print(f"Info: No episode logs listed for agent config: '{agent_config_filepath}'.")
            continue

        for episode_log_path in episode_log_filepaths:
            if not os.path.isfile(episode_log_path):
                print(f"Warning: Episode log file not found: '{episode_log_path}'. Skipping.")
                continue

            log_basename = os.path.basename(episode_log_path)
            try:
                episode_id_str = log_basename.replace("episode_", "").replace("_log.jsonl", "")
            except Exception:
                episode_id_str = log_basename # Fallback

            ep_total_steps = 0
            ep_total_reward = 0
            ep_total_perf_score = 0.0
            ep_infos_list = []
            ep_agent_observations_list = []
            
            episode_lines_data = []
            sum_step_time_taken_s = 0.0
            found_step_time_taken_s_in_episode = False
            final_total_time_taken_from_last_step_info = None

            try:
                with open(episode_log_path, 'r') as f_episode_log:
                    for line_num, line in enumerate(f_episode_log):
                        try:
                            step_data = json.loads(line)
                            episode_lines_data.append(step_data)
                        except json.JSONDecodeError:
                            # print(f"Warning: Could not decode JSON from line {line_num + 1} in '{log_basename}'. Skipping line.")
                            continue
                
                if not episode_lines_data:
                    # print(f"Warning: No valid JSON lines found in '{log_basename}'. Skipping this log.")
                    continue

                ep_total_steps = len(episode_lines_data)

                for step_idx, step_data_dict in enumerate(episode_lines_data):
                    ep_total_reward += step_data_dict.get("reward", 0)
                    ep_total_perf_score += step_data_dict.get("perf_score", 0.0)
                    
                    raw_obs = step_data_dict.get("agent_observation")
                    parsed_obs = _try_parse_json_string_for_function(raw_obs, "agent_observation", log_basename, str(step_idx+1))
                    ep_agent_observations_list.append(parsed_obs)

                    raw_info = step_data_dict.get("info")
                    parsed_info = _try_parse_json_string_for_function(raw_info, "info", log_basename, str(step_idx+1))
                    ep_infos_list.append(parsed_info)

                    if "time_taken_s" in step_data_dict:
                        try:
                            sum_step_time_taken_s += float(step_data_dict["time_taken_s"])
                            found_step_time_taken_s_in_episode = True
                        except (ValueError, TypeError):
                            pass 

                    if step_idx == ep_total_steps - 1: # Last step
                        if isinstance(parsed_info, dict) and "total_time_taken" in parsed_info:
                            try:
                                final_total_time_taken_from_last_step_info = float(parsed_info["total_time_taken"])
                            except (ValueError, TypeError):
                                pass # Keep it None if not a valid float

                ep_total_time_taken = None
                if final_total_time_taken_from_last_step_info is not None:
                    ep_total_time_taken = final_total_time_taken_from_last_step_info
                elif found_step_time_taken_s_in_episode:
                    ep_total_time_taken = sum_step_time_taken_s
                
                episode_data_point = {
                    "episode_id": episode_id_str,
                    "total_steps": ep_total_steps,
                    "total_reward": ep_total_reward,
                    "total_perf_score": ep_total_perf_score,
                    "total_time_taken": ep_total_time_taken,
                    "agent_observations": ep_agent_observations_list,
                    "infos": ep_infos_list
                }
                grouped_results[group_key].append(episode_data_point)

            except Exception as e:
                print(f"Error processing episode log '{episode_log_path}': {e}")
                continue # Skip to next episode log file
                
    return grouped_results


def calculate_average_performance(extracted_results_map):
    """
    Calculates average performance metrics from the extracted_results_map.

    Args:
        extracted_results_map (dict): The output from process_evaluation_run_map,
                                      where keys are tuples (game_name, model_name,
                                      observation_mode, harness_bool) and values are
                                      lists of episode data dictionaries.

    Returns:
        list: A list of dictionaries, where each dictionary represents a group
              and its averaged 'total_steps', 'total_reward', 'total_perf_score',
              along with the group identifiers and 'num_episodes'.
    """
    averaged_data_list = []

    if not extracted_results_map:
        print("Warning: The provided extracted_results_map is empty. No averages will be calculated.")
        return averaged_data_list

    for group_key, episodes_list in extracted_results_map.items():
        game_name, model_name, observation_mode, harness = group_key
        
        if not episodes_list:
            print(f"Info: No episodes found for group: {group_key}. Skipping averaging for this group.")
            # Optionally, you could add an entry with 0 episodes and NaN/0 averages
            averaged_data_list.append({
                "game_name": game_name,
                "model_name": model_name,
                "observation_mode": observation_mode,
                "harness": harness,
                "num_episodes": 0,
                "avg_total_steps": 0, # or float('nan')
                "avg_total_reward": 0, # or float('nan')
                "avg_total_perf_score": 0, # or float('nan')
                "avg_total_time_taken": 0 # or float('nan')
            })
            continue

        num_episodes = len(episodes_list)
        sum_steps = 0
        sum_reward = 0
        sum_perf_score = 0
        sum_time_taken = 0
        episodes_with_time = 0


        for episode_data in episodes_list:
            sum_steps += episode_data.get("total_steps", 0)
            sum_reward += episode_data.get("total_reward", 0)
            sum_perf_score += episode_data.get("total_perf_score", 0.0)
            
            time_taken = episode_data.get("total_time_taken")
            if time_taken is not None:
                sum_time_taken += time_taken
                episodes_with_time += 1
        
        avg_steps = sum_steps / num_episodes if num_episodes > 0 else 0
        avg_reward = sum_reward / num_episodes if num_episodes > 0 else 0
        avg_perf_score = sum_perf_score / num_episodes if num_episodes > 0 else 0
        avg_time_taken = sum_time_taken / episodes_with_time if episodes_with_time > 0 else None


        averaged_data_list.append({
            "game_name": game_name,
            "model_name": model_name,
            "observation_mode": observation_mode,
            "harness": harness, # This is already boolean from the group_key
            "num_episodes": num_episodes,
            "avg_total_steps": avg_steps,
            "avg_total_reward": avg_reward,
            "avg_total_perf_score": avg_perf_score,
            "avg_total_time_taken": avg_time_taken
        })
        
    return averaged_data_list

def combine_benchmark_with_updates(df_benchmark, df_local_updates):
    """
    Combines a benchmark DataFrame with local update DataFrame.
    Updates existing models or adds new ones.
    """
    if df_local_updates is None or len(df_local_updates) == 0:
        print("No local updates provided. Returning benchmark DataFrame as is.")
        return df_benchmark.copy()
    
    if df_benchmark.empty:
        print("Benchmark DataFrame is empty. Returning local updates DataFrame as is.")
        return df_local_updates.copy()

    # Set the multi-index for both DataFrames to easily update and combine
    keys = ['model_name', 'harness', 'game_name']
    
    # Ensure keys are present in both dataframes
    for key_col in keys:
        if key_col not in df_benchmark.columns:
            print(f"Warning: Key column '{key_col}' not found in benchmark DataFrame. Returning local updates.")
            return df_local_updates.copy()
        if key_col not in df_local_updates.columns:
            print(f"Warning: Key column '{key_col}' not found in local updates DataFrame. Returning benchmark data.")
            return df_benchmark.copy()
            
    # Index the DataFrames by the key columns
    df_local_indexed = df_local_updates.set_index(keys)
    df_benchmark_indexed = df_benchmark.set_index(keys)
    
    # Update the benchmark data with local updates
    df_combined_indexed = df_benchmark_indexed.copy()
    df_combined_indexed.update(df_local_indexed)
    
    # Add rows that are entirely new in local updates
    new_indices = df_local_indexed.index.difference(df_benchmark_indexed.index)
    if not new_indices.empty:
        df_combined_indexed = df_combined_indexed.append(df_local_indexed.loc[new_indices])
    
    # Reset the index to return the DataFrame to its original form
    df_combined = df_combined_indexed.reset_index()
    
    print(f"Combined {len(df_local_updates)} local updates with benchmark data.")
    return df_combined

def update_benchmark_average(df_benchmark_average, df_update_model_average):
    """
    Updates benchmark average data with new model averages.
    If model exists, update its scores. If not, add it.
    """
    if df_update_model_average is None or df_update_model_average.empty:
        print("No model update averages provided. Returning benchmark averages as is.")
        return df_benchmark_average.copy()
        
    if df_benchmark_average.empty:
        print("Benchmark DataFrame is empty. Returning local updates DataFrame as is.")
        return df_update_model_average.copy()

    # Set the multi-index for both DataFrames to easily update and combine
    keys = ['model_name', 'harness']
    
    # Ensure keys are present in both dataframes
    for key_col in keys:
        if key_col not in df_benchmark_average.columns:
            print(f"Warning: Key column '{key_col}' not found in benchmark DataFrame. Returning local updates.")
            return df_update_model_average.copy()
        if key_col not in df_update_model_average.columns:
            print(f"Warning: Key column '{key_col}' not found in local updates DataFrame. Returning benchmark data.")
            return df_benchmark_average.copy()
            
    # Index the DataFrames by the key columns
    df_local_indexed = df_update_model_average.set_index(keys)
    df_benchmark_indexed = df_benchmark_average.set_index(keys)
    
    # Update the benchmark data with local updates
    df_combined_indexed = df_benchmark_indexed.copy()
    df_combined_indexed.update(df_local_indexed)
    
    # Add rows that are entirely new in local updates
    new_indices = df_local_indexed.index.difference(df_benchmark_indexed.index)
    if not new_indices.empty:
        df_combined_indexed = df_combined_indexed.append(df_local_indexed.loc[new_indices])
    
    # Reset the index to return the DataFrame to its original form
    df_combined = df_combined_indexed.reset_index()
    
    print(f"Updated benchmark averages with {len(df_update_model_average)} model averages.")
    return df_combined

def visualize_model_scores(score_dict, title="Model Performance Distribution"):
    """
    Visualizes model scores distribution in a bar chart.
    
    Args:
        score_dict: Dictionary with model names as keys and scores as values
        title: Title for the plot
    """
    if not score_dict:
        print("No scores to visualize.")
        return
        
    models = list(score_dict.keys())
    scores = list(score_dict.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(models, scores)
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, scores)):
        if isinstance(score, (int, float)) and not np.isnan(score):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{score:.2f}', ha='center', va='bottom')
    
    plt.title(title)
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def polynomial_analysis(x_values, y_values, max_degree=3):
    """
    Performs polynomial regression analysis.
    
    Args:
        x_values: List of x values
        y_values: List of corresponding y values  
        max_degree: Maximum degree of polynomial to test
        
    Returns:
        Dictionary with coefficients and R² scores for each degree
    """
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    
    x_array = np.array(x_values).reshape(-1, 1)
    y_array = np.array(y_values)
    
    results = {}
    
    for degree in range(1, max_degree + 1):
        # Create polynomial features
        poly_features = PolynomialFeatures(degree=degree)
        x_poly = poly_features.fit_transform(x_array)
        
        # Fit model
        model = LinearRegression()
        model.fit(x_poly, y_array)
        
        # Predict and calculate R²
        y_pred = model.predict(x_poly)
        r2 = r2_score(y_array, y_pred)
        
        results[degree] = {
            'coefficients': model.coef_,
            'intercept': model.intercept_,
            'r2_score': r2
        }
    
    return results

def factorize_polynomial(coefficients):
    """
    Attempts to factorize a polynomial given its coefficients.
    
    Args:
        coefficients: List of polynomial coefficients in ascending order of powers
        
    Returns:
        Factored form representation as string
    """
    # Remove leading zeros
    coeffs = np.array(coefficients)
    coeffs = coeffs[coeffs != 0] if len(coeffs) > 1 else coeffs
    
    if len(coeffs) <= 1:
        return f"{coeffs[0] if len(coeffs) > 0 else 0}"
    
    # Find roots of the polynomial
    try:
        roots = np.roots(coeffs[::-1])  # np.roots expects coefficients in descending order
        
        # Check if roots are real
        real_roots = []
        for root in roots:
            if np.isreal(root):
                real_roots.append(np.real(root))
        
        if len(real_roots) == len(roots):
            # All roots are real, can factorize
            factored = f"{coeffs[-1]}"  # Leading coefficient
            for root in real_roots:
                if root >= 0:
                    factored += f" * (x - {root:.3f})"
                else:
                    factored += f" * (x + {abs(root):.3f})"
            return factored
        else:
            return "Cannot factorize: complex roots present"
            
    except Exception as e:
        return f"Factorization failed: {str(e)}"

def normalize_column_safely(column_series):
    numeric_series = pd.to_numeric(column_series, errors='coerce')
    min_val = numeric_series.min()
    max_val = numeric_series.max()

    if pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
        result_series = pd.Series([pd.NA] * len(numeric_series), index=numeric_series.index, dtype=float)
        valid_numeric_mask = numeric_series.notna()
        result_series[valid_numeric_mask] = 1.0 # Default to 1 if no range or all same
        return result_series
    else:
        return (numeric_series - min_val) / (max_val - min_val) * 99 + 1

def hex_to_rgba(hex_color, alpha=0.2):
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f'rgba({r}, {g}, {b}, {alpha})'

def get_random_color(existing_colors_hex: set) -> str:
    """Generates a random hex color string, trying to avoid existing ones."""
    attempts = 0
    while attempts < 100:
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        hex_color = f'#{r:02x}{g:02x}{b:02x}'
        if hex_color not in existing_colors_hex:
            return hex_color
        attempts += 1
    return f'#{random.randint(0,255):02x}{random.randint(0,255):02x}{random.randint(0,255):02x}'

def prepare_dataframe_for_plots(source_df: pd.DataFrame, selected_games: list, selected_models: list, harness_bool_to_use: bool) -> pd.DataFrame:
    if source_df.empty:
        print("Input source_df is empty. Cannot prepare for plots.")
        return pd.DataFrame()
    
    # Filter by harness, games, and models
    df_filtered = source_df[
        (source_df['harness'] == harness_bool_to_use) & 
        (source_df['game_name'].isin(selected_games)) &
        (source_df['model_name'].isin(selected_models))
    ].copy()
    
    if df_filtered.empty:
        print(f"Warning: No data found for harness={harness_bool_to_use}, selected games and models.")
        return pd.DataFrame()
    
    # Pivot the data
    try:
        df_pivot = df_filtered.pivot_table(
            index='model_name', columns='game_name', values='avg_score_from_rank'
        ).reset_index().rename(columns={'model_name': 'Player'})
    except Exception as e:
        print(f"Error during pivoting: {e}")
        return pd.DataFrame()
    
    if df_pivot.empty:
        print(f"Pivoted DataFrame is empty for harness={harness_bool_to_use}.")
        return pd.DataFrame()
    
    # Normalize each game column using the provided function
    for game_name in selected_games:
        if game_name in df_pivot.columns:
            df_pivot[game_name] = normalize_column_safely(df_pivot[game_name])
    
    return df_pivot

def create_comparison_radar_chart(df: pd.DataFrame, model_colors: dict, selected_games: list, harness_status_str: str, highlight_models: list = None):
    import plotly.graph_objects as go
    
    if df.empty:
        return go.Figure().update_layout(title_text=f"No data for Radar Chart ({harness_status_str})")
    
    # Check available game columns
    available_games = [game for game in selected_games if game in df.columns]
    if not available_games:
        return go.Figure().update_layout(title_text=f"No game columns for Radar Chart ({harness_status_str})")
    
    # Fill NaN values with 1
    for game in available_games:
        df[game] = df[game].fillna(1)
    
    fig = go.Figure()
    
    if 'Player' not in df.columns:
        print("Error: 'Player' column not found for radar chart.")
        return fig
    
    sorted_players = sorted(df['Player'].unique())
    
    for player in sorted_players:
        player_row_df = df[df['Player'] == player]
        if player_row_df.empty:
            continue
        
        player_row = player_row_df.iloc[0]
        r_values = [player_row.get(game, 1) for game in available_games]
        r_values = [val if pd.notna(val) else 1 for val in r_values]
        
        is_highlighted = highlight_models and player in highlight_models
        model_color_hex = model_colors.get(player, '#808080')
        if not isinstance(model_color_hex, str) or not model_color_hex.startswith('#'):
            model_color_hex = '#808080'
        
        line_props = dict(color='red', width=3) if is_highlighted else dict(color=model_color_hex, width=1.5)
        marker_props = dict(color='red', size=8, line=dict(color='darkred', width=2)) if is_highlighted else dict(color=model_color_hex, size=4, line=dict(color='#B0B0B0', width=1))
        fill_color = 'rgba(255,0,0,0.4)' if is_highlighted else hex_to_rgba(model_color_hex, 0.2)
        opacity = 0.9 if is_highlighted else 0.7
        
        fig.add_trace(go.Scatterpolar(
            r=r_values + [r_values[0]],
            theta=available_games + [available_games[0]],
            mode='lines+markers',
            name=player,
            line=line_props,
            marker=marker_props,
            fill='toself',
            fillcolor=fill_color,
            opacity=opacity,
            hovertemplate=f'<b>{player}</b><br>Game: %{{theta}}<br>Score: %{{r:.2f}} <extra></extra>'
        ))
    
    fig.update_layout(
        title_text=f'Model Performance Radar ({harness_status_str}) - Normalized Scores (1-100 Scale)',
        title_x=0.5,
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[1, 100],
                gridcolor='lightgray',
                tickvals=list(range(10, 101, 10)),
                tickformat=".0f"
            ),
            angularaxis=dict(
                tickfont=dict(size=10),
                gridcolor='lightgray'
            )
        ),
        legend=dict(
            title="Models",
            orientation="v",
            yanchor="top",
            y=1.02,
            xanchor="left",
            x=1.05
        ),
        width=800,
        height=600,
        margin=dict(l=100, r=200, t=100, b=50)
    )
    
    return fig

def create_comparison_bar_chart(df: pd.DataFrame, model_colors: dict, selected_games: list, harness_status_str: str, highlight_models: list = None):
    import plotly.graph_objects as go
    
    if df.empty:
        return go.Figure().update_layout(title_text=f"No data for Bar Chart ({harness_status_str})")
    
    # Check available game columns
    available_games = [game for game in selected_games if game in df.columns]
    if not available_games:
        return go.Figure().update_layout(title_text=f"No game columns for Bar Chart ({harness_status_str})")
    
    # Fill NaN values with 1
    for game in available_games:
        df[game] = df[game].fillna(1)
    
    fig = go.Figure()
    
    if 'Player' not in df.columns:
        print("Error: 'Player' column not found for bar chart.")
        return fig
    
    sorted_players = sorted(df['Player'].unique())
    
    for player_name in sorted_players:
        player_row_df = df[df['Player'] == player_name]
        if player_row_df.empty:
            continue
        
        player_row = player_row_df.iloc[0]
        y_values = [player_row.get(game, 1) for game in available_games]
        y_values = [val if pd.notna(val) else 1 for val in y_values]
        
        model_color_hex = model_colors.get(player_name, '#808080')
        if not isinstance(model_color_hex, str) or not model_color_hex.startswith('#'):
            model_color_hex = '#808080'
        
        is_highlighted = highlight_models and player_name in highlight_models
        opacity = 1.0 if is_highlighted else 0.7
        line_width = 2 if is_highlighted else 0
        
        fig.add_trace(go.Bar(
            name=player_name,
            x=available_games,
            y=y_values,
            marker_color=model_color_hex,
            opacity=opacity,
            marker_line_width=line_width,
            marker_line_color='red' if is_highlighted else '#333333',
            hovertemplate=f'<b>{player_name}</b><br>Game: %{{x}}<br>Score: %{{y:.2f}} <extra></extra>'
        ))
    
    fig.update_layout(
        barmode='group',
        title_text=f'Model Performance Comparison ({harness_status_str}) - Normalized Scores (1-100 Scale)',
        title_x=0.5,
        xaxis_title="Game",
        yaxis_title="Normalized Score (1-100)",
        legend_title_text='Models',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.02,
            xanchor="left",
            x=1.02
        ),
        width=max(800, 150 * len(available_games) * min(5, len(sorted_players)) + 250),
        height=600,
        margin=dict(l=50, r=200, t=100, b=50)
    )
    fig.update_yaxes(range=[0, 105])
    
    return fig

def create_game_specific_horizontal_bar_charts(df: pd.DataFrame, model_colors: dict, selected_games: list, harness_status_str: str, highlight_models: list = None):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    if df.empty:
        print(f"No data for Game-Specific Bar Charts ({harness_status_str})")
        return []
    
    # Check available game columns
    available_games = [game for game in selected_games if game in df.columns]
    if not available_games:
        print(f"No game columns for Game-Specific Bar Charts ({harness_status_str})")
        return []
    
    # Fill NaN values with 1
    for game in available_games:
        df[game] = df[game].fillna(1)
    
    if 'Player' not in df.columns:
        print("Error: 'Player' column not found for game-specific bar charts.")
        return []
    
    figures = []
    
    # Loop through each game to create separate horizontal bar charts
    for game_name in available_games:
        # Get data for this specific game
        game_data = []
        for _, row in df.iterrows():
            player_name = row['Player']
            score = row.get(game_name, 1)
            if pd.notna(score):
                game_data.append({
                    'player': player_name,
                    'score': score
                })
        
        if not game_data:
            print(f"No data for game: {game_name}")
            continue
        
        # Sort by score (high to low)
        game_data.sort(key=lambda x: x['score'], reverse=True)
        
        # Create horizontal bar chart for this game
        fig = go.Figure()
        
        players = [item['player'] for item in game_data]
        scores = [item['score'] for item in game_data]
        
        # Reverse the order so highest scores appear at the top
        players_reversed = players[::-1]
        scores_reversed = scores[::-1]
        
        # Create bars with colors and highlighting - using the defined model colors
        bar_colors = []
        bar_opacities = []
        bar_line_widths = []
        bar_line_colors = []
        
        for player in players_reversed:
            # Get the color from the defined model_colors dictionary
            model_color_hex = model_colors.get(player, '#808080')  # Default gray if not found
            
            # Ensure it's a valid hex color
            if not isinstance(model_color_hex, str) or not model_color_hex.startswith('#'):
                model_color_hex = '#808080'
            
            is_highlighted = highlight_models and player in highlight_models
            
            bar_colors.append(model_color_hex)
            bar_opacities.append(1.0 if is_highlighted else 0.7)
            bar_line_widths.append(3 if is_highlighted else 0)
            bar_line_colors.append('red' if is_highlighted else model_color_hex)  # Use model color for border too
            
            # Debug print to verify colors are being used
            # print(f"Player: {player}, Color: {model_color_hex}, Highlighted: {is_highlighted}")
        
        fig.add_trace(go.Bar(
            x=scores_reversed,
            y=players_reversed,
            orientation='h',
            marker=dict(
                color=bar_colors,  # Using the defined colors from model_colors_for_plots
                opacity=bar_opacities,
                line=dict(
                    width=bar_line_widths,
                    color=bar_line_colors
                )
            ),
            hovertemplate='<b>%{y}</b><br>Score: %{x:.2f}<extra></extra>',
            showlegend=False
        ))
        
        fig.update_layout(
            title_text=f'{game_name} Performance ({harness_status_str}) - Sorted High to Low',
            title_x=0.5,
            xaxis_title="Normalized Score (1-100)",
            yaxis_title="Models",
            width=800,
            height=max(400, 40 * len(players) + 100),  # Dynamic height based on number of players
            margin=dict(l=150, r=50, t=80, b=50),
            yaxis=dict(
                categoryorder='array',
                categoryarray=players_reversed  # Reversed order so highest appears at top
            )
        )
        fig.update_xaxes(range=[0, 105])
        
        figures.append(fig)
    
    return figures

def generate_model_performance_plots(final_updated_df: pd.DataFrame, average_results: pd.DataFrame = None, model_colors_url: str = None):
    """
    Generates radar and bar charts for model performance comparison.
    
    Args:
        final_updated_df: DataFrame with columns ['model_name', 'harness', 'game_name', 'avg_score_from_rank']
        average_results: Optional DataFrame with local results to highlight
        model_colors_url: Optional URL to fetch model colors JSON
    """
    import plotly.graph_objects as go
    import requests
    import json
    import random
    
    # Default model and game names
    show_games = list(final_updated_df['game_name'].unique())
    show_models = list(final_updated_df['model_name'].unique())

    highlight_models = []
    if average_results is not None and not average_results.empty:
        if 'model_name' in average_results.columns:
            highlight_models = list(average_results['model_name'].unique())

    # Model colors configuration
    model_colors_for_plots = {}
    used_colors_from_json = set()

    # Load model colors
    if model_colors_url:
        print(f"Attempting to load model colors from URL...")
        try:
            response = requests.get(model_colors_url)
            response.raise_for_status()
            model_colors_from_url = response.json()
            model_colors_for_plots.update(model_colors_from_url)
            used_colors_from_json.update(model_colors_from_url.values())
            print(f"Successfully loaded model colors from URL. {len(model_colors_for_plots)} colors loaded.")
        except:
            print("Warning: Could not download model colors. Will use random colors for all models.")

    # Assign colors to all models
    if not final_updated_df.empty:
        all_models_in_data = final_updated_df['model_name'].unique()
        for model_name_from_data in all_models_in_data:
            if model_name_from_data not in model_colors_for_plots:
                new_color = get_random_color(used_colors_from_json)
                model_colors_for_plots[model_name_from_data] = new_color
                used_colors_from_json.add(new_color)

        print(f"Models to highlight: {highlight_models}")
        print(f"Games to show: {show_games}")
        print(f"Models to show: {show_models}")

        # Generate and show plots for both harness conditions
        for harness_boolean_val, harness_title_str in [(True, "Harness True"), (False, "Harness False")]:
            print(f"\n--- Generating plots for {harness_title_str} ---")
            
            df_for_plotting = prepare_dataframe_for_plots(
                source_df=final_updated_df,
                selected_games=show_games,
                selected_models=show_models,
                harness_bool_to_use=harness_boolean_val
            )
            
            if df_for_plotting.empty:
                print(f"No data to plot for {harness_title_str}. Skipping.")
                continue
            
            # Create and show radar chart
            fig_radar = create_comparison_radar_chart(
                df=df_for_plotting,
                model_colors=model_colors_for_plots,
                selected_games=show_games,
                harness_status_str=harness_title_str,
                highlight_models=highlight_models
            )
            print(f"Displaying Radar Chart for {harness_title_str}:")
            fig_radar.show()

            # Create and show game-specific horizontal bar charts
            game_bar_figures = create_game_specific_horizontal_bar_charts(
                df=df_for_plotting,
                model_colors=model_colors_for_plots,
                selected_games=show_games,
                harness_status_str=harness_title_str,
                highlight_models=highlight_models
            )
            
            for i, fig_bar in enumerate(game_bar_figures):
                game_name = show_games[i] if i < len(show_games) else f"Game {i+1}"
                print(f"Displaying Horizontal Bar Chart for {game_name} ({harness_title_str}):")
                fig_bar.show()

    else:
        print("'final_updated_df' variable not found or is empty. Cannot generate plots.")

def convert_local_averages_to_dataframe(local_averaged_results_list: list) -> pd.DataFrame:
    """
    Converts a list of local averaged results (dictionaries) into a Pandas DataFrame
    with columns 'model_name', 'harness', 'game_name', and 'avg_score_from_rank'.
    The 'avg_score_from_rank' column is populated from the 'avg_total_perf_score'
    field of the input dictionaries.
    """
    if not local_averaged_results_list:
        print("Warning: The local_averaged_results_list is empty. Returning an empty DataFrame.")
        return pd.DataFrame(columns=['model_name', 'harness', 'game_name', 'avg_score_from_rank'])

    # Create a list of dictionaries with only the desired columns and new key name
    formatted_list_for_df = []
    for item in local_averaged_results_list:
        if not isinstance(item, dict):
            print(f"Warning: Skipping an item in local_averaged_results_list as it's not a dictionary: {item}")
            continue
        
        # Ensure all necessary keys are present in the item before trying to access them
        required_keys = ['model_name', 'harness', 'game_name', 'avg_total_perf_score']
        if not all(key in item for key in required_keys):
            print(f"Warning: Skipping an item due to missing one or more required keys ({', '.join(required_keys)}): {item}")
            continue
            
        formatted_list_for_df.append({
            "model_name": item['model_name'],
            "harness": item['harness'],  # Assuming this is already boolean
            "game_name": item['game_name'],
            "avg_score_from_rank": item['avg_total_perf_score'] # Mapping local score to this new column name
        })
    
    if not formatted_list_for_df:
        print("Warning: No valid data could be extracted from local_averaged_results_list. Returning an empty DataFrame.")
        return pd.DataFrame(columns=['model_name', 'harness', 'game_name', 'avg_score_from_rank'])

    df_local_formatted = pd.DataFrame(formatted_list_for_df)
    print(f"Created DataFrame from local averaged results with {len(df_local_formatted)} rows and columns: {list(df_local_formatted.columns)}")
    return df_local_formatted

def load_and_average_benchmark_rank_data(url: str) -> pd.DataFrame:
    """
    Fetches benchmark rank data from a URL, calculates average scores,
    and returns it as a Pandas DataFrame.
    The 'harness' column in the output DataFrame will be boolean.
    'avg_score_from_rank' is the column with the calculated average from the benchmark file.
    """
    benchmark_rank_data_dict = {}
    print(f"Attempting to download benchmark data from: {url}")
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        benchmark_rank_data_dict = response.json() # requests can directly parse to dict
        print("Benchmark data successfully downloaded and parsed.")
    except requests.exceptions.RequestException as e:
        print(f"Error: Could not download data from {url}. Exception: {e}")
        return pd.DataFrame() # Return empty DataFrame on error
    except json.JSONDecodeError as e: # response.json() can also raise this if content is not valid JSON
        print(f"Error: Could not parse JSON from the URL response. Exception: {e}")
        return pd.DataFrame() # Return empty DataFrame on error
    except Exception as e: # Catch any other unexpected errors
        print(f"An unexpected error occurred while fetching or loading benchmark data: {e}")
        return pd.DataFrame()

    # Calculate averages
    rank_averages_list_of_dicts = []
    if not benchmark_rank_data_dict:
        print("Warning: Empty benchmark_rank_data_dict received. Cannot calculate averages from it.")
        return pd.DataFrame()

    for model_name, harness_data in benchmark_rank_data_dict.items():
        if not isinstance(harness_data, dict):
            # print(f"Skipping model '{model_name}' in benchmark data: harness_data is not a dict.")
            continue
        for harness_status_str, game_data in harness_data.items():
            if not isinstance(game_data, dict):
                # print(f"Skipping harness '{harness_status_str}' for model '{model_name}': game_data is not a dict.")
                continue
            
            # Convert harness status string to boolean for consistent merging
            harness_bool = True if harness_status_str == "harness_true" else False
            
            for game_name, scores_list in game_data.items():
                avg_score = pd.NA # Use pd.NA for missing numeric data
                if scores_list and isinstance(scores_list, list):
                    # Filter out non-numeric scores before averaging
                    numeric_scores = [s for s in scores_list if isinstance(s, (int, float))]
                    if numeric_scores: # Ensure there's at least one numeric score
                        avg_score = sum(numeric_scores) / len(numeric_scores)
                    # else:
                        # print(f"No numeric scores found for {model_name}/{harness_status_str}/{game_name} in benchmark.")
                # else:
                    # print(f"Scores_list is empty or not a list for {model_name}/{harness_status_str}/{game_name} in benchmark.")
                    
                rank_averages_list_of_dicts.append({
                    "model_name": model_name,
                    "harness": harness_bool, 
                    "game_name": game_name,
                    "avg_score_from_rank": avg_score  # This will be the averaged score from the rank file
                })
    
    if not rank_averages_list_of_dicts:
        print("No averages could be calculated from the benchmark rank data (list is empty).")
        return pd.DataFrame()
        
    df_benchmark_avg = pd.DataFrame(rank_averages_list_of_dicts)
    print(f"Created DataFrame from benchmark rank averages with {len(df_benchmark_avg)} rows.")
    return df_benchmark_avg

def combine_and_update_averages(df_benchmark: pd.DataFrame, df_local_updates: pd.DataFrame) -> pd.DataFrame:
    """
    Combines benchmark averages with local update averages.
    Local updates will override benchmark data for the same (model_name, harness, game_name) keys.
    All unique entries from both DataFrames will be included.

    Args:
        df_benchmark: DataFrame with columns ['model_name', 'harness', 'game_name', 'avg_score_from_rank']
                      from the benchmark data.
        df_local_updates: DataFrame with columns ['model_name', 'harness', 'game_name', 'avg_score_from_rank']
                          from the local averaged data (where 'avg_score_from_rank' holds local scores).

    Returns:
        A Pandas DataFrame with the combined and updated averages.
    """
    if df_local_updates.empty and df_benchmark.empty:
        print("Both benchmark and local update DataFrames are empty. Returning an empty DataFrame.")
        return pd.DataFrame(columns=['model_name', 'harness', 'game_name', 'avg_score_from_rank'])

    if df_local_updates.empty:
        print("Local updates DataFrame is empty. Returning benchmark DataFrame as is.")
        return df_benchmark.copy() # Return a copy to avoid modifying the original

    if df_benchmark.empty:
        print("Benchmark DataFrame is empty. Returning local updates DataFrame as is.")
        return df_local_updates.copy()

    # Set the multi-index for both DataFrames to easily update and combine
    keys = ['model_name', 'harness', 'game_name']
    
    # Ensure keys are present in both dataframes
    for key_col in keys:
        if key_col not in df_benchmark.columns:
            print(f"Warning: Key column '{key_col}' not found in benchmark DataFrame. Returning local updates.")
            return df_local_updates.copy()
        if key_col not in df_local_updates.columns:
            print(f"Warning: Key column '{key_col}' not found in local updates DataFrame. Returning benchmark data.")
            return df_benchmark.copy()
            
    df_benchmark_indexed = df_benchmark.set_index(keys)
    df_local_updates_indexed = df_local_updates.set_index(keys)

    # Update the benchmark data with local data.
    # Rows in local_updates will overwrite rows in benchmark_indexed if the index matches.
    # Rows in local_updates not in benchmark_indexed will be added.
    df_combined_indexed = df_benchmark_indexed.copy() # Start with a copy of benchmark
    df_combined_indexed.update(df_local_updates_indexed) # Update with local, overwrites existing, does not add new from local

    # Add new rows from local_updates_indexed that were not in df_benchmark_indexed
    # This requires finding rows in local that are not in benchmark and concatenating.
    # A more straightforward way for 'outer join' like behavior with local priority is:
    
    # Re-think: A simpler approach for "local overrides, then add unique from benchmark"
    # 1. Take all local data.
    # 2. Find benchmark data that is NOT in local (based on keys) and append it.

    # Set index on local updates
    df_local_final = df_local_updates.set_index(keys)

    # Filter benchmark data to get only rows NOT present in local updates
    # This uses the index of df_local_final to exclude matching rows from df_benchmark
    df_benchmark_only = df_benchmark.set_index(keys)[~df_benchmark.set_index(keys).index.isin(df_local_final.index)]
    
    # Concatenate local data (which has priority) with the unique benchmark data
    df_final_combined = pd.concat([df_local_final, df_benchmark_only])
    
    # Reset index to get 'model_name', 'harness', 'game_name' back as columns
    df_final_combined = df_final_combined.reset_index()

    print(f"Combined DataFrame created. Local updates took priority. Final rows: {len(df_final_combined)}")
    return df_final_combined

def run_polynomial_analysis_notebook(
    final_updated_df: pd.DataFrame, 
    OTHER_TASK_RANK_URL: str,
    DEFAULT_MODEL_MATCH: list = None,
    selected_games: list = None, 
    selected_models: list = None, 
    harness_filter: bool = None, 
    max_degree: int = 3
):
    """
    Comprehensive polynomial analysis function that handles model ranking data properly.
    Modified to work seamlessly without file I/O.
    
    Args:
        final_updated_df: DataFrame with model performance data
        OTHER_TASK_RANK_URL: URL to fetch other task ranking data
        DEFAULT_MODEL_MATCH: List of default models for analysis
        selected_games: List of games to analyze (if None, uses all games)
        selected_models: List of models to analyze (if None, uses DEFAULT_MODEL_MATCH)
        harness_filter: Boolean to filter by harness (if None, analyzes both)
        max_degree: Maximum polynomial degree to test
        
    Returns:
        Dictionary with comprehensive polynomial analysis results
    """
    from itertools import product
    from scipy.stats import rankdata
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    import requests
    import json
    
    # Use provided DEFAULT_MODEL_MATCH or fallback to predefined list
    if DEFAULT_MODEL_MATCH is None:
        DEFAULT_MODELS_FOR_OTHER_TASKS = [
            "claude-3-5-sonnet-20241022",
            "claude-3-7-sonnet-20250219 (thinking)",
            "gemini-2.5-pro-preview-05-06", 
            "llama-4-maverick-17b-128e-instruct-fp8",
            "gpt-4o-2024-11-20",
            "o1-2024-12-17",
            "o3-2025-04-16",
            "o4-mini-2025-04-16"
        ]
    else:
        DEFAULT_MODELS_FOR_OTHER_TASKS = DEFAULT_MODEL_MATCH
    
    def build_batches(rank_matrices):
        batches = []
        valid_matrices = [m for m in rank_matrices if isinstance(m, np.ndarray) and m.shape[0] > 0]
        if len(valid_matrices) != len(rank_matrices):
            print("Warning: Invalid/empty rank matrices for build_batches. Using valid ones.")
            if not valid_matrices: return []
            rank_matrices = valid_matrices
        for row_idxs in product(*(range(m.shape[0]) for m in rank_matrices)):
            batches.append(np.stack([m[i] for m, i in zip(rank_matrices, row_idxs)], axis=0).T)
        return batches

    class RankingPredictor:
        def __init__(self, degree: int, lr: float = 1e-4, epochs: int = 1000, shuffle: bool = True, scale: float = 8.0):
            self.degree, self.lr, self.epochs, self.shuffle, self.scale = degree, lr, epochs, shuffle, scale
            self._phi = PolynomialFeatures(degree=degree, include_bias=True)
            self._w = None
        
        def _transform(self, R, fit=False): 
            return self._phi.fit_transform(R / self.scale) if fit else self._phi.transform(R / self.scale)
        
        def fit(self, R_list, G_list):
            if not isinstance(R_list, (list, tuple)): R_list, G_list = [R_list], [G_list]
            valid_indices = [i for i, r_matrix in enumerate(R_list) if r_matrix.shape[0] > 0]
            if not valid_indices:
                print("Error: All R matrices in R_list are empty. Cannot fit PolynomialFeatures.")
                self._w = np.zeros(1) 
                return
            R_list_filtered = [R_list[i] for i in valid_indices]
            G_list_filtered = [G_list[i] for i in valid_indices]
            if not R_list_filtered:
                print("Error: R_list_filtered is empty after checking individual matrices. Cannot fit.")
                self._w = np.zeros(1)
                return
            X_list = [self._transform(R_list_filtered[0], fit=True)] + \
                     [self._transform(R) for R in R_list_filtered[1:]]
            self._w = np.zeros(X_list[0].shape[1])
            for _ in range(self.epochs):
                order = np.random.permutation(len(X_list)) if self.shuffle else range(len(X_list))
                for i in order:
                    X, G = X_list[i], G_list_filtered[i]
                    if X.shape[0] == 0: continue 
                    self._w -= self.lr * (2.0 * X.T @ (X @ self._w - G))
                    self._w = np.maximum(self._w, 0.0)
        
        def predict(self, R):
            if self._w is None: raise ValueError("Predictor not fitted.")
            if R.shape[0] == 0: return np.array([])
            return self._transform(R) @ self._w
        
        def evaluate(self, R, G, norm_type='L2', normalization='mean'):
            if R.shape[0] == 0 or G.shape[0] == 0: 
                return (np.nan, 0.0) if self.degree == 1 else np.nan
            pred = self.predict(R)
            if pred.shape[0] == 0 : 
                 return (np.nan, 0.0) if self.degree == 1 else np.nan
            min_len = min(len(pred), len(G))
            pred = pred[:min_len]
            G = G[:min_len]
            if min_len == 0: 
                return (np.nan, 0.0) if self.degree == 1 else np.nan
            if norm_type == 'L1': res = np.sum(np.abs(pred - G))
            elif norm_type == 'Linf': res = np.max(np.abs(pred - G))
            elif norm_type == 'L0.5': res = np.sum(np.abs(pred - G) ** 0.5) ** 2
            else: res = np.linalg.norm(pred - G, ord=2)
            eps = np.finfo(float).eps
            if normalization == 'mean': factor = np.mean(G) + eps if len(G) > 0 else eps
            elif normalization == 'max': factor = np.max(G) + eps if len(G) > 0 else eps
            elif normalization == 'std': factor = np.std(G) + eps if len(G) > 1 else eps
            elif normalization == 'range': factor = (np.max(G) - np.min(G)) + eps if len(G) > 0 else eps
            else: factor = 1
            norm_res = res / factor if factor != 0 else res 
            if self.degree == 1:
                if len(pred) < 2 or len(G) < 2: r = np.nan
                else:
                    pred_std, G_std = np.std(pred) + eps, np.std(G) + eps
                    r = 0.0 if pred_std * G_std == 0 else np.clip(np.corrcoef(pred, G)[0, 1], -1.0, 1.0)
                return norm_res, r
            return norm_res

    def _scores_to_ranks(scores_list, higher_is_better=True):
        s = np.array(scores_list, dtype=float); m = np.isnan(s)
        s[m] = -np.inf if higher_is_better else np.inf
        if np.all(np.isinf(s)) or np.all(np.isnan(s)) or len(s[~np.isinf(s) & ~np.isnan(s)]) == 0 :
            return np.full(len(s), np.nan)
        ranks = rankdata(-s if higher_is_better else s, method='average')
        return ranks

    def _get_ranks_for_game(game_name, model_names, model_perf_data, harness_status_str):
        scores = []
        for model_name in model_names:
            try:
                harness_key = harness_status_str
                game_scores = model_perf_data.get(model_name, {}).get(harness_key, {}).get(game_name, [])
                numeric_scores = [s for s in game_scores if isinstance(s, (int, float)) and pd.notna(s)]
                scores.append(np.mean(numeric_scores) if numeric_scores else np.nan)
            except Exception as e:
                scores.append(np.nan)
        raw_ranks = _scores_to_ranks(scores, higher_is_better=True)
        return raw_ranks

    def load_other_task_ranks_from_url(url, model_names_list_for_matching):
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"Warning: Could not load other task ranks from {url}: {e}")
            # Return dummy data for demonstration
            num_models = len(model_names_list_for_matching)
            return {
                "knowledge": np.random.rand(3, num_models),
                "math": np.random.rand(3, num_models),
                "coding": np.random.rand(3, num_models),
                "visual": np.random.rand(3, num_models),
                "puzzle": np.random.rand(3, num_models)
            }
        
        num_m_expected = len(model_names_list_for_matching)
        if num_m_expected == 0:
            print("Warning: model_names_list_for_matching is empty in load_other_task_ranks_from_url.")
            return {}
        if not data: return {}
        processed_ranks = {}
        for category, rank_vectors in data.items():
            if not isinstance(rank_vectors, list): continue
            valid_category_vectors = []
            for single_rank_vector in rank_vectors:
                if isinstance(single_rank_vector, list) and len(single_rank_vector) == num_m_expected:
                    try:
                        valid_category_vectors.append([float(x) for x in single_rank_vector])
                    except (ValueError, TypeError): pass
            if valid_category_vectors:
                processed_ranks[category] = np.array(valid_category_vectors, dtype=float)
        if not processed_ranks:
            print(f"Warning: No categories from {url} had rank vectors matching {num_m_expected} models.")
        return processed_ranks

    def prepare_model_perf_data_from_df(df_input):
        model_perf_data = {}
        for _, row in df_input.iterrows():
            model_name = row['model_name']
            harness_status = bool(row['harness']) 
            game_name = row['game_name']
            score = row['avg_score_from_rank'] 
            harness_key = 'harness_true' if harness_status else 'harness_false'
            model_perf_data.setdefault(model_name, {})
            model_perf_data[model_name].setdefault(harness_key, {})
            model_perf_data[model_name][harness_key].setdefault(game_name, [])
            if pd.notna(score):
                model_perf_data[model_name][harness_key][game_name].append(float(score))
            else:
                 model_perf_data[model_name][harness_key][game_name].append(np.nan)
        return model_perf_data

    # Main analysis logic
    working_df = final_updated_df.copy()
    
    # Set up model order - use selected_models if provided, otherwise use DEFAULT_MODELS_FOR_OTHER_TASKS
    if selected_models:
        working_df = working_df[working_df['model_name'].isin(selected_models)].copy()
        current_model_order = selected_models
    else:
        current_model_order = DEFAULT_MODELS_FOR_OTHER_TASKS
        working_df = working_df[working_df['model_name'].isin(current_model_order)].copy()

    if not current_model_order:
        print("Error: No models selected for analysis. Exiting.")
        return {}

    all_model_perf_data = prepare_model_perf_data_from_df(working_df)
    all_other_task_ranks = load_other_task_ranks_from_url(OTHER_TASK_RANK_URL, current_model_order)

    if not all_model_perf_data:
        print("Error: Model performance data is empty. Exiting analysis.")
        return {}
    if not all_other_task_ranks:
        print("Error: Other task ranks data is empty. Exiting analysis.")
        return {}

    # Set up target games
    if selected_games:
        target_games_to_process = selected_games
    else:
        available_games = set()
        harness_status_for_G = "harness_true" if harness_filter is None or harness_filter else "harness_false"
        for model_name in all_model_perf_data:
            if harness_status_for_G in all_model_perf_data.get(model_name, {}):
                available_games.update(all_model_perf_data[model_name][harness_status_for_G].keys())
        target_games_to_process = sorted(list(available_games)) if available_games else []

    # Category combinations
    category_combinations = [
        {"name": "Comb1_KnowPuzVisMathCode", "keys": ["knowledge", "puzzle", "visual", "math", "coding"]},
        {"name": "Comb2_KnowVisMathCode", "keys": ["knowledge", "visual", "math", "coding"]},
        {"name": "Comb3_LangPhyVisMathCode", "keys": ["language", "physics", "visual", "math", "coding"]},
        {"name": "Comb4_KnowMathCode", "keys": ["knowledge", "math", "coding"]}
    ]

    valid_category_combinations = []
    for comb in category_combinations:
        if comb["keys"] and all(key in all_other_task_ranks for key in comb["keys"]):
            valid_category_combinations.append(comb)
    
    if not valid_category_combinations:
        print("Warning: No valid R category combinations. Using available categories.")
        available_keys = list(all_other_task_ranks.keys())
        if len(available_keys) >= 2:
            valid_category_combinations = [{"name": "Available_Categories", "keys": available_keys[:min(4, len(available_keys))]}]

    all_polynomial_results = {}
    
    # Process each game
    for target_game_name_for_G_loop in target_games_to_process:
        harness_status_for_G = "harness_true" if harness_filter is None or harness_filter else "harness_false"
        G_current_ranks_float = _get_ranks_for_game(target_game_name_for_G_loop, current_model_order, all_model_perf_data, harness_status_for_G)

        if G_current_ranks_float is None or len(G_current_ranks_float) != len(current_model_order):
            print(f"Error: G ranks for '{target_game_name_for_G_loop}'. Skipping."); continue
        
        valid_g_indices = ~np.isnan(G_current_ranks_float)
        if not np.any(valid_g_indices): continue

        G_final_for_fit = G_current_ranks_float[valid_g_indices].astype(int)
        models_for_this_G_fit = [current_model_order[i] for i, is_valid in enumerate(valid_g_indices) if is_valid]
        
        all_polynomial_results.setdefault(target_game_name_for_G_loop, {})

        for comb_info in valid_category_combinations:
            comb_name, comb_keys = comb_info["name"], comb_info["keys"]
            
            R_sources_unfiltered_for_comb = [all_other_task_ranks[k] for k in comb_keys if k in all_other_task_ranks]
            if len(R_sources_unfiltered_for_comb) != len(comb_keys): continue

            R_sources_for_fit = []
            valid_R_source_for_comb = True
            for r_matrix_unfiltered in R_sources_unfiltered_for_comb:
                if r_matrix_unfiltered.ndim == 1:
                    if len(r_matrix_unfiltered) == len(current_model_order):
                        filtered_r = r_matrix_unfiltered[valid_g_indices]
                        if np.any(~np.isnan(filtered_r)): R_sources_for_fit.append(filtered_r.reshape(-1, 1))
                        else: valid_R_source_for_comb = False; break
                    else: valid_R_source_for_comb = False; break
                elif r_matrix_unfiltered.ndim == 2:
                    if r_matrix_unfiltered.shape[1] == len(current_model_order):
                        filtered_r_matrix = r_matrix_unfiltered[:, valid_g_indices]
                        if np.any(np.sum(~np.isnan(filtered_r_matrix), axis=1) > 0): R_sources_for_fit.append(filtered_r_matrix)
                        else: valid_R_source_for_comb = False; break
                    else: valid_R_source_for_comb = False; break
                else: valid_R_source_for_comb = False; break
            
            if not valid_R_source_for_comb or len(R_sources_for_fit) != len(comb_keys): continue
            
            temp_R_sources = []
            for r_mat in R_sources_for_fit:
                if r_mat.ndim == 1: r_mat = r_mat.reshape(-1,1)
                if r_mat.shape[0] > 0: temp_R_sources.append(r_mat)
            R_sources_for_fit = temp_R_sources

            if len(R_sources_for_fit) != len(comb_keys) : continue

            R_batches = build_batches(R_sources_for_fit)
            if not R_batches: continue
            
            G_targets_for_fit = [G_final_for_fit] * len(R_batches)
            actual_epochs = 1000  # Fixed epochs for simplicity

            predictor = RankingPredictor(degree=max_degree, lr=5e-3, epochs=actual_epochs, scale=float(len(models_for_this_G_fit)))
            try:
                predictor.fit(R_batches, G_targets_for_fit)
            except Exception as e_fit:
                print(f"Error during predictor.fit for {target_game_name_for_G_loop}/{comb_name}: {e_fit}")
                continue

            current_run_results = {
                "config": {
                    "poly_degree": max_degree,
                    "model_order_used": models_for_this_G_fit,
                    "R_categories_used": comb_keys,
                    "target_game_name": target_game_name_for_G_loop,
                    "target_harness_status": harness_status_for_G,
                }, 
                "evaluation": {}, 
                "feature_weights": {}
            }

            eval_results = []
            for Rb_idx, Rb_eval in enumerate(R_batches):
                 if Rb_eval.shape[0] > 0 and G_final_for_fit.shape[0] > 0 and Rb_eval.shape[0] == G_final_for_fit.shape[0]:
                    eval_results.append(predictor.evaluate(Rb_eval, G_final_for_fit))

            if not eval_results:
                current_run_results["evaluation"]["avg_residual_error"] = np.nan
                if max_degree == 1: current_run_results["evaluation"]["avg_pearson_r"] = np.nan
            elif max_degree == 1:
                valid_evals = [e for e in eval_results if isinstance(e, tuple) and len(e) == 2 and pd.notna(e[0]) and pd.notna(e[1])]
                if valid_evals:
                    current_run_results["evaluation"]["avg_residual_error"] = np.mean([e[0] for e in valid_evals])
                    current_run_results["evaluation"]["avg_pearson_r"] = np.mean([e[1] for e in valid_evals])
                else: current_run_results["evaluation"].update({"avg_residual_error": np.nan, "avg_pearson_r": np.nan})
            else:
                valid_evals = [e for e in eval_results if pd.notna(e)]
                if valid_evals: current_run_results["evaluation"]["avg_residual_error"] = np.mean(valid_evals)
                else: current_run_results["evaluation"]["avg_residual_error"] = np.nan
            
            w_vals = predictor._w
            f_names = []
            try:
                if R_batches and R_batches[0].shape[1] == len(comb_keys):
                     f_names = predictor._phi.get_feature_names_out(input_features=comb_keys)
                elif w_vals is not None: f_names = [f"feat_{j}" for j in range(len(w_vals))]
            except Exception:
                if w_vals is not None: f_names = [f"feat_{j}" for j in range(len(w_vals))]
            
            if w_vals is not None:
                current_run_results["feature_weights"] = {f_names[i]: w_vals[i] for i in range(len(w_vals))}
            
            entry_key = f"{harness_status_for_G}_deg{max_degree}"
            all_polynomial_results[target_game_name_for_G_loop].setdefault(comb_name, {})
            all_polynomial_results[target_game_name_for_G_loop][comb_name][entry_key] = current_run_results
            
    return all_polynomial_results

def visualize_polynomial_category_weights(polynomial_results: dict, save_plots: bool = False, comb_list: list = None):
    """
    Visualizes polynomial analysis results with one bar chart per game within each combination.
    Each game's bar chart shows all features (knowledge, physics, math, etc.) on x-axis.
    
    Args:
        polynomial_results: Results from run_polynomial_analysis_notebook
        save_plots: Whether to save plots (optional, default False for seamless workflow)
        comb_list: List of combination names to visualize (default: ["Comb3_LangPhyVisMathCode"])
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Set default combination list if not provided
    if comb_list is None:
        comb_list = ["Comb3_LangPhyVisMathCode"]
    
    if not polynomial_results:
        print("No polynomial results to visualize")
        return
    
    # Organize data by combination first
    combination_data = {}
    all_games = list(polynomial_results.keys())
    
    for combination_name in comb_list:
        combination_data[combination_name] = {}
        
        for game_name in all_games:
            game_data = polynomial_results[game_name]
            if combination_name in game_data:
                config_key = list(game_data[combination_name].keys())[0]
                config_data = game_data[combination_name][config_key]
                
                # Extract linear weights only
                feature_weights = config_data.get('feature_weights', {})
                linear_weights = {}
                
                for feature_name, weight in feature_weights.items():
                    if feature_name != '1' and '^' not in feature_name and ' ' not in feature_name:
                        linear_weights[feature_name] = weight
                
                combination_data[combination_name][game_name] = {
                    'linear_weights': linear_weights,
                    'error': config_data.get('evaluation', {}).get('avg_residual_error', np.nan),
                    'categories': config_data.get('config', {}).get('R_categories_used', [])
                }
    
    # Filter out combinations with no data
    valid_combinations = [comb for comb in comb_list if combination_data.get(comb)]
    
    if not valid_combinations:
        print("No valid combinations found with data")
        return
    
    print(f"Visualizing {len(valid_combinations)} combinations across {len(all_games)} games")
    
    # Define colors for categories (consistent across all plots)
    all_categories = set()
    for comb_data in combination_data.values():
        for game_data in comb_data.values():
            all_categories.update(game_data['linear_weights'].keys())
    
    category_colors = {}
    colors = plt.cm.Set3(np.linspace(0, 1, len(all_categories)))
    for i, cat in enumerate(sorted(all_categories)):
        category_colors[cat] = colors[i]
    
    # Process each combination separately
    for combination_name in valid_combinations:
        comb_data = combination_data[combination_name]
        
        # Get games with data for this combination
        games_with_data = []
        for game_name in all_games:
            if game_name in comb_data and comb_data[game_name]['linear_weights']:
                games_with_data.append(game_name)
        
        if not games_with_data:
            print(f"No games with data for {combination_name}")
            continue
        
        # Get all features for this combination
        all_features_in_comb = set()
        for game_name in games_with_data:
            all_features_in_comb.update(comb_data[game_name]['linear_weights'].keys())
        
        features_sorted = sorted(all_features_in_comb)
        
        if not features_sorted:
            print(f"No features with data for {combination_name}")
            continue
        
        n_games = len(games_with_data)
        
        # Create subplots for each game in this combination
        if n_games == 1:
            fig, axes = plt.subplots(1, 1, figsize=(12, 6))
            axes = [axes]
        elif n_games == 2:
            fig, axes = plt.subplots(1, 2, figsize=(24, 6))
        elif n_games <= 4:
            fig, axes = plt.subplots(2, 2, figsize=(24, 12))
            axes = axes.flatten()
        elif n_games <= 6:
            fig, axes = plt.subplots(2, 3, figsize=(36, 12))
            axes = axes.flatten()
        elif n_games <= 9:
            fig, axes = plt.subplots(3, 3, figsize=(36, 18))
            axes = axes.flatten()
        else:
            # For more than 9 games, use a 4x4 grid (max 16 games)
            fig, axes = plt.subplots(4, 4, figsize=(48, 24))
            axes = axes.flatten()
        
        fig.suptitle(f'{combination_name} - Feature Weights by Game', fontsize=16, fontweight='bold')
        
        # Plot each game separately
        for game_idx, game_name in enumerate(games_with_data):
            if game_idx >= len(axes):
                break
                
            ax = axes[game_idx]
            game_data = comb_data[game_name]
            linear_weights = game_data['linear_weights']
            error = game_data['error']
            
            # Prepare data for this game
            feature_values = []
            feature_colors = []
            
            for feature in features_sorted:
                weight = linear_weights.get(feature, 0)
                feature_values.append(weight)
                feature_colors.append(category_colors.get(feature, '#888888'))
            
            # Create bar chart for this game
            x_positions = np.arange(len(features_sorted))
            bars = ax.bar(x_positions, feature_values, color=feature_colors, alpha=0.8, 
                         edgecolor='black', linewidth=0.5)
            
            # Add value labels on bars
            for bar, value, feature in zip(bars, feature_values, features_sorted):
                if abs(value) > 0.001:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, 
                           height + (0.02 * max(abs(v) for v in feature_values) if height >= 0 else -0.02 * max(abs(v) for v in feature_values)), 
                           f'{value:.3f}', ha='center', 
                           va='bottom' if height >= 0 else 'top', 
                           fontsize=9, fontweight='bold')
            
            # Customize the plot
            ax.set_xticks(x_positions)
            ax.set_xticklabels([feature.title() for feature in features_sorted], 
                              fontsize=10, rotation=45, ha='right')
            ax.set_ylabel('Linear Weight', fontsize=11)
            ax.set_title(f'{game_name}\n(Error: {error:.3f})', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Set y-axis limits with some padding
            if feature_values:
                max_abs_weight = max(abs(v) for v in feature_values)
                if max_abs_weight > 0:
                    ax.set_ylim(-max_abs_weight * 1.2, max_abs_weight * 1.2)
            
            # Add horizontal line at y=0
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(games_with_data), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if save_plots:
            safe_filename = f'polynomial_weights_{combination_name}'.replace(' ', '_').replace('(', '').replace(')', '')
            plt.savefig(f'{safe_filename}.png', dpi=300, bbox_inches='tight')
            print(f"Saved plot: {safe_filename}.png")
        
        plt.show()
        
        # Print summary for this combination
        print(f"\n{combination_name} Summary:")
        print(f"  Games: {len(games_with_data)}")
        print(f"  Features: {', '.join(features_sorted)}")
        
        # Calculate average error across games
        errors = [comb_data[game]['error'] for game in games_with_data if not np.isnan(comb_data[game]['error'])]
        if errors:
            avg_error = np.mean(errors)
            print(f"  Average error: {avg_error:.4f}")
        
        # Show strongest feature weights across all games
        feature_totals = {}
        for feature in features_sorted:
            total_abs_weight = sum(abs(comb_data[game]['linear_weights'].get(feature, 0)) for game in games_with_data)
            feature_totals[feature] = total_abs_weight
        
        if feature_totals:
            strongest_feature = max(feature_totals.items(), key=lambda x: x[1])
            print(f"  Strongest overall feature: {strongest_feature[0]} (total abs weight: {strongest_feature[1]:.3f})")
    
    print("\nPolynomial analysis visualization complete.")

def normalize_game_columns(benchmark_df: pd.DataFrame, columns_to_normalize: list = None) -> pd.DataFrame:
    """
    Normalize specified game columns using the formula: (x - x.min()) / (x.max() - x.min()) * 99 + 1
    
    Args:
        benchmark_df: DataFrame with benchmark scores
        columns_to_normalize: List of column names to normalize. 
                            Defaults to ["Sokoban", "Tetris", "2048", "Candy Crush", "Ace Attorney"]
    
    Returns:
        DataFrame with normalized columns
    """
    if columns_to_normalize is None:
        columns_to_normalize = ["Sokoban", "Tetris", "2048", "Candy Crush", "Ace Attorney"]
    
    normalized_df = benchmark_df.copy()
    
    # Only normalize columns that exist in the DataFrame
    existing_columns = [col for col in columns_to_normalize if col in normalized_df.columns]
    
    if existing_columns:
        print(f"Normalizing columns: {existing_columns}")
        normalized_df[existing_columns] = normalized_df[existing_columns].apply(
            lambda x: (x - x.min()) / (x.max() - x.min()) * 99 + 1
        )
    else:
        print("No columns to normalize found in DataFrame")
    
    return normalized_df

def generate_tsne_visualization(benchmark_df: pd.DataFrame, model_names: list, save_plot: bool = False):
    """
    Generate t-SNE visualization to explore model clustering based on performance patterns.
    
    Args:
        benchmark_df: DataFrame with benchmark scores
        model_names: List of model names corresponding to dataframe rows
        save_plot: Whether to save the plot
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    
    print("Generating t-SNE visualization...")
    
    # Prepare data - remove columns with too many NaN values and fill remaining NaNs
    data_for_tsne = benchmark_df.copy()
    
    # Remove columns with more than 50% missing values
    missing_threshold = 0.5
    valid_columns = []
    for col in data_for_tsne.columns:
        missing_ratio = data_for_tsne[col].isna().sum() / len(data_for_tsne)
        if missing_ratio <= missing_threshold:
            valid_columns.append(col)
    
    data_for_tsne = data_for_tsne[valid_columns]
    print(f"Using {len(valid_columns)} benchmarks for t-SNE (removed {len(benchmark_df.columns) - len(valid_columns)} with >50% missing)")
    
    # Fill remaining NaN values with column mean
    data_for_tsne = data_for_tsne.fillna(data_for_tsne.mean())
    
    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_for_tsne)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(data_scaled)-1))
    tsne_results = tsne.fit_transform(data_scaled)
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot points
    colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
    
    for i, (model_name, color) in enumerate(zip(model_names, colors)):
        ax.scatter(tsne_results[i, 0], tsne_results[i, 1], 
                  color=color, s=100, alpha=0.7, edgecolors='black', linewidth=1)
        ax.annotate(model_name.replace('-', '-\n'), 
                   (tsne_results[i, 0], tsne_results[i, 1]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, ha='left', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3))
    
    ax.set_title('t-SNE Visualization of Model Performance\nAcross Multiple Benchmarks', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE Component 1', fontsize=12)
    ax.set_ylabel('t-SNE Component 2', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add explanation text
    explanation = f"Models clustered by performance similarity across {len(valid_columns)} benchmarks\nCloser models have similar performance patterns"
    ax.text(0.02, 0.98, explanation, transform=ax.transAxes, 
           fontsize=10, va='top', ha='left',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig('tsne_model_comparison.png', dpi=300, bbox_inches='tight')
        print("Saved t-SNE plot: tsne_model_comparison.png")
    
    plt.show()
    
    print("t-SNE visualization complete")


def generate_benchmark_correlation_matrix(benchmark_df: pd.DataFrame, save_plot: bool = False):
    """
    Generate correlation matrix heatmap for benchmark performance.
    
    Args:
        benchmark_df: DataFrame with benchmark scores
        save_plot: Whether to save the plot
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    
    print("Generating benchmark correlation matrix...")
    
    # Prepare data - only use columns with sufficient data
    correlation_data = benchmark_df.copy()
    
    # Remove columns with more than 30% missing values for correlation analysis
    missing_threshold = 0.3
    valid_columns = []
    for col in correlation_data.columns:
        missing_ratio = correlation_data[col].isna().sum() / len(correlation_data)
        if missing_ratio <= missing_threshold:
            valid_columns.append(col)
    
    correlation_data = correlation_data[valid_columns]
    print(f"Using {len(valid_columns)} benchmarks for correlation analysis")
    
    # Calculate correlation matrix
    correlation_matrix = correlation_data.corr()
    
    # Create the heatmap
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    
    # Generate the heatmap
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  # Mask upper triangle
    sns.heatmap(correlation_matrix, 
                mask=mask,
                annot=True, 
                cmap='RdBu_r',
                center=0,
                square=True,
                fmt='.2f',
                cbar_kws={"shrink": 0.8},
                ax=ax)
    
    ax.set_title('Benchmark Performance Correlation Matrix\n(Lower Triangle Only)', 
                fontsize=14, fontweight='bold')
    
    # Rotate labels for better readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig('benchmark_correlation_matrix.png', dpi=300, bbox_inches='tight')
        print("Saved correlation matrix: benchmark_correlation_matrix.png")
    
    plt.show()
    
    # Print high correlations
    print("\nHighest correlations (>0.7):")
    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_val = correlation_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                high_corr_pairs.append((correlation_matrix.columns[i], 
                                      correlation_matrix.columns[j], 
                                      corr_val))
    
    high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    for bench1, bench2, corr in high_corr_pairs[:10]:  # Top 10
        print(f"  {bench1} ↔ {bench2}: {corr:.3f}")
    
    print("Correlation analysis complete")


def generate_latent_ability_decomposition(benchmark_df: pd.DataFrame, model_names: list, n_components: int = 3, save_plot: bool = False):
    """
    Generate latent ability decomposition using PCA to identify underlying model capabilities.
    
    Args:
        benchmark_df: DataFrame with benchmark scores
        model_names: List of model names corresponding to dataframe rows
        n_components: Number of latent components to extract
        save_plot: Whether to save the plot
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    print(f"Generating latent ability decomposition with {n_components} components...")
    
    # Prepare data
    decomposition_data = benchmark_df.copy()
    
    # Remove columns with more than 40% missing values
    missing_threshold = 0.4
    valid_columns = []
    for col in decomposition_data.columns:
        missing_ratio = decomposition_data[col].isna().sum() / len(decomposition_data)
        if missing_ratio <= missing_threshold:
            valid_columns.append(col)
    
    decomposition_data = decomposition_data[valid_columns]
    
    # Fill NaN values with column mean
    decomposition_data = decomposition_data.fillna(decomposition_data.mean())
    
    print(f"Using {len(valid_columns)} benchmarks for decomposition")
    
    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(decomposition_data)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    pca_results = pca.fit_transform(data_scaled)
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Model scores on latent components
    ax1 = plt.subplot(2, 3, 1)
    colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
    
    for i, (model_name, color) in enumerate(zip(model_names, colors)):
        ax1.scatter(pca_results[i, 0], pca_results[i, 1], 
                   color=color, s=100, alpha=0.7, edgecolors='black')
        ax1.annotate(model_name.split('-')[0], 
                    (pca_results[i, 0], pca_results[i, 1]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, ha='left', va='bottom')
    
    ax1.set_title('PCA: Component 1 vs Component 2', fontsize=12, fontweight='bold')
    ax1.set_xlabel(f'Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax1.set_ylabel(f'Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Component loadings heatmap
    ax2 = plt.subplot(2, 3, 2)
    loadings = pca.components_[:n_components].T
    loadings_df = pd.DataFrame(loadings, index=valid_columns, 
                              columns=[f'PC{i+1}' for i in range(n_components)])
    
    im = ax2.imshow(loadings_df.T, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax2.set_title('Component Loadings', fontsize=12, fontweight='bold')
    ax2.set_yticks(range(n_components))
    ax2.set_yticklabels([f'PC{i+1}' for i in range(n_components)])
    ax2.set_xticks(range(len(valid_columns)))
    ax2.set_xticklabels(valid_columns, rotation=45, ha='right')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
    cbar.set_label('Loading Strength')
    
    # Plot 3: Explained variance
    ax3 = plt.subplot(2, 3, 3)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    ax3.bar(range(1, n_components+1), pca.explained_variance_ratio_, 
           alpha=0.7, label='Individual')
    ax3.plot(range(1, n_components+1), cumvar, 'ro-', label='Cumulative')
    ax3.set_title('Explained Variance', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Principal Component')
    ax3.set_ylabel('Variance Explained')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Model abilities on each component
    ax4 = plt.subplot(2, 3, 4)
    model_abilities = pd.DataFrame(pca_results, 
                                  index=model_names, 
                                  columns=[f'PC{i+1}' for i in range(n_components)])
    
    # Plot PC1 scores as horizontal bar chart
    pc1_scores = model_abilities['PC1'].sort_values(ascending=True)
    bars = ax4.barh(range(len(pc1_scores)), pc1_scores.values, 
                   color=[colors[model_names.index(name)] for name in pc1_scores.index])
    ax4.set_yticks(range(len(pc1_scores)))
    ax4.set_yticklabels([name.split('-')[0] for name in pc1_scores.index])
    ax4.set_title(f'PC1 Abilities ({pca.explained_variance_ratio_[0]:.1%} var)', 
                 fontsize=12, fontweight='bold')
    ax4.set_xlabel('PC1 Score')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: 3D view if we have 3+ components
    if n_components >= 3:
        ax5 = plt.subplot(2, 3, 5, projection='3d')
        for i, (model_name, color) in enumerate(zip(model_names, colors)):
            ax5.scatter(pca_results[i, 0], pca_results[i, 1], pca_results[i, 2],
                       color=color, s=100, alpha=0.7, edgecolors='black')
        ax5.set_title('3D PCA View', fontsize=12, fontweight='bold')
        ax5.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax5.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax5.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
    
    # Plot 6: Component interpretation
    ax6 = plt.subplot(2, 3, 6)
    top_loadings_per_pc = []
    for pc in range(min(3, n_components)):
        pc_loadings = np.abs(loadings_df[f'PC{pc+1}'])
        top_features = pc_loadings.nlargest(3)
        top_loadings_per_pc.append(f"PC{pc+1}:\n" + "\n".join([f"  {feat}: {val:.2f}" 
                                                               for feat, val in top_features.items()]))
    
    interpretation_text = "\n\n".join(top_loadings_per_pc)
    ax6.text(0.05, 0.95, interpretation_text, transform=ax6.transAxes,
            fontsize=10, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    ax6.set_title('Top Features per Component', fontsize=12, fontweight='bold')
    ax6.axis('off')
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig('latent_ability_decomposition.png', dpi=300, bbox_inches='tight')
        print("Saved latent ability plot: latent_ability_decomposition.png")
    
    plt.show()
    
    # Print summary
    print(f"\nLatent Ability Summary:")
    print(f"Total variance explained by {n_components} components: {cumvar[-1]:.1%}")
    for i in range(n_components):
        print(f"  PC{i+1}: {pca.explained_variance_ratio_[i]:.1%}")
    
    print("Latent ability decomposition complete")
    
    return {
        'pca_model': pca,
        'pca_results': pca_results,
        'loadings': loadings_df,
        'explained_variance': pca.explained_variance_ratio_,
        'valid_columns': valid_columns
    }

# Add these functions at the end of the file

def factorize_with_scipy_no_bias(Y, d, lambda_l1=0.01, lambda_l2_m=0.01, lambda_l2_b=0.01):
    """
    Low-rank matrix factorization using scipy optimization with improved NaN handling.
    """
    import numpy as np
    from scipy.optimize import minimize
    
    T, S = Y.shape

    def unpack(x):
        M = x[:T*d].reshape(T, d)
        B = x[T*d:T*d + S*d].reshape(S, d)
        return M, B

    def loss_fn(x):
        M, B = unpack(x)
        Y_hat = M @ B.T # change this line to handle null values in Y, so for positions if nan values in Y, don't need to compute the Y_hat
            
        mask = ~np.isnan(Y)
        
        loss = np.sum((Y[mask] - Y_hat[mask])**2)
        l1 = lambda_l1 * np.sum(np.abs(B))
        l2 = lambda_l2_m * np.sum(M**2) + lambda_l2_b * np.sum(B**2) 
        
        # Soft constraint loss: encourage M[2, k] <= M[3, k] for all k
        # soft_constraint = np.sum(np.maximum(M[2] - M[3], 0.0)**2)
        # pairwise = lambda_pair * soft_constraint

        return loss + l1 + l2

    x0 = np.random.rand((T + S) * d)
    
    bounds = [(0, None)] * len(x0)
    res = minimize(loss_fn, x0, bounds=bounds, method='L-BFGS-B', options={
        'maxfun': 50000,     # maximum number of function evaluations
        'maxiter': 50000,    # maximum number of iterations
        'disp': True        # optional: print progress
    })
    success, loss = res.success, res.fun

    M_opt, B_opt = unpack(res.x)
    return M_opt, B_opt, success, loss


def factorize_with_scipy_bias(Y, d, lambda_l1=0.01, lambda_l2_m=0.01, lambda_l2_b=0.01):
    """
    Low-rank matrix factorization with bias terms using scipy optimization.
    """
    import numpy as np
    from scipy.optimize import minimize
    
    T, S = Y.shape

    def unpack(x):
        M = x[:T*d].reshape(T, d)
        B = x[T*d:T*d + S*d].reshape(S, d)
        v = x[T*d + S*d:].reshape(S)  # Benchmark bias
        return M, B, v
    
    def loss_fn(x):
        M, B, v = unpack(x)
        Y_hat = M @ B.T + v[None, :]  # Add benchmark bias
        
        mask = ~np.isnan(Y)

        loss = np.sum((Y[mask] - Y_hat[mask])**2)
        l1 = lambda_l1 * np.sum(np.abs(B))
        l2 = lambda_l2_m * np.sum(M**2) + lambda_l2_b * np.sum(B**2) 
        
        return loss + l1 + l2

    # Initial values
    x0 = np.random.rand((T + S) * d + S)
    
    bounds = [(0, None)] * len(x0)
    res = minimize(loss_fn, x0, bounds=bounds, method='L-BFGS-B',options={
        'maxfun': 50000,     # maximum number of function evaluations
        'maxiter': 50000,    # maximum number of iterations
        'disp': True        # optional: print progress
    })
    success, loss = res.success, res.fun

    M_opt, B_opt, v_opt = unpack(res.x)
    return M_opt, B_opt, v_opt, success, loss


def generate_simple_tsne_plots(benchmark_data: dict, save_plot: bool = False):
    """
    Generate simple t-SNE plots: scores-based and ranks-based side by side.
    
    Args:
        benchmark_data: Dictionary with benchmark scores
        save_plot: Whether to save the plot
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.manifold import TSNE
    from scipy.stats import rankdata
    
    print("Generating simple t-SNE plots...")
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    titles = ["t-SNE based on scores", "t-SNE based on ranks"]

    # Convert to numpy array, forcing all values to be float (NaNs for invalid values)
    values = np.array([np.array(row, dtype=np.float64) for row in benchmark_data.values()], dtype=np.float64)
    keys = list(benchmark_data.keys())  # Start with all keys

    # Remove rows with any NaN values
    mask = ~np.isnan(values).any(axis=1)
    values = values[mask]
    keys = [keys[i] for i in range(len(keys)) if mask[i]]  # Only keep keys for valid rows

    # Normalize each column (feature-wise) instead of row-wise
    min_vals = np.nanmin(values, axis=0, keepdims=True)
    max_vals = np.nanmax(values, axis=0, keepdims=True)
    range_vals = np.where(max_vals - min_vals == 0, 1, max_vals - min_vals)
    normal_values = (values - min_vals) / range_vals

    # Apply t-SNE (reduce to 2D for visualization)
    for i in range(2):
        tsne = TSNE(n_components=2, perplexity=7, random_state=0)
        tsne_result = tsne.fit_transform(normal_values if i == 0 else np.apply_along_axis(rankdata, axis=1, arr=normal_values))
        
        ax[i].scatter(tsne_result[:, 0], tsne_result[:, 1])
        for j, key in enumerate(keys):
            ax[i].text(tsne_result[j, 0], tsne_result[j, 1], key, fontsize=9)
        
        ax[i].set_title(titles[i])
        ax[i].set_xlabel("t-SNE 1")
        ax[i].set_ylabel("t-SNE 2")
        ax[i].set_xticks([])
        ax[i].set_yticks([])

    plt.tight_layout()
    
    if save_plot:
        plt.savefig('tsne.png', dpi=300, bbox_inches='tight')
        print("Saved t-SNE plots: tsne.png")
    
    plt.show()
    print("t-SNE visualization complete")


def generate_simple_correlation_matrix(benchmark_df: pd.DataFrame, save_plot: bool = False):
    """
    Generate simple correlation matrix heatmap with upper triangle masked.
    
    Args:
        benchmark_df: DataFrame with benchmark scores (should be normalized)
        save_plot: Whether to save the plot
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    print("Generating simple correlation matrix...")
    
    corr_matrix = benchmark_df.corr(method='spearman')
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    plt.figure(figsize=(11, 9))
    sns.heatmap(
        corr_matrix,
        mask=mask,  # apply the mask here
        annot=False,
        fmt=".2f",
        cmap="coolwarm",
        center=0.4,
        linewidths=0.5,
        vmin=-0.2,
        vmax=1.0
    )
    plt.title("Benchmark Correlation Matrix")
    
    if save_plot:
        plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
        print("Saved correlation matrix: correlation_matrix.png")
    
    plt.show()
    print("Correlation matrix complete")


def generate_simple_latent_factorization(benchmark_data: dict, benchmark_df: pd.DataFrame, feature_num: int = 4, save_plot: bool = False):
    """
    Generate simple latent ability factorization with horizontal bar charts.
    
    Args:
        benchmark_data: Dictionary with benchmark scores  
        benchmark_df: DataFrame with benchmark scores (for column names)
        feature_num: Number of latent features
        save_plot: Whether to save the plot
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    from scipy.stats import rankdata
    from itertools import chain
    
    print(f"Generating latent factorization with {feature_num} features...")
    
    # Convert scores to ranks
    benchmark_ranks = {}
    for benchmark, scores in benchmark_data.items():
        scores_array = np.array(scores, dtype=np.float64)
        valid_mask = ~np.isnan(scores_array)
        ranks = np.full_like(scores_array, fill_value=np.nan, dtype=np.float64)
        ranks[valid_mask] = rankdata(scores_array[valid_mask], method='average')
        benchmark_ranks[benchmark] = ranks.tolist()
    
    benchmark_df_rank = pd.DataFrame(benchmark_ranks)
    
    # Apply factorization
    np.random.seed(12)  
    Y = (benchmark_df_rank * 0.2).to_numpy()
    bias = False

    M_hat, B_hat, success, loss = factorize_with_scipy_no_bias(Y, feature_num, lambda_l1=0.08, lambda_l2_m=0.01, lambda_l2_b=0.00)
    print(f"Factorization: success={success}, loss={loss:.4f}")

    np.set_printoptions(suppress=True, precision=2)

    # Normalize
    normalized = B_hat / B_hat.sum(axis=1, keepdims=True)

    print("benchmark:")
    for bench, feature_normalized in zip(benchmark_df.columns, normalized):
        formatted_normalized = " ".join(f"{x:.2f}" for x in feature_normalized)
        print(f"[{formatted_normalized}] {bench}")
    
    feature_names = [f"Feature_{i+1}" for i in range(normalized.shape[1])]
    bench_names = list(benchmark_df.columns)

    # rows = features, columns = benchmarks
    bench_feature_df = pd.DataFrame(
        normalized.T,           # transpose ⇒ shape (feature_num, n_benchmarks)
        index=feature_names,     # feature_num feature rows
        columns=bench_names      # one column per benchmark
    )
    
    # Configuration
    BASELINE = ['Sokoban', 'Tetris', '2048', 'Candy Crush', 'Ace Attorney']
    TOP_K = 3  # highest three

    # Color dictionary
    model_colors = {
        'Multi Challenge': '#66c2a5', 'MMLU-Pro': '#fc8d62', 'GPQA': '#8da0cb',
        'Sokoban': '#e78ac3', 'SMB': '#a6d854', 'Tetris': '#ffd92f',
        '2048': '#e5c494', 'Candy Crush': '#b3b3b3', 'Ace Attorney': '#1f78b4',
        'nyt-connections': '#fb8072', 'Math 500': '#80b1d3', 'AIME 2025': '#fdb462',
        'MGSM': '#bebada', 'VISTA': '#ffed6f', 'HLE': '#bc80bd',
        'MMMU': '#fccde5', 'EnigmaEval': '#ccebc5', 'HLE(Text)': '#d9d9d9',
        'Emma-mini Physics': '#1b9e77', 'PHYBench': '#d95f02', 'chatbot Arena_vision': '#7570b3',
        'EMMA-mini Math': '#66a61e', 'LiveBench-Math': '#e6ab02',
        'LiveBench-Coding': '#a6761d', 'BigCodeBench Pass@1': '#666666', 'Aider polyglot coding': '#1f78b4',
        'LiveBench-Language': '#33a02c'
    }

    # Build labels + weights for every feature
    labels_per_feat = []
    weights_per_feat = []

    for feat_name, row in bench_feature_df.iterrows():
        # pick top‑K non‑baseline benchmarks
        available_other = [col for col in row.index if col not in BASELINE and col in row.index]
        top_other = (
            row[available_other]      # only consider non-baseline columns that exist
               .sort_values(ascending=False)
               .head(TOP_K)
               .index.tolist()
        )
        ordered_labels = top_other + [b for b in BASELINE if b in row.index]  # only include existing baseline columns
        labels_per_feat.append(ordered_labels)
        weights_per_feat.append(row[ordered_labels].values)

    # Guarantee a colour for every label
    needed = set(chain(*labels_per_feat)) - model_colors.keys()
    if needed:
        extra_palette = sns.color_palette("tab20", len(needed))
        model_colors.update(dict(zip(needed, extra_palette)))

    # Plot
    n_feat = len(labels_per_feat)
    fig, axes = plt.subplots(1, n_feat, figsize=(3.6 * n_feat, 3))
    if n_feat == 1:
        axes = [axes]  # Make it iterable
    plt.subplots_adjust(wspace=0.4)

    for i, (ax, labels, weights) in enumerate(zip(axes, labels_per_feat, weights_per_feat)):
        bar_colors = [model_colors.get(l, '#cccccc') for l in labels]
        ax.barh(labels, weights, color=bar_colors)

        ax.set_title(f"Feature {i+1}", fontsize=12)
        ax.set_xlabel("Weight", fontsize=12)
        ax.tick_params(axis='both', labelsize=11)
        ax.set_xlim(0, weights.max() * 1.1)

        ax.invert_yaxis()
        ax.spines[['top', 'right']].set_visible(False)
        
        # dashed line separating "top‑K" and baseline for quick visual cue
        baseline_count = len([b for b in BASELINE if b in labels])
        if baseline_count > 0:
            ax.hlines(y=len(labels) - baseline_count - 0.5,
                      xmin=0, xmax=1, colors='gray', linestyles='--',
                      transform=ax.get_yaxis_transform())

    plt.tight_layout()
    
    if save_plot:
        plt.savefig("low_rank_consistent_colours.png", dpi=300, bbox_inches='tight')
        print("Saved latent factorization plot")
    
    plt.show()
    print("Latent factorization complete")
    
    return {
        'M_hat': M_hat,
        'B_hat': B_hat, 
        'normalized': normalized,
        'bench_feature_df': bench_feature_df,
        'success': success,
        'loss': loss
    }


