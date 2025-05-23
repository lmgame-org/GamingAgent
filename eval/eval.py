import json
import os
from typing import Dict, List
import yaml
# import argparse # No longer needed
import numpy as np # For median calculation if needed

# Import GameLogProcessor and the helper functions using a relative import
from .utils import (
    GameLogProcessor, 
    load_json_file, 
    save_json_file, 
    update_model_perf_rank, 
    update_game_perf_data
)

# Import plotting utilities
from .plot_utils import (
    prepare_dataframe_for_plots,
    create_comparison_radar_chart,
    create_comparison_bar_chart
)

# Import Replay Utilities
from .replay_utils import generate_2048_median_score_replay # For the new specific 2048 replay

# Default values for run configurations (used if eval_config.yaml is missing/incomplete)
DEFAULT_GAME_LIST_CONFIG = [
    "twenty_forty_eight"
]

DEFAULT_MODEL_LIST_CONFIG = [
    "claude-3-5-sonnet-20241022"
]
DEFAULT_FORCE_MODEL_PERF_RANK = False
DEFAULT_FORCE_GAME_PERF = False
DEFAULT_GENERATE_BAR_PLOT = True
DEFAULT_GENERATE_RADAR_CHART = True
DEFAULT_GENERATE_REPLAYS = False


# --- Configuration Loading ---
GAME_SPECIFIC_CONFIG_FILE_PATH = "eval/game_eval_config.yaml" # Renamed
RUN_CONFIG_FILE_PATH = "eval/eval_config.yaml" # New config for run settings

def load_game_eval_config(config_path: str) -> Dict: # Renamed function
    default_game_config = {
        "game_specific_configs": {
            "twenty_forty_eight": {"score_transformation": None, "display_name": "2048"}
        },
        "default_game_display_name_mapping": {}
    }
    if not os.path.exists(config_path):
        print(f"Warning: Game-specific config file {config_path} not found. Using default game configurations.")
        return default_game_config
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            if config is None: # Handle empty YAML file
                print(f"Warning: Game-specific config file {config_path} is empty. Using default game configurations.")
                return default_game_config
            # Ensure essential keys exist, falling back to defaults if not
            if "game_specific_configs" not in config:
                config["game_specific_configs"] = default_game_config["game_specific_configs"]
            if "default_game_display_name_mapping" not in config:
                config["default_game_display_name_mapping"] = default_game_config["default_game_display_name_mapping"]
            return config
    except yaml.YAMLError as e_yaml:
        print(f"Error: Could not parse YAML from {config_path}: {e_yaml}. Using default game configurations.")
        return default_game_config
    except Exception as e:
        print(f"An unexpected error occurred while loading {config_path}: {e}. Using default game configurations.")
        return default_game_config

def load_run_config(config_path: str) -> Dict:
    """Loads run configurations from the YAML file."""
    default_run_settings = {
        "game_list": DEFAULT_GAME_LIST_CONFIG,
        "model_list": DEFAULT_MODEL_LIST_CONFIG,
        "force_model_perf_rank": DEFAULT_FORCE_MODEL_PERF_RANK,
        "force_game_perf": DEFAULT_FORCE_GAME_PERF,
        "generate_bar_plot": DEFAULT_GENERATE_BAR_PLOT,
        "generate_radar_chart": DEFAULT_GENERATE_RADAR_CHART,
        "generate_replays": DEFAULT_GENERATE_REPLAYS
    }
    if not os.path.exists(config_path):
        print(f"Warning: Run configuration file {config_path} not found. Using default run settings.")
        return default_run_settings
    try:
        with open(config_path, 'r') as f:
            run_config = yaml.safe_load(f)
            if run_config is None:
                print(f"Warning: Run configuration file {config_path} is empty. Using default run settings.")
                return default_run_settings
            # Apply defaults for any missing keys
            for key, value in default_run_settings.items():
                if key not in run_config:
                    print(f"Warning: '{key}' not found in {config_path}. Using default value: {value}")
                    run_config[key] = value
            return run_config
    except yaml.YAMLError as e_yaml:
        print(f"Error: Could not parse YAML from {config_path}: {e_yaml}. Using default run settings.")
        return default_run_settings
    except Exception as e:
        print(f"An unexpected error occurred while loading {config_path}: {e}. Using default run settings.")
        return default_run_settings


# Define paths
MODEL_PERF_RANK_FILE = "eval/perf/model_perf_rank.json"
DETAILED_GAME_PERF_FILE = "eval/perf/game_perf.json"
PLOT_OUTPUT_DIR = "eval/perf/plots"
VIDEO_OUTPUT_BASE_DIR = "eval/perf/video"
MODEL_COLORS_FILE = "eval/assets/model_colors.json"

# Main execution logic
def main():
    # Load run configurations from eval_config.yaml
    run_config = load_run_config(RUN_CONFIG_FILE_PATH)
    game_list_to_process = run_config.get("game_list", DEFAULT_GAME_LIST_CONFIG)
    model_list_to_process = run_config.get("model_list", DEFAULT_MODEL_LIST_CONFIG)
    force_model_perf_rank_flag = run_config.get("force_model_perf_rank", DEFAULT_FORCE_MODEL_PERF_RANK)
    force_game_perf_flag = run_config.get("force_game_perf", DEFAULT_FORCE_GAME_PERF)
    generate_bar_plot_flag = run_config.get("generate_bar_plot", DEFAULT_GENERATE_BAR_PLOT)
    generate_radar_chart_flag = run_config.get("generate_radar_chart", DEFAULT_GENERATE_RADAR_CHART)
    generate_replays_flag = run_config.get("generate_replays", DEFAULT_GENERATE_REPLAYS)

    # Load game-specific configurations from game_eval_config.yaml
    game_config_data = load_game_eval_config(GAME_SPECIFIC_CONFIG_FILE_PATH)
    game_specific_configs = game_config_data.get("game_specific_configs", {})

    print(f"Running evaluation with settings from {RUN_CONFIG_FILE_PATH} (and defaults for missing values):")
    print(f"  Games to process: {game_list_to_process}")
    print(f"  Models to process: {model_list_to_process}")
    print(f"  Force update model_perf_rank: {force_model_perf_rank_flag}")
    print(f"  Force update game_perf: {force_game_perf_flag}")
    print(f"  Generate Bar Plot: {generate_bar_plot_flag}")
    print(f"  Generate Radar Chart: {generate_radar_chart_flag}")
    print(f"  Generate Replays: {generate_replays_flag}")
    print(f"  Using game_specific_configs from {GAME_SPECIFIC_CONFIG_FILE_PATH} (or defaults if file/key is missing).")

    os.makedirs(os.path.dirname(MODEL_PERF_RANK_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(DETAILED_GAME_PERF_FILE), exist_ok=True)

    model_colors = {}
    try:
        with open(MODEL_COLORS_FILE, 'r') as f:
            model_colors = json.load(f)
        print(f"Successfully loaded model colors from {MODEL_COLORS_FILE}")
    except FileNotFoundError:
        print(f"Warning: Model colors file not found at {MODEL_COLORS_FILE}. Using default colors.")
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from {MODEL_COLORS_FILE}. Using default colors.")

    all_detailed_game_perf_data = load_json_file(DETAILED_GAME_PERF_FILE, default_data={})

    for game_name in game_list_to_process:
        print(f"\nProcessing game: {game_name}")
        game_display_name = game_specific_configs.get(game_name, {}).get("display_name", game_name)
        current_game_config = game_specific_configs.get(game_name, {})
        score_transformation_rule_for_game = current_game_config.get("score_transformation")
        if score_transformation_rule_for_game:
            print(f"  Using score transformation rule: '{score_transformation_rule_for_game}' for {game_name}")

        current_game_detailed_data = all_detailed_game_perf_data.get(game_display_name, {})

        for model_prefix_to_search in model_list_to_process:
            print(f"  Processing model prefix/authoritative name: {model_prefix_to_search}")
            processor = GameLogProcessor(
                game_name=game_name, 
                model_name_prefix_for_search=model_prefix_to_search,
                authoritative_model_name=model_prefix_to_search,
                score_transformation_rule=score_transformation_rule_for_game
            )

            if not processor.raw_data:
                print(f"    No data collected by processor for {game_name} with model prefix {model_prefix_to_search}. Skipping updates.")
                continue

            # 1. Update model_perf_rank.json
            print(f"    Updating {MODEL_PERF_RANK_FILE}...")
            model_perf_rank_data = load_json_file(MODEL_PERF_RANK_FILE, default_data={})
            new_model_perf_subset = processor.generate_model_perf_update()
            if new_model_perf_subset:
                updated_model_perf_rank_data = update_model_perf_rank(
                    model_perf_rank_data, 
                    new_model_perf_subset, 
                    game_name, # game_name here is the internal key for the transformation logic
                    force=force_model_perf_rank_flag
                )
                save_json_file(updated_model_perf_rank_data, MODEL_PERF_RANK_FILE)
            else:
                print(f"    No new data from processor to update {MODEL_PERF_RANK_FILE} for {model_prefix_to_search}.")

            # 2. Update detailed game performance data for game_perf.json
            print(f"    Updating detailed performance for {game_display_name} in {DETAILED_GAME_PERF_FILE}...")
            # For game_perf.json, we want the raw scores, which generate_game_perf_update provides.
            new_game_perf_subset_for_model = processor.generate_game_perf_update()
            if new_game_perf_subset_for_model:
                current_game_detailed_data = update_game_perf_data(
                    current_game_detailed_data, # Pass the data for the current game (e.g., "2048")
                    new_game_perf_subset_for_model, # New data for the current model for this game
                    force=force_game_perf_flag
                )
            else:
                print(f"    No new detailed data from processor for {model_prefix_to_search} in {game_name}.")
        
        # After processing all models for the current game, update the main dictionary
        all_detailed_game_perf_data[game_display_name] = current_game_detailed_data
        print(f"Finished processing for game: {game_name} ({game_display_name})")

    # After processing all games, save the consolidated detailed game performance data once
    print(f"\nSaving all detailed game performance data to {DETAILED_GAME_PERF_FILE}...")
    save_json_file(all_detailed_game_perf_data, DETAILED_GAME_PERF_FILE)

    # --- Plotting section --- 
    if generate_bar_plot_flag or generate_radar_chart_flag:
        print("\n--- Generating Performance Plots ---")
        target_games_for_plot = game_list_to_process # Use the actual games processed
        print(f"Target games for plots: {target_games_for_plot}")

        current_plot_output_dir = PLOT_OUTPUT_DIR
        if model_list_to_process: 
            first_model_name_for_dir = model_list_to_process[0]
            # Replace hyphens with underscores, then take first 15 chars for directory name
            model_name_prefix_for_dir = first_model_name_for_dir.replace("-", "_")[:15]
            if model_name_prefix_for_dir: # Ensure not empty after slicing
                current_plot_output_dir = os.path.join(PLOT_OUTPUT_DIR, model_name_prefix_for_dir)
        
        os.makedirs(current_plot_output_dir, exist_ok=True)
        print(f"Plots will be saved to: {current_plot_output_dir}")

        if not target_games_for_plot:
            print("No target games defined for plotting. Skipping plot generation.")
        else:
            models_for_plot_data = model_list_to_process # Use the list of models processed in this run
            for harness_status in ["harness_true", "harness_false"]:
                print(f"\nGenerating plots for {harness_status}...")
                df_plot = prepare_dataframe_for_plots(
                    rank_data_path=MODEL_PERF_RANK_FILE,
                    selected_games=target_games_for_plot, 
                    game_specific_configs=game_specific_configs, 
                    harness_status_to_use=harness_status
                )

                if df_plot.empty:
                    print(f"  No data available for {harness_status} after preparation. Skipping plot generation.")
                    continue

                # Common plot filename prefix
                plot_filename_prefix = f"comparison_{harness_status}"
                if model_list_to_process: # Add model info to filename if specific models were processed
                    # Create a concise string from model list for filename
                    models_str = "_".join(m.replace("-", "_")[:10] for m in model_list_to_process) # Shorter, more filename-friendly
                    plot_filename_prefix = f"{models_str}_{plot_filename_prefix}"

                # Generate Bar Chart
                if generate_bar_plot_flag:
                    print(f"  Generating Bar Chart for {harness_status}...")
                    # Get selected_games_display_names for the plotting functions
                    selected_games_display_names_for_plot = []
                    for game_key in target_games_for_plot:
                        display_name = game_specific_configs.get(game_key, {}).get("display_name", game_key)
                        selected_games_display_names_for_plot.append(display_name)
                    
                    # Highlight models: use all models present in the dataframe for this plot
                    highlight_models_for_plot = df_plot['Player'].unique().tolist()

                    fig_bar = create_comparison_bar_chart(
                        df=df_plot, 
                        model_colors=model_colors,
                        selected_games_display_names=selected_games_display_names_for_plot,
                        harness_status=harness_status,
                        highlight_models=highlight_models_for_plot
                    )
                    bar_chart_path_html = os.path.join(current_plot_output_dir, f"{plot_filename_prefix}_bar.html")
                    bar_chart_path_png = os.path.join(current_plot_output_dir, f"{plot_filename_prefix}_bar.png")
                    try:
                        fig_bar.write_html(bar_chart_path_html)
                        fig_bar.write_image(bar_chart_path_png, width=max(800, 200 * len(selected_games_display_names_for_plot)), height=600)
                        print(f"    Bar chart saved to {bar_chart_path_html} and {bar_chart_path_png}")
                    except Exception as e:
                        print(f"    Error saving bar chart for {harness_status}: {e}")
                        print(f"    Make sure you have 'kaleido' (or 'plotly-orca') and 'plotly' installed: pip install kaleido plotly")

                # Generate Radar Chart
                if generate_radar_chart_flag:
                    print(f"  Generating Radar Chart for {harness_status}...")
                    # Get selected_games_display_names for the plotting functions (can reuse from bar chart section if already generated)
                    if not 'selected_games_display_names_for_plot' in locals() or not selected_games_display_names_for_plot:
                        selected_games_display_names_for_plot = []
                        for game_key in target_games_for_plot:
                            display_name = game_specific_configs.get(game_key, {}).get("display_name", game_key)
                            selected_games_display_names_for_plot.append(display_name)

                    # Highlight models (can reuse from bar chart section if already generated)
                    if not 'highlight_models_for_plot' in locals() or not highlight_models_for_plot:
                        highlight_models_for_plot = df_plot['Player'].unique().tolist()

                    fig_radar = create_comparison_radar_chart(
                        df=df_plot, 
                        model_colors=model_colors,
                        selected_games_display_names=selected_games_display_names_for_plot,
                        harness_status=harness_status,
                        highlight_models=highlight_models_for_plot
                    )
                    radar_chart_path_html = os.path.join(current_plot_output_dir, f"{plot_filename_prefix}_radar.html")
                    radar_chart_path_png = os.path.join(current_plot_output_dir, f"{plot_filename_prefix}_radar.png")
                    try:
                        fig_radar.write_html(radar_chart_path_html)
                        fig_radar.write_image(radar_chart_path_png, width=1000, height=700)
                        print(f"    Radar chart saved to {radar_chart_path_html} and {radar_chart_path_png}")
                    except Exception as e:
                        print(f"    Error saving radar chart for {harness_status}: {e}")
                        print(f"    Make sure you have 'kaleido' (or 'plotly-orca') and 'plotly' installed: pip install kaleido plotly")
    else:
        print("\nSkipping Performance Plot Generation as both flags are false.")

    # --- Video Replay Generation --- 
    if generate_replays_flag:
        print("\n--- Generating Video Replays (if supported) ---")
        os.makedirs(VIDEO_OUTPUT_BASE_DIR, exist_ok=True) # Ensure base video dir exists

        for game_name_for_replay in game_list_to_process: # Iterate through games processed in this run
            # Get the display name as used in game_perf.json keys
            game_display_name_for_replay = game_specific_configs.get(game_name_for_replay, {}).get("display_name", game_name_for_replay)

            if game_name_for_replay == "twenty_forty_eight":
                print(f"  Preparing to generate 2048 median replays for game: {game_name_for_replay} (display name in JSON: {game_display_name_for_replay})")
                for model_name in model_list_to_process:
                    print(f"    Processing model: {model_name} for 2048 median replay")
                    for harness_key in ["harness_true", "harness_false"]:
                        print(f"      Generating for harness: {harness_key}")
                        try:
                            # Call the function from replay_utils for 2048
                            generate_2048_median_score_replay(
                                game_perf_json_path=DETAILED_GAME_PERF_FILE,
                                model_name_prefix=model_name, # This is the model name used as key in game_perf.json
                                game_display_name=game_display_name_for_replay, # This is the game name used as key
                                harness_status_key=harness_key,
                                video_output_base_dir=VIDEO_OUTPUT_BASE_DIR
                                # frame_duration will use default from replay_utils
                            )
                        except Exception as e_median_replay:
                            print(f"      Error calling generate_2048_median_score_replay for {model_name}, {harness_key}: {e_median_replay}")
            else:
                print(f"  Replay generation is not currently supported for game: '{game_name_for_replay}'. Skipping replay for this game.")
    else:
        print("\nSkipping Video Replay Generation as flag is false.")

if __name__ == "__main__":
    main()
    print("\nEvaluation script finished.")

# Example of how to run if you have cache populated:
# Assuming cache/twenty_forty_eight/claude_3_5_sonn_20250521_181702/ exists with logs
# and eval/utils.py has GameLogProcessor and helper functions, and eval/__init__.py exists:
# Run from the directory *above* eval, e.g., your project root (GamingAgent):
# python -m eval.eval