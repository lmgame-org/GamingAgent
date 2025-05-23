import json
import os
import argparse # Import argparse
# Import GameLogProcessor and the helper functions using a relative import
from .utils import (
    GameLogProcessor, 
    load_json_file, 
    save_json_file, 
    update_model_perf_rank, 
    update_game_perf_data
)

# Predefined lists (as they were in the original file)
game_list = [
    "twenty_forty_eight"
]

model_list = [
    "claude-3-5-sonnet-20241022" # This will be used as a prefix (first 15 chars by GameLogProcessor)
]

# Define paths
MODEL_PERF_RANK_FILE = "eval/perf/model_perf_rank.json"
GAME_PERF_DIR = "eval/perf/game_perf"

# Main execution logic
def main():
    parser = argparse.ArgumentParser(description="Run evaluation script to process game logs.")
    parser.add_argument("--force", action="store_true", help="Force update existing entries in JSON files.")
    args = parser.parse_args()

    print(f"Running evaluation with force update: {args.force}")

    # Ensure the main perf directory exists (save_json_file will also ensure specific file dirs exist)
    os.makedirs(os.path.dirname(MODEL_PERF_RANK_FILE), exist_ok=True)
    os.makedirs(GAME_PERF_DIR, exist_ok=True) # Ensure the general game_perf directory exists

    for game_name in game_list:
        print(f"\nProcessing game: {game_name}")
        
        game_perf_file_path = os.path.join(GAME_PERF_DIR, f"{game_name}.json")
        current_game_perf_data = load_json_file(game_perf_file_path, default_data={})

        for model_prefix_to_search in model_list:
            print(f"  Processing model prefix/authoritative name: {model_prefix_to_search}")
            
            processor = GameLogProcessor(
                game_name=game_name, 
                model_name_prefix_for_search=model_prefix_to_search, # Used for finding dirs (can be prefix)
                authoritative_model_name=model_prefix_to_search     # Used as the key for reporting (exact name)
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
                    game_name,
                    force=False# Pass force flag
                )
                save_json_file(updated_model_perf_rank_data, MODEL_PERF_RANK_FILE)
            else:
                print(f"    No new data from processor to update {MODEL_PERF_RANK_FILE} for {model_prefix_to_search}.")

            # 2. Update game_perf/<game_name>.json
            print(f"    Updating {game_perf_file_path}...")
            new_game_perf_subset = processor.generate_game_perf_update()

            if new_game_perf_subset:
                current_game_perf_data = update_game_perf_data(
                    current_game_perf_data, 
                    new_game_perf_subset,
                    force=True # Pass force flag
                )
                save_json_file(current_game_perf_data, game_perf_file_path)
            else:
                print(f"    No new data from processor to update {game_perf_file_path} for {model_prefix_to_search}.")
        
        print(f"Finished processing for game: {game_name}")

if __name__ == "__main__":
    main()
    print("\nEvaluation script finished.")

# Example of how to run if you have cache populated:
# Assuming cache/twenty_forty_eight/claude_3_5_sonn_20250521_181702/ exists with logs
# and eval/utils.py has GameLogProcessor and helper functions, and eval/__init__.py exists:
# Run from the directory *above* eval, e.g., your project root (GamingAgent):
# python -m eval.eval