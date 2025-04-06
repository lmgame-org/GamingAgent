import time
import numpy as np
import concurrent.futures
import argparse
from collections import deque

from games.candy.workers import candy_crush_worker
from tools.utils import str2bool
import os
import shutil
# System prompt remains constant
system_prompt = (  
    "You are a highly intelligent Candy Crush gameplay agent trained to achieve the highest possible score "  
    "using a limited number of moves. Your goal is to analyze the board carefully and identify the best adjacent " 
)

state_reader_system_prompt = (
    "You are a game state reader for Candy Crush. Your task is to analyze the game board and identify the current state."
)

CACHE_DIR = "cache/candy_crush"
if os.path.exists(CACHE_DIR):
    shutil.rmtree(CACHE_DIR)

os.makedirs(CACHE_DIR)


def main():
    """
    Spawns a number of short-term and/or long-term workers based on user-defined parameters.
    """
    print("\n" + "="*80)
    print("Candy Crush Automation Agent")
    print("="*80)

    parser = argparse.ArgumentParser(
        description="Candy Crush grid processing with configurable parameters."
    )
    parser.add_argument("--api_provider", type=str, default="deepseek",
                        help="API provider to use (anthropic, openai, gemini).")
    parser.add_argument("--model_name", type=str, default="deepseek-reasoner", # "claude-3-5-sonnet-20241022",
                        help="Model name.")
    parser.add_argument("--state_reader_api_provider", type=str, default="anthropic",
                        help="Game state reader API provider to use.")
    parser.add_argument("--state_reader_model_name", type=str, default="claude-3-7-sonnet-20250219",
                        help="Game state reader model name.")
    parser.add_argument("--modality", type=str, default="text-only",
                        choices=["vision-only", "vision-text", "text-only"],
                        help="Employ vision-only, text-only or vision-text reasoning mode.")
    parser.add_argument("--thinking", type=str, default=True, help="Whether to use deep thinking.")
    parser.add_argument("--memory", type=str, default=False, help="whether to use memory.")
    parser.add_argument("--crop_left", type=int, default=237, help="Pixels to crop from the left.")
    parser.add_argument("--crop_right", type=int, default=762, help="Pixels to crop from the right.")
    parser.add_argument("--crop_top", type=int, default=175, help="Pixels to crop from the top.")
    parser.add_argument("--crop_bottom", type=int, default=290, help="Pixels to crop from the bottom.")
    # mingjia half screen 237 762 175 290
    # yuxuan 975 465 275 215
    parser.add_argument("--grid_rows", type=int, default=8, help="Number of grid rows.")
    parser.add_argument("--grid_cols", type=int, default=8, help="Number of grid columns.")
    parser.add_argument("--moves", type=int, default=26, help="Number of moves")
    
    args = parser.parse_args()

    print("\nConfiguration:")
    print(f"   - API Provider: {args.api_provider}")
    print(f"   - Model: {args.model_name}")
    # print(f"   - State Reader API: {args.state_reader_api_provider}")
    # print(f"   - State Reader Model: {args.state_reader_model_name}")
    print(f"   - Modality: {args.modality}")
    print(f"   - Deep Thinking: {args.thinking}")
    print(f"   - Memory: {args.memory}")
    print(f"   - Grid Size: {args.grid_rows}x{args.grid_cols}")
    print(f"   - Total Moves: {args.moves}")
    print("\n" + "-"*80)

    prev_responses = deque(maxlen=7)
        
    count = 0

    try:
        while True:
            if count == args.moves:
                print("\n" + "="*80)
                print(f"Game Complete! {args.model_name} finished {args.moves} moves.")
                print("="*80)
                break

            start_time = time.time()

            print(f"\nMove {count + 1}/{args.moves}")
            print("-"*40)

            # Execute the Candy Crush worker
            latest_response = candy_crush_worker(system_prompt, state_reader_system_prompt,
                                                args.api_provider, args.model_name, 
                                                args.state_reader_api_provider, args.state_reader_model_name,
                                                args.modality, str2bool(args.thinking),
                                                args.crop_left, args.crop_right, args.crop_top, args.crop_bottom, 
                                                args.grid_rows, args.grid_cols, 
                                                prev_response=" ".join(prev_responses) if args.memory else None,
                                                move_count=count)

            if latest_response:
                prev_responses.append(latest_response)
                count += 1
            elapsed_time = time.time() - start_time
            
            print(f"\nMove completed in {elapsed_time:.2f} seconds")
            print("Waiting 3 seconds before next move...")
            time.sleep(3)

            if prev_responses:
                print("\nPrevious moves history:")
                print("-"*40)
                for i, resp in enumerate(prev_responses, 1):
                    print(f"Move {i}: {resp}")

    except KeyboardInterrupt:
        print("\n" + "="*80)
        print("Stopping Candy Crush automation. Goodbye!")
        print("="*80)


if __name__ == "__main__":
    main()