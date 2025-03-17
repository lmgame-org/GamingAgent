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

CACHE_DIR = "cache/candy_crush"
if os.path.exists(CACHE_DIR):
    shutil.rmtree(CACHE_DIR)

os.makedirs(CACHE_DIR)


def main():
    """
    Spawns a number of short-term and/or long-term workers based on user-defined parameters.
    """
    parser = argparse.ArgumentParser(
        description="Candy Crush grid processing with configurable parameters."
    )
    parser.add_argument("--api_provider", type=str, default="openai",
                        help="API provider to use (anthropic, openai, gemini).")
    parser.add_argument("--model_name", type=str, default="gpt-4-turbo",
                        help="Model name.")
    parser.add_argument("--modality", type=str, default="text-only", choices=["text-only", "vision-text"], help="modality used.")
    parser.add_argument("--thinking", type=str, default=True, help="Whether to use deep thinking.")
    parser.add_argument("--crop_left", type=int, default=880, help="Pixels to crop from the left.")
    parser.add_argument("--crop_right", type=int, default=660, help="Pixels to crop from the right.")
    parser.add_argument("--crop_top", type=int, default=350, help="Pixels to crop from the top.")
    parser.add_argument("--crop_bottom", type=int, default=230, help="Pixels to crop from the bottom.")
    parser.add_argument("--grid_rows", type=int, default=8, help="Number of grid rows.")
    parser.add_argument("--grid_cols", type=int, default=8, help="Number of grid columns.")
    parser.add_argument("--moves", type=int, default=50, help="Number of moves")
    
    args = parser.parse_args()

    prev_responses = deque(maxlen=7)

    count = 0

    try:
        while True:
            if count == args.moves:
                print(f"{args.model_name} finished {args.moves}.")
                break

            start_time = time.time()

            # Execute the Candy Crush worker
            latest_response = candy_crush_worker(system_prompt, args.api_provider, args.model_name, args.modality, str2bool(args.thinking),
                                                 args.crop_left, args.crop_right, args.crop_top, args.crop_bottom, 
                                                 args.grid_rows, args.grid_cols, " ".join(prev_responses))
            # break
            if latest_response:
                prev_responses.append(latest_response)
            elapsed_time = time.time() - start_time
            time.sleep(3)
            count+=1
            print("[debug] previous message:")
            print("\n".join(prev_responses))

    except KeyboardInterrupt:
        print("\n[INFO] Stopping Candy Crush automation. Goodbye!")


if __name__ == "__main__":
    main()