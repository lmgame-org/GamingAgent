import time
from collections import deque
import numpy as np
import concurrent.futures
import argparse
import json

from games.space_invaders.worker import space_invaders_worker

system_prompt = (
    "You are a highly intelligent Atari 2600 Space Invaders gameplay agent, search for and determine the best action to take given each game state. Prioritize survival over score maximization."
)

def main():
    parser = argparse.ArgumentParser(
        description="Space Invaders automation agent."
    )
    parser.add_argument("--api_provider", type=str, default="openai",
                        help="API provider (anthropic, openai, gemini).")
    parser.add_argument("--model_name", type=str, default="gpt-4-turbo",
                        help="AI model to use.")
    parser.add_argument("--crop_top", type=int, default=50, help="Pixels to crop from the top.")
    parser.add_argument("--crop_bottom", type=int, default=400, help="Pixels to crop from the bottom.")
    parser.add_argument("--crop_left", type=int, default=50, help="Pixels to crop from the left.")
    parser.add_argument("--crop_right", type=int, default=700, help="Pixels to crop from the right.")
    parser.add_argument("--loop_interval", type=float, default=0.5,
                        help="Seconds between each AI decision loop.")
    
    args = parser.parse_args()

    # Record the last 5 strategy, prevent duplication
    prev_responses = deque(maxlen=5)

    try:
        while True:
            start_time = time.time()
            latest_reponse = space_invaders_worker(
                system_prompt, args.api_provider, args.model_name,
                args.crop_top, args.crop_bottom, args.crop_left, args.crop_right,
                " ".join(prev_responses)
            )
            if latest_reponse:
                try:
                    response_data = json.loads(latest_reponse)
                    prev_responses.append(json.dumps(response_data)) 
                except json.JSONDecodeError:
                    print("[ERROR] AI Response is not valid JSON. Skipping this response.")
            
            elapsed_time = time.time() - start_time
            sleep_time = max(0, args.loop_interval - elapsed_time)
            time.sleep(sleep_time)

            print(f"[INFO] Action executed in {elapsed_time:.2f} seconds.")
    except KeyboardInterrupt:
        print("\n[INFO] Stopping Space Invaders automation. Goodbye!")

if __name__ == "__main__":
    main()
