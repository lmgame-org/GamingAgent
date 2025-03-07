import time
import numpy as np
import concurrent.futures
import argparse
from collections import deque

from games.boxxel.workers import boxxel_worker
# System prompt remains constant
system_prompt = (
    "You are an expert AI agent specialized in solving Sokoban puzzles optimally. "
    "Your goal is to push all boxes onto the designated dock locations while avoiding deadlocks. "
    
    # "You control a worker ('@') who can move in four directions (up, down, left, right) but can only push boxes ('$') if positioned correctly. "
    # "You cannot pull boxes—only push them. If a box gets stuck against a wall with no way to reposition it, a restart may be necessary. "
    
    # "\n\n### Sokoban Rules ###\n"
    # "1. The game takes place on a grid-based level.\n"
    # "2. The level consists of different elements:\n"
    # "   - Wall (#): Impassable.\n"
    # "   - Dock (?): Goal position for boxes.\n"
    # "   - Worker (@): Your character.\n"
    # "   - Box ($): Movable by pushing only.\n"
    # "   - Box on Dock (*): A correctly placed box.\n"
    # "3. You can push boxes, but only if there is an empty floor space or dock behind them.\n"
    # "4. The objective is to place all boxes onto dock squares.\n"
    # "5. If a box becomes stuck in a corner or against a wall with no way to reposition it, restart is required ('R').\n"
    # "6. Undo the last move using ('D') if needed.\n"

    # "\n### Movement Constraints ###\n"
    # "1. You can only push a box when directly adjacent to it and facing the direction you wish to move.\n"
    # "2. Allowed box movements:\n"
    # "   - **Upward push**: If the player is below the box (e.g., Player at (3,2), Box at (2,2)), move up to push the box to (1,2), and the player moves to (2,2).\n"
    # "   - **Leftward push**: If the player is to the right of the box, push it left if the space allows.\n"
    # "   - **Rightward push**: If the player is to the left of the box, push it right if there is an open space.\n"
    # "   - **Downward push**: If the player is above the box, push it downward into an open space.\n"
    # "3. You **cannot** move through walls or boxes unless pushing is valid.\n"
    # "4. If pushing a box into a dock, it becomes a correctly placed box (*).\n"

    # "\n### Strategy for Optimal Moves ###\n"
    # "1. **Plan ahead**: Avoid moving boxes into positions where they cannot be recovered.\n"
    # "2. **Corner awareness**: Never push a box against a wall or into a corner unless it's a final placement.\n"
    # "3. **Dock positioning**: Prioritize moving boxes towards their nearest dock (?).\n"
    # "4. **Undo wisely**: If a mistake is made, use 'D' to unmove before committing to a restart.\n"
    # "5. **Minimize moves**: Solve the level in as few moves as possible.\n"
)


def main():
    parser = argparse.ArgumentParser(description="Boxxel AI Agent")
    parser.add_argument("--api_provider", type=str, default="openai", help="API provider to use.")
    parser.add_argument("--model_name", type=str, default="gpt-4-turbo", help="LLM model name.")
    parser.add_argument("--loop_interval", type=float, default=3, help="Time in seconds between moves.")
    parser.add_argument("--level", type=int, default=1, help="Time in seconds between moves.")
    args = parser.parse_args()

    prev_responses = deque(maxlen=7)
    count = 0
    try:
        while True:
            start_time = time.time()
            latest_response = boxxel_worker(system_prompt, args.api_provider, args.model_name
                                            , " ".join(prev_responses), level = args.level)
            if count == 3:
                break
            count +=1
            if latest_response:
                prev_responses.append(latest_response)
            elapsed_time = time.time() - start_time
            time.sleep(1)
            print(f"[INFO] Move executed in {elapsed_time:.2f} seconds.")
    except KeyboardInterrupt:
        print("\nStopped by user.")

if __name__ == "__main__":
    main()