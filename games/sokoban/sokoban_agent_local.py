import time
import numpy as np
import concurrent.futures
import argparse
from collections import deque, Counter

import os
import json
import re
import pyautogui

from games.sokoban.workers_local import sokoban_worker

CACHE_DIR = "cache/sokoban"

from collections import Counter

def majority_vote_move(moves_list, prev_move=None):
    """
    Returns the majority-voted move from moves_list.
    If there's a tie for the top count, and if prev_move is among those tied moves,
    prev_move is chosen. Otherwise, pick the first move from the tie.
    """
    if not moves_list:
        return None

    c = Counter(moves_list)
    
    # c.most_common() -> list of (move, count) sorted by count descending, then by move
    counts = c.most_common()
    top_count = counts[0][1]  # highest vote count

    tie_moves = [m for m, cnt in counts if cnt == top_count]

    if len(tie_moves) > 1 and prev_move:
        if prev_move in tie_moves:
            return prev_move
        else:
            return tie_moves[0]
    else:
        return tie_moves[0]


# System prompt remains constant
system_prompt = (
    "You are an expert AI agent specialized in solving Sokoban puzzles optimally. "
    "Your goal is to push all boxes onto the designated dock locations while avoiding deadlocks. "
)


def main():
    parser = argparse.ArgumentParser(description="sokoban AI Agent")
    parser.add_argument("--api_provider", type=str, default="openai", help="API provider to use.")
    parser.add_argument("--model_name", type=str, default="o3-mini", help="LLM model name.")
    parser.add_argument("--modality", type=str, default="text-only", choices=["text-only", "vision-text"],
                        help="modality used.")
    parser.add_argument("--thinking", type=str, default=True, help="Whether to use deep thinking.")
    parser.add_argument("--starting_level", type=int, default=1, help="Starting level for the Sokoban game.")
    parser.add_argument("--num_threads", type=int, default=10, help="Number of parallel threads to launch.")
    args = parser.parse_args()

    prev_responses = deque(maxlen=10)
    level = None

    def perform_move(move):
        key_map = {
            "up": "up",
            "down": "down",
            "left": "left",
            "right": "right",
            "restart": 'R',
            "unmove": 'D'
        }
        if move in key_map:
            pyautogui.press(key_map[move])
            print(f"Performed move: {move}")
        else:
            print(f"[WARNING] Invalid move: {move}")

    try:
        while True:
            current_level_path = os.path.join(CACHE_DIR, "current_level.json")
            with open(current_level_path, 'r') as f:
                level_dict = json.load(f)
                level = level_dict["level"]
            
            start_time = time.time()

            # Self-consistency launch, to disable, set "--num_threads 1"
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
                futures = []
                for _ in range(args.num_threads):
                    futures.append(
                        executor.submit(
                            sokoban_worker,
                            system_prompt,
                            args.api_provider,
                            args.model_name,
                            "\n".join(prev_responses),
                            thinking=args.thinking,
                            modality=args.modality,
                            level=level
                        )
                    )
                
                # Wait until all threads finish
                concurrent.futures.wait(futures)
                results = [f.result() for f in futures]
            
            print("all threads finished execution...")
            print(results)

            # Find the shortest solution length among all threads
            shortest_length = min(len(mlist) for mlist in results)

            # ------------------------- action ------------------------ #
            # For each position up to that shortest length, do a majority vote
            final_moves = []
            collected_thoughts_per_move = []
            # Iterate over all possible future steps
            for i in range(shortest_length):
                # Collect the i-th move and thought from each thread (with sufficient actions predicted)
                move_thought_pairs = [sol[i] for sol in results if len(sol) > i]
                
                # Vote
                move_candidates = [pair["move"] for pair in move_thought_pairs]
                move_candidate_count = {}
                for move_candidate in move_candidates:
                    if move_candidate in move_candidate_count.keys():
                        move_candidate_count[move_candidate] += 1
                    else:
                        move_candidate_count[move_candidate] = 1
                
                print(move_candidate_count)

                if final_moves:
                    chosen_move = majority_vote_move(move_candidates, final_moves[-1])
                else:
                    chosen_move = majority_vote_move(move_candidates)
                final_moves.append(chosen_move)

                # Iterate over all valid threads for this step
                # Gather all thoughts from the threads whose move == chosen_move
                matched_thoughts = [pair["thought"] for pair in move_thought_pairs 
                                     if pair["move"] == chosen_move]

                matched_thought = matched_thoughts[0]

                collected_thoughts_per_move.append(matched_thought)

            # Loop through every move in the order they appear
            for move in final_moves:
                move = move.strip().lower()

                # Perform the move
                perform_move(move)
            # ------------------------- action ------------------------ #

            # HACK: temporary memory module
            if final_moves:
                assert len(final_moves) == len(collected_thoughts_per_move), "move and thought length disagree, regex operation errored out."
                for move, matched_thought in zip(final_moves, collected_thoughts_per_move):
                    latest_response = "step executed:\n" + f"move: {move}, thought: {matched_thought}" + "\n"
                    prev_responses.append(latest_response)

            print("[debug] previous message:")
            print("\n".join(prev_responses))
            elapsed_time = time.time() - start_time
            time.sleep(1)
            print(f"[INFO] Move executed in {elapsed_time:.2f} seconds.")
    except KeyboardInterrupt:
        print("\nStopped by user.")

if __name__ == "__main__":
    main()