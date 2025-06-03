import time
import numpy as np
import concurrent.futures
import argparse
from collections import deque, Counter
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
import datetime

import os
import json
import re
import pyautogui
# from games.ace_attorney.reflection_worker import ReflectionTracker


from games.ace_attorney.workers import (
    ace_attorney_worker, 
    perform_move, 
    ace_evidence_worker, 
    short_term_memory_worker,
    vision_only_reasoning_worker,
    long_term_memory_worker,
    memory_retrieval_worker,
    vision_only_ace_attorney_worker,
    check_end_statement,
    check_skip_conversation,
    handle_skip_conversation
)
from tools.utils import str2bool, encode_image, log_output, get_annotate_img, capture_game_window, log_game_event
from collections import Counter

# Global base cache directory
BASE_CACHE_DIR = "cache/ace_attorney"

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
    "You are an expert AI agent specialized in playing Ace Attorney games. Your goal is to solve cases by gathering evidence, cross-examining witnesses, and presenting the correct evidence at the right time to prove your client's innocence. "
)

def main():
    # reflection = ReflectionTracker()

    parser = argparse.ArgumentParser(description="Ace Attorney AI Agent")
    parser.add_argument("--api_provider", type=str, default="anthropic", help="API provider to use.")
    parser.add_argument("--model_name", type=str, default="claude-3-7-sonnet-20250219", help="LLM model name.")
    parser.add_argument("--modality", type=str, default="vision-text", 
                       choices=["text-only", "vision-text", "vision-only"],
                       help="modality used.")
    parser.add_argument("--thinking", type=str, default="False", help="Whether to use deep thinking.")
    parser.add_argument("--episode_name", type=str, default="The_First_Turnabout", 
                       help="Name of the current episode being played.")
    parser.add_argument("--num_threads", type=int, default=1, help="Number of parallel threads to launch.")
    args = parser.parse_args()

    prev_response = ""

    # Create timestamped cache directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Handle model names with forward slashes
    model_name_for_cache = args.model_name.split('/')[-1] if '/' in args.model_name else args.model_name
    if "claude" in args.model_name:
        cache_dir = os.path.join(BASE_CACHE_DIR, f"{timestamp}_{args.episode_name}_{args.modality}_{args.api_provider}_{model_name_for_cache}_{args.thinking}")
    else:
        cache_dir = os.path.join(BASE_CACHE_DIR, f"{timestamp}_{args.episode_name}_{args.modality}_{args.api_provider}_{model_name_for_cache}")
    
    # Create the cache directory if it doesn't exist
    os.makedirs(BASE_CACHE_DIR, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    # Also ensure the base cache directory exists (for backward compatibility)
    print(f"Using cache directory: {cache_dir}")

    thinking_bool = str2bool(args.thinking)

    print("--------------------------------Start Evidence Worker--------------------------------")
    evidence_result = ace_evidence_worker(
        system_prompt,
        args.api_provider,
        args.model_name,
        prev_response,
        thinking=thinking_bool,
        modality=args.modality,
        episode_name=args.episode_name,
        cache_dir=cache_dir
    )
    decision_state = None

    try:
        while True:
            start_time = time.time()

            # Choose the appropriate worker based on modality
            if args.modality == "vision-only":
                worker_func = vision_only_ace_attorney_worker
            else:
                worker_func = ace_attorney_worker

            # Self-consistency launch with 1-second interval between threads
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
                futures = []
                for i in range(args.num_threads):
                    futures.append(
                        executor.submit(
                            worker_func,
                            system_prompt,
                            args.api_provider,
                            args.model_name,
                            prev_response,
                            thinking=thinking_bool,
                            modality=args.modality,
                            episode_name=args.episode_name,
                            decision_state=decision_state,
                            cache_dir=cache_dir
                        )
                    )
                    if i < args.num_threads - 1:  # Don't sleep after the last thread
                        time.sleep(2)  # Add 1-second interval between launching threads
                
                # Wait until all threads finish
                concurrent.futures.wait(futures)
                results = [f.result() for f in futures]
            
            # Check for skip conversation in the first result's dialog
            if results and results[0] and "dialog" in results[0]:
                dialog = results[0]["dialog"]
                skip_dialogs = check_skip_conversation(dialog, args.episode_name)
                if skip_dialogs:
                    print("\n" + "="*70)
                    print("=== Skip Conversation Detected ===")
                    print(f"├── Episode: {args.episode_name}")
                    print(f"├── Number of dialogs to skip: {len(skip_dialogs)}")
                    print("└── Starting skip sequence...")
                    print("="*70 + "\n")
                    
                    # Handle the skip conversation
                    skip_result = handle_skip_conversation(
                        system_prompt,
                        args.api_provider,
                        args.model_name,
                        prev_response,
                        thinking_bool,
                        args.modality,
                        args.episode_name,
                        dialog,
                        skip_dialogs,
                        cache_dir=cache_dir
                    )
                    
                    if skip_result:
                        # Replace all results with the skip result
                        results = [skip_result]
            
            print("\n" + "="*70)
            print("=== Analysis Results ===")
            print("="*70)
            for i, result in enumerate(results, 1):
                if result and "move" in result and "thought" in result and "game_state" in result:
                    print(f"\nThread {i} Analysis:")
                    print(f"├── Game State: {result['game_state']}")
                    print(f"├── Move: {result['move'].strip().lower()}")
                    print(f"├── Thought Process:")
                    print(f"│   ├── Primary Reasoning: {result['thought']}")
                    if "dialog" in result:
                        if isinstance(result['dialog'], dict) and 'name' in result['dialog'] and 'text' in result['dialog']:
                            print(f"│   ├── Dialog Context: {result['dialog']['name']}: {result['dialog']['text']}")
                        else:
                            print(f"│   ├── Dialog Context: {result['dialog']}")
                    if "evidence" in result and result["evidence"]:
                        print(f"│   ├── Evidence Context: {result['evidence']['name']}: {result['evidence']['description']}")
                    if "scene" in result and result["scene"]:
                        print(f"│   └── Scene Context: {result['scene'][:200]}...")
                else:
                    print(f"\nThread {i}: Invalid result")
            print("\n" + "="*70)

            # Collect all moves and thoughts from the results
            moves = []
            thoughts = []
            game_states = []
            dialogs = []
            evidences = []
            scenes = []
            
            for result in results:
                if result and "move" in result and "thought" in result and "game_state" in result:
                    moves.append(result["move"].strip().lower())
                    thoughts.append(result["thought"])
                    game_states.append(result["game_state"])
                    dialogs.append(result.get("dialog", {}))
                    evidences.append(result.get("evidence", {}))
                    scenes.append(result.get("scene", ""))

            if not moves:
                print("[WARNING] No valid moves found in results")
                continue

            # Print vote counts with reasoning
            move_counts = Counter(moves)
            print("\n=== Move Analysis ===")
            for move, count in move_counts.most_common():
                print(f"├── Move: {move}")
                print(f"│   ├── Votes: {count}")
                # Find all thoughts associated with this move
                move_indices = [i for i, m in enumerate(moves) if m == move]
                print(f"│   ├── Supporting Thoughts:")
                for idx in move_indices:
                    print(f"│   │   ├── Thought: {thoughts[idx]}")
                    if dialogs[idx]:
                        if isinstance(dialogs[idx], dict) and 'name' in dialogs[idx] and 'text' in dialogs[idx]:
                            print(f"│   │   ├── Dialog: {dialogs[idx]['name']}: {dialogs[idx]['text']}")
                        else:
                            print(f"│   │   ├── Dialog: {dialogs[idx]}")
                    if evidences[idx]:
                        print(f"│   │   ├── Evidence: {evidences[idx]['name']}: {evidences[idx]['description']}")
                    print(f"│   │   └── Scene: {scenes[idx][:150]}...")
            print("└──" + "─"*66)

            # Perform majority vote on moves
            chosen_move = majority_vote_move(moves)
            chosen_idx = moves.index(chosen_move)
            chosen_thought = thoughts[chosen_idx]
            chosen_game_state = game_states[chosen_idx]
            chosen_dialog = dialogs[chosen_idx]
            chosen_evidence = evidences[chosen_idx]
            chosen_scene = scenes[chosen_idx]

            print("\n=== Final Decision ===")
            print(f"├── Game State: {chosen_game_state}")
            print(f"├── Chosen Move: {chosen_move}")
            print(f"├── Decision Reasoning:")
            print(f"│   ├── Primary Thought: {chosen_thought}")
            if chosen_dialog:
                if isinstance(chosen_dialog, dict) and 'name' in chosen_dialog and 'text' in chosen_dialog:
                    print(f"│   ├── Dialog Context: {chosen_dialog['name']}: {chosen_dialog['text']}")
                else:
                    print(f"│   ├── Dialog Context: {chosen_dialog}")
            if chosen_evidence:
                print(f"│   ├── Evidence Context: {chosen_evidence['name']}: {chosen_evidence['description']}")
            print(f"│   └── Scene Context: {chosen_scene[:200]}...")
            print(f"└── Execution Status: Pending")
            print("="*70 + "\n")
            
            # Log the final decision
            log_game_event(f"Final Decision - State: {chosen_game_state}, Move: {chosen_move}, Thought: {chosen_thought}, Dialog: {chosen_dialog}, Evidence: {chosen_evidence}, Scene: {chosen_scene[:150]}...", 
                          cache_dir=cache_dir)

            # Perform the chosen move
            perform_move(chosen_move)
            
            # Check if we've reached the end statement
            if check_end_statement(chosen_dialog, args.episode_name):
                print("\n=== End Statement Reached ===")
                print(f"Ending episode: {args.episode_name}")
                break
            
            # Update previous response with game state, move, thought and scene
            prev_response = f"game_state: {chosen_game_state}\nmove: {chosen_move}\nthought: {chosen_thought}"

            # Update short-term memory with the chosen response
            short_term_memory_worker(
                system_prompt,
                args.api_provider,
                args.model_name,
                prev_response,
                thinking=thinking_bool,
                modality=args.modality,
                episode_name=args.episode_name,
                cache_dir=cache_dir
            )

            # Record presented evidence into long-term memory as dialog format
            if chosen_move == "x" and chosen_evidence and chosen_evidence.get("name"):
                presentation_dialog = {
                    "name": "Phoenix",
                    "text": f"I present the {chosen_evidence['name']}."
                }
                long_term_memory_worker(
                    system_prompt,
                    args.api_provider,
                    args.model_name,
                    prev_response,
                    thinking=thinking_bool,
                    modality=args.modality,
                    episode_name=args.episode_name,
                    dialog=presentation_dialog,
                    cache_dir=cache_dir
                )
            if dialog == {
                    "name": "Mia",
                    "text": "Read this note out loud."
                }:
                evidence_new={
                    "name": "Mia's Memo",
                    "text": "A list of people's names in Mia's handwriting.",
                    "description": "A light-colored document filled with typed text, viewed at an angle, displayed on a gray background within a highlighted evidence slot."

                }
                long_term_memory_worker(
                    system_prompt,
                    args.api_provider,
                    args.model_name,
                    prev_response,
                    thinking=thinking_bool,
                    modality=args.modality,
                    episode_name=args.episode_name,
                    evidence=evidence_new,
                    cache_dir=cache_dir
                )
                dialog_new = {
                    "name": "Phoenix",
                    "text": f"I revceive a new evidence 'Mia's Memo'."
                }
                long_term_memory_worker(
                    system_prompt,
                    args.api_provider,
                    args.model_name,
                    prev_response,
                    thinking=thinking_bool,
                    modality=args.modality,
                    dialog=dialog_new,
                    cache_dir=cache_dir
                )


            if chosen_move == "z" and decision_state and decision_state.get("has_options"):
                decision_state = None  # Reset after confirming choice
            else:
                # Keep the state if returned by worker
                for result in results:
                    if "decision_state" in result:
                        decision_state = result["decision_state"]

            elapsed_time = time.time() - start_time
            time.sleep(4)
            print(f"[INFO] Move executed in {elapsed_time:.2f} seconds\n")
    except KeyboardInterrupt:
        print("\nStopped by user.")

if __name__ == "__main__":
    main()