import time
import numpy as np
import concurrent.futures
import argparse
from collections import deque, Counter
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
import datetime
import json

import os
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
    normalize_content,
    vision_only_ace_attorney_worker,
    check_end_statement,
    check_skip_conversation,
    handle_skip_conversation,
    evaluate_present_evidence
)
from tools.utils import str2bool, encode_image, log_output, get_annotate_img, capture_game_window, log_game_event
from collections import Counter

# Global base cache directory
BASE_CACHE_DIR = "cache/ace_attorney"

# Load prompts from JSON file
with open("games/ace_attorney/ace_attorney_prompts.json", 'r', encoding='utf-8') as f:
    PROMPTS = json.load(f)

system_prompt = PROMPTS["system_prompt"]

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
    parser.add_argument("--use_mapping_background", type=str2bool, default=True, 
                       help="Whether to use background transcript from mapping.json instead of ace_attorney_1.json")
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
    # cache_dir = "cache/ace_attorney/20250428_113921_The_First_Turnabout_vision-text_openai_o4-mini-2025-04-16"
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
    move_history = deque(maxlen=10) # Initialize move history (adjust maxlen as needed)

    try:
        while True:
            start_time = time.time()

            # In multi-thread mode, run multiple instances in parallel
            with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
                futures = []
                for _ in range(args.num_threads):
                    if args.modality == "vision-only":
                        futures.append(executor.submit(
                            vision_only_reasoning_worker,
                            system_prompt,
                            args.api_provider,
                            args.model_name,
                            prev_response,
                            thinking=thinking_bool,
                            modality=args.modality,
                            episode_name=args.episode_name,
                            cache_dir=cache_dir,
                            use_mapping_background=args.use_mapping_background
                        ))
                    elif args.modality == "vision-text":
                        futures.append(executor.submit(
                            ace_attorney_worker,
                            system_prompt,
                            args.api_provider,
                            args.model_name,
                            prev_response,
                            thinking=thinking_bool,
                            modality=args.modality,
                            episode_name=args.episode_name,
                            decision_state=None,
                            cache_dir=cache_dir,
                            use_mapping_background=args.use_mapping_background
                        ))
                    else:  # text-only
                        futures.append(executor.submit(
                            vision_only_ace_attorney_worker,
                            system_prompt,
                            args.api_provider,
                            args.model_name,
                            prev_response,
                            thinking=thinking_bool,
                            modality=args.modality,
                            episode_name=args.episode_name,
                            cache_dir=cache_dir,
                            use_mapping_background=args.use_mapping_background
                        ))
                
                # Get results from all threads
                results = [future.result() for future in futures]
            
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
            
            # --- Evaluate 'x' move BEFORE logging and executing --- 
            final_move_to_perform = chosen_move # Start with the LLM's choice
            if chosen_move == 'x':
                print("--- Initiating 'x' Move Evaluation ---")
                
                # 1. Get Normalized Current Statement
                if isinstance(chosen_dialog, dict) and chosen_dialog.get("name") and chosen_dialog.get("text"):
                    raw_statement = f"{chosen_dialog['name']}: {chosen_dialog['text']}"
                    normalized_statement = normalize_content(raw_statement, args.episode_name, cache_dir)
                else:
                    normalized_statement = normalize_content(str(chosen_dialog), args.episode_name, cache_dir)
                print(f"   Normalized Statement for Eval: {normalized_statement}")

                # 2. Get Normalized Scene
                normalized_scene = normalize_content(chosen_scene, args.episode_name, cache_dir)
                print(f"   Normalized Scene for Eval: {normalized_scene[:100]}...")

                # 3. Get Normalized Evidence Details (from memory context of the chosen result)
                # We need the memory context that led to the chosen 'x' move
                chosen_memory_context = ""
                if chosen_idx < len(results) and "memory_context" in results[chosen_idx]:
                    chosen_memory_context = results[chosen_idx]["memory_context"]
                
                normalized_evidence_details = "Error: Could not extract evidence details."
                if chosen_memory_context:
                    try:
                        # Memory context should already be normalized by memory_retrieval_worker
                        evidences_section = chosen_memory_context.split("Collected Evidences:")[1].strip()
                        collected_evidences_lines = [e for e in evidences_section.split("\n") if e.strip()]
                        normalized_evidence_details = "\n".join(collected_evidences_lines)
                        print(f"   Normalized Evidence for Eval: Extracted successfully.")
                    except IndexError:
                         print(f"   Normalized Evidence for Eval: Failed to extract from context.")
                         normalized_evidence_details = ""
                else:
                    print(f"   Normalized Evidence for Eval: No memory context found for chosen result.")
                    normalized_evidence_details = ""

                # 4. Call the evaluator
                evaluation = evaluate_present_evidence(
                    move_history=list(move_history), # Pass the actual history as a list
                    current_thought=chosen_thought,
                    game_state=chosen_game_state, # Already normalized
                    current_statement=normalized_statement,
                    scene=normalized_scene, 
                    evidence_details=normalized_evidence_details,
                    api_provider="openai", # Hardcoded for evaluator
                    model_name="o3-2025-04-16" # Specific O3 model
                )

                # --- Log Evaluation Result --- 
                eval_log_file = os.path.join(cache_dir, "evaluator_log.txt")
                log_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_entry = f"[{log_timestamp}] Eval Target Move: 'x' | Result: {evaluation} | Statement: {normalized_statement} | Thought: {chosen_thought}\n"
                try:
                    with open(eval_log_file, 'a', encoding='utf-8') as f:
                        f.write(log_entry)
                except Exception as e:
                    print(f"[ERROR] Failed to write to evaluator log: {e}")
                # --- End Logging ---

                # 5. Override move if evaluation fails - REMOVED/COMMENTED OUT
                # if evaluation == 0:
                #     print("--- Evaluator Result: 'x' move deemed illogical. Overriding to 'z'. ---")
                #     final_move_to_perform = 'z' 
                #     # Optionally update the thought to reflect override? 
                #     # chosen_thought += " [Evaluator Override: Invalid 'x', changed to 'z']"
                # else:
                #     print("--- Evaluator Result: 'x' move deemed logical. Proceeding. ---")
                print("="*70 + "\n") # Separator after evaluation block

            # Log the final decision (always using the original chosen_move now)
            log_game_event(f"Final Decision - State: {chosen_game_state}, Move: {chosen_move}, Thought: {chosen_thought}, Dialog: {chosen_dialog}, Evidence: {chosen_evidence}, Scene: {chosen_scene[:150]}...", 
                          cache_dir=cache_dir)

            # Update move history with the chosen move/thought *before* performing
            move_history.append({"move": chosen_move, "thought": chosen_thought})

            # Perform the chosen move
            perform_move(chosen_move)
            
            # Check if we've reached the end statement
            if check_end_statement(chosen_dialog, args.episode_name):
                print("\n=== End Statement Reached ===")
                print(f"Ending episode: {args.episode_name}")
                break
            
            # Update previous response with game state, move, thought and scene
            prev_response = f"game_state: {chosen_game_state}\ncurrent_statement: {chosen_dialog}\nmove: {final_move_to_perform}\nthought: {chosen_thought}"

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
            if final_move_to_perform == "x" and chosen_evidence and chosen_evidence.get("name"):
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


            if final_move_to_perform == "z" and decision_state and decision_state.get("has_options"):
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