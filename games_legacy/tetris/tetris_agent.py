import time
import numpy as np
import concurrent.futures
import argparse

from games.tetris.workers import tetris_worker
from games.tetris.speculators import tetris_speculator

system_prompt = (
    "You are an expert AI agent specialized in playing Tetris gameplay, search for and execute optimal moves given each game state. Prioritize line clearing over speed."
)

tetris_board_reader_system_prompt = (
    "You are an expert AI agent specialized in converting a Tetris game grid to a text table."
)

tetris_board_aggregator_system_prompt = (
    "You are an expert AI agent specialized in aggregating subtable into a bigger text table, please take ptach size offsets into consideration."
)

speculator_system_prompt = (
    "You are an expert AI agent specialized in playing Tetris gameplay, search for and execute optimal moves given each game state. Prioritize line clearing over speed."
)

def main():
    """
    Spawns a number of short-term and/or long-term Tetris workers based on user-defined parameters.
    Each worker will analyze the Tetris board and choose moves accordingly.
    """
    parser = argparse.ArgumentParser(
        description="Tetris gameplay agent with configurable concurrent workers."
    )
    parser.add_argument("--api_provider", type=str, default="anthropic",
                        help="API provider to use.")
    parser.add_argument("--model_name", type=str, default="claude-3-7-sonnet-20250219",
                        help="Model name.")
    parser.add_argument("--board_reader_api_provider", type=str, default="anthropic",
                        help="board reader API provider to use.")
    parser.add_argument("--board_reader_model_name", type=str, default="claude-3-7-sonnet-20250219",
                        help="Board reader model name.")
    parser.add_argument("--modality", type=str, default="vision-text", 
                        choices=["vision", "vision-text", "text-only"],
                        help="Employ vision reasoning or vision-text reasoning mode.")
    parser.add_argument("--concurrency_interval", type=float, default=2,
                        help="Interval in seconds between workers.")
    parser.add_argument("--api_response_latency_estimate", type=float, default=6,
                        help="Estimated API response latency in seconds.")
    parser.add_argument("--speculator_count", type=int, default=1,
                        help="Number of speculators.")
    parser.add_argument("--speculation_size", type=int, default=10,
                        help="Max number of state-action pairs to consider when generating new plannnig prompt.")
    parser.add_argument("--speculation_delay", type=float, default=60,
                        help="Number of seconds before first planning prompt is generated.")
    parser.add_argument("--control_time", type=float, default=4,
                        help="Worker control time.")
    parser.add_argument("--input_type", type=str, default="read-from-game-backend",
                        help="Game state input type.")
    parser.add_argument("--policy", type=str, default="fixed", 
                        choices=["fixed"],
                        help="Worker policy")
    parser.add_argument("--cache_folder", type=str, default="/Users/lhu/workspace/game_arena_env/Python-Tetris-Game-Pygame/cache/tetris",
                        help="game state path.")
    

    args = parser.parse_args()
    # FIXME (lanxiang): scale the number of speculators
    assert args.speculator_count == 1, f"more than 1 speculator worker ({args.speculator_count} is passed) is not yet supported."

    worker_span = args.control_time + args.concurrency_interval
    num_threads = int(args.api_response_latency_estimate // worker_span)
    
    if args.api_response_latency_estimate % worker_span != 0:
        num_threads += 1
    
    # Create an offset list
    offsets = [i * (args.control_time + args.concurrency_interval) for i in range(num_threads)]
    
    num_threads += args.speculator_count

    print(f"Starting with {num_threads} threads, ({args.speculator_count} thread[s] are speculator[s]) using policy '{args.policy}'...")
    print(f"API Provider: {args.api_provider}, Model Name: {args.model_name}")

    # Spawn workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # TODO: add multiple speculator if needed
        if args.speculator_count:
            executor.submit(
                tetris_speculator, 0, args.speculation_delay, speculator_system_prompt,
                args.api_provider, args.model_name, args.speculation_size, 20
            )
        for i in range(num_threads-1):
            if args.policy == "fixed":
                executor.submit(
                    tetris_worker, i, offsets[i], system_prompt, tetris_board_reader_system_prompt, tetris_board_aggregator_system_prompt,
                    args.api_provider, args.model_name, 
                    args.board_reader_api_provider, args.board_reader_model_name, 
                    args.modality, input_type=args.input_type,
                    plan_seconds=args.control_time, cache_folder=args.cache_folder
                )
            else:
                raise NotImplementedError(f"policy: {args.policy} not implemented.")

        try:
            while True:
                time.sleep(0.25)
        except KeyboardInterrupt:
            print("\nMain thread interrupted. Exiting all threads...")

if __name__ == "__main__":
    main()