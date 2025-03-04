import time
import numpy as np
import concurrent.futures
import argparse

from games.tetris.workers import worker_tetris
from games.tetris.speculators import speculator_tetris

system_prompt = (
    "You are an expert AI agent specialized in playing Tetris gameplay, search for and execute optimal moves given each game state. Prioritize line clearing over speed."
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
    parser.add_argument("--concurrency_interval", type=float, default=1,
                        help="Interval in seconds between workers.")
    parser.add_argument("--api_response_latency_estimate", type=float, default=5,
                        help="Estimated API response latency in seconds.")
    parser.add_argument("--speculator_count", type=int, default=1,
                        help="Number of speculators.")
    parser.add_argument("--speculation_size", type=int, default=10,
                        help="Max number of state-action pairs to consider when generating new plannnig prompt.")
    parser.add_argument("--speculation_delay", type=float, default=30,
                        help="Number of seconds before first planning prompt is generated.")
    parser.add_argument("--control_time", type=float, default=4,
                        help="Worker control time.")
    parser.add_argument("--policy", type=str, default="fixed", 
                        choices=["fixed"],
                        help="Worker policy")

    args = parser.parse_args()

    # FIXME (lanxiang): scale the number of speculators
    assert args.speculator_count == 1, f"more than 1 speculator worker is not yet supported."

    worker_span = args.control_time + args.concurrency_interval
    num_threads = int(args.api_response_latency_estimate // worker_span)
    
    if args.api_response_latency_estimate % worker_span != 0:
        num_threads += 1
    
    # Create an offset list
    offsets = [i * (args.control_time + args.concurrency_interval) for i in range(num_threads)]
    
    
    num_threads += args.speculator_count

    print(f"Starting with {num_threads} threads ({args.speculator_count} threads are speculators) using policy '{args.policy}'...")
    print(f"API Provider: {args.api_provider}, Model Name: {args.model_name}")

    # Spawn workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        executor.submit(
            speculator_tetris, 0, args.speculation_delay, system_prompt,
            args.api_provider, args.model_name, args.speculation_size, 20
        )
        for i in range(num_threads-1):
            if args.policy == "fixed":
                executor.submit(
                    worker_tetris, i, offsets[i], system_prompt,
                    args.api_provider, args.model_name, args.control_time
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