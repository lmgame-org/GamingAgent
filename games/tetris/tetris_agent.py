import time
import concurrent.futures
import argparse

from games.tetris.workers import tetris_worker

system_prompt = (
    "You are an expert AI agent specialized in playing Tetris gameplay, search for and execute optimal moves given each game state. Prioritize line clearing over speed."
)

tetris_board_reader_system_prompt = (
    "You are an expert AI agent specialized in converting a Tetris game grid to a text table."
)

tetris_board_aggregator_system_prompt = (
    "You are an expert AI agent specialized in aggregating subtable into a bigger text table, please take ptach size offsets into consideration."
)

def main():
    """
    Spawns a number of Tetris worker threads (without a speculator).
    Each worker analyzes the Tetris board and chooses moves accordingly.
    """
    parser = argparse.ArgumentParser(
        description="Tetris gameplay agent with configurable concurrent workers (speculator disabled)."
    )
    parser.add_argument("--api_provider", type=str, default="anthropic",
                        help="API provider to use.")
    parser.add_argument("--model_name", type=str, default="claude-3-7-sonnet-20250219",
                        help="Model name.")
    parser.add_argument("--board_reader_api_provider", type=str, default="anthropic",
                        help="Board reader API provider to use.")
    parser.add_argument("--board_reader_model_name", type=str, default="claude-3-7-sonnet-20250219",
                        help="Board reader model name.")
    parser.add_argument("--modality", type=str, default="vision-text", 
                        choices=["vision", "vision-text", "text-only"],
                        help="Employ vision reasoning or vision-text reasoning mode.")
    parser.add_argument("--concurrency_interval", type=float, default=2,
                        help="Interval in seconds between workers.")
    parser.add_argument("--api_response_latency_estimate", type=float, default=6,
                        help="Estimated API response latency in seconds.")
    parser.add_argument("--control_time", type=float, default=4,
                        help="Worker control time.")
    parser.add_argument("--input_type", type=str, default="read-from-game-backend",
                        help="Game state input type.")
    parser.add_argument("--policy", type=str, default="fixed", 
                        choices=["fixed"],
                        help="Worker policy")
    parser.add_argument("--cache_folder", type=str, default="games/tetris/Python-Tetris-Game-Pygame/cache/tetris",
                        help="Game state path.")

    args = parser.parse_args()

    # Determine number of worker threads based on the API latency estimate.
    worker_span = args.control_time + args.concurrency_interval
    num_threads = int(args.api_response_latency_estimate // worker_span)
    if args.api_response_latency_estimate % worker_span != 0:
        num_threads += 1

    # Create an offset list for each worker.
    offsets = [i * (args.control_time + args.concurrency_interval) for i in range(num_threads)]

    print(f"Starting with {num_threads} worker threads using policy '{args.policy}'...")
    print(f"API Provider: {args.api_provider}, Model Name: {args.model_name}")

    # Spawn worker threads.
    # with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
    #     for i in range(num_threads):
    #         if args.policy == "fixed":
    #             executor.submit(
    #                 tetris_worker,
    #                 i,
    #                 offsets[i],
    #                 system_prompt,
    #                 tetris_board_reader_system_prompt,
    #                 tetris_board_aggregator_system_prompt,
    #                 args.api_provider,
    #                 args.model_name,
    #                 args.board_reader_api_provider,
    #                 args.board_reader_model_name,
    #                 args.modality,
    #                 input_type=args.input_type,
    #                 plan_seconds=args.control_time,
    #                 cache_folder=args.cache_folder
    #             )
    #         else:
    #             raise NotImplementedError(f"policy: {args.policy} not implemented.")

    #     try:
    #         while True:
    #             time.sleep(0.25)
    #     except KeyboardInterrupt:
    #         print("\nMain thread interrupted. Exiting all threads...")

    if args.policy == "fixed":
        # Directly call tetris_worker with a single worker id (0) and offset 0
        tetris_worker(
            system_prompt,
            args.api_provider,
            args.model_name)
    else:
        raise NotImplementedError(f"policy: {args.policy} not implemented.")

if __name__ == "__main__":
    main()
