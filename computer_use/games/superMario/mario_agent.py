import time
import numpy as np
import concurrent.futures
import argparse

from games.superMario.workers import worker_short, worker_long

# System prompt remains constant
system_prompt = (
    "You are an intelligent Super Mario gameplay agent that controls Mario, search for and execute optimal path given each game state. Prioritize survival over speed."
)

def main():
    """
    Spawns a number of short-term and/or long-term workers based on user-defined parameters.
    """
    parser = argparse.ArgumentParser(
        description="Super Mario gameplay agent with configurable concurrent workers."
    )
    parser.add_argument("--api_provider", type=str, default="anthropic",
                        help="API provider to use.")
    parser.add_argument("--model_name", type=str, default="claude-3-7-sonnet-20250219",
                        help="Model name.")
    parser.add_argument("--concurrency_interval", type=float, default=0.5,
                        help="Interval in seconds between starting workers.")
    parser.add_argument("--api_response_latency_estimate", type=float, default=7.0,
                        help="Estimated API response latency in seconds.")
    parser.add_argument("--policy", type=str, default="alternate", choices=["mixed", "alternate", "long", "short"],
                        help="Worker policy: 'long', or 'short'. In 'long' or 'short' modes only those workers are enabled.")

    args = parser.parse_args()

    num_threads = int(args.api_response_latency_estimate / args.concurrency_interval)
    offsets = [i * args.concurrency_interval for i in range(num_threads)]

    print(f"Starting with {num_threads} threads using policy '{args.policy}'...")
    print(f"API Provider: {args.api_provider}, Model Name: {args.model_name}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        for i in range(num_threads):
            if args.policy == "mixed":
                if i % 2 == 0:
                    executor.submit(worker_long, i, offsets[i], system_prompt, args.api_provider, args.model_name)
                executor.submit(worker_short, i, offsets[i], system_prompt, args.api_provider, args.model_name)
            if args.policy == "alternate":
                # Alternate between long and short workers.
                if i % 2 == 0:
                    executor.submit(worker_long, i, offsets[i], system_prompt, args.api_provider, args.model_name)
                else:    
                    executor.submit(worker_short, i, offsets[i], system_prompt, args.api_provider, args.model_name)
            elif args.policy == "long":
                executor.submit(worker_long, i, offsets[i], system_prompt, args.api_provider, args.model_name)
            elif args.policy == "short":
                executor.submit(worker_short, i, offsets[i], system_prompt, args.api_provider, args.model_name)

        try:
            while True:
                time.sleep(0.25)
        except KeyboardInterrupt:
            print("\nMain thread interrupted. Exiting all threads...")

if __name__ == "__main__":
    main()