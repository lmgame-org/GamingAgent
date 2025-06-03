#!/usr/bin/env python3
"""
Example script to run the 2048 agent.
Demonstrates how to initialize and run the agent with different configurations.
"""

import argparse
import os
from gamingagent.agents.twentyFortyEight_agent import TwentyFortyEightAgent

def run_with_args(args):
    """Run the 2048 agent with command line arguments."""
    # Create the agent with specified options
    agent = TwentyFortyEightAgent(
        model_name=args.model,
        harness=(not args.no_harness),  # Default is to use harness
        max_memory=args.memory,
        config_path=args.config
    )
    
    # Initialize the environment with specified parameters
    env = agent.env_init(
        render_mode="rgb_array" if not args.human else "human",
        size=4,  # Standard 4x4 board
        max_pow=16  # Allow up to 2^16 (65536)
    )
    
    if env is None:
        print("Failed to initialize environment. Make sure gymnasium_2048 is installed.")
        print("Try: pip install gymnasium-2048")
        return
    
    print(f"Running game with model: {args.model}")
    print(f"{'Using perception-memory-reasoning pipeline' if not args.no_harness else 'Using base module only'}")
    print(f"Cache directory: {agent.cache_dir}")
    
    # Run the game with specified number of steps
    game_summary = agent.run_game(max_steps=args.steps)
    
    # Print final summary
    print("\n=== Game Summary ===")
    print(f"Steps: {game_summary['total_steps']}")
    print(f"Score: {game_summary['final_score']}")
    print(f"Max Tile: {game_summary['max_tile']}")
    print(f"Reward: {game_summary['total_reward']}")
    print(f"Images saved to: {agent.observations_dir}")

def main():
    """Main function to parse arguments and run the agent."""
    parser = argparse.ArgumentParser(description="Run the 2048 agent")
    
    parser.add_argument("--model", type=str, default="claude-3-7-sonnet-latest",
                        help="Model to use for inference (default: claude-3-7-sonnet-latest)")
    
    parser.add_argument("--steps", type=int, default=100,
                        help="Maximum number of steps to run (default: 100)")
    
    parser.add_argument("--no-harness", action="store_true",
                        help="Use base module only instead of perception-memory-reasoning pipeline")
    
    parser.add_argument("--memory", type=int, default=20,
                        help="Maximum number of memory entries to store (default: 20)")
    
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config file with prompts")
    
    parser.add_argument("--human", action="store_true",
                        help="Use human rendering mode (display window)")
    
    args = parser.parse_args()
    run_with_args(args)

if __name__ == "__main__":
    main() 