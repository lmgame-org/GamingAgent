import argparse
import subprocess
import sys
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Run AI Agent in Game Environment")
    parser.add_argument("--game", type=str, default="all", help="Name of the game to run or 'all'")
    parser.add_argument("--platform", type=str, default="pygame", help="Platform type: 'pygame' or 'emulator'")
    parser.add_argument("--model", type=str, default="gpt-4", help="Model name (e.g., gpt-4, o3-mini)")
    parser.add_argument("--provider", type=str, default="openai", help="API provider (openai, anthropic, etc.)")
    parser.add_argument("--max-steps", type=int, default=100, help="Maximum steps to run per game")
    parser.add_argument("--headless", action="store_true", help="Run without display interaction")
    return parser.parse_args()

def maybe_launch_game(game, platform):
    if platform == "pygame" and game in ["sokoban", "tile_2048", "all"]:
        game_path = os.path.join("src", "pygame", f'{game}',f"{game}.py")
        if os.path.exists(game_path):
            print(f"[main.py] Launching game window: {game_path}")
            subprocess.Popen([sys.executable, game_path])
        else:
            print(f"[main.py] Game file not found: {game_path}")
            
def main():
    args = parse_args()

    if args.platform == "pygame":
        maybe_launch_game(args.game, args.platform)
        from src.game_env import run_game_env
        run_game_env(
            game=args.game,
            model_name=args.model,
            provider=args.provider,
            max_steps=args.max_steps,
            headless=args.headless,
        )
    # elif args.platform == "emulator":
    #     from src.env.game_emulator_env import run_game_env
    #     run_game_env(
    #         game=args.game,
    #         model_name=args.model,
    #         provider=args.provider,
    #         max_steps=args.max_steps,
    #         headless=args.headless,
    #     )
    else:
        raise ValueError(f"[main.py] Unsupported platform: {args.platform}")

if __name__ == "__main__":
    main()
