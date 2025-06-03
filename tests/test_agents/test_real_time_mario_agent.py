import argparse
import retro
from gamingagent.envs.real_time_video_game_env import RealTimeVideoGameEnv
from gamingagent.agents.real_time_mario_agent import RealTimeMarioAgent
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--game",
        default="SuperMarioBros-Nes",
        help="the name or path for the game to run",
    )
    parser.add_argument(
        "--state",
        default=retro.State.DEFAULT,
        help="the initial state file to load, minus the extension",
    )
    parser.add_argument(
        "--scenario",
        "-s",
        default="scenario",
        help="the scenario file to load, minus the extension",
    )
    parser.add_argument("--record", "-r", action="store_true", help="record bk2 movies")
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=1,
        help="increase verbosity (can be specified multiple times)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="count",
        default=0,
        help="decrease verbosity (can be specified multiple times)",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=30.0,
        help="target frames per second for the game",
    )
    parser.add_argument(
        "--api-provider",
        type=str,
        default="anthropic",
        help="API provider to use",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="claude-3-opus-20240229",
        help="Model name to use",
    )
    args = parser.parse_args()

    # Create real-time environment
    env = RealTimeVideoGameEnv(
        game=args.game,
        state=args.state,
        scenario=args.scenario,
        record=args.record,
        render_mode="human",
        target_fps=args.target_fps
    )

    # Create real-time Mario agent
    agent = RealTimeMarioAgent(
        env=env,
        game_name=args.game,
        api_provider=args.api_provider,
        model_name=args.model_name
    )
    
    verbosity = args.verbose - args.quiet
    
    try:
        # Reset environment and start simulation thread
        observation = env.reset()
        print("Environment reset complete")
        
        # Print button mapping for debugging
        print("Button mapping:", env.buttons)
        print("Number of buttons:", env.num_buttons)
        
        # For FPS monitoring
        last_fps_print = time.time()
        frame_count = 0
        
        # Start the agent's worker threads
        agent.start()
        print("Agent worker threads started")
        
        # Main loop to execute actions
        while True:
            # Get current frame
            current_frame = env.get_latest_frame()
            if current_frame is None:
                continue
                
            # Get action from agent
            action = agent.select_action(current_frame)
            
            # Execute action
            step_result = env.step(action)
            frame_count += 1
            
            # Print FPS and action info every second
            current_time = time.time()
            if current_time - last_fps_print >= 1.0:
                fps = frame_count / (current_time - last_fps_print)
                if verbosity > 0:
                    print(f"Game running at {fps:.1f} FPS")
                    if isinstance(step_result, tuple):
                        obs, reward, terminated, truncated, info = step_result
                        print(f"Step: {env.step_count}")
                        print(f"Action state: {action}")
                        if info:
                            print(f"Info: {info}")
                frame_count = 0
                last_fps_print = current_time
            
            # Check if episode is done
            if isinstance(step_result, tuple):
                _, _, terminated, truncated, _ = step_result
                if terminated or truncated:
                    print("Episode done, resetting...")
                    env.reset()
            
    except KeyboardInterrupt:
        print("\nStopping agent...")
    finally:
        agent.close()
        env.close()
        print("Cleanup complete")

if __name__ == "__main__":
    main()
