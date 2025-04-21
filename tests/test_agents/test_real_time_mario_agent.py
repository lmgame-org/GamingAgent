import argparse
import retro
from gamingagent.envs.retro_env import RealTimeClassicVideoGameEnv
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
        default=60.0,
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
    env = RealTimeClassicVideoGameEnv(
        game=args.game,
        state=args.state,
        scenario=args.scenario,
        record=args.record,
        render_mode="human",
        target_fps=args.target_fps,
        frame_skip=2  # Default frame skip for short worker
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
        while True:
            # Reset environment and start simulation thread
            observation = env.reset()
            t = 0
            total_reward = 0
            
            while True:
                # Get action from agent
                action = agent.select_action(observation)
                
                # Take step in environment
                step_result = env.step(action)
                if step_result is None:
                    time.sleep(0.1)  # Wait for simulation thread
                    continue
                    
                observation, reward, terminated, truncated, info = step_result
                t += 1
                total_reward += reward
                
                # Print info periodically
                if t % 10 == 0 and verbosity > 1:
                    infostr = ""
                    if info:
                        infostr = ", info: " + ", ".join(
                            ["%s=%i" % (k, v) for k, v in info.items()],
                        )
                    print(f"t={t}{infostr}, fps={env.get_fps():.1f}")
                
                # Print rewards
                if verbosity > 0:
                    if reward > 0:
                        print(f"t={t} got reward: {reward}, current reward: {total_reward}")
                    elif reward < 0:
                        print(f"t={t} got penalty: {reward}, current reward: {total_reward}")
                
                # Check if episode is done
                if terminated or truncated:
                    if verbosity >= 0:
                        print(f"done! total reward: time={t}, reward={total_reward}")
                        input("press enter to continue")
                        print()
                    break
                    
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        agent.close()

if __name__ == "__main__":
    main()
