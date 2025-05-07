import argparse
import retro
from gamingagent.envs.classic_video_game_env import ClassicVideoGameEnv
from gamingagent.agents.mario_agent import MarioAgent

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
    args = parser.parse_args()

    # Create environment
    env = ClassicVideoGameEnv(
        game=args.game,
        state=args.state,
        scenario=args.scenario,
        record=args.record,
        render_mode="human"
    )

    # Create Mario agent
    agent = MarioAgent(
        env=env,
        game_name=args.game,
        api_provider="anthropic",
        model_name="claude-3-opus-20240229"
    )
    
    verbosity = args.verbose - args.quiet
    
    try:
        while True:
            observation = env.reset()
            t = 0
            total_reward = 0
            
            while True:
                # Get action from agent
                action = agent.select_action(observation)
                
                # Take step in environment
                observation, reward, terminated, truncated, info = env.step(action)
                t += 1
                total_reward += reward
                
                # Print info periodically
                if t % 10 == 0 and verbosity > 1:
                    infostr = ""
                    if info:
                        infostr = ", info: " + ", ".join(
                            ["%s=%i" % (k, v) for k, v in info.items()],
                        )
                    print(f"t={t}{infostr}")
                
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

if __name__ == "__main__":
    main()
