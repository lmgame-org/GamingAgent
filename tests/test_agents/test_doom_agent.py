import argparse
from gamingagent.envs.custom_05_doom.Doom_env import DoomEnvWrapper
from gamingagent.agents.base_agent import BaseAgent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--game_config_path",
        default="configs/custom_05_doom/config.yaml",
        help="Path to the game configuration file",
    )
    parser.add_argument(
        "--api_provider",
        default="openai",
        help="API provider for the agent",
    )
    parser.add_argument(
        "--model_name",
        default="gpt-4o",
        help="Model name for the agent",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=1,
        help="Increase verbosity (can be specified multiple times)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="count",
        default=0,
        help="Decrease verbosity (can be specified multiple times)",
    )
    args = parser.parse_args()

    # Create environment
    env = DoomEnvWrapper(
        game_config_path=args.game_config_path,
        observation_mode="vision",
        base_log_dir="cache/doom/test_run",
        render_mode_human=True
    )

    # Create Doom agent
    agent = BaseAgent(
        game_name="doom",
        model_name=args.model_name,
        config_path=args.game_config_path,
        harness=True,
        max_memory=10,
        cache_dir="cache/doom/test_run",
        observation_mode="vision"
    )

    # Fixing the action_mapping issue
    if not isinstance(env.adapter.move_to_action_idx, dict) or not env.adapter.move_to_action_idx:
        env.adapter.move_to_action_idx = {
            "move_left": 0,
            "move_right": 1,
            "attack": 2
        }

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