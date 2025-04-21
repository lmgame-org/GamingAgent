import argparse
import retro
import asyncio
from gamingagent.envs.retro_env import ClassicVideoGameEnv
from gamingagent.agents.mario_agent import MarioAgent
from gamingagent.providers import APIProviderManager

async def run_episode(env, agent, episode_num, verbosity):
    """Run a single episode with the agent."""
    observation = env.reset()
    t = 0
    total_reward = 0
    
    # Start recording if enabled
    agent.start_recording(episode_num)
    
    while True:
        # Get action from agent asynchronously
        action = await agent.select_action_async(observation)
        
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
            
    # Stop recording
    agent.stop_recording()
    return total_reward

async def main():
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
        "--model",
        default="claude-3-opus-20240229",
        help="the model to use for the API provider",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1000,
        help="maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="temperature for the API provider",
    )
    parser.add_argument(
        "--concurrency-interval",
        type=float,
        default=1.0,
        help="interval between worker starts in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=4,
        help="number of worker threads (default: 4)",
    )
    parser.add_argument(
        "--api-response-latency",
        type=float,
        default=7.0,
        help="estimated API response latency in seconds",
    )
    parser.add_argument(
        "--frame-delay",
        type=float,
        default=0.016,  # 60 FPS
        help="frame delay in seconds (controls game speed)",
    )
    args = parser.parse_args()

    print(f"Initializing environment with game: {args.game}")
    # Create environment
    env = ClassicVideoGameEnv(
        game=args.game,
        state=args.state,
        scenario=args.scenario,
        record=args.record,
        render_mode="human",
        frame_delay=args.frame_delay
    )

    print("Initializing API provider manager...")
    # Initialize API provider manager
    provider_manager = APIProviderManager()
    provider_manager.initialize_providers()
    
    # Verify Anthropic provider is available
    if not provider_manager.anthropic:
        raise RuntimeError("Anthropic provider not initialized. Please check your API key.")
    print("API provider manager initialized successfully")

    print(f"Creating Mario agent with model: {args.model}")
    # Create Mario agent
    agent = MarioAgent(
        env=env,
        provider_manager=provider_manager,
        model_name=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        num_threads=args.num_threads,
        concurrency_interval=args.concurrency_interval,
        api_response_latency_estimate=args.api_response_latency,
        record_bk2=args.record
    )
    
    verbosity = args.verbose - args.quiet
    
    try:
        episode = 0
        while True:
            total_reward = await run_episode(env, agent, episode, verbosity)
            episode += 1
            
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up
        await agent.close_async()
        env.close()

if __name__ == "__main__":
    asyncio.run(main())
