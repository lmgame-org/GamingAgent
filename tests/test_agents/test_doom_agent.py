import argparse
from gamingagent.envs.custom_05_doom.Doom_env import DoomEnvWrapper
from gamingagent.agents.base_agent import BaseAgent
# Import our agent modules
from tests.test_agents.modules.base_module import Base_module
from tests.test_agents.modules.perception import PerceptionModule
from tests.test_agents.modules.memory import MemoryModule
from tests.test_agents.modules.reasoning import ReasoningModule
from tools.serving import APIManager
from gamingagent.modules.perception_module import PerceptionModule
from gamingagent.modules.memory_module import MemoryModule
from gamingagent.modules.reasoning_module import ReasoningModule


class DoomAgent:
    def __init__(self, model_name="gpt-4o", agent_mode="full"):
        self.model_name = model_name
        self.agent_mode = agent_mode
        self.last_action = None

        self.perception_module = None
        self.memory_module = None
        self.reasoning_module = None

        if self.agent_mode == "full":
            self.perception_module = PerceptionModule(model_name, "cache/doom")
            self.memory_module = MemoryModule(model_name, "cache/doom/memory.json", "cache/doom")
            self.reasoning_module = ReasoningModule(model_name)
        elif self.agent_mode == "memory_reasoning":
            self.memory_module = MemoryModule(model_name, "cache/doom/memory.json", "cache/doom")
            self.reasoning_module = ReasoningModule(model_name)

    async def get_action(self, observation):
        if self.agent_mode == "full":
            perception_data = self.perception_module.process_observation(observation)  # Updated from analyze_frame to process_observation
            self.memory_module.add_game_state(perception_data, self.last_action)
            memory_summary = self.memory_module.get_memory_summary()
            action_plan = await self.reasoning_module.plan_action(perception_data, memory_summary)
            self.last_action = action_plan["move"]
            return action_plan
        elif self.agent_mode == "memory_reasoning":
            memory_summary = self.memory_module.get_memory_summary()
            action_plan = await self.reasoning_module.plan_action(None, memory_summary)
            self.last_action = action_plan["move"]
            return action_plan


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
        game_name="doom",  # Added the required game_name argument
        game_config_path=args.game_config_path,
        observation_mode="vision",
        base_log_dir="cache/doom/test_run",
        render_mode_human=True
    )

    # Fixing the action_mapping issue
    if not isinstance(env.adapter.move_to_action_idx, dict) or not env.adapter.move_to_action_idx:
        env.adapter.move_to_action_idx = {
            "move_up": 0,
            "move_down": 1,
            "move_left": 2,
            "move_right": 3,
            "attack": 4  # Updated to match the available_buttons
        }

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

    # Initialize Perception and Reasoning Modules
    perception_module = PerceptionModule(model_name=args.model_name, cache_dir="cache/doom/test_run")
    reasoning_module = ReasoningModule(model_name=args.model_name, cache_dir="cache/doom/test_run")

    # Attach modules to the agent
    agent.perception_module = perception_module
    agent.reasoning_module = reasoning_module

    verbosity = args.verbose - args.quiet

    try:
        while True:
            observation = env.reset()
            t = 0
            total_reward = 0

            while True:
                # Get action from agent
                action = agent.get_action(observation)  # Updated from select_action to get_action

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


async def test_doom_agent():
    env = DoomEnvWrapper(
        game_name="doom",
        game_config_path="configs/custom_05_doom/config.yaml",
        observation_mode="vision",
        base_log_dir="cache/doom/test_run",
        render_mode_human=True
    )
    agent = DoomAgent(model_name="gpt-4o", agent_mode="full")

    observation, _ = env.reset()  # Extract observation from the tuple
    done = False
    while not done:
        action_plan = await agent.get_action(observation)
        print(f"Action: {action_plan['move']}, Thought: {action_plan['thought']}")
        observation, reward, done, info = env.step(action_plan["move"])

    env.close()


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_doom_agent())