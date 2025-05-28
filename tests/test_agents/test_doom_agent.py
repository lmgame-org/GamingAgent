import argparse
import os
from datetime import datetime
from gamingagent.envs.custom_05_doom.Doom_env import DoomEnvWrapper
from tests.test_agents.modules.doom_base_module import DoomBaseModule
from gamingagent.modules.perception_module import PerceptionModule
from gamingagent.modules.memory_module import MemoryModule
from gamingagent.modules.reasoning_module import ReasoningModule
import asyncio
import json

class DoomAgent:
    def __init__(self, model_name, cache_dir, use_perception=False, use_memory=False, use_reasoning=False, config_path=None):
        self.base_module = DoomBaseModule()
        self.perception_module = PerceptionModule(model_name, cache_dir) if use_perception else None
        self.memory_module = MemoryModule(model_name, os.path.join(cache_dir, "memory.json"), cache_dir) if use_memory else None
        
        # Load config if provided
        config = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        
        # Initialize reasoning module with config
        if use_reasoning:
            reasoning_config = config.get("reasoning_module", {})
            self.reasoning_module = ReasoningModule(
                model_name=model_name,
                cache_dir=cache_dir,
                system_prompt=reasoning_config.get("system_prompt", ""),
                prompt=reasoning_config.get("prompt", "")
            )
        else:
            self.reasoning_module = None
            
        self.last_action = None

    async def get_action(self, observation):
        perception_data = self.perception_module.process_observation(observation) if self.perception_module else None

        if self.memory_module and perception_data:
            # Convert perception data to game state dictionary
            game_state = {
                "img_path": observation.img_path if hasattr(observation, "img_path") else None,
                "textual_representation": observation.textual_representation if hasattr(observation, "textual_representation") else "",
                "perception_data": perception_data
            }
            
            # Process observation through memory module
            observation = self.memory_module.process_observation(observation, game_state)
            
            # Update memory with action and thought
            if self.last_action:
                observation = self.memory_module.update_action_memory(
                    observation, 
                    self.last_action, 
                    "Previous action"
                )
            memory_summary = self.memory_module.get_memory_summary(observation)
        else:
            memory_summary = None

        if self.reasoning_module:
            # Create a combined observation with all available data
            combined_observation = observation
            if perception_data:
                combined_observation.processed_visual_description = perception_data
            if memory_summary:
                combined_observation.game_trajectory = memory_summary.get("game_trajectory", "")
                combined_observation.reflection = memory_summary.get("reflection", "")
            
            action_plan = await self.reasoning_module.plan_action(combined_observation)
            self.last_action = action_plan["action"]
            return action_plan

        action = self.base_module.process_observation(observation)
        return {"action": action, "thought": "Base only"}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Enable all modules (perception, memory, reasoning)")
    parser.add_argument("--use_perception", action="store_true", help="Enable perception module")
    parser.add_argument("--use_memory", action="store_true", help="Enable memory module")
    parser.add_argument("--use_reasoning", action="store_true", help="Enable reasoning module")
    return parser.parse_args()

async def run():
    args = parse_args()
    
    # Set up cache directory
    cache_dir = os.path.join("doom", "cache", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(cache_dir, exist_ok=True)
    
    # Initialize the environment
    env = DoomEnvWrapper(
        game_name="doom",
        observation_mode="both",  # Get both vision and text observations
        base_log_dir=cache_dir,
        render_mode_human=False
    )
    
    # Initialize agent with all modules enabled
    agent = DoomAgent(
        model_name="gpt-4o",
        cache_dir=cache_dir,
        use_perception=True,  # Enable perception module
        use_memory=True,      # Enable memory module
        use_reasoning=True,   # Enable reasoning module
        config_path="configs/custom_05_doom/module_prompts.json"  # Add config path
    )
    
    # Reset the environment
    observation, info = env.reset()
    done = False
    total_reward = 0
    step_count = 0
    max_steps = 100  # Prevent infinite loops
    
    while not done and step_count < max_steps:
        # Get action from agent
        action_plan = await agent.get_action(observation)
        print(f"\nStep {step_count + 1}")
        print(f"State - Health: {info.get('health')}, Ammo: {info.get('ammo')}, Kills: {info.get('kills')}")
        print(f"Action: {action_plan['action']}")
        print(f"Thought: {action_plan['thought']}")
        
        # Take the action
        observation, reward, done, info = env.step(action_plan["action"])
        total_reward += reward
        step_count += 1
        
        if done:
            print(f"\nEpisode finished after {step_count} steps")
            print(f"Total reward: {total_reward}")
            break
    
    env.close()

if __name__ == "__main__":
    asyncio.run(run())