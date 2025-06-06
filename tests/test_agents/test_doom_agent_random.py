import argparse
import os
import json
from datetime import datetime
import sys
import logging
import subprocess
from PIL import Image
import numpy as np

"""
Simple random agent for Doom that makes random actions without any complex modules.
"""
# Add the GamingAgent root directory to the Python path
gaming_agent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, gaming_agent_dir)

from gamingagent.envs.custom_05_doom.Doom_env import DoomEnvWrapper
from tests.test_agents.modules.doom_base_module import DoomBaseModule
import asyncio
from gamingagent.modules.core_module import GameTrajectory

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model name (not used for random agent)")
    parser.add_argument("--max_steps", type=int, default=100, help="Maximum number of steps to run")
    return parser.parse_args()

class DoomAgent:
    def __init__(self):
        """Initialize the Doom Agent with basic random functionality."""
        # Set up logging
        self.logger = logging.getLogger('doom_agent')
        self.logger.setLevel(logging.INFO)
        
        # Initialize base module for random actions
        self.base = DoomBaseModule()
        
        # Track game state
        self.steps = 0
        
    async def get_action(self, observation):
        """
        Get a random action based on the current observation.
        
        Args:
            observation: The current game observation (not used for random actions)
            
        Returns:
            dict: Dictionary containing action and thought
        """
        try:
            # Get random action from base module
            action_index = self.base.process_observation(observation)
            
            # Convert action index to action name
            action_names = ["move_left", "move_right", "attack"]
            action_name = action_names[action_index]
            
            # Create action plan
            action_plan = {
                "action": action_name,
                "thought": f"Random action: {action_name}"
            }
            
            # Update step counter
            self.steps += 1
            
            return action_plan
            
        except Exception as e:
            self.logger.error(f"Error in get_action: {e}", exc_info=True)
            return {
                "action": "attack",
                "thought": f"Error occurred: {str(e)}. Defaulting to attack."
            }

async def run():
    args = parse_args()

    try:
        # Initialize environment
        env = DoomEnvWrapper(
            game_name="VizdoomBasic-v0",
            config_dir_path="gamingagent/envs/custom_05_doom",
            observation_mode="both",
            base_log_dir="cache/doom",
            render_mode_human=False,
            record_video=True,
            video_dir="videos/doom",
            model_name=args.model,
            headless=True
        )

        # Initialize agent
        agent = DoomAgent()

        # Reset environment
        observation, info = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        try:
            while not done and step_count < args.max_steps:
                # Get random action from agent
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

        finally:
            # Close environment
            env.close()
            print("\nEnvironment closed.")

    except Exception as e:
        print(f"Error in run: {str(e)}")
        print("Please check:")
        print("1. All required directories have proper permissions")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)