import argparse
import os
import json
from datetime import datetime
from gamingagent.envs.custom_05_doom.Doom_env import DoomEnvWrapper
from tests.test_agents.modules.doom_base_module import DoomBaseModule
from gamingagent.modules.perception_module import PerceptionModule
from gamingagent.modules.memory_module import MemoryModule
from gamingagent.modules.reasoning_module import ReasoningModule
from gamingagent.modules.core_module import GameTrajectory
import asyncio

def load_model_config():
    """Load model configuration from file."""
    config_path = os.path.join("configs", "model_config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}

def get_model_config(model_name):
    """Get the appropriate model configuration based on the model name."""
    config = load_model_config()
    if not config or "models" not in config:
        raise ValueError("Model configuration not found")
    
    if model_name not in config["models"]:
        raise ValueError(f"Unsupported model: {model_name}")
    
    model_info = config["models"][model_name]
    api_key = os.getenv(model_info["env_var"])
    if not api_key:
        raise ValueError(f"Environment variable {model_info['env_var']} not set")
    
    return model_info["name"]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Enable all modules (perception, memory, reasoning)")
    parser.add_argument("--use_perception", action="store_true", help="Enable perception module")
    parser.add_argument("--use_memory", action="store_true", help="Enable memory module")
    parser.add_argument("--use_reasoning", action="store_true", help="Enable reasoning module")
    
    # Load available models from config
    config = load_model_config()
    available_models = list(config.get("models", {}).keys()) if config else ["gpt-4"]
    
    parser.add_argument("--model", type=str, default=config.get("default_model", "gpt-4"), 
                      choices=available_models,
                      help="Model to use for the agent")
    return parser.parse_args()

class DoomAgent:
    def __init__(self, model_name, cache_dir, use_perception=False, use_memory=False, use_reasoning=False, config_path=None):
        self.base_module = DoomBaseModule()
        
        # Load config if provided
        config = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        
        # Create module-specific cache directories
        perception_cache_dir = os.path.join(cache_dir, "perception") if use_perception else None
        memory_cache_dir = os.path.join(cache_dir, "memory") if use_memory else None
        reasoning_cache_dir = os.path.join(cache_dir, "reasoning") if use_reasoning else None
        
        # Create directories if they don't exist
        for dir_path in [perception_cache_dir, memory_cache_dir, reasoning_cache_dir]:
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
        
        # Initialize perception module with config
        if use_perception:
            perception_config = config.get("perception_module", {})
            self.perception_module = PerceptionModule(
                model_name=model_name,
                cache_dir=perception_cache_dir,
                system_prompt=perception_config.get("system_prompt", ""),
                prompt=perception_config.get("prompt", "")
            )
            print(f"Initialized perception module with cache dir: {perception_cache_dir}")
        else:
            self.perception_module = None
        
        # Initialize memory module with config
        if use_memory:
            memory_config = config.get("memory_module", {})
            self.memory_module = MemoryModule(
                model_name=model_name,
                cache_dir=memory_cache_dir,
                system_prompt=memory_config.get("system_prompt", ""),
                prompt=memory_config.get("prompt", "")
            )
            print(f"Initialized memory module with cache dir: {memory_cache_dir}")
        else:
            self.memory_module = None
        
        # Initialize reasoning module with config
        if use_reasoning:
            reasoning_config = config.get("reasoning_module", {})
            self.reasoning_module = ReasoningModule(
                model_name=model_name,
                cache_dir=reasoning_cache_dir,
                system_prompt=reasoning_config.get("system_prompt", ""),
                prompt=reasoning_config.get("prompt", "")
            )
            print(f"Initialized reasoning module with cache dir: {reasoning_cache_dir}")
        else:
            self.reasoning_module = None
            
        self.last_action = None
        self.last_thought = None

    async def get_action(self, observation):
        # Process observation through perception module if available
        perception_data = self.perception_module.process_observation(observation) if self.perception_module else None
        
        # Create game state dictionary with all available information
        game_state = {
            "perception_data": perception_data,
            "last_action": self.last_action,
            "last_thought": self.last_thought,
            "health": getattr(observation, "health", None),
            "ammo": getattr(observation, "ammo", None),
            "kills": getattr(observation, "kills", None),
            "textual_representation": getattr(observation, "textual_representation", ""),
            "img_path": getattr(observation, "img_path", None)
        }
        
        # Ensure observation has game_trajectory
        if not hasattr(observation, "game_trajectory"):
            observation.game_trajectory = GameTrajectory(max_length=10)
        
        # Process observation through memory module if available
        if self.memory_module:
            # Pass game_state as a dictionary to memory module
            observation = self.memory_module.process_observation(observation, game_state)
            
            # Update memory with last action and thought
            if self.last_action or self.last_thought:
                observation = self.memory_module.update_action_memory(
                    observation,
                    self.last_action,
                    self.last_thought
                )
        
        # Get action from reasoning module if available, otherwise use base module
        if self.reasoning_module:
            # Get memory summary if available
            memory_summary = None
            if self.memory_module:
                memory_summary = self.memory_module.get_memory_summary(observation)
                if memory_summary:
                    # Create a new GameTrajectory object with the memory summary
                    trajectory = GameTrajectory(max_length=10)
                    if memory_summary.get("game_trajectory"):
                        trajectory.add(memory_summary["game_trajectory"])
                    observation.game_trajectory = trajectory
                    observation.reflection = memory_summary.get("reflection", "")
            
            action_plan = self.reasoning_module.plan_action(observation)
        else:
            action_plan = self.base_module.get_action(observation)
            
        # Update last action and thought
        if isinstance(action_plan, dict):
            self.last_action = action_plan.get("action")
            self.last_thought = action_plan.get("thought")
        else:
            self.last_action = action_plan
            self.last_thought = None
            
        # Ensure we return a dictionary with action and thought
        if not isinstance(action_plan, dict):
            action_plan = {"action": action_plan, "thought": "Base module action"}
            
        # Ensure action is a single string, not a list
        if isinstance(action_plan["action"], list):
            action_plan["action"] = action_plan["action"][0]
        elif isinstance(action_plan["action"], str):
            # Remove any brackets and quotes from the action string
            action = action_plan["action"].strip()
            if action.startswith("[") and action.endswith("]"):
                action = action[1:-1]
            if action.startswith("'") and action.endswith("'"):
                action = action[1:-1]
            if action.startswith('"') and action.endswith('"'):
                action = action[1:-1]
            action_plan["action"] = action
            
        return action_plan

async def run():
    args = parse_args()
    
    # Get model configuration
    model_name = get_model_config(args.model)
    print(f"Using model: {model_name}")
    
    # Set up cache directory
    cache_dir = os.path.join("doom", "cache", f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
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
        model_name=model_name,
        cache_dir=cache_dir,
        use_perception=args.use_perception or args.all,
        use_memory=args.use_memory or args.all,
        use_reasoning=args.use_reasoning or args.all,
        config_path="configs/custom_05_doom/module_prompts.json"
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