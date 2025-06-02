"""
Doom Agent with LLM Integration using the Harness Design

This agent uses the core modules (perception, memory, reasoning) to play Doom,
with LLM-based decision making and proper game state tracking.
"""

import os
import sys
import json
import time
import logging
import numpy as np
from PIL import Image
from datetime import datetime
import asyncio
import argparse
import subprocess

# Add the GamingAgent root directory to the Python path
gaming_agent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, gaming_agent_dir)

from gamingagent.envs.custom_05_doom.Doom_env import DoomEnvWrapper
from gamingagent.modules.perception_module import PerceptionModule
from gamingagent.modules.memory_module import MemoryModule
from gamingagent.modules.reasoning_module import ReasoningModule
from gamingagent.modules.core_module import GameTrajectory, Observation
from tools.serving import APIManager

def load_module_prompts():
    """Load module prompts from config file."""
    config_paths = [
        os.path.join("configs", "custom_05_doom", "module_prompts.json"),
        os.path.join("GamingAgent", "configs", "custom_05_doom", "module_prompts.json"),
        os.path.join(os.path.dirname(__file__), "..", "..", "configs", "custom_05_doom", "module_prompts.json")
    ]
    
    for config_path in config_paths:
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load config from {config_path}: {e}")
    
    raise ValueError("No module prompts configuration file found.")

def load_model_config():
    """Load model configuration from file."""
    config_paths = [
        os.path.join("configs", "model_config.json"),
        os.path.join("GamingAgent", "configs", "model_config.json"),
        os.path.join(os.path.dirname(__file__), "..", "..", "configs", "model_config.json")
    ]
    
    for config_path in config_paths:
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load config from {config_path}: {e}")
    
    raise ValueError("No model configuration file found.")

class DoomAgentHarness:
    """Doom Agent using the harness design with LLM integration."""
    
    def __init__(self, model_name: str = "gpt-4o", reasoning_effort: str = "high"):
        """Initialize the Doom Agent with harness design."""
        # Create timestamp for this run
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Set up simple output directory structure
        self.output_dir = os.path.join("doom_agent_output", self.timestamp)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up logging
        self._setup_logging()
        self.logger.info(f"Initializing DoomAgentHarness with model: {model_name}")
        
        # Initialize API manager
        self.api_manager = APIManager(
            game_name="doom",
            base_cache_dir=self.output_dir
        )
        
        # Initialize environment
        try:
            self.logger.info("Initializing Doom environment...")
            self.env = DoomEnvWrapper(
                game_name="VizdoomBasic-v0",
                config_dir_path="gamingagent/envs/custom_05_doom",
                observation_mode="both",
                base_log_dir=self.output_dir,
                render_mode_human=False,
                record_video=True,
                video_dir=self.output_dir,
                model_name=model_name,
                headless=True
            )
            self.logger.info("Doom environment initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Doom environment: {e}")
            raise
        
        # Initialize core modules
        self.perception = PerceptionModule()
        self.memory = MemoryModule()
        self.reasoning = ReasoningModule()
        
        # Load module prompts
        self.module_prompts = load_module_prompts()
        
        # Game state tracking
        self.steps = 0
        self.available_actions = ["move_left", "move_right", "attack"]
        self.last_action = None
        self.trajectory = GameTrajectory(max_length=10)
        
        # Create module logs directory
        self.module_logs_dir = os.path.join(self.output_dir, "module_logs")
        os.makedirs(self.module_logs_dir, exist_ok=True)
        
        self.logger.info(f"Initialized DoomAgentHarness with model: {model_name}")
    
    def _setup_logging(self):
        """Set up logging for the agent."""
        self.logger = logging.getLogger('doom_agent_harness')
        self.logger.setLevel(logging.INFO)
        
        # Create file handler
        log_file = os.path.join(self.output_dir, "agent.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    async def get_action(self, observation: Observation, info: dict) -> dict:
        """Get action from LLM based on current state."""
        try:
            self.logger.info(f"Step {self.steps + 1} starting")
            
            # Create step log file
            step_log_path = os.path.join(self.module_logs_dir, f"step_{self.steps:04d}.txt")
            
            # 1. Perception: Process the observation
            self.logger.info("="*80)
            self.logger.info("PERCEPTION MODULE - Starting processing")
            
            # Debug logging for observation
            self.logger.info(f"Observation type: {type(observation)}")
            if hasattr(observation, 'img_path'):
                self.logger.info(f"Image path: {observation.img_path}")
                # Verify frame exists and is readable
                try:
                    with Image.open(observation.img_path) as img:
                        self.logger.info(f"Current frame dimensions: {img.size}")
                        # Store frame data for comparison
                        self.last_frame = np.array(img)
                except Exception as e:
                    self.logger.error(f"Error reading current frame: {e}")
                    self.last_frame = None
            
            # Initialize processed_observation with basic info
            processed_observation = Observation(
                img_path=observation.img_path if hasattr(observation, 'img_path') else None,
                textual_representation=info.get('textual_representation', ''),
                processed_visual_description='',
                game_trajectory=GameTrajectory(max_length=10),
                reflection=''
            )
            
            # Check if we have an image path in the observation
            if processed_observation.img_path:
                try:
                    # Get system message and prompt template for perception
                    perception_prompt = self.module_prompts.get("perception_module", {})
                    system_message = str(perception_prompt.get("system_prompt", ""))
                    prompt_template = str(perception_prompt.get("prompt", ""))
                    
                    # Format user message for perception with image path
                    user_message = prompt_template.format(
                        textual_representation=f"""Health: {info.get('health', 0)}
Ammo: {info.get('ammo', 0)}
Kills: {info.get('kills', 0)}
Position: ({info.get('position_x', 0)}, {info.get('position_y', 0)})
Angle: {info.get('angle', 0)}
Status: {'Finished' if info.get('is_episode_finished', False) else 'In Progress'}
Image Path: {processed_observation.img_path}"""
                    )
                    
                    # Get visual description from LLM
                    response = self.api_manager.vision_text_completion(
                        model_name="gpt-4o",
                        system_prompt=system_message,
                        prompt=user_message,
                        image_path=processed_observation.img_path,
                        reasoning_effort="medium"
                    )
                    
                    # Update processed observation with visual description
                    processed_observation.processed_visual_description = response
                    
                    # Log the visual description
                    self.logger.info(f"Generated visual description: {response}")
                    
                    # Write to step log file
                    with open(step_log_path, 'w') as f:
                        f.write("=== PERCEPTION MODULE ===\n")
                        f.write(f"Visual Description:\n{response}\n\n")
                except Exception as e:
                    self.logger.error(f"Error in perception processing: {e}")
                    processed_observation.processed_visual_description = "Error processing visual input"
            else:
                self.logger.warning("No image path in observation")
                processed_observation.processed_visual_description = "No image available for processing"
            
            # 2. Memory: Update and retrieve from memory
            self.logger.info("="*80)
            self.logger.info("MEMORY MODULE - Starting processing")
            
            # Format game state
            game_state = {
                "health": int(info.get("health", 0)),
                "ammo": int(info.get("ammo", 0)),
                "kills": int(info.get("kills", 0)),
                "position_x": float(info.get("position_x", 0)),
                "position_y": float(info.get("position_y", 0)),
                "angle": float(info.get("angle", 0)),
                "is_episode_finished": bool(info.get("is_episode_finished", False)),
                "is_player_dead": bool(info.get("is_player_dead", False))
            }
            
            # Update memory
            processed_observation = self.memory.process_observation(
                observation=processed_observation,
                game_state=game_state
            )
            
            # Get memory summary
            memory_summary = self.memory.get_memory_summary(processed_observation)
            
            # Generate reflection if none exists
            if not memory_summary.get("reflection") or memory_summary.get("reflection") == "N/A":
                try:
                    # Get system message and prompt template for memory
                    memory_prompt = self.module_prompts.get("memory_module", {})
                    system_message = str(memory_prompt.get("system_prompt", ""))
                    prompt_template = str(memory_prompt.get("prompt", ""))
                    
                    # Format user message for memory
                    user_message = prompt_template.format(
                        game_trajectory=str(memory_summary.get("game_trajectory", "")),
                        textual_representation=f"""Health: {game_state['health']}
Ammo: {game_state['ammo']}
Kills: {game_state['kills']}
Position: ({game_state['position_x']}, {game_state['position_y']})
Angle: {game_state['angle']}
Status: {'Finished' if game_state['is_episode_finished'] else 'In Progress'}"""
                    )
                    
                    # Get reflection from LLM
                    response = self.api_manager.text_only_completion(
                        model_name="gpt-4o",
                        system_prompt=system_message,
                        prompt=user_message,
                        temperature=0.7
                    )
                    
                    memory_summary["reflection"] = response.strip()
                    self.logger.info(f"Generated reflection: {response.strip()}")
                except Exception as e:
                    self.logger.error(f"Error in memory reflection: {e}")
                    memory_summary["reflection"] = "Error generating reflection"
            
            # Log memory output
            with open(step_log_path, 'a') as f:
                f.write("=== MEMORY MODULE ===\n")
                f.write(f"Game Trajectory: {memory_summary.get('game_trajectory', '')}\n")
                f.write(f"Reflection: {memory_summary.get('reflection', '')}\n\n")
            
            # 3. Reasoning: Get action from LLM
            self.logger.info("="*80)
            self.logger.info("REASONING MODULE - Starting processing")
            
            # Get system message and prompt template
            reasoning_prompt = self.module_prompts.get("reasoning_module", {})
            system_message = str(reasoning_prompt.get("system_prompt", ""))
            prompt_template = str(reasoning_prompt.get("prompt", ""))
            
            # Format user message
            user_message = prompt_template.format(
                textual_representation=f"""Health: {game_state['health']}
Ammo: {game_state['ammo']}
Kills: {game_state['kills']}
Position: ({game_state['position_x']}, {game_state['position_y']})
Angle: {game_state['angle']}
Status: {'Finished' if game_state['is_episode_finished'] else 'In Progress'}""",
                memory=str(memory_summary.get("game_trajectory", "")),
                trajectory=str(self.trajectory.get() or "No previous game states available."),
                processed_visual_description=str(processed_observation.processed_visual_description or ""),
                game_trajectory=str(memory_summary.get("game_trajectory", "")),
                reflection=str(memory_summary.get("reflection", ""))
            )
            
            # Get action from LLM
            response = self.api_manager.text_only_completion(
                model_name="gpt-4o",
                system_prompt=system_message,
                prompt=user_message,
                temperature=0.7
            )
            
            # Parse response
            lines = response.strip().split('\n')
            action_line = next((line for line in lines if line.startswith('action:')), None)
            thought_line = next((line for line in lines if line.startswith('thought:')), None)
            
            action = action_line.split('action:')[1].strip().lower() if action_line else "attack"
            thought = thought_line.split('thought:')[1].strip() if thought_line else "No reasoning provided"
            
            # Validate action
            if action not in self.available_actions:
                action = "attack"
                thought = f"Invalid action received: {action}. Defaulting to attack."
            
            # Log reasoning output
            with open(step_log_path, 'a') as f:
                f.write("=== REASONING MODULE ===\n")
                f.write(f"Action: {action}\n")
                f.write(f"Thought: {thought}\n")
            
            # Update state tracking
            self.last_action = action
            self.steps += 1
            
            # Update trajectory
            trajectory_entry = {
                "action": action,
                "thought": thought,
                "state": game_state,
                "timestamp": datetime.now().isoformat()
            }
            self.trajectory.add(json.dumps(trajectory_entry))
            
            # Save frame AFTER action is executed
            if hasattr(observation, 'img_path'):
                frame_dir = os.path.join(self.output_dir, "frames")
                os.makedirs(frame_dir, exist_ok=True)
                frame_path = os.path.join(frame_dir, f"frame_{self.steps:04d}.png")
                
                # Copy the current frame to the new path
                if os.path.exists(observation.img_path):
                    import shutil
                    shutil.copy2(observation.img_path, frame_path)
                    # Update observation with the new frame path
                    observation.img_path = frame_path
                    self.logger.info(f"Saved frame to: {frame_path}")
                    
                    # Verify frame exists and is readable
                    try:
                        with Image.open(frame_path) as img:
                            self.logger.info(f"Frame dimensions: {img.size}")
                            # Compare with previous frame if available
                            if hasattr(self, 'last_frame') and self.last_frame is not None:
                                current_frame = np.array(img)
                                frame_diff = np.sum(np.abs(current_frame - self.last_frame))
                                self.logger.info(f"Frame difference from previous: {frame_diff}")
                                if frame_diff == 0:
                                    self.logger.warning("Frame has not changed from previous step!")
                    except Exception as e:
                        self.logger.error(f"Error reading frame: {e}")
                else:
                    self.logger.error(f"Source frame does not exist: {observation.img_path}")
            
            return {
                "action": action,
                "thought": thought,
                "perception": {
                    "textual_representation": str(processed_observation.textual_representation),
                    "processed_visual_description": str(processed_observation.processed_visual_description),
                    "img_path": str(processed_observation.img_path) if processed_observation.img_path else None
                },
                "memory": {
                    "memory": str(memory_summary.get("game_trajectory", "")),
                    "reflection": str(memory_summary.get("reflection", "")),
                    "current_state": str(memory_summary.get("current_state", ""))
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in get_action: {str(e)}", exc_info=True)
            return {
                "action": "attack",
                "thought": f"Error occurred: {str(e)}. Defaulting to attack."
            }
    
    async def run_episode(self, max_steps: int = 100):
        """Run a single episode of the game."""
        try:
            # Reset environment
            observation, info = self.env.reset()
            done = False
            total_reward = 0
            step = 0
            
            while not done and step < max_steps:
                # Get action from agent
                action_plan = await self.get_action(observation, info)
                
                # Execute action
                observation, reward, done, info = self.env.step(action_plan["action"])
                total_reward += reward
                
                step += 1
                
            self.logger.info(f"Episode finished after {step} steps")
            self.logger.info(f"Total reward: {total_reward}")
            
        except Exception as e:
            self.logger.error(f"Error in run_episode: {str(e)}")
        finally:
            if self.env:
                self.env.close()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4o",
                      help="Model to use for the agent")
    parser.add_argument("--reasoning_effort", type=str, default="high",
                      choices=["low", "medium", "high"],
                      help="Level of reasoning effort")
    parser.add_argument("--max_steps", type=int, default=50,
                      help="Maximum number of steps to run")
    return parser.parse_args()

async def main():
    args = parse_args()
    
    # Create and run the agent
    agent = DoomAgentHarness(
        model_name=args.model,
        reasoning_effort=args.reasoning_effort
    )
    await agent.run_episode(max_steps=args.max_steps)

if __name__ == "__main__":
    asyncio.run(main())
