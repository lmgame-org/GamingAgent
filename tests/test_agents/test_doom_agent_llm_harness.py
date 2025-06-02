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
import cv2

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
        self.output_dir = os.path.join("GamingAgent", "doom_agent_output", model_name + "_" + self.timestamp)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create logs directory
        self.logs_dir = os.path.join(self.output_dir, "logs")
        os.makedirs(self.logs_dir, exist_ok=True)
        
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
        
        self.logger.info(f"Initialized DoomAgentHarness with model: {model_name}")
        self.logger.info(f"Logs directory: {self.logs_dir}")
    
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
            
            # Initialize step log
            step_log = {
                "step": self.steps,
                "timestamp": datetime.now().isoformat(),
                "perception": {},
                "memory": {},
                "reasoning": {}
            }
            
            # 1. Perception: Process the observation
            self.logger.info("="*80)
            self.logger.info("PERCEPTION MODULE - Starting processing")
            
            # Debug logging for observation
            self.logger.info(f"Observation type: {type(observation)}")
            frame_stats = {}
            if hasattr(observation, 'img_path'):
                self.logger.info(f"Image path: {observation.img_path}")
                # Verify frame exists and is readable
                try:
                    with Image.open(observation.img_path) as img:
                        self.logger.info(f"Current frame dimensions: {img.size}")
                        # Store frame data for comparison
                        self.last_frame = np.array(img)
                        # Log frame statistics
                        frame_stats = {
                            "mean_pixel_value": float(np.mean(self.last_frame)),
                            "std_pixel_value": float(np.std(self.last_frame)),
                            "min_pixel_value": float(np.min(self.last_frame)),
                            "max_pixel_value": float(np.max(self.last_frame))
                        }
                except Exception as e:
                    self.logger.error(f"Error reading current frame: {e}")
                    self.last_frame = None
                    frame_stats = {"error": str(e)}
            
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
                    
                    # Update perception log
                    step_log["perception"] = {
                        "frame_stats": frame_stats,
                        "system_message": system_message,
                        "user_message": user_message,
                        "visual_description": response,
                        "img_path": processed_observation.img_path,
                        "textual_representation": processed_observation.textual_representation
                    }
                        
                except Exception as e:
                    self.logger.error(f"Error in perception processing: {e}")
                    processed_observation.processed_visual_description = "Error processing visual input"
                    step_log["perception"] = {
                        "error": str(e),
                        "frame_stats": frame_stats
                    }
            else:
                self.logger.warning("No image path in observation")
                processed_observation.processed_visual_description = "No image available for processing"
                step_log["perception"] = {
                    "error": "No image path in observation",
                    "frame_stats": frame_stats
                }
            
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
            
            # Add img_path to game_state for memory module
            if processed_observation.img_path:
                game_state["img_path"] = processed_observation.img_path
            
            # Log current game state
            self.logger.info("Current Game State:")
            self.logger.info(json.dumps(game_state, indent=2))
            
            # Step 1: Process observation with memory and generate reflection
            self.logger.info("Processing observation with memory...")
            processed_observation = self.memory.process_observation(
                observation=processed_observation,
                game_state=game_state
            )
            self.logger.info(f"Generated reflection: {processed_observation.reflection}")
            
            # Step 2: Get memory summary for reasoning
            self.logger.info("Getting memory summary...")
            memory_summary = self.memory.get_memory_summary(processed_observation)
            self.logger.info("Memory Summary:")
            self.logger.info(f"Game Trajectory: {memory_summary.get('game_trajectory', '')}")
            self.logger.info(f"Current State: {memory_summary.get('current_state', '')}")
            
            # Update memory log
            step_log["memory"] = {
                "game_state": game_state,
                "last_action": self.last_action,
                "game_trajectory": memory_summary.get("game_trajectory", ""),
                "reflection": processed_observation.reflection,
                "current_state": memory_summary.get("current_state", ""),
                "img_path": processed_observation.img_path
            }
            
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
                reflection=str(processed_observation.reflection)
            )
            
            # Log reasoning inputs
            self.logger.info("Reasoning Inputs:")
            self.logger.info(f"System Message: {system_message}")
            self.logger.info(f"User Message: {user_message}")
            
            # Get action from LLM
            self.logger.info("Getting action from LLM...")
            response = self.api_manager.text_only_completion(
                model_name="gpt-4o",
                system_prompt=system_message,
                prompt=user_message,
                temperature=0.7
            )
            self.logger.info(f"Raw LLM Response: {response}")
            
            # Parse response
            lines = response.strip().split('\n')
            action_line = next((line for line in lines if line.startswith('action:')), None)
            thought_line = next((line for line in lines if line.startswith('thought:')), None)
            
            action = action_line.split('action:')[1].strip().lower() if action_line else "attack"
            thought = thought_line.split('thought:')[1].strip() if thought_line else "No reasoning provided"
            
            # Log parsed action and thought
            self.logger.info("Parsed Action and Thought:")
            self.logger.info(f"Action: {action}")
            self.logger.info(f"Thought: {thought}")
            
            # Step 3: Update action memory with the new action and thought
            self.logger.info("Updating action memory...")
            processed_observation = self.memory.update_action_memory(
                observation=processed_observation,
                action=action,
                thought=thought
            )
            
            # Update reasoning log
            step_log["reasoning"] = {
                "system_message": system_message,
                "user_message": user_message,
                "raw_response": response,
                "parsed_action": action,
                "parsed_thought": thought,
                "game_state": game_state,
                "img_path": processed_observation.img_path
            }
            
            # Update trajectory with new state
            self.trajectory.append(game_state)
            
            # Write step log
            try:
                step_num = f"{self.steps:04d}"
                step_log_path = os.path.join(self.logs_dir, f"step_{step_num}.json")
                with open(step_log_path, 'w') as f:
                    json.dump(step_log, f, indent=2)
                self.logger.info(f"Successfully wrote step log to {step_log_path}")
            except Exception as e:
                self.logger.error(f"Error writing step log: {e}")
            
            # Update state tracking
            self.last_action = action
            self.steps += 1
            
            # Log final action decision
            self.logger.info("="*80)
            self.logger.info("Final Action Decision:")
            self.logger.info(f"Action: {action}")
            self.logger.info(f"Thought: {thought}")
            self.logger.info(f"Game state at action: {json.dumps(game_state, indent=2)}")
            self.logger.info("="*80)
            
            # Return action and thought
            return action, thought
            
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

    def save_step_log(self, step_log: dict):
        """Save step log in both JSON and readable text format."""
        try:
            # Save JSON log
            step_num = f"{step_log['step']:04d}"
            step_log_path = os.path.join(self.logs_dir, f"step_{step_num}.json")
            with open(step_log_path, 'w') as f:
                json.dump(step_log, f, indent=2)
            
            # Save formatted text log
            text_log_path = os.path.join(self.logs_dir, f"step_{step_num}.txt")
            with open(text_log_path, 'w') as f:
                f.write("="*80 + "\n")
                f.write(f"STEP {step_log['step']} - {step_log['timestamp']}\n")
                f.write("="*80 + "\n\n")
                
                # Perception Section
                f.write("PERCEPTION MODULE\n")
                f.write("-"*40 + "\n")
                if "frame_stats" in step_log["perception"]:
                    f.write("Frame Statistics:\n")
                    for stat, value in step_log["perception"]["frame_stats"].items():
                        f.write(f"  {stat}: {value}\n")
                f.write("\nVisual Description:\n")
                f.write(f"{step_log['perception'].get('visual_description', 'No visual description')}\n\n")
                
                # Memory Section
                f.write("MEMORY MODULE\n")
                f.write("-"*40 + "\n")
                f.write("Game State:\n")
                for key, value in step_log["memory"]["game_state"].items():
                    f.write(f"  {key}: {value}\n")
                f.write("\nLast Action: {}\n".format(step_log["memory"].get("last_action", "None")))
                f.write("\nGame Trajectory:\n")
                f.write(f"{step_log['memory'].get('game_trajectory', 'No trajectory')}\n")
                f.write("\nReflection:\n")
                f.write(f"{step_log['memory'].get('reflection', 'No reflection')}\n")
                f.write("\nCurrent State:\n")
                f.write(f"{step_log['memory'].get('current_state', 'No current state')}\n\n")
                
                # Reasoning Section
                f.write("REASONING MODULE\n")
                f.write("-"*40 + "\n")
                f.write("System Message:\n")
                f.write(f"{step_log['reasoning'].get('system_message', 'No system message')}\n\n")
                f.write("User Message:\n")
                f.write(f"{step_log['reasoning'].get('user_message', 'No user message')}\n\n")
                f.write("Raw Response:\n")
                f.write(f"{step_log['reasoning'].get('raw_response', 'No response')}\n\n")
                f.write("Parsed Action: {}\n".format(step_log["reasoning"].get("parsed_action", "None")))
                f.write("Parsed Thought:\n")
                f.write(f"{step_log['reasoning'].get('parsed_thought', 'No thought')}\n\n")
                
                # Final Decision
                f.write("="*80 + "\n")
                f.write("FINAL DECISION\n")
                f.write("-"*40 + "\n")
                f.write(f"Action: {step_log['reasoning'].get('parsed_action', 'None')}\n")
                f.write(f"Thought: {step_log['reasoning'].get('parsed_thought', 'No thought')}\n")
                f.write("="*80 + "\n")
            
            self.logger.info(f"Successfully wrote step logs to {step_log_path} and {text_log_path}")
        except Exception as e:
            self.logger.error(f"Error writing step logs: {e}")

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
