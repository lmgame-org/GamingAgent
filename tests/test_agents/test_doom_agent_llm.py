# test doom agent with llm

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
from tests.test_agents.modules.doom_base_module import DoomBaseModule
from tools.serving import APIManager

# Load module prompts
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

# Create cache directory for this run
CACHE_DIR = os.path.join("cache", "doom_experiments", datetime.now().strftime("%Y%m%d_%H%M%S"))
GAME_LOG_FILE = os.path.join(CACHE_DIR, "game_log.jsonl")
DATA_LOG_FILE = os.path.join(CACHE_DIR, "data_log.jsonl")
MEMORY_FILE = os.path.join(CACHE_DIR, "memory.json")
os.makedirs(CACHE_DIR, exist_ok=True)

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
    
    raise ValueError("No model configuration file found. Please ensure model_config.json exists in one of the config directories.")

def setup_logging():
    """Set up logging configuration."""
    logger = logging.getLogger('doom_agent_llm')
    logger.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    return logger

# Initialize logging
logger = setup_logging()

class DoomAgentLLM:
    def __init__(self, agent_mode: str = "full", model_name: str = "gpt-4o"):
        """Initialize the Doom agent with LLM integration."""
        self.agent_mode = agent_mode
        self.model_name = model_name
        
        # Create timestamp for this run
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Set up output directory structure
        self.output_dir = os.path.join("doom_agent_output", self.timestamp)
        os.makedirs(self.output_dir, exist_ok=True)
        os.chmod(self.output_dir, 0o777)  # Ensure full permissions
        
        # Create model-specific directory
        self.model_dir = os.path.join(self.output_dir, self.model_name, self.timestamp)
        os.makedirs(self.model_dir, exist_ok=True)
        os.chmod(self.model_dir, 0o777)  # Ensure full permissions
        
        # Create recordings directory
        self.recordings_dir = os.path.join(self.model_dir, "recordings")
        os.makedirs(self.recordings_dir, exist_ok=True)
        os.chmod(self.recordings_dir, 0o777)  # Ensure full permissions
        
        # Create logs directory
        self.logs_dir = os.path.join(self.model_dir, "logs")
        os.makedirs(self.logs_dir, exist_ok=True)
        os.chmod(self.logs_dir, 0o777)  # Ensure full permissions
        
        # Set up logging
        self._setup_logging()
        
        # Now we can use the logger
        self.logger.info(f"Initializing DoomAgentLLM with model: {model_name}")
        
        # Initialize API manager
        self.api_manager = APIManager(
            game_name="doom",
            base_cache_dir=self.model_dir
        )
        
        # Initialize environment
        try:
            self.env = DoomEnvWrapper(
                game_name="VizdoomBasic-v0",
                config_dir_path="gamingagent/envs/custom_05_doom",
                observation_mode="both",  # Get both vision and text observations
                base_log_dir=self.output_dir,
                render_mode_human=False,
                record_video=True,  # Enable video recording
                video_dir=self.recordings_dir,  # Use recordings directory for video
                model_name=model_name,
                headless=True  # Set to False if GUI rendering is required
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize Doom environment: {e}")
            raise
        
        # Initialize components
        self.base = DoomBaseModule()
        
        # Game state tracking
        self.steps = 0
        self.available_actions = ["move_left", "move_right", "attack"]
        self.last_action = None
        
        # Set up signal handler for graceful shutdown
        self._setup_signal_handling()
        
        self.logger.info(f"Initialized DoomAgentLLM with mode: {agent_mode}, model: {model_name}")
    
    def _setup_logging(self):
        """Set up logging for the agent."""
        # Set up main logger
        self.logger = logging.getLogger('doom_agent_llm')
        self.logger.setLevel(logging.INFO)
        
        # Create file handler
        log_file = os.path.join(self.logs_dir, f"agent_{self.timestamp}.log")
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
    
    def _setup_signal_handling(self):
        """Set up signal handlers for graceful shutdown."""
        import signal
        
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, attempting graceful shutdown...")
            try:
                # Clean up environment
                if self.env:
                    self.env.close()
                    self.logger.info("Environment closed successfully")
            except Exception as e:
                self.logger.error(f"Error during shutdown: {e}")
            finally:
                sys.exit(0)
        
        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def get_action(self, observation, info):
        """Get action from LLM based on current state."""
        try:
            # Get current game state
            game_state = {
                "health": info.get("health", 0),
                "ammo": info.get("ammo", 0),
                "kills": info.get("kills", 0),
                "position_x": info.get("position_x", 0),
                "position_y": info.get("position_y", 0),
                "angle": info.get("angle", 0),
                "is_episode_finished": info.get("is_episode_finished", False),
                "is_player_dead": info.get("is_player_dead", False)
            }
            
            # Load module prompts
            module_prompts = load_module_prompts()
            base_module = module_prompts.get("base_module", {})
            
            # Get system message and prompt template
            system_message = base_module.get("system_prompt", "")
            prompt_template = base_module.get("prompt", "")
            
            # Format user message with game state
            user_message = prompt_template.format(
                textual_representation=f"""Health: {game_state['health']}
Ammo: {game_state['ammo']}
Kills: {game_state['kills']}
Position: ({game_state['position_x']}, {game_state['position_y']})
Angle: {game_state['angle']}
Status: {'Finished' if game_state['is_episode_finished'] else 'In Progress'}"""
            )
            
            # Get action from LLM
            response = self.api_manager.text_only_completion(
                model_name=self.model_name,
                system_prompt=system_message,
                prompt=user_message,
                temperature=0.7
            )
            
            # Parse response to get action and thought
            try:
                # Split response into lines and find action line
                lines = response.strip().split('\n')
                action_line = next((line for line in lines if line.startswith('action:')), None)
                thought_line = next((line for line in lines if line.startswith('thought:')), None)
                
                if action_line:
                    action = action_line.split('action:')[1].strip().lower()
                else:
                    self.logger.warning("No action found in response, defaulting to attack")
                    action = "attack"
                
                if thought_line:
                    thought = thought_line.split('thought:')[1].strip()
                else:
                    thought = "No reasoning provided"
                
            except Exception as e:
                self.logger.error(f"Error parsing LLM response: {e}")
                action = "attack"
                thought = f"Error parsing response: {str(e)}"
            
            # Validate action
            valid_actions = ["move_left", "move_right", "attack"]
            if action not in valid_actions:
                self.logger.warning(f"Invalid action received: {action}. Defaulting to attack.")
                action = "attack"
                thought = f"Invalid action received: {action}. Defaulting to attack."
            
            # Log action details
            self.logger.info(f"Action selected: {action}")
            self.logger.info(f"Thought: {thought}")
            self.logger.info(f"Game state at action: {game_state}")
            
            return {
                "action": action,
                "thought": thought
            }
            
        except Exception as e:
            self.logger.error(f"Error in get_action: {str(e)}")
            return {
                "action": "attack",
                "thought": f"Error occurred: {str(e)}. Defaulting to attack."
            }
    
    def _reconstruct_video(self):
        """Reconstruct video from saved frames using ffmpeg."""
        try:
            # Find the frames directory in the recordings folder
            # The path structure is: doom_agent_output/{timestamp}/{model_name}/{timestamp}/recordings/episode_1_frames
            model_dir = os.path.join(self.output_dir, self.model_name, self.timestamp)
            recordings_dir = os.path.join(model_dir, "recordings")
            self.logger.info(f"Looking for frames in recordings directory: {recordings_dir}")
            
            frames_dir = os.path.join(recordings_dir, "episode_1_frames")
            if not os.path.exists(frames_dir):
                self.logger.warning(f"Frames directory not found: {frames_dir}")
                return
            self.logger.info(f"Found frames directory: {frames_dir}")

            # Get all frame files
            frame_files = sorted([f for f in os.listdir(frames_dir) if f.startswith('frame_') and f.endswith('.png')])
            self.logger.info(f"Found {len(frame_files)} frame files in {frames_dir}")
            if not frame_files:
                self.logger.warning("No frames found for video reconstruction")
                return

            # Create frame list file for ffmpeg
            frame_list_path = os.path.join(frames_dir, "frame_list.txt")
            with open(frame_list_path, 'w') as f:
                for frame_file in frame_files:
                    frame_path = os.path.join(frames_dir, frame_file)
                    f.write(f"file '{frame_path}'\n")
                    self.logger.debug(f"Added frame to list: {frame_path}")

            # Set output video path
            output_video = os.path.join(model_dir, f"episode_{self.timestamp}.mp4")
            self.logger.info(f"Will create video at: {output_video}")

            # Get frame resolution from first frame
            first_frame_path = os.path.join(frames_dir, frame_files[0])
            self.logger.info(f"Reading first frame from: {first_frame_path}")
            first_frame = Image.open(first_frame_path)
            width, height = first_frame.size
            self.logger.info(f"Frame resolution: {width}x{height}")

            # Run ffmpeg to create video
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', frame_list_path,
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-r', '30',
                '-vf', f'scale={width}:{height}',
                output_video
            ]
            self.logger.info(f"Running ffmpeg command: {' '.join(ffmpeg_cmd)}")

            # Execute ffmpeg command
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                self.logger.error(f"FFmpeg error: {result.stderr}")
                raise RuntimeError(f"FFmpeg failed: {result.stderr}")
            
            self.logger.info(f"Video reconstructed successfully: {output_video}")

            # Save video metadata
            metadata = {
                "timestamp": self.timestamp,
                "total_frames": len(frame_files),
                "resolution": f"{width}x{height}",
                "video_path": output_video
            }
            metadata_path = os.path.join(model_dir, "video_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            self.logger.info(f"Saved video metadata to: {metadata_path}")

            # Clean up temporary files
            try:
                os.remove(frame_list_path)
                for frame_file in frame_files:
                    os.remove(os.path.join(frames_dir, frame_file))
                os.rmdir(frames_dir)
                self.logger.info("Cleaned up temporary frame files")
            except Exception as e:
                self.logger.warning(f"Error cleaning up temporary files: {e}")

        except Exception as e:
            self.logger.error(f"Error reconstructing video: {e}", exc_info=True)

    async def run_episode(self, max_steps=100):
        """Run a single episode of the game."""
        try:
            # Reset environment
            observation, info = self.env.reset()
            done = False
            total_reward = 0
            step = 0
            
            # Log initial state
            self.logger.info(f"Initial state - Health: {info.get('health')}, Ammo: {info.get('ammo')}, Kills: {info.get('kills')}")
            
            while not done and step < max_steps:
                try:
                    # Get action from agent
                    action_plan = await self.get_action(observation, info)
                    
                    # Log pre-action state
                    self.logger.info(f"Pre-action state - Health: {info.get('health')}, Ammo: {info.get('ammo')}, Kills: {info.get('kills')}")
                    self.logger.info(f"Executing action: {action_plan['action']}")
                    
                    # Execute action
                    observation, reward, done, info = self.env.step(action_plan["action"])
                    total_reward += reward
                    
                    # Log post-action state
                    self.logger.info(f"Post-action state - Health: {info.get('health')}, Ammo: {info.get('ammo')}, Kills: {info.get('kills')}")
                    self.logger.info(f"Action: {action_plan['action']}")
                    self.logger.info(f"Thought: {action_plan['thought']}")
                    self.logger.info(f"Reward: {reward}")
                    
                    step += 1
                    
                except Exception as e:
                    self.logger.error(f"Error during step {step}: {str(e)}")
                    raise
                
            self.logger.info(f"Episode finished after {step} steps")
            self.logger.info(f"Total reward: {total_reward}")
            
        except Exception as e:
            self.logger.error(f"Error in run_episode: {str(e)}")
        finally:
            # Clean up environment and reconstruct video
            if self.env:
                try:
                    self.logger.info("Closing environment and reconstructing video...")
                    self.env.close()
                    self.logger.info("Environment closed successfully")
                    # Reconstruct video from saved frames
                    self._reconstruct_video()
                except Exception as e:
                    self.logger.error(f"Error during environment cleanup: {e}", exc_info=True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4o",
                      help="Model to use for the agent")
    parser.add_argument("--mode", type=str, default="full",
                      choices=["base", "full"],
                      help="Agent mode to use")
    parser.add_argument("--max_steps", type=int, default=50,
                      help="Maximum number of steps to run")
    return parser.parse_args()

async def main():
    args = parse_args()
    
    # Create and run the agent
    agent = DoomAgentLLM(model_name=args.model, agent_mode=args.mode)
    await agent.run_episode(max_steps=args.max_steps)

if __name__ == "__main__":
    asyncio.run(main())
