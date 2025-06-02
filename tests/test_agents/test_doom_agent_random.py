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
gameplay that is random, no thinking, no reasoning, no memory, no perception, no core module
"""
# Add the GamingAgent root directory to the Python path
gaming_agent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, gaming_agent_dir)

from gamingagent.envs.custom_05_doom.Doom_env import DoomEnvWrapper
from tests.test_agents.modules.doom_base_module import DoomBaseModule
from gamingagent.modules.perception_module import PerceptionModule
from gamingagent.modules.memory_module import MemoryModule
from gamingagent.modules.reasoning_module import ReasoningModule
from gamingagent.modules.core_module import GameTrajectory
import asyncio

def load_model_config():
    """Load model configuration from file."""
    # Try multiple possible config paths
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
    
    print("Warning: No model config found, using default configuration")
    return {
        "models": {
            "gpt-4": {
                "name": "gpt-4",
                "provider": "openai",
                "env_var": "OPENAI_API_KEY",
                "supports_vision": True,
                "supports_system_messages": True,
                "max_tokens": 8192,
                "supports_thinking": True
            }
        },
        "default_model": "gpt-4"
    }

def get_model_config(model_name):
    """Get the appropriate model configuration based on the model name."""
    try:
        config = load_model_config()
        if not config or "models" not in config:
            raise ValueError("Model configuration not found or invalid")
        
        if model_name not in config["models"]:
            available_models = list(config["models"].keys())
            raise ValueError(f"Unsupported model: {model_name}. Available models: {available_models}")
        
        model_info = config["models"][model_name]
        api_key = os.getenv(model_info["env_var"])
        if not api_key:
            raise ValueError(f"Environment variable {model_info['env_var']} not set. Please set it before running the agent.")
        
        print(f"Found configuration for model: {model_name}")
        print(f"Using environment variable: {model_info['env_var']}")
        print(f"Model supports vision: {model_info.get('supports_vision', False)}")
        print(f"Model supports system messages: {model_info.get('supports_system_messages', False)}")
        print(f"Max tokens: {model_info.get('max_tokens', 16384)}")
        
        return model_name
        
    except Exception as e:
        print(f"Error in get_model_config: {str(e)}")
        raise

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
    args = parser.parse_args()
    
    # Ensure model name is a string
    if isinstance(args.model, list):
        args.model = args.model[0]
    args.model = str(args.model)
    
    return args

class DoomAgent:
    def __init__(self, model_name="gpt-4o", reasoning_effort="high", thinking=True):
        """
        Initialize the Doom Agent with basic functionality.
        
        Args:
            model_name (str): Name of the model to use
            reasoning_effort (str): Level of reasoning effort (low, medium, high)
            thinking (bool): Whether to enable thinking mode
        """
        # Create timestamp for this run
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Set up output directory structure
        self.output_dir = os.path.join("doom_agent_output", self.timestamp)
        
        # Create logs directory first
        self.logs_dir = os.path.join(self.output_dir, "logs")
        os.makedirs(self.logs_dir, exist_ok=True)
        os.chmod(self.logs_dir, 0o777)  # Ensure full permissions
        
        # Set up logging
        self._setup_logging()
        
        # Now we can use the logger
        self.logger.info(f"Initializing DoomAgent with model: {model_name}")
        
        # Set up remaining output directories
        self._setup_output_directories()
        
        # Verify API key is set
        config = load_model_config()
        model_info = config["models"][model_name]
        api_key = os.getenv(model_info["env_var"])
        if not api_key:
            raise ValueError(f"Environment variable {model_info['env_var']} not set. Please set it before running the agent.")
        self.logger.info(f"API key found for {model_name}")
        
        # Initialize base module
        self.base = DoomBaseModule()
        
        # Track game state
        self.last_action = None
        self.available_actions = ["move_left", "move_right", "attack"]
        self.steps = 0
        
        self.logger.info("DoomAgent initialization complete")
        
    def _setup_output_directories(self):
        """Set up the directory structure for recording agent outputs."""
        # Main directories (logs_dir already created)
        self.metadata_dir = os.path.join(self.output_dir, "metadata")  # Store all metadata
        
        # Create main directories with full permissions
        os.makedirs(self.metadata_dir, exist_ok=True)
        os.chmod(self.metadata_dir, 0o777)  # Ensure full permissions
        
        # Create stage-specific metadata directories
        self.stage_dirs = {
            'initial': os.path.join(self.metadata_dir, "initial"),
            'perception': os.path.join(self.metadata_dir, "perception"),
            'memory': os.path.join(self.metadata_dir, "memory"),
            'reasoning': os.path.join(self.metadata_dir, "reasoning"),
            'final': os.path.join(self.metadata_dir, "final")
        }
        
        # Create stage directories with full permissions
        for dir_path in self.stage_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
            os.chmod(dir_path, 0o777)  # Ensure full permissions
            
        self.logger.info(f"Created output directories in {self.output_dir}")
        self.logger.info(f"Logs directory: {self.logs_dir}")
        self.logger.info(f"Metadata directory: {self.metadata_dir}")
        
    def _setup_logging(self):
        """Set up logging for the agent."""
        # Set up main logger
        self.logger = logging.getLogger('doom_agent')
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
        
    def _setup_module_logger(self, module_name):
        """Set up logging for a specific module."""
        logger = logging.getLogger(f'doom_agent.{module_name}')
        logger.setLevel(logging.INFO)
        
        # Create module-specific log file
        log_file = os.path.join(self.logs_dir, f"{module_name}_{self.timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
        return logger
        
    def _save_frame(self, frame, step_num):
        """
        Save a game frame with metadata.
        
        Args:
            frame: The frame to save
            step_num: The step number
        """
        try:
            if isinstance(frame, np.ndarray):
                # Save frame
                frame_path = os.path.join(self.frames_dir, f"frame_{step_num:04d}.png")
                img = Image.fromarray(frame)
                img.save(frame_path)
                
                # Save frame metadata
                frame_metadata = {
                    "step": step_num,
                    "timestamp": datetime.now().isoformat(),
                    "frame_path": frame_path,
                    "resolution": frame.shape[:2],
                    "format": "RGB"
                }
                
                metadata_path = os.path.join(self.frames_dir, f"frame_{step_num:04d}_metadata.json")
                with open(metadata_path, 'w') as f:
                    json.dump(frame_metadata, f, indent=2)
                    
                self.logger.debug(f"Saved frame {step_num} to {frame_path}")
                return frame_path
            else:
                self.logger.warning(f"Frame is not a numpy array, cannot save")
                return None
                
        except Exception as e:
            self.logger.error(f"Error saving frame: {e}", exc_info=True)
            return None
            
    def _record_observation(self, data, stage):
        """
        Record observation data for debugging.
        
        Args:
            data: The data to record
            stage (str): The stage of processing
        """
        try:
            # Get the appropriate directory for this stage
            stage_dir = self.stage_dirs[stage]
            
            # Add metadata to the data
            if isinstance(data, dict):
                data = {
                    **data,
                    "step": self.steps,
                    "timestamp": datetime.now().isoformat(),
                    "stage": stage
                }
            
            # Save data as JSON
            output_file = os.path.join(stage_dir, f"{stage}_{self.steps:04d}.json")
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            # Log the recording
            self.logger.debug(f"Recorded {stage} observation to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error recording {stage} observation: {e}", exc_info=True)
            
    async def get_action(self, observation):
        """
        Get the next action based on the current observation.
        
        Args:
            observation: The current game observation
            
        Returns:
            dict: Dictionary containing action and thought
        """
        try:
            # Log step start
            self.logger.info(f"Step {self.steps + 1} starting")
            
            # Get action from base module
            action_index = self.base.process_observation(observation)
            
            # Convert action index to action name
            action_names = ["move_left", "move_right", "attack"]
            action_name = action_names[action_index]
            
            # Create action plan
            action_plan = {
                "action": action_name,  # Send the action name as a string
                "move": action_name,  # For logging
                "thought": f"Base module action: {action_name}"
            }
            
            # Update state tracking
            self.last_action = action_name
            self.steps += 1
            
            # Record final action
            final_action = {
                "action": action_name,  # Store the action name
                "move": action_name,
                "thought": action_plan["thought"],
                "step": self.steps,
                "timestamp": datetime.now().isoformat()
            }
            self._record_observation(final_action, 'final')
            
            # Log step completion
            self.logger.info(f"Step {self.steps} completed - Action: {action_name}")
            
            return final_action
            
        except Exception as e:
            self.logger.error(f"Error in get_action: {e}", exc_info=True)
            error_action = {
                "action": "skip",  # No action
                "move": "skip",
                "thought": f"Error occurred: {str(e)}",
                "step": self.steps,
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
            self._record_observation(error_action, 'final')
            return error_action
            
    def _convert_move_to_action(self, move):
        """
        Convert the base module's move to a game action.
        
        Args:
            move (str): The move from the base module
            
        Returns:
            str: The game action to take
        """
        # Convert move to action
        if move == "left":
            return "move_left"
        elif move == "right":
            return "move_right"
        elif move == "attack":
            return "attack"
        else:
            print(f"Invalid move '{move}', defaulting to move_right")
            return "move_right"
            
    def _reconstruct_video(self):
        """
        Reconstruct video from saved frames using ffmpeg.
        """
        try:
            # Get all frame files
            frame_files = sorted([f for f in os.listdir(self.frames_dir) if f.endswith('.png')])
            if not frame_files:
                self.logger.warning("No frames found to reconstruct video")
                return
                
            # Create frame list file for ffmpeg
            frame_list_path = os.path.join(self.frames_dir, "frame_list.txt")
            with open(frame_list_path, 'w') as f:
                for frame_file in frame_files:
                    abs_path = os.path.abspath(os.path.join(self.frames_dir, frame_file))
                    f.write(f"file '{abs_path}'\n")
            
            # Set output video path
            video_path = os.path.join(self.output_dir, f"gameplay_{self.timestamp}.mp4")
            
            # Get frame resolution from first frame
            first_frame = Image.open(os.path.join(self.frames_dir, frame_files[0]))
            width, height = first_frame.size
            
            # Use ffmpeg to create video
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', frame_list_path,
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-r', '30',  # 30 FPS
                '-vf', f'scale={width}:{height}',
                video_path
            ]
            
            subprocess.run(ffmpeg_cmd, check=True)
            
            # Save video metadata
            video_metadata = {
                "start_time": self.timestamp,
                "end_time": datetime.now().isoformat(),
                "total_frames": len(frame_files),
                "resolution": [width, height],
                "fps": 30.0,
                "video_path": video_path
            }
            
            metadata_path = os.path.join(self.output_dir, f"video_metadata_{self.timestamp}.json")
            with open(metadata_path, 'w') as f:
                json.dump(video_metadata, f, indent=2)
            
            self.logger.info(f"Created video: {video_path}")
            self.logger.info(f"Total frames: {len(frame_files)}")
            
            # Clean up temporary files
            try:
                os.remove(frame_list_path)
            except Exception as e:
                self.logger.warning(f"Error cleaning up temporary files: {e}")
            
            return video_path
            
        except Exception as e:
            self.logger.error(f"Error reconstructing video: {e}", exc_info=True)
            return None

async def run():
    args = parse_args()

    try:
        # Get model configuration
        model_name = get_model_config(args.model)
        print(f"Using model: {model_name}")

        # Verify API key is set
        config = load_model_config()
        model_info = config["models"][model_name]
        api_key = os.getenv(model_info["env_var"])
        if not api_key:
            raise ValueError(f"Environment variable {model_info['env_var']} not set. Please set it before running the agent.")
        print(f"API key found for {model_name}")

        # Set up cache directory with full permissions
        cache_dir = os.path.join("doom", "cache", f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(cache_dir, exist_ok=True)
        os.chmod(cache_dir, 0o777)  # Ensure full permissions
        print(f"Using cache directory: {cache_dir}")

        # Create video directory if recording is enabled
        video_dir = os.path.join(cache_dir, "recordings")
        os.makedirs(video_dir, exist_ok=True)
        os.chmod(video_dir, 0o777)  # Ensure full permissions
        print(f"Video recording enabled. Videos will be saved to: {video_dir}")

        # Initialize the environment with updated parameters
        env = DoomEnvWrapper(
            game_name="doom",
            config_dir_path="gamingagent/envs/custom_05_doom",
            observation_mode="both",  # Get both vision and text observations
            base_log_dir=cache_dir,
            render_mode_human=False,
            record_video=True,  # Enable video recording
            video_dir=video_dir,
            model_name=model_name,
            headless=True  # Set to False if GUI rendering is required
        )

        # Initialize agent with all modules enabled
        agent = DoomAgent(
            model_name=model_name,  # Pass the model name from config
            reasoning_effort="high",
            thinking=True
        )

        # Reset the environment
        observation, info = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        max_steps = 100  # Prevent infinite loops

        try:
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

        finally:
            # Close environment - this will handle video cleanup
            env.close()
            print("\nEnvironment closed.")

    except Exception as e:
        print(f"Error in run: {str(e)}")
        print("Please check:")
        print("1. API key is set correctly in environment variables")
        print("2. Model configuration is valid")
        print("3. All required directories have proper permissions")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)