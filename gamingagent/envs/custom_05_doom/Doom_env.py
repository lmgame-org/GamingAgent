from __future__ import annotations

import os
import json
import logging
import signal
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional
import glob
import subprocess
import shutil
import time

import numpy as np
import cv2
import vizdoom as vzd
from vizdoom import DoomGame, Mode, ScreenResolution, ScreenFormat, Button, GameVariable
from gamingagent.envs.gym_env_adapter import GymEnvAdapter
from gamingagent.modules.core_module import Observation

# Disable audio and PipeWire completely
os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['ALSA_CARD'] = 'none'
os.environ['PULSE_SERVER'] = 'none'
os.environ['PIPEWIRE_RUNTIME_DIR'] = 'none'

# Set display environment variable
if 'DISPLAY' not in os.environ:
    os.environ['DISPLAY'] = ':0'

__all__ = ["DoomEnvWrapper"]

class DoomEnvWrapper:
    """Wrapper for the Doom environment with enhanced logging and recording capabilities."""
    
    _DEFAULT_ENV_ID = "VizdoomBasic-v0"
    _DEFAULT_RESOLUTION = (320, 240)
    _SCALED_RESOLUTION = (2048, 1536)
    _FPS = 30.0
    
    def __init__(
        self,
        game_name: str,
        config_dir_path: str = "gamingagent/envs/custom_05_doom",
        observation_mode: str = "vision",
        base_log_dir: str = "cache/doom",
        render_mode_human: bool = False,
        record_video: bool = False,
        video_dir: str = "videos/doom",
        model_name: str = "default",
        headless: bool = False
    ) -> None:
        """Initialize the Doom environment wrapper."""
        # Set basic attributes first
        self.game_name = game_name
        self.config_dir_path = os.path.abspath(config_dir_path)  # Use absolute path
        self.observation_mode = observation_mode.lower()
        self.base_log_dir = base_log_dir
        self.render_mode_human = render_mode_human
        self.record_video = record_video
        self.video_dir = video_dir
        self.model_name = self._normalize_model_name(model_name)
        self.headless = headless
        
        # Set up directories and logging first
        self._setup_directories()
        self._setup_logging()
        
        # Now we can use the logger
        self.logger.info(f"Initializing Doom environment for model: {self.model_name}")
        
        # Set up Xvfb for headless mode if needed
        self._xvfb = None
        if self.headless:
            try:
                from xvfbwrapper import Xvfb
                self._xvfb = Xvfb(width=640, height=480, colordepth=24)
                self._xvfb.start()
                self.logger.info("[DoomEnvWrapper] Started Xvfb for headless mode")
            except ImportError:
                self.logger.warning("[DoomEnvWrapper] xvfbwrapper not installed. Running without Xvfb.")
            except Exception as e:
                self.logger.error(f"[DoomEnvWrapper] Error setting up Xvfb: {e}")
        
        # Load configuration
        cfg_file = os.path.join(self.config_dir_path, "game_env_config.json")
        self._cfg = self._load_config(cfg_file)
        if not self._cfg:
            self.logger.error(f"Failed to load config from {cfg_file}")
            raise FileNotFoundError(f"Config file not found: {cfg_file}")
        
        # Set up signal handling
        self._setup_signal_handling()
        
        # Initialize game components
        self._init_game_components()
        
        # Set up video recording if enabled
        if self.record_video:
            self._setup_video_writer()
            
        self.logger.info(f"[DoomEnvWrapper] Initialized environment for model: {self.model_name}")
        self.logger.info(f"[DoomEnvWrapper] Run directory: {self.run_dir}")

    def _load_config(self, cfg_file: str) -> Dict[str, Any]:
        """Load configuration from file."""
        # Try multiple possible config paths
        config_paths = [
            cfg_file,  # Try the provided path first
            os.path.join("GamingAgent", "gamingagent", "envs", "custom_05_doom", "game_env_config.json"),
            os.path.join(os.path.dirname(__file__), "game_env_config.json"),  # Try relative to this file
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "gamingagent", "envs", "custom_05_doom", "game_env_config.json")
        ]
        
        for path in config_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        config = json.load(f)
                        self.logger.info(f"Loaded config from: {path}")
                        return config
                except Exception as e:
                    self.logger.error(f"Error loading config from {path}: {e}")
        
        self.logger.error(f"Failed to load config from any of these paths: {config_paths}")
        return {}

    def _normalize_model_name(self, model_name: str) -> str:
        """Normalize and validate model name."""
        model_name = model_name.lower()
        if model_name == "default":
            model_name = os.environ.get("MODEL_NAME", "default").lower()
        return model_name

    def _setup_signal_handling(self) -> None:
        """Set up signal handling for graceful shutdown."""
        self._original_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_sigint)

    def _setup_directories(self) -> None:
        """Set up directory structure for logging and recording."""
        # Create model-specific directory
        self.model_dir = os.path.join(self.base_log_dir, self.model_name)
        self._verify_directory_permissions(self.model_dir)
        
        # Create run-specific directory
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.model_dir, self.run_timestamp)
        
        # Create subdirectories
        self.recordings_dir = os.path.join(self.run_dir, "recordings")
        self.screenshots_dir = os.path.join(self.run_dir, "screenshots")
        self.logs_dir = os.path.join(self.run_dir, "logs")
        
        for directory in [self.run_dir, self.recordings_dir, self.screenshots_dir, self.logs_dir]:
            os.makedirs(directory, exist_ok=True)

    def _verify_directory_permissions(self, directory: str) -> None:
        """Verify write permissions for a directory."""
        try:
            os.makedirs(directory, exist_ok=True)
            test_file = os.path.join(directory, ".test")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
        except (OSError, IOError) as e:
            raise RuntimeError(f"Cannot write to directory {directory}: {e}")

    def _setup_logging(self) -> None:
        """Set up logging configuration."""
        # Create logger
        self.logger = logging.getLogger(f"doom_{self.model_name}")
        
        # Set log level from environment or default to INFO
        log_level = os.environ.get("DOOM_LOG_LEVEL", "INFO").upper()
        self.logger.setLevel(getattr(logging, log_level))
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Create handlers
        handlers = [
            logging.FileHandler(os.path.join(self.logs_dir, "doom_env.log")),
            logging.StreamHandler()
        ]
        
        # Configure handlers with detailed format
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )
        
        for handler in handlers:
            handler.setLevel(getattr(logging, log_level))
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.logger.info(f"Initializing Doom environment for model: {self.model_name}")
        self.logger.info(f"Run directory: {self.run_dir}")
        self.logger.info(f"Log level: {log_level}")

    def _init_game_components(self) -> None:
        """Initialize game-related components."""
        # Initialize adapter first
        self.adapter = GymEnvAdapter(
            game_name=self.game_name,
            observation_mode=self.observation_mode,
            agent_cache_dir=self.run_dir,
            game_specific_config_path=os.path.join(self.config_dir_path, "game_env_config.json"),
        )
        
        # Initialize game state
        self._game = None
        self.current_frame = None
        self.current_info = {}
        self.current_episode = 1
        self.frame_count = 0
        self.video_writer = None  # Initialize video_writer attribute
        
        # Initialize game
        self._initialize_env()
        
        self.logger.info("[DoomEnvWrapper] Game components initialized successfully")

    def _setup_video_writer(self) -> None:
        """Set up video writer for recording gameplay."""
        if not self.record_video:
            return
            
        try:
            # Create recordings directory if it doesn't exist
            os.makedirs(self.recordings_dir, exist_ok=True)
            os.chmod(self.recordings_dir, 0o777)  # Ensure full permissions
            
            # Create temporary frames directory
            self.frames_dir = os.path.join(self.recordings_dir, f"episode_{self.current_episode}_frames")
            os.makedirs(self.frames_dir, exist_ok=True)
            os.chmod(self.frames_dir, 0o777)  # Ensure full permissions
            
            # Create metadata file
            metadata_path = os.path.join(self.recordings_dir, f"episode_{self.current_episode}_metadata.json")
            self.metadata = {
                "start_time": datetime.now().isoformat(),
                "model_name": self.model_name,
                "episode": self.current_episode,
                "resolution": self._DEFAULT_RESOLUTION,
                "fps": self._FPS,
                "frames": [],
                "video_path": None
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            
            self.logger.info(f"[DoomEnvWrapper] Started video recording for episode {self.current_episode}")
            self.logger.info(f"[DoomEnvWrapper] Frames directory: {self.frames_dir}")
            self.logger.info(f"[DoomEnvWrapper] Metadata file: {metadata_path}")
            
            # Start recording immediately
            if self.current_frame is not None:
                self._write_frame_to_video(self.current_frame)
            
        except Exception as e:
            self.logger.error(f"[DoomEnvWrapper] Error setting up video writer: {e}", exc_info=True)
            self.record_video = False

    def _write_frame_to_video(self, frame, action=None, reward=None, info=None):
        """Write a frame to the video file.
        
        Args:
            frame: The frame to write
            action: Optional action that was taken
            reward: Optional reward received
            info: Optional game state info
        """
        if not self.record_video or frame is None:
            return
            
        try:
            # Save frame with sequential number instead of timestamp
            frame_num = len(self.metadata["frames"])
            frame_path = os.path.join(self.frames_dir, f"frame_{frame_num:04d}.png")
            
            # Ensure frame is in correct format
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            
            # Save frame
            cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            # Update metadata with additional info
            frame_info = {
                "frame_number": frame_num,
                "frame_path": frame_path,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add optional info if provided
            if action is not None:
                frame_info["action"] = action
            if reward is not None:
                frame_info["reward"] = reward
            if info is not None:
                frame_info["game_state"] = info
            
            self.metadata["frames"].append(frame_info)
            
            # Update metadata file
            metadata_path = os.path.join(self.recordings_dir, f"episode_{self.current_episode}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            
            self.logger.info(f"[DoomEnvWrapper] Wrote frame {frame_num} to video")
            
        except Exception as e:
            self.logger.error(f"[DoomEnvWrapper] Error writing frame to video: {e}", exc_info=True)

    def _cleanup_video_writer(self):
        """Clean up video writer and create final video."""
        if not self.record_video or not hasattr(self, 'frames_dir'):
            return
            
        try:
            # Create output video path
            output_path = os.path.join(self.recordings_dir, f"episode_{self.current_episode}.mp4")
            
            # Check if we have any frames
            if not os.path.exists(self.frames_dir):
                self.logger.warning(f"[DoomEnvWrapper] Frames directory does not exist: {self.frames_dir}")
                return
                
            frame_files = sorted([f for f in os.listdir(self.frames_dir) if f.endswith('.png')])
            if not frame_files:
                self.logger.warning(f"[DoomEnvWrapper] No frames found in {self.frames_dir}")
                return
            
            self.logger.info(f"[DoomEnvWrapper] Found {len(frame_files)} frames to convert")
            
            # Get frame resolution from first frame
            first_frame = cv2.imread(os.path.join(self.frames_dir, frame_files[0]))
            height, width = first_frame.shape[:2]
            
            # Create a temporary video file for each frame
            temp_video = os.path.join(self.frames_dir, "temp_video.mp4")
            
            # First, create a video from the frames
            ffmpeg_cmd1 = [
                'ffmpeg', '-y',
                '-framerate', str(self._FPS),
                '-i', os.path.join(self.frames_dir, 'frame_%04d.png'),
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-r', str(self._FPS),
                '-vf', f'scale={width}:{height}',
                temp_video
            ]
            
            # Run first ffmpeg command
            self.logger.info("[DoomEnvWrapper] Creating initial video from frames...")
            result = subprocess.run(ffmpeg_cmd1, capture_output=True, text=True)
            if result.returncode != 0:
                self.logger.error(f"[DoomEnvWrapper] FFmpeg error (step 1): {result.stderr}")
                raise RuntimeError(f"FFmpeg failed (step 1): {result.stderr}")
            
            # Verify temp video was created
            if not os.path.exists(temp_video):
                raise RuntimeError("Temporary video file was not created")
            self.logger.info(f"[DoomEnvWrapper] Created temporary video: {temp_video}")
            
            # Then, optimize the video for web playback
            ffmpeg_cmd2 = [
                'ffmpeg', '-y',
                '-i', temp_video,
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                '-preset', 'medium',
                '-crf', '23',
                output_path
            ]
            
            # Run second ffmpeg command
            self.logger.info("[DoomEnvWrapper] Optimizing video for web playback...")
            result = subprocess.run(ffmpeg_cmd2, capture_output=True, text=True)
            if result.returncode != 0:
                self.logger.error(f"[DoomEnvWrapper] FFmpeg error (step 2): {result.stderr}")
                raise RuntimeError(f"FFmpeg failed (step 2): {result.stderr}")
            
            # Verify final video was created
            if not os.path.exists(output_path):
                raise RuntimeError("Final video file was not created")
            
            # Get video information
            ffprobe_cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height,r_frame_rate,duration',
                '-of', 'json',
                output_path
            ]
            
            probe_result = subprocess.run(ffprobe_cmd, capture_output=True, text=True)
            if probe_result.returncode == 0:
                video_info = json.loads(probe_result.stdout)
                self.logger.info(f"[DoomEnvWrapper] Video info: {json.dumps(video_info, indent=2)}")
            
            # Update metadata with video info
            self.metadata["video_path"] = output_path
            self.metadata["end_time"] = datetime.now().isoformat()
            self.metadata["total_frames"] = len(frame_files)
            self.metadata["resolution"] = [width, height]
            self.metadata["fps"] = self._FPS
            self.metadata["duration"] = len(frame_files) / self._FPS
            
            metadata_path = os.path.join(self.recordings_dir, f"episode_{self.current_episode}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            
            self.logger.info(f"[DoomEnvWrapper] Created video: {output_path}")
            self.logger.info(f"[DoomEnvWrapper] Total frames: {len(frame_files)}")
            self.logger.info(f"[DoomEnvWrapper] Resolution: {width}x{height}")
            self.logger.info(f"[DoomEnvWrapper] FPS: {self._FPS}")
            self.logger.info(f"[DoomEnvWrapper] Duration: {len(frame_files)/self._FPS:.2f} seconds")
            
            # Clean up temporary files
            try:
                if os.path.exists(temp_video):
                    os.remove(temp_video)
                for frame_file in frame_files:
                    os.remove(os.path.join(self.frames_dir, frame_file))
                os.rmdir(self.frames_dir)
            except Exception as e:
                self.logger.warning(f"[DoomEnvWrapper] Error cleaning up temporary files: {e}")
            
        except Exception as e:
            self.logger.error(f"[DoomEnvWrapper] Error creating video: {e}", exc_info=True)
            # Keep the frames directory in case of error
            self.logger.info(f"[DoomEnvWrapper] Frames preserved in: {self.frames_dir}")

    def _save_frame(self, frame: np.ndarray, episode_id: int, step_num: int) -> Optional[str]:
        """Save a frame to the screenshots directory.
        
        Args:
            frame: The frame to save
            episode_id: Current episode ID
            step_num: Current step number
            
        Returns:
            Path to the saved frame, or None if saving failed
        """
        try:
            if frame is None:
                self.logger.error("[DoomEnvWrapper] Cannot save frame: frame is None")
                return None
                
            if not isinstance(frame, np.ndarray):
                self.logger.error(f"[DoomEnvWrapper] Invalid frame type: {type(frame)}")
                return None
                
            if frame.size == 0:
                self.logger.error("[DoomEnvWrapper] Cannot save frame: frame is empty")
                return None
                
            self.logger.debug(f"[DoomEnvWrapper] Saving frame shape: {frame.shape}")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            
            # Save original frame
            screenshot_path = os.path.join(self.screenshots_dir, f"step_{timestamp}.png")
            cv2.imwrite(screenshot_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            self.logger.debug(f"[DoomEnvWrapper] Saved original frame to: {screenshot_path}")
            
            # Save scaled frame
            scaled_path = os.path.join(self.screenshots_dir, f"step_{timestamp}_scaled.png")
            scaled_frame = cv2.resize(frame, self._SCALED_RESOLUTION)
            cv2.imwrite(scaled_path, cv2.cvtColor(scaled_frame, cv2.COLOR_RGB2BGR))
            self.logger.debug(f"[DoomEnvWrapper] Saved scaled frame to: {scaled_path}")
            
            return scaled_path
            
        except Exception as e:
            self.logger.error(f"[DoomEnvWrapper] Error saving frame: {e}")
            self.logger.error(f"[DoomEnvWrapper] Frame shape: {frame.shape if frame is not None else 'None'}")
            self.logger.error(f"[DoomEnvWrapper] Frame type: {type(frame)}")
            return None

    def _handle_sigint(self, signum: int, frame: Any) -> None:
        """Handle SIGINT (Ctrl+C) gracefully."""
        self.logger.info("Received SIGINT, cleaning up...")
        self.close()
        signal.signal(signal.SIGINT, self._original_sigint_handler)
        raise KeyboardInterrupt()

    def _initialize_env(self) -> None:
        """Initialize the Doom environment."""
        try:
            self.logger.info("Initializing Doom environment...")
            self._game = DoomGame()
            
            # Configure game settings
            self.logger.debug("Configuring game settings...")
            self._configure_game_settings()
            
            # Initialize game
            self.logger.debug("Initializing game...")
            self._game.init()
            self.logger.info("Successfully initialized Doom environment")
            
        except Exception as e:
            self.logger.error(f"Error initializing Doom environment: {e}", exc_info=True)
            if self._game is not None:
                self._game.close()
                self._game = None
            raise

    def _configure_game_settings(self) -> None:
        """Configure game settings."""
        # Set screen resolution and format
        self._game.set_screen_resolution(ScreenResolution.RES_640X480)
        self._game.set_screen_format(ScreenFormat.RGB24)
        
        # Set rendering options
        self._game.set_window_visible(not self.headless)  # Show window unless in headless mode
        self._game.set_render_hud(self._cfg.get("rendering_options", {}).get("render_hud", True))
        self._game.set_render_crosshair(self._cfg.get("rendering_options", {}).get("render_crosshair", False))
        self._game.set_render_weapon(self._cfg.get("rendering_options", {}).get("render_weapon", True))
        self._game.set_render_decals(self._cfg.get("rendering_options", {}).get("render_decals", False))
        self._game.set_render_particles(self._cfg.get("rendering_options", {}).get("render_particles", False))
        self._game.set_render_effects_sprites(self._cfg.get("rendering_options", {}).get("render_effects_sprites", False))
        self._game.set_render_messages(self._cfg.get("rendering_options", {}).get("render_messages", False))
        self._game.set_render_corpses(self._cfg.get("rendering_options", {}).get("render_corpses", False))
        self._game.set_render_screen_flashes(self._cfg.get("rendering_options", {}).get("render_screen_flashes", False))
        self._game.set_render_minimal_hud(self._cfg.get("rendering_options", {}).get("render_minimal_hud", True))
        
        # Set game mode and settings
        self._game.set_mode(Mode.PLAYER)
        self._game.set_living_reward(self._cfg.get("rewards", {}).get("living_reward", -1))
        self._game.set_doom_skill(self._cfg.get("doom_skill", 3))  # Lower difficulty
        
        # Set episode settings
        self._game.set_episode_start_time(self._cfg.get("episode_settings", {}).get("episode_start_time", 14))
        self._game.set_episode_timeout(self._cfg.get("episode_settings", {}).get("episode_timeout", 600))
        
        # Set up available buttons - only basic actions
        self._game.set_available_buttons([
            Button.MOVE_LEFT,
            Button.MOVE_RIGHT,
            Button.ATTACK
        ])
        
        # Set up available game variables
        self._game.set_available_game_variables([
            GameVariable.AMMO2,
            GameVariable.KILLCOUNT,
            GameVariable.POSITION_X,
            GameVariable.POSITION_Y,
            GameVariable.ANGLE,
            GameVariable.HEALTH
        ])
        
        # Set up action repeat for more consistent movement
        self._game.set_ticrate(self._cfg.get("episode_settings", {}).get("ticrate", 20))
        
        # Set up enemy spawning
        self._game.set_doom_map("map01")  # Use the first map
        
        # Set scenario path - use absolute path to ensure it works
        scenario_path = os.path.join(os.path.dirname(vzd.__file__), "scenarios", "basic.wad")
        if not os.path.exists(scenario_path):
            self.logger.error(f"Scenario file not found at: {scenario_path}")
            raise FileNotFoundError(f"Scenario file not found at: {scenario_path}")
        self._game.set_doom_scenario_path(scenario_path)

    def _text_repr(self) -> str:
        """Generate a concise textual representation of the current game state."""
        if not self.current_info:
            return "No game state available."

        ammo = self.current_info.get('ammo', 'N/A')
        kills = self.current_info.get('kills', 'N/A')
        health = self.current_info.get('health', 'N/A')
        pos_x = self.current_info.get('position_x', 'N/A')
        pos_y = self.current_info.get('position_y', 'N/A')
        angle = self.current_info.get('angle', 'N/A')
        is_episode_finished = self.current_info.get('is_episode_finished', True)

        return "\n".join([
            f"State: Health={health}, Ammo={ammo}, Kills={kills}",
            f"Position: ({pos_x}, {pos_y}), Angle={angle}",
            f"Status: {'Finished' if is_episode_finished else 'In Progress'}",
            f"Actions: move_left, move_right, attack",
            f"Rewards: +106 kill, -5 shot, +1 alive",
            f"Goal: Kill monster or timeout (600 tics)"
        ])

    def _buttons_from_str(self, action_str: Optional[str]) -> List[int]:
        """Convert action string to button presses."""
        if action_str is None:
            self.logger.warning("[DoomEnvWrapper] Received None action, using no-op")
            return [0] * 3  # Only 3 buttons: move_left, move_right, attack
            
        action_map = {
            "move_left": [1, 0, 0],
            "move_right": [0, 1, 0],
            "attack": [0, 0, 1]
        }
        
        # Log the action mapping
        self.logger.info(f"[DoomEnvWrapper] Converting action '{action_str}' to buttons")
        buttons = action_map.get(action_str.lower(), [0] * 3)
        self.logger.info(f"[DoomEnvWrapper] Mapped to buttons: {buttons}")
        
        return buttons

    def _extract_game_specific_info(self) -> Dict[str, Any]:
        """Extract game-specific information from the current state."""
        if not self._game:
            return {}
            
        state = self._game.get_state()
        if state is None or state.game_variables is None or len(state.game_variables) == 0:
            return {}
            
        # Extract all available game variables
        info = {
            "ammo": int(state.game_variables[0]),
            "kills": int(state.game_variables[1]) if len(state.game_variables) > 1 else 0,
            "position_x": float(state.game_variables[2]) if len(state.game_variables) > 2 else 0.0,
            "position_y": float(state.game_variables[3]) if len(state.game_variables) > 3 else 0.0,
            "angle": float(state.game_variables[4]) if len(state.game_variables) > 4 else 0.0,
            "health": int(state.game_variables[5]) if len(state.game_variables) > 5 else 100
        }
        
        # Add additional state information
        info.update({
            "is_episode_finished": self._game.is_episode_finished(),
            "is_player_dead": self._game.is_player_dead()
        })
        
        return info

    def reset(self, *, seed: int | None = None, episode_id: int = 1, **kwargs) -> Tuple[Observation, Dict[str, Any]]:
        """Reset the environment and return the initial observation."""
        try:
            self._initialize_env()
            self.adapter.reset_episode(episode_id)
            self.current_episode = episode_id
            self.frame_count = 0
            
            # Set up video recording before starting the episode
            if self.record_video:
                self._setup_video_writer()
            
            self.logger.info(f"Starting episode {episode_id}")
            self._game.new_episode()

            # Get initial state
            state = self._game.get_state()
            self.current_frame = state.screen_buffer if state and state.screen_buffer is not None else None
            self.current_info = self._extract_game_specific_info()

            # Handle observation
            img_path = None
            if self.adapter.observation_mode in ("vision", "both") and self.current_frame is not None:
                img_path = self._save_frame(self.current_frame, episode_id, 0)
                # Write initial frame to video
                if self.record_video:
                    self._write_frame_to_video(self.current_frame)

            obs = self.adapter.create_agent_observation(
                img_path=img_path,
                text_representation=self._text_repr()
            )

            return obs, self.current_info.copy()
            
        except Exception as e:
            self.logger.error(f"Error in reset: {e}")
            self.close()
            raise

    def step(self, agent_action_str: Optional[str]) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """Execute an action and return the next observation."""
        try:
            # Get previous state for comparison
            prev_info = self.current_info.copy()
            
            # Convert action string to button presses
            buttons = self._buttons_from_str(agent_action_str)
            self.logger.info(f"[DoomEnvWrapper] Executing action with buttons: {buttons}")
            
            # Execute action
            reward = self._game.make_action(buttons)
            done = self._game.is_episode_finished()
            
            # Update state
            if not done:
                state = self._game.get_state()
                if state is None:
                    self.logger.error("[DoomEnvWrapper] Game state is None after action")
                    raise RuntimeError("Game state is None after action")
                    
                if state.screen_buffer is None:
                    self.logger.error("[DoomEnvWrapper] Screen buffer is None after action")
                    raise RuntimeError("Screen buffer is None after action")
                    
                self.current_frame = state.screen_buffer
                self.current_info = self._extract_game_specific_info()
                
                # Log state changes
                self.logger.info(f"[DoomEnvWrapper] State changes:")
                self.logger.info(f"  Position: ({prev_info.get('position_x', 0)}, {prev_info.get('position_y', 0)}) -> ({self.current_info.get('position_x', 0)}, {self.current_info.get('position_y', 0)})")
                self.logger.info(f"  Angle: {prev_info.get('angle', 0)} -> {self.current_info.get('angle', 0)}")
                self.logger.info(f"  Ammo: {prev_info.get('ammo', 0)} -> {self.current_info.get('ammo', 0)}")
                self.logger.info(f"  Kills: {prev_info.get('kills', 0)} -> {self.current_info.get('kills', 0)}")
                self.logger.info(f"  Reward: {reward}")

                # Write frame to video with progress information
                if self.record_video:
                    self._write_frame_to_video(
                        self.current_frame,
                        action=agent_action_str,
                        reward=reward,
                        info=self.current_info
                    )
            else:
                self.current_frame = None
                self.current_info = {}
                self.logger.info(f"[DoomEnvWrapper] Episode {self.current_episode} finished")

            # Handle observation
            img_path = None
            if self.adapter.observation_mode in ("vision", "both") and self.current_frame is not None:
                img_path = self._save_frame(self.current_frame, self.current_episode, self.adapter.current_step_num)
                self.logger.info(f"[DoomEnvWrapper] Saved frame to: {img_path}")

            obs = self.adapter.create_agent_observation(
                img_path=img_path,
                text_representation=self._text_repr()
            )

            return obs, reward, done, self.current_info.copy()
            
        except Exception as e:
            self.logger.error(f"[DoomEnvWrapper] Error in step: {e}", exc_info=True)
            self.close()
            raise

    def render(self) -> None:
        """Render the game for human viewing."""
        if self._game:
            self._game.render()

    def close(self) -> None:
        """Clean up resources when the environment is closed."""
        try:
            # Ensure video recording is cleaned up
            if hasattr(self, 'record_video') and self.record_video:
                self._cleanup_video_writer()
            
            # Close the DoomGame instance
            if hasattr(self, '_game') and self._game is not None:
                self._game.close()
                self._game = None
            
            # Close Xvfb if it was started
            if hasattr(self, '_xvfb') and self._xvfb is not None:
                self._xvfb.stop()
                self._xvfb = None
            
            self.logger.info("Environment closed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during environment cleanup: {e}")
            # Try to ensure video is saved even if there's an error
            if hasattr(self, 'record_video') and self.record_video:
                try:
                    self._cleanup_video_writer()
                except:
                    pass