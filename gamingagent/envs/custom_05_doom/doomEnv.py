from __future__ import annotations

import os
import json
import logging
import sys
import time
import faulthandler
from typing import Any, Dict, List, Tuple, Optional
import datetime

import numpy as np
import vizdoom as vzd
import gymnasium as gym
from gymnasium import spaces

from gamingagent.envs.gym_env_adapter import GymEnvAdapter
from gamingagent.modules.core_module import Observation

# Enable fault handler for better crash information
faulthandler.enable()

# Set minimal environment variables to prevent SDL initialization issues
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["ALSA_CARD"] = "none"
os.environ["PULSE_SERVER"] = "none"
os.environ["PIPEWIRE_RUNTIME_DIR"] = "none"

# Set display environment variable
if 'DISPLAY' not in os.environ:
    os.environ['DISPLAY'] = ':0'

def log_memory_usage():
    """Log current memory usage."""
    process = psutil.Process(os.getpid())
    print(f"[{time.time()}] Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB", file=sys.stderr)

def log_system_info():
    """Log system information."""
    print(f"[{time.time()}] Python version: {sys.version}", file=sys.stderr)
    print(f"[{time.time()}] VizDoom version: {vzd.__version__}", file=sys.stderr)
    print(f"[{time.time()}] Environment variables:", file=sys.stderr)
    for var in ['DISPLAY', 'SDL_VIDEODRIVER', 'SDL_AUDIODRIVER']:
        print(f"[{time.time()}] {var}: {os.environ.get(var)}", file=sys.stderr)

__all__ = ["DoomEnvWrapper"]

class DoomEnvWrapper(gym.Env):
    """Wrapper for the Doom environment.
    
    This wrapper provides a Gymnasium-compatible interface to the VizDoom environment.
    It handles game initialization, state management, and action execution.
    
    Attributes:
        game_name (str): Name of the game
        config_dir_path (str): Path to the configuration directory
        observation_mode (str): Mode of observation ("vision", "text", or "both")
        base_log_dir (str): Base directory for logging
        render_mode (str): Rendering mode
        model_name (str): Name of the model
        headless (bool): Whether to run in headless mode
        record_video (bool): Whether to record video
        video_dir (str): Directory to save videos
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        game_name: str,
        config_dir_path: str = "gamingagent/envs/custom_05_doom",
        observation_mode: str = "vision",
        base_log_dir: str = "cache/doom",
        render_mode: str | None = None,
        model_name: str = "default",
        headless: bool = True,
        record_video: bool = False,
        video_dir: str | None = None,
        render_mode_human: bool = False,
        debug: bool = False
    ) -> None:
        """Initialize the Doom environment wrapper.
        
        Args:
            game_name: Name of the game
            config_dir_path: Path to the configuration directory
            observation_mode: Mode of observation ("vision", "text", or "both")
            base_log_dir: Base directory for logging
            render_mode: Rendering mode
            model_name: Name of the model
            headless: Whether to run in headless mode
            record_video: Whether to record video
            video_dir: Directory to save videos
            render_mode_human: Whether to render in human mode
        """
        super().__init__()
        
        self.debug = debug
        if self.debug:
            log_system_info()
            log_memory_usage()
        
        print("[DoomEnvWrapper] Starting initialization...", file=sys.stderr)
        
        # Basic attributes
        self.game_name = game_name
        self.config_dir_path = os.path.abspath(config_dir_path)
        self.observation_mode = observation_mode.lower()
        self.base_log_dir = base_log_dir
        self.render_mode = "human" if render_mode_human else render_mode
        self.model_name = model_name
        self.headless = headless
        self.record_video = record_video
        self.video_dir = video_dir
        
        # Initialize episode tracking
        self.current_episode_id = 1
        self.current_step_num = 0
        
        # Create video directory if needed
        if self.record_video and self.video_dir:
            os.makedirs(self.video_dir, exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger(f"doom_{model_name}")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        print("[DoomEnvWrapper] Loading configuration...", file=sys.stderr)
        
        # Load configuration
        cfg_file = os.path.join(self.config_dir_path, "game_env_config.json")
        self._cfg = self._load_config(cfg_file)
        if not self._cfg:
            raise FileNotFoundError(f"Failed to load config from {cfg_file}")
        
        print("[DoomEnvWrapper] Initializing adapter...", file=sys.stderr)
        
        # Initialize adapter
        try:
            self.adapter = GymEnvAdapter(
                game_name=self.game_name,
                observation_mode=self.observation_mode,
                agent_cache_dir=self.base_log_dir,
                game_specific_config_path=cfg_file,
                max_steps_for_stuck=self._cfg.get("max_unchanged_steps_for_termination", 30)
            )
        except Exception as e:
            print(f"[DoomEnvWrapper] Error initializing adapter: {e}", file=sys.stderr)
            raise
        
        print("[DoomEnvWrapper] Initializing game components...", file=sys.stderr)
        
        # Initialize game components
        self._init_game_components()
        
        # State tracking
        self.current_frame = None
        self.current_info = {}
        
        print("[DoomEnvWrapper] Setting up observation and action spaces...", file=sys.stderr)
        
        # Define observation and action spaces
        screen_res = self._cfg.get("rendering_options", {}).get("screen_resolution", "RES_320X240")
        if screen_res == "RES_640X480":
            screen_shape = (480, 640, 3)  # Height, Width, Channels
        else:
            screen_shape = (240, 320, 3)  # Default to smaller resolution
            
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=screen_shape,
            dtype=np.uint8
        )
        
        # Define action space based on available buttons
        available_buttons = self._cfg.get("available_buttons", ["MOVE_LEFT", "MOVE_RIGHT", "ATTACK"])
        self.action_space = spaces.Discrete(len(available_buttons))
        
        print("[DoomEnvWrapper] Initialization complete.", file=sys.stderr)

    def _load_config(self, cfg_file: str) -> Dict[str, Any]:
        """Load configuration from file.
        
        Args:
            cfg_file: Path to the configuration file
            
        Returns:
            Dict containing the configuration
            
        Raises:
            FileNotFoundError: If the config file cannot be found
        """
        try:
            print(f"[DoomEnvWrapper] Loading config from: {cfg_file}", file=sys.stderr)
            with open(cfg_file, 'r') as f:
                config = json.load(f)
                self.logger.info(f"Loaded config from: {cfg_file}")
                return config
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return {}

    def _validate_game_state(self):
        """Validate game state."""
        if not self.game:
            raise RuntimeError("Game instance is None")
        if not hasattr(self.game, 'get_state'):
            raise RuntimeError("Game instance is not properly initialized")

    def _init_game_components(self) -> None:
        """Initialize the Doom game components."""
        try:
            print(f"[{time.time()}] Creating game instance...", file=sys.stderr)
            self.game = vzd.DoomGame()
            
            print(f"[{time.time()}] Setting basic settings...", file=sys.stderr)
            try:
                # Basic settings - following basic example
                self.game.set_window_visible(not self.headless)
                self.game.set_sound_enabled(False)  # Keep sound disabled as in basic example
                
                # Set scenario path first - exactly as in basic example
                scenario_name = self._cfg.get("doom_scenario_path", "basic.wad")
                vizdoom_dir = os.path.dirname(vzd.__file__)
                scenario_path = os.path.join(vizdoom_dir, "scenarios", scenario_name)
                
                if not os.path.exists(scenario_path):
                    raise FileNotFoundError(f"Scenario file not found: {scenario_path}")
                
                self.game.set_doom_scenario_path(scenario_path)
                
                # Set map - exactly as in basic example
                self.game.set_doom_map(self._cfg.get("doom_map", "map01"))
                
                # Screen settings - exactly as in basic example
                screen_res = self._cfg.get("rendering_options", {}).get("screen_resolution", "RES_320X240")
                self.game.set_screen_resolution(getattr(vzd.ScreenResolution, screen_res))
                self.game.set_screen_format(vzd.ScreenFormat.RGB24)
                
                # Set available buttons - exactly as in basic example
                available_buttons = self._cfg.get("available_buttons", ["MOVE_LEFT", "MOVE_RIGHT", "ATTACK"])
                self.game.set_available_buttons([getattr(vzd.Button, btn) for btn in available_buttons])
                
                # Set available game variables - exactly as in basic example
                available_vars = self._cfg.get("available_game_variables", [
                    "AMMO2",
                    "KILLCOUNT",
                    "POSITION_X",
                    "POSITION_Y",
                    "ANGLE",
                    "HEALTH"
                ])
                # Store the mapping of variable names to their indices for later use
                self.game_variable_indices = {
                    var: idx for idx, var in enumerate(available_vars)
                }
                self.game.set_available_game_variables([getattr(vzd.GameVariable, var) for var in available_vars])
                
                # Set episode settings - exactly as in basic example
                self.game.set_episode_timeout(self._cfg.get("episode_settings", {}).get("episode_timeout", 600))
                self.game.set_episode_start_time(self._cfg.get("episode_settings", {}).get("episode_start_time", 14))
                
                # Set rewards - exactly as in basic example
                rewards = self._cfg.get("rewards", {})
                self.game.set_living_reward(rewards.get("living_reward", -1))
                
                # Set game mode - exactly as in basic example
                self.game.set_mode(vzd.Mode.PLAYER)

                # Set rendering options
                self.game.set_render_hud(True)  # Always show HUD
                self.game.set_render_crosshair(False)
                self.game.set_render_weapon(True)  # Always show weapon
                self.game.set_render_decals(True)  # Show bullet impacts
                self.game.set_render_particles(True)  # Show particle effects
                self.game.set_render_effects_sprites(True)  # Show effect sprites
                self.game.set_render_messages(False)
                self.game.set_render_corpses(False)
                self.game.set_render_screen_flashes(True)  # Show muzzle flash
                self.game.set_render_minimal_hud(False)  # Show full HUD to display ammo
                
            except vzd.ViZDoomErrorException as e:
                print(f"[{time.time()}] VizDoom error in settings: {e}", file=sys.stderr)
                raise
            except Exception as e:
                print(f"[{time.time()}] Unexpected error in settings: {e}", file=sys.stderr)
                raise
            
            print(f"[{time.time()}] Initializing game engine...", file=sys.stderr)
            try:
                # Initialize exactly as in basic example
                self.game.init()
            except vzd.ViZDoomErrorException as e:
                print(f"[{time.time()}] VizDoom error in initialization: {e}", file=sys.stderr)
                raise
            except Exception as e:
                print(f"[{time.time()}] Unexpected error in initialization: {e}", file=sys.stderr)
                raise
            
            print(f"[{time.time()}] Game initialization successful.", file=sys.stderr)
            if self.debug:
                log_memory_usage()
                
        except Exception as e:
            print(f"[{time.time()}] Error during game initialization: {e}", file=sys.stderr)
            self.logger.error(f"Failed to initialize Doom game: {e}")
            if hasattr(self, 'game'):
                try:
                    self.game.close()
                except:
                    pass
            raise

    def _buttons_from_str(self, action_str: Optional[str]) -> List[bool]:
        """Convert action string to button presses.
        
        Args:
            action_str: Action string to convert
            
        Returns:
            List of button states (True or False)
        """
        if not action_str or action_str.lower() == "none":
            self.logger.info("No action provided, using no-op")
            return [False, False, False]  # No action
            
        action_str = action_str.strip().lower()
        
        # Map actions to button combinations exactly as in the VizDoom example
        # [MOVE_LEFT, MOVE_RIGHT, ATTACK]
        action_map = {
            "move_left": [True, False, False],
            "move_right": [False, True, False],
            "attack": [False, False, True]
        }
        
        buttons = action_map.get(action_str, [False, False, False])
        self.logger.info(f"Converting action '{action_str}' to buttons: {buttons}")
        self.logger.info(f"Button meanings: [MOVE_LEFT, MOVE_RIGHT, ATTACK]")
        
        return buttons

    def _extract_game_specific_info(self) -> Dict[str, Any]:
        """Extract game-specific information from the current state.
        
        Returns:
            Dict containing game state information
        """
        if not self.game:
            self.logger.error("Game instance is None")
            return {}
            
        state = self.game.get_state()
        if state is None:
            self.logger.error("Game state is None")
            return {}
        if state.game_variables is None or len(state.game_variables) == 0:
            self.logger.error("Game variables not available")
            return {}
            
        # Use the stored mapping to get variable values
        try:
            info = {
                "ammo": int(state.game_variables[self.game_variable_indices["AMMO2"]]),
                "kills": int(state.game_variables[self.game_variable_indices["KILLCOUNT"]]),
                "position_x": float(state.game_variables[self.game_variable_indices["POSITION_X"]]),
                "position_y": float(state.game_variables[self.game_variable_indices["POSITION_Y"]]),
                "angle": float(state.game_variables[self.game_variable_indices["ANGLE"]]),
                "health": int(state.game_variables[self.game_variable_indices["HEALTH"]])
            }
        except (KeyError, IndexError) as e:
            self.logger.error(f"Error accessing game variables: {e}")
            self.logger.error(f"Available variables: {self.game_variable_indices}")
            self.logger.error(f"Game variables array length: {len(state.game_variables)}")
            return {}
        
        # Add additional state information
        info.update({
            "is_episode_finished": self.game.is_episode_finished(),
            "is_player_dead": self.game.is_player_dead()
        })
        
        self.logger.info(f"Extracted game info: {info}")
        return info

    def _text_repr(self) -> str:
        """Generate a concise textual representation of the current game state.
        
        Returns:
            String representation of the game state
        """
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
            f"Rewards: +106 kill, -5 shot, -1 alive",
            f"Goal: Kill monster or timeout (600 tics)"
        ])

    def reset(self, *, seed: int | None = None, episode_id: int = 1, **kwargs) -> Tuple[Observation, Dict[str, Any]]:
        """Reset the environment to start a new episode.
        
        Args:
            seed: Random seed for reproducibility
            episode_id: ID of the current episode
            **kwargs: Additional arguments
            
        Returns:
            Tuple containing:
                - Observation object with game state
                - Dictionary with additional information
        """
        try:
            print(f"[{time.time()}] Resetting environment for episode {episode_id}...", file=sys.stderr)
            
            # Update episode tracking
            self.current_episode_id = episode_id
            self.current_step_num = 0
            
            # Reset the game
            self.game.new_episode()
            
            # Get initial state
            state = self.game.get_state()
            if state is None:
                raise RuntimeError("Failed to get initial game state")
            
            # Save current frame
            self.current_frame = state.screen_buffer
            
            # Create observation
            obs = Observation()
            obs.raw_observation = self.current_frame
            obs.textual_representation = self._text_repr()
            obs.processed_visual_description = ""  # Will be filled by perception module
            
            # Save frame and set image path
            if self.observation_mode in ("vision", "both"):
                img_path = self._save_game_frame(self.current_frame, self.current_episode_id, self.current_step_num)
                obs.img_path = img_path
            
            # Initialize game trajectory
            from gamingagent.modules.core_module import GameTrajectory
            obs.game_trajectory = GameTrajectory(max_length=100)
            
            # Add initial state to trajectory
            ts = datetime.datetime.now().isoformat(timespec="seconds")
            initial_entry = (
                f"##Turn Hash\n[{ts}]\n"
                f"###Obs\n{obs.textual_representation}\n"
                f"###Thought\nInitializing new episode\n"
            )
            obs.game_trajectory.add(initial_entry)
            
            # Extract game info
            self.current_info = self._extract_game_specific_info()
            self.current_info['episode_id'] = self.current_episode_id
            
            # Store current observation for next step
            self.last_observation = obs
            
            print(f"[{time.time()}] Environment reset complete.", file=sys.stderr)
            return obs, self.current_info
            
        except Exception as e:
            print(f"[{time.time()}] Error during reset: {e}", file=sys.stderr)
            self.logger.error(f"Failed to reset environment: {e}")
            raise

    def _save_game_frame(self, frame: np.ndarray, episode_id: int, step_num: int) -> str:
        """
        Save a game frame to the game_frames directory with episode and step information.
        
        Args:
            frame: The frame to save as a numpy array
            episode_id: Current episode ID
            step_num: Current step number
            
        Returns:
            str: Path to the saved frame
        """
        # Create a separate game_frames directory under base_log_dir
        game_frames_dir = os.path.join(self.base_log_dir, "game_frames")
        os.makedirs(game_frames_dir, exist_ok=True)
        
        # Create episode-specific directory under game_frames
        episode_dir = os.path.join(game_frames_dir, f"episode_{episode_id:03d}")
        os.makedirs(episode_dir, exist_ok=True)
        
        # Create path with episode and step numbers
        path = os.path.join(episode_dir, f"step_{step_num:04d}.png")
        
        try:
            from PIL import Image
            Image.fromarray(frame).save(path)
            
            # Save metadata alongside the frame
            metadata_path = os.path.join(episode_dir, f"step_{step_num:04d}_metadata.json")
            metadata = {
                "episode_id": episode_id,
                "step_num": step_num,
                "timestamp": datetime.datetime.now().isoformat(),
                "frame_path": path,
                "game_state": {
                    "ammo": self.game.get_game_variable(vzd.GameVariable.AMMO2),
                    "health": self.game.get_game_variable(vzd.GameVariable.HEALTH),
                    "position_x": self.game.get_game_variable(vzd.GameVariable.POSITION_X),
                    "position_y": self.game.get_game_variable(vzd.GameVariable.POSITION_Y),
                    "angle": self.game.get_game_variable(vzd.GameVariable.ANGLE)
                }
            }
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            print(f"[DoomEnvWrapper] Warning: could not save game frame to {path}: {e}")
            
        return path

    def step(
        self,
        agent_action_str: Optional[str],
        thought_process: str = "",
        time_taken_s: float = 0.0
    ) -> Tuple[Observation, float, bool, bool, Dict[str, Any], float]:
        """Execute an action and return the next observation."""
        try:
            # Store previous state for comparison
            prev_state = self._extract_game_specific_info()
            
            # Convert action to buttons
            buttons = self._buttons_from_str(agent_action_str)
            self.logger.info(f"Executing action with buttons: {buttons}")
            
            # Execute action and get reward
            reward = self.game.make_action(buttons)
            
            # Get new state
            state = self.game.get_state()
            if state is None:
                raise RuntimeError("Failed to get game state after action")
                
            # Update current state
            self.current_frame = state.screen_buffer
            self.current_info = self._extract_game_specific_info()
            self.current_info['episode_id'] = self.current_episode_id
            
            # Verify state changed
            if prev_state:
                changes = []
                for key in ['ammo', 'kills', 'position_x', 'position_y', 'angle', 'health']:
                    if key in prev_state and key in self.current_info:
                        if prev_state[key] != self.current_info[key]:
                            changes.append(f"{key}: {prev_state[key]} -> {self.current_info[key]}")
                if changes:
                    self.logger.info(f"State changes detected: {', '.join(changes)}")
                else:
                    self.logger.warning("No state changes detected after action")
            
            # Check if episode is finished
            done = self.game.is_episode_finished()
            
            # Log state changes
            self.logger.info(f"State #{state.number}")
            self.logger.info(f"Game variables: {state.game_variables}")
            self.logger.info(f"Reward: {reward}")
            
            # Create observation
            obs = Observation()
            obs.raw_observation = self.current_frame
            obs.textual_representation = self._text_repr()
            obs.processed_visual_description = ""  # Will be filled by perception module
            
            # Save frame and set image path
            if self.observation_mode in ("vision", "both") and self.current_frame is not None:
                img_path = self._save_game_frame(self.current_frame, self.current_episode_id, self.current_step_num)
                obs.img_path = img_path
            
            # Maintain game trajectory
            if hasattr(self, 'last_observation') and hasattr(self.last_observation, 'game_trajectory'):
                obs.game_trajectory = self.last_observation.game_trajectory
            else:
                from gamingagent.modules.core_module import GameTrajectory
                obs.game_trajectory = GameTrajectory(max_length=100)
            
            # Add current step to trajectory
            ts = datetime.datetime.now().isoformat(timespec="seconds")
            step_entry = (
                f"##Turn Hash\n[{ts}]\n"
                f"###Obs\n{obs.textual_representation}\n"
                f"###Action\n{agent_action_str if agent_action_str else 'no_action'}\n"
                f"###Thought\n{thought_process if thought_process else 'No thought provided'}\n"
            )
            obs.game_trajectory.add(step_entry)
            
            # Store current observation for next step
            self.last_observation = obs
            
            # Create game state dict for memory module
            game_state = {
                # Base module fields
                "textual_representation": obs.textual_representation,
                
                # Perception module fields
                "img_path": obs.img_path,
                
                # Memory module fields
                "prev_context": self.last_observation.textual_representation if hasattr(self, 'last_observation') else "",
                "current_observation": obs.textual_representation,
                "perception_data": obs.processed_visual_description,
                
                # Reasoning module fields
                "processed_visual_description": obs.processed_visual_description,
                "game_trajectory": obs.game_trajectory,
                "reflection": "",  # Will be filled by memory module
                
                # Environment state tracking
                "done": done
            }
            
            # Update current info with game state
            self.current_info.update(game_state)
            
            # Increment step counter after successful step
            self.current_step_num += 1
            
            return obs, reward, done, False, self.current_info.copy(), 0.0
            
        except Exception as e:
            self.logger.error(f"Error in step: {e}")
            raise

    def render(self) -> None:
        """Render the game.
        
        This method renders the game if not in headless mode.
        """
        if not self.headless and self.game:
            self.game.render()

    def close(self) -> None:
        """Clean up resources.
        
        This method closes the game instance and cleans up any resources.
        """
        if hasattr(self, 'game'):
            self.game.close()
