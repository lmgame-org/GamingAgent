from __future__ import annotations

import os
import json
import logging
import sys
import time
import faulthandler
from typing import Any, Dict, List, Tuple, Optional
import datetime
import random
import cv2
import psutil  # Add psutil import for memory usage logging

import numpy as np
import vizdoom as vzd
import gymnasium as gym
from gymnasium import spaces

from gamingagent.envs.gym_env_adapter import GymEnvAdapter
from gamingagent.modules.core_module import Observation, GameTrajectory

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
        # Initialize base class first
        super().__init__()
        
        self.debug = debug
        if self.debug:
            log_system_info()
            log_memory_usage()
        
        print("[DoomEnvWrapper] Starting initialization...", file=sys.stderr)
        
        # Initialize our attributes
        self.logger = logging.getLogger(__name__)
        self.game_trajectory = []
        self.last_observation = None
        self.button_mapping = {}
        self.current_ammo = 50  # Initialize ammo count
        self.episode_dir = None  # Will be set in reset()
        
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
        
        # Set up run directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create video directory if needed
        if self.record_video and self.video_dir:
            os.makedirs(self.video_dir, exist_ok=True)
        
        # Set up logging
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
        available_buttons = self._cfg.get("available_buttons", [
            "move_left",
            "move_right",
            "attack"
        ])
        print(f"[{time.time()}] Setting available buttons: {available_buttons}", file=sys.stderr)
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
            # Set environment variables for headless mode
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            os.environ["SDL_AUDIODRIVER"] = "dummy"
            
            print(f"[{time.time()}] Creating game instance...", file=sys.stderr)
            self.game = vzd.DoomGame()
            
            print(f"[{time.time()}] Setting basic settings...", file=sys.stderr)
            try:
                # Basic settings - following basic example
                self.game.set_window_visible(False)  # Always false for headless mode
                self.game.set_sound_enabled(False)  # Keep sound disabled as in basic example
                
                # Set scenario path first - exactly as in basic example
                scenario_name = self._cfg.get("doom_scenario_path", "basic.wad")
                scenario_path = os.path.join(vzd.scenarios_path, scenario_name)
                
                if not os.path.exists(scenario_path):
                    raise FileNotFoundError(f"Scenario file not found: {scenario_path}")
                
                self.game.set_doom_scenario_path(scenario_path)
                
                # Set map - exactly as in basic example
                self.game.set_doom_map(self._cfg.get("doom_map", "map01"))
                
                # Screen settings - exactly as in basic example
                screen_res = self._cfg.get("rendering_options", {}).get("screen_resolution", "RES_320X240")
                self.game.set_screen_resolution(getattr(vzd.ScreenResolution, screen_res))
                self.game.set_screen_format(vzd.ScreenFormat.RGB24)
                
                # Enable additional buffers for better state tracking
                self.game.set_depth_buffer_enabled(True)
                self.game.set_labels_buffer_enabled(True)
                self.game.set_automap_buffer_enabled(True)
                self.game.set_objects_info_enabled(True)
                self.game.set_sectors_info_enabled(True)
                
                # Set rendering options
                rendering_options = self._cfg.get("rendering_options", {})
                self.game.set_render_hud(rendering_options.get("render_hud", True))
                self.game.set_render_crosshair(rendering_options.get("render_crosshair", False))
                self.game.set_render_weapon(rendering_options.get("render_weapon", True))
                self.game.set_render_decals(rendering_options.get("render_decals", True))
                self.game.set_render_particles(rendering_options.get("render_particles", True))
                self.game.set_render_effects_sprites(rendering_options.get("render_effects_sprites", True))
                self.game.set_render_messages(rendering_options.get("render_messages", False))
                self.game.set_render_corpses(rendering_options.get("render_corpses", False))
                self.game.set_render_screen_flashes(rendering_options.get("render_screen_flashes", True))
                self.game.set_render_minimal_hud(rendering_options.get("render_minimal_hud", False))
                
                # Set game mode and settings
                self.game.set_mode(vzd.Mode.PLAYER)
                self.game.set_living_reward(self._cfg.get("rewards", {}).get("living_reward", -1))
                self.game.set_doom_skill(self._cfg.get("doom_skill", 3))
                
                # Set episode settings
                episode_settings = self._cfg.get("episode_settings", {})
                self.game.set_episode_start_time(episode_settings.get("episode_start_time", 14))
                self.game.set_episode_timeout(episode_settings.get("episode_timeout", 600))
                self.game.set_ticrate(episode_settings.get("ticrate", 20))
                
                # Set available buttons and game variables
                available_buttons = self._cfg.get("available_buttons", ["move_left", "move_right", "attack"])
                self.game.set_available_buttons([getattr(vzd.Button, btn.upper()) for btn in available_buttons])
                
                # Set available game variables
                self.game.set_available_game_variables([
                    vzd.GameVariable.HEALTH,
                    vzd.GameVariable.AMMO2,  # Using AMMO2 for pistol ammo
                    vzd.GameVariable.POSITION_X,
                    vzd.GameVariable.POSITION_Y,
                    vzd.GameVariable.ANGLE
                ])
                
                # Initialize the game
                print(f"[{time.time()}] Initializing game engine...", file=sys.stderr)
                self.game.init()
                
                # Store button mapping for later use
                self.button_mapping = {
                    "move_left": [True, False, False],
                    "move_right": [False, True, False],
                    "attack": [False, False, True]
                }
                
            except vzd.ViZDoomErrorException as e:
                print(f"[{time.time()}] VizDoom error in settings: {e}", file=sys.stderr)
                raise
            except Exception as e:
                print(f"[{time.time()}] Unexpected error in settings: {e}", file=sys.stderr)
                raise
            
            print(f"[{time.time()}] Game engine initialized successfully", file=sys.stderr)
            
        except Exception as e:
            print(f"[{time.time()}] Failed to initialize game components: {e}", file=sys.stderr)
            raise

    def _buttons_from_str(self, action_str: str) -> List[bool]:
        """Convert action string to button presses.
        
        Args:
            action_str: Action string to convert
            
        Returns:
            List of boolean values representing button presses
        """
        if action_str not in self.button_mapping:
            raise ValueError(f"Unknown action: {action_str}")
        return self.button_mapping[action_str]

    def _extract_game_specific_info(self) -> Dict[str, Any]:
        """Extract game-specific information from the current state."""
        info = {}
        
        # Get game variables
        try:
            game_vars = {
                "ammo2": self.game.get_game_variable(vzd.GameVariable.AMMO2),
                "health": self.game.get_game_variable(vzd.GameVariable.HEALTH),
                "position_x": self.game.get_game_variable(vzd.GameVariable.POSITION_X),
                "position_y": self.game.get_game_variable(vzd.GameVariable.POSITION_Y),
                "angle": self.game.get_game_variable(vzd.GameVariable.ANGLE)
            }
            
            # Map game variables to their values
            for var_name in self._cfg.get("available_game_variables", []):
                if var_name.lower() in game_vars:
                    info[var_name.lower()] = game_vars[var_name.lower()]
            
            # Add episode status
            info['is_episode_finished'] = self.game.is_episode_finished()
            
            # Add timestamp
            info['timestamp'] = datetime.datetime.now().isoformat()
            
        except Exception as e:
            self.logger.error(f"Error extracting game info: {e}")
            info = {
                'ammo2': 0,
                'health': 0,
                'position_x': 0,
                'position_y': 0,
                'angle': 0,
                'is_episode_finished': True,
                'timestamp': datetime.datetime.now().isoformat()
            }
        
        return info

    def _text_repr(self) -> str:
        """Get text representation of current state.
        
        Returns:
            String representation of current state
        """
        try:
            state = self.game.get_state()
            if state is None:
                return "No state available"
                
            # Get game variables
            health = self.game.get_game_variable(vzd.GameVariable.HEALTH)
            ammo = self.game.get_game_variable(vzd.GameVariable.AMMO2)
            pos_x = self.game.get_game_variable(vzd.GameVariable.POSITION_X)
            pos_y = self.game.get_game_variable(vzd.GameVariable.POSITION_Y)
            angle = self.game.get_game_variable(vzd.GameVariable.ANGLE)
            
            # Format state info
            state_info = (
                f"Health: {health:.1f}\n"
                f"Ammo: {ammo:.1f}\n"
                f"Position: ({pos_x:.1f}, {pos_y:.1f})\n"
                f"Angle: {angle:.1f}\n"
                f"Episode Finished: {self.game.is_episode_finished()}\n"
                f"Player Dead: {self.game.is_player_dead()}"
            )
            
            return state_info
            
        except Exception as e:
            self.logger.error(f"[DoomEnvWrapper] Error getting text representation: {e}")
            return "Error getting state information"

    def reset(self, *, seed: int | None = None, episode_id: int = 1, **kwargs) -> Tuple[Observation, Dict[str, Any]]:
        """Reset the environment for a new episode.
        
        Args:
            seed: Random seed
            episode_id: Episode ID
            
        Returns:
            Tuple of (observation, info)
        """
        # Reset base environment
        super().reset(seed=seed)
        self.game.new_episode()
        
        # Reset adapter for new episode
        self.adapter.reset_episode(episode_id)
        
        # Reset ammo count
        self.current_ammo = 50
        
        # Get initial state
        state = self._get_game_state()
        
        # Capture initial frame
        frame_path = self._capture_frame()
        if not frame_path:
            self.logger.error("[DoomEnvWrapper] Failed to capture initial frame")
            frame_path = os.path.join(self.adapter.agent_observations_dir, "initial_frame.png")
            # Create a blank frame if capture fails
            blank_frame = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.imwrite(frame_path, blank_frame)
        
        # Initialize game trajectory
        self.game_trajectory = GameTrajectory(max_length=100)
        
        # Create initial observation
        observation = Observation(
            img_path=frame_path,
            game_trajectory=self.game_trajectory,
            reflection=None,
            processed_visual_description=None,
            textual_representation=None
        )
        
        # Store initial observation
        self.last_observation = observation
        
        return observation, state

    def _capture_frame(self) -> str:
        """Capture current frame and save it.
        
        Returns:
            Path to saved frame image
        """
        try:
            state = self.game.get_state()
            if state is None:
                return ""
                
            # Get screen buffer
            screen = state.screen_buffer
            if screen is None:
                return ""
                
            # Convert from BGR to RGB
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
                
            # Generate unique filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            frame_path = os.path.join(self.adapter.agent_observations_dir, f"step_{timestamp}.png")
            
            # Save frame
            cv2.imwrite(frame_path, screen)
            return frame_path
            
        except Exception as e:
            self.logger.error(f"[DoomEnvWrapper] Error capturing frame: {e}")
            return ""

    def step(
        self,
        agent_action_str: Optional[str],
        thought_process: str = "",
        time_taken_s: float = 0.0,
        use_random_action: bool = False
    ) -> Tuple[Observation, float, bool, bool, Dict[str, Any], float]:
        """Execute one step in the environment.
        
        Args:
            agent_action_str: Action string from agent
            thought_process: Agent's thought process
            time_taken_s: Time taken by agent to decide
            use_random_action: Whether to use random action
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info, performance_score)
        """
        self._validate_game_state()
        self.adapter.increment_step()

        # Get current state before action
        prev_info = self._get_game_state()
        prev_ammo = prev_info.get('ammo2', 0)

        # Convert action string to button list
        if use_random_action:
            action_str = random.choice(self._cfg.get("available_buttons", ["move_left", "move_right", "attack"]))
        else:
            action_str = agent_action_str

        buttons = self._buttons_from_str(action_str)
        self.logger.info(f"[DoomEnvWrapper] Executing action '{action_str}' with buttons: {buttons}")

        # Get the game's ticrate from config
        ticrate = self._cfg.get("episode_settings", {}).get("ticrate", 20)
        frame_time = 1.0 / ticrate

        # Maximum attempts to get state changes
        max_attempts = 3
        state_changed = False
        reward = 0
        current_info = None

        for attempt in range(max_attempts):
            # Execute action with multiple tics to ensure state updates
            reward = self.game.make_action(buttons, tics=8)  # Advance 8 tics per action
            
            # Longer delay to allow action to process
            time.sleep(frame_time * 2)  # Wait 2 frames worth of time
            
            # Get new state
            state = self.game.get_state()
            if state is None:
                self.logger.error(f"Failed to get game state after action (attempt {attempt + 1}/{max_attempts})")
                continue
                
            # Update current state
            current_info = self._get_game_state()
            
            # Verify state changed
            if prev_info:
                changes = []
                for key in ['ammo2', 'health', 'position_x', 'position_y', 'angle']:
                    if key in prev_info and key in current_info:
                        if prev_info[key] != current_info[key]:
                            changes.append(f"{key}: {prev_info[key]} -> {current_info[key]}")
                if changes:
                    self.logger.info(f"State changes detected: {', '.join(changes)}")
                    state_changed = True
                    break
                else:
                    self.logger.warning(f"No state changes detected after action (attempt {attempt + 1}/{max_attempts})")
                    time.sleep(frame_time * 2)  # Wait longer before next attempt

        if not state_changed:
            self.logger.error("Failed to detect state changes after maximum attempts")
            current_info = prev_info  # Use previous state if no changes detected

        # Capture frame after state update
        frame_path = self._capture_frame()

        # Update game trajectory
        ts = datetime.datetime.now().isoformat(timespec="seconds")
        trajectory_entry = (
            f"##Turn Hash\n[{ts}]\n"
            f"###Obs\n{self._text_repr()}\n"
            f"###Thought\n{thought_process}\n"
            f"###Action\n{action_str}\n"
        )
        self.game_trajectory.add(trajectory_entry)

        # Format thought process for observation
        formatted_thought = (
            f"Current State:\n"
            f"- Health: {current_info.get('health', 0)}\n"
            f"- Ammo: {current_info.get('ammo2', 0)}\n"
            f"- Position: ({current_info.get('position_x', 0)}, {current_info.get('position_y', 0)})\n"
            f"- Angle: {current_info.get('angle', 0)}\n\n"
            f"Action Analysis:\n"
            f"- Action taken: {action_str}\n"
            f"- Reward received: {reward}\n"
            f"- State changes: {', '.join([f'{k}: {v}' for k, v in current_info.items() if k in ['health', 'ammo2', 'position_x', 'position_y', 'angle']])}\n\n"
            f"Combat Strategy:\n{thought_process}"
        )

        # Create observation
        observation = Observation(
            img_path=frame_path,
            game_trajectory=self.game_trajectory,
            reflection=formatted_thought,
            processed_visual_description=self._text_repr(),
            textual_representation=None
        )

        # Store current observation for next step
        self.last_observation = observation

        # Check if episode is done
        is_episode_finished = self.game.is_episode_finished()

        # Set the performance score as the reward
        performance_score = reward

        # Log step data using adapter
        self.adapter.log_step_data(
            agent_action_str=action_str,
            thought_process=formatted_thought,
            reward=reward,
            info=current_info,
            terminated=is_episode_finished,
            truncated=False,
            time_taken_s=time_taken_s,
            perf_score=performance_score,
            agent_observation=observation
        )

        return observation, reward, is_episode_finished, False, current_info, performance_score

    def render(self) -> None:
        """Render the game.
        
        This method renders the game by getting the current game state and displaying it.
        For human viewing, it uses pygame to display the screen buffer.
        """
        if not self.game:
            return
            
        state = self.game.get_state()
        if state is None:
            return
            
        # Get screen buffer
        screen = state.screen_buffer
        if screen is None:
            return
            
        # Convert from BGR to RGB
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
        
        # For human viewing, display using pygame
        if self.render_mode == "human":
            import pygame
            
            # Initialize pygame if needed
            if not hasattr(self, 'window_surface'):
                pygame.init()
                pygame.display.set_caption("ViZDoom")
                self.window_surface = pygame.display.set_mode(screen.shape[:2][::-1])
                self.clock = pygame.time.Clock()
            
            # Convert screen buffer to pygame surface and display
            surf = pygame.surfarray.make_surface(screen.transpose(1, 0, 2))
            self.window_surface.blit(surf, (0, 0))
            pygame.display.update()
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])

    def close(self) -> None:
        """Clean up resources.
        
        This method closes the game instance and cleans up any resources.
        """
        if hasattr(self, 'game'):
            self.game.close()
        if hasattr(self, 'adapter'):
            self.adapter.close_log_file()

    def _get_game_state(self) -> Dict[str, Any]:
        """Get current game state information.
        
        Returns:
            Dictionary containing game state information
        """
        try:
            state = self.game.get_state()
            if state is None:
                self.logger.error("[DoomEnvWrapper] Failed to get game state")
                return {}
                
            # Create state dictionary with proper variable mapping
            state_info = {
                "health": self.game.get_game_variable(vzd.GameVariable.HEALTH),
                "ammo2": self.game.get_game_variable(vzd.GameVariable.AMMO2),
                "position_x": self.game.get_game_variable(vzd.GameVariable.POSITION_X),
                "position_y": self.game.get_game_variable(vzd.GameVariable.POSITION_Y),
                "angle": self.game.get_game_variable(vzd.GameVariable.ANGLE),
                "is_episode_finished": self.game.is_episode_finished(),
                "is_player_dead": self.game.is_player_dead()
            }
            
            return state_info
            
        except Exception as e:
            self.logger.error(f"[DoomEnvWrapper] Error getting game state: {e}")
            return {}