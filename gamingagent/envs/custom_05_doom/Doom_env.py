from __future__ import annotations

import os, json
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from vizdoom import DoomGame, Mode, ScreenResolution, ScreenFormat, gymnasium_wrapper, Button, GameVariable
from gamingagent.envs.gym_env_adapter import GymEnvAdapter
from gamingagent.modules.core_module import Observation
import gymnasium as gym


__all__ = ["DoomEnvWrapper"]

class DoomEnvWrapper:
    _DEFAULT_ENV_ID = "Doom-Snes"  # Default environment ID for Doom

    def __init__(
        self,
        game_name: str, # e.g., "super_mario_bros"
        config_dir_path: str = "gamingagent/envs/custom_05_doom",
        observation_mode: str = "vision",
        base_log_dir: str = "cache/doom/default_run",
        render_mode_human: bool = False,
    ) -> None:
        
        super().__init__()

        # ── load JSON wrapper config ──
        cfg_file = os.path.join(config_dir_path, "game_env_config.json")
        if os.path.isfile(cfg_file):
            with open(cfg_file, "r") as f:
                _cfg = json.load(f)
        else:
            _cfg = {}
        
        self.game_name = game_name
        self.env_id: str = _cfg.get("env_id", self._DEFAULT_ENV_ID)
        self.env_init_kwargs: Dict[str, Any] = _cfg.get("env_init_kwargs", {})
        self.base_log_dir = base_log_dir
        self.render_mode_human = _cfg.get("render_mode_human", False)
        
        # Load rendering options
        self.rendering_options = _cfg.get("rendering_options", {})
        self.episode_settings = _cfg.get("episode_settings", {})

        # Adapter for logging and observation handling
        self.adapter = GymEnvAdapter(
            game_name="Doom",
            observation_mode=observation_mode,
            agent_cache_dir=self.base_log_dir,
            game_specific_config_path=cfg_file,
        )

        # DoomGame instance
        self._game: Optional[DoomGame] = None
        self.current_frame: Optional[np.ndarray] = None
        self.current_info: Dict[str, Any] = {}

    def _initialize_env(self) -> None:
        """
        Initialize the Doom environment using vizdoom.
        """
        self._game = DoomGame()
        
        # Set screen resolution and format from config
        self._game.set_screen_resolution(ScreenResolution.RES_320X240)
        self._game.set_screen_format(ScreenFormat.RGB24)
        
        # Set rendering options from config
        self._game.set_window_visible(self.rendering_options.get("window_visible", False))
        self._game.set_render_hud(self.rendering_options.get("render_hud", True))
        self._game.set_render_crosshair(self.rendering_options.get("render_crosshair", False))
        self._game.set_render_weapon(self.rendering_options.get("render_weapon", True))
        self._game.set_render_decals(self.rendering_options.get("render_decals", False))
        self._game.set_render_particles(self.rendering_options.get("render_particles", False))
        
        # Set game mode and settings
        self._game.set_mode(Mode.PLAYER)
        self._game.set_living_reward(self.rendering_options.get("living_reward", 1))
        self._game.set_doom_skill(self.episode_settings.get("doom_skill", 5))
        
        # Set episode settings
        self._game.set_episode_start_time(self.episode_settings.get("episode_start_time", 14))
        self._game.set_episode_timeout(self.episode_settings.get("episode_timeout", 300))
        
        # Set up available buttons
        self._game.set_available_buttons([
            Button.MOVE_UP,
            Button.MOVE_DOWN,
            Button.MOVE_LEFT,
            Button.MOVE_RIGHT,
            Button.ATTACK
        ])
        
        # Set up game variables to track
        self._game.set_available_game_variables([
            GameVariable.HEALTH,
            GameVariable.AMMO2,
            GameVariable.KILLCOUNT
        ])
        
        self._game.init()

    def reset(self, *, seed: int | None = None, episode_id: int = 1, **kwargs) -> Observation:
        """
        Reset the environment and return the initial observation.
        """
        # Initialize the environment if not already done
        self._initialize_env()

        # Reset the adapter for the new episode
        self.adapter.reset_episode(episode_id)

        # Start a new episode in the Doom game
        self._game.new_episode()

        # Get the initial frame and game-specific info
        state = self._game.get_state()
        if state and state.screen_buffer is not None:
            # The screen buffer should be in RGB24 format (240, 320, 3)
            self.current_frame = state.screen_buffer
            if self.current_frame.shape != (240, 320, 3):
                print(f"[DoomEnvWrapper] Warning: Unexpected screen buffer shape: {self.current_frame.shape}")
                self.current_frame = None
        else:
            self.current_frame = None
        self.current_info = self._extract_game_specific_info()

        # Handle observation modes and save the initial frame if needed
        img_path = None
        if self.adapter.observation_mode in ("vision", "both") and self.current_frame is not None:
            try:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(self.adapter._create_agent_observation_path(episode_id, 0)), exist_ok=True)
                # Ensure the frame is in the correct format for saving
                if self.current_frame.dtype != np.uint8:
                    self.current_frame = self.current_frame.astype(np.uint8)
                img_path = self.adapter.save_frame_and_get_path(self.current_frame)
            except Exception as e:
                print(f"[DoomEnvWrapper] Error saving frame: {e}")

        # Create the initial observation for the agent
        obs = self.adapter.create_agent_observation(
            img_path=img_path, text_representation=self._text_repr()
        )

        return obs, self.current_info.copy()

    def step(self, agent_action_str: Optional[str]) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """
        Execute an action in the environment and return the next observation, reward, done flag, and info.
        """
        if self._game is None:
            raise RuntimeError("Call reset() before step()")

        # Map the action string to button presses
        act = self._buttons_from_str(agent_action_str)

        # Step the environment
        reward = self._game.make_action(act)
        done = self._game.is_episode_finished()

        if not done:
            # Update the current frame and game-specific info
            state = self._game.get_state()
            if state and state.screen_buffer is not None:
                # The screen buffer should be in RGB24 format (240, 320, 3)
                self.current_frame = state.screen_buffer
                if self.current_frame.shape != (240, 320, 3):
                    print(f"[DoomEnvWrapper] Warning: Unexpected screen buffer shape: {self.current_frame.shape}")
                    self.current_frame = None
            else:
                self.current_frame = None
            self.current_info = self._extract_game_specific_info()
        else:
            # Clear the frame and info if the episode is finished
            self.current_frame = None
            self.current_info = {}

        # Adjust rewards based on config
        if reward == 106:  # Kill monster reward
            reward += 106
        elif reward == -5:  # Shot penalty
            reward -= 5

        # Handle observation modes and save the frame if needed
        img_path = None
        if self.adapter.observation_mode in ("vision", "both") and self.current_frame is not None:
            try:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(self.adapter._create_agent_observation_path(self.adapter.current_episode_id, self.adapter.current_step_num)), exist_ok=True)
                # Ensure the frame is in the correct format for saving
                if self.current_frame.dtype != np.uint8:
                    self.current_frame = self.current_frame.astype(np.uint8)
                img_path = self.adapter.save_frame_and_get_path(self.current_frame)
            except Exception as e:
                print(f"[DoomEnvWrapper] Error saving frame: {e}")

        # Create the observation for the agent
        observation = self.adapter.create_agent_observation(
            img_path=img_path, text_representation=self._text_repr()
        )

        return observation, reward, done, self.current_info.copy()

    # ───────────────────── info / other helpers ─────────────────────
    def _buttons_from_str(self, action_str: Optional[str]) -> List[int]:
        """
        Map an action string (e.g., "move_left") to a button vector for vizdoom.
        """
        buttons = [0] * self._game.get_available_buttons_size()
        if action_str:
            # Map string actions to Button enums
            action_to_button = {
                "move_up": Button.MOVE_UP,
                "move_down": Button.MOVE_DOWN,
                "move_left": Button.MOVE_LEFT,
                "move_right": Button.MOVE_RIGHT,
                "attack": Button.ATTACK
            }
            
            # Get the Button enum for the action
            button = action_to_button.get(action_str.lower())
            if button is None:
                raise ValueError(f"Invalid action: {action_str}. Valid actions are: {list(action_to_button.keys())}")
            
            # Set the corresponding button to 1
            buttons[self._game.get_available_buttons().index(button)] = 1
            
        return buttons

    def _extract_game_specific_info(self) -> Dict[str, Any]:
        """
        Extract game-specific information from the current state.
        """
        state = self._game.get_state()
        if state is None or state.game_variables is None:
            return {}

        # Get game variables in the order they were set up
        health = state.game_variables[0] if len(state.game_variables) > 0 else None
        ammo = state.game_variables[1] if len(state.game_variables) > 1 else None
        kills = state.game_variables[2] if len(state.game_variables) > 2 else None

        return {
            "health": health,
            "ammo": ammo,
            "kills": kills,
        }

    def _text_repr(self) -> str:
        """
        Generate a textual representation of the current game state.
        """
        if not self.current_info:
            return "No game state available."

        health = self.current_info.get('health', 'N/A')
        ammo = self.current_info.get('ammo', 'N/A')
        kills = self.current_info.get('kills', 'N/A')

        return f"Health: {health}, Ammo: {ammo}, Kills: {kills}"

    def render(self) -> None:
        """
        Render the game for human viewing.
        """
        if self._game:
            self._game.render()

    def close(self) -> None:
        """
        Close the environment and clean up resources.
        """
        if self._game:
            self._game.close()