from __future__ import annotations

import os, json
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from vizdoom import DoomGame, Mode, ScreenResolution, gymnasium_wrapper
from gamingagent.envs.gym_env_adapter import GymEnvAdapter
from gamingagent.modules.core_module import Observation
import gymnasium as gym


__all__ = ["DoomEnvWrapper"]

class DoomEnvWrapper:
    def __init__(
        self,
        game_name: str, # e.g., "super_mario_bros"
        game_config_path: str = "configs/custom_05_doom/config.yaml",
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
        self.game_config_path = game_config_path
        self.base_log_dir = base_log_dir
        self.render_mode_human = _cfg.get("render_mode_human", False)

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
        self._game.load_config(self.game_config_path)
        self._game.set_screen_resolution(ScreenResolution.RES_320X240)
        self._game.set_window_visible(self.render_mode_human)
        self._game.set_mode(Mode.PLAYER)
        self._game.set_living_reward(1)  # Matches living_reward in config
        self._game.set_doom_skill(5)  # Matches doom_skill in config
        self._game.init()
        
        
    # ───────────────────── Gym API ──────────────────────
    def reset(self, *, seed: int | None = None, episode_id: int = 1, **kwargs) -> Observation:
        """
        Reset the environment and return the initial observation.
        """
        # Initialize the environment if not already done
        self._initialize_env()

        # Reset the adapter for the new episode
        self.adapter.reset_episode(episode_id)

        # Reset the raw environment and extract the initial frame and info
        self.current_frame, _ = self._game.reset(seed=seed)
        self.current_info = self._extract_info()

        # Start a new episode in the Doom game
        self._game.new_episode()

        # Handle observation modes and save the initial frame if needed
        img_path = None
        if self.adapter.observation_mode in ("vision", "both"):
            img_path = self.adapter.save_frame_and_get_path(self.current_frame)

        # Create the initial observation for the agent
        obs = self.adapter.create_agent_observation(
            img_path=img_path, text_representation=self._text_repr()
        )

        # Update the current frame and game-specific info
        self.current_frame = self._game.get_state().screen_buffer
        self.current_info = self._extract_game_specific_info()

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
            self.current_frame = state.screen_buffer if state else None
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
            img_path = self.adapter.save_frame_and_get_path(self.current_frame)

        # Create the observation for the agent
        observation = self.adapter.create_agent_observation(
            img_path=img_path, text_representation=self._text_repr()
        )

        return observation, reward, done, self.current_info.copy()

    # ───────────────────── info / other helpers ─────────────────────
    def _buttons_from_str(self, action_str: Optional[str]) -> List[int]:
        """
        Map an action string (e.g., "MOVE_LEFT||MOVE_RIGHT||MOVE_UP||MOVE_DOWN||ATTACK") to a button vector for vizdoom.
        """
        buttons = [0] * self._game.get_available_buttons_size()
        if action_str:
            for action in action_str.split("||"):
                if action in self._game.get_available_buttons():
                    buttons[self._game.get_available_buttons().index(action)] = 1
                else:
                    raise ValueError(f"Invalid action: {action} is not in available buttons: {self._game.get_available_buttons()}")
        return buttons

    def _extract_game_specific_info(self) -> Dict[str, Any]:
        """
        Extract game-specific information from the current state.
        """
        state = self._game.get_state()
        if state is None or state.game_variables is None:
            return {}

        return {
            "health": state.game_variables[0] if len(state.game_variables) > 0 else None,
            "ammo": state.game_variables[1] if len(state.game_variables) > 1 else None,
            "kills": state.game_variables[2] if len(state.game_variables) > 2 else None,
        }

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