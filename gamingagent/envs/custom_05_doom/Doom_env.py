from __future__ import annotations

import os, json
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from vizdoom import DoomGame, Mode, ScreenResolution
from gamingagent.envs.gym_env_adapter import GymEnvAdapter
from gamingagent.modules.core_module import Observation

__all__ = ["DoomEnvWrapper"]

class DoomEnvWrapper:
    def __init__(
        self,
        game_config_path: str = "configs/custom_05_doom/doom_config.cfg",
        observation_mode: str = "vision",
        base_log_dir: str = "cache/doom/default_run",
        render_mode_human: bool = False,
    ) -> None:
        """
        Initialize the Doom environment wrapper.
        """
        self.game_config_path = game_config_path
        self.observation_mode = observation_mode
        self.base_log_dir = base_log_dir
        self.render_mode_human = render_mode_human

        # Adapter for logging and observation handling
        self.adapter = GymEnvAdapter(
            game_name="Doom",
            observation_mode=self.observation_mode,
            agent_cache_dir=self.base_log_dir,
            game_specific_config_path=os.path.join(os.path.dirname(self.game_config_path), "module_prompts.json"),
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
        self._game.set_screen_resolution(ScreenResolution.RES_640X480)
        self._game.set_window_visible(self.render_mode_human)
        self._game.set_mode(Mode.PLAYER)
        self._game.init()

    def reset(self) -> Observation:
        """
        Reset the environment and return the initial observation.
        """
        if self._game is None:
            self._initialize_env()

        self._game.new_episode()
        self.current_frame = self._game.get_state().screen_buffer
        self.current_info = self._extract_game_specific_info()
        return self.adapter.reset(self.current_frame, self.current_info)

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
            self.current_frame = self._game.get_state().screen_buffer
            self.current_info = self._extract_game_specific_info()
        else:
            self.current_frame = None
            self.current_info = {}

        # Create the observation
        observation = self.adapter.step(
            self.current_frame, self.current_info, agent_action_str, reward
        )

        return observation, reward, done, self.current_info

    def _buttons_from_str(self, action_str: Optional[str]) -> List[int]:
        """
        Map an action string (e.g., "MOVE_FORWARD||TURN_LEFT") to a button vector for vizdoom.
        """
        buttons = [0] * self._game.get_available_buttons_size()
        if action_str:
            for action in action_str.split("||"):
                if action in self._game.get_available_buttons():
                    buttons[self._game.get_available_buttons().index(action)] = 1
        return buttons

    def _extract_game_specific_info(self) -> Dict[str, Any]:
        """
        Extract game-specific information from the current state.
        """
        state = self._game.get_state()
        if state is None:
            return {}

        return {
            "health": state.game_variables[0],
            "ammo": state.game_variables[1],
            "kills": state.game_variables[2],
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