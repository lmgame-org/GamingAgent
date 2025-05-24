import os
import json
import hashlib
from typing import Any, Dict, Tuple, Optional

import gymnasium as gym
import numpy as np
from PIL import Image

from gamingagent.modules.core_module import Observation

# Retro is required to emulate the original NES "1942" ROM
import retro

class NineteenFortyTwoEnvWrapper:
    """Environment wrapper for Capcom's classic shooter **1942 (NES)**.

    This class is *self‑contained*
    It borrows the ergonomic conveniences of `TwentyFortyEightEnvWrapper`:
    * episode / step logging
    * optional vision or text observations (or both)
    * action‑string → environment‑action mapping with graceful *skip* handling
    * stuck‑state detection (terminates if the video frame is unchanged for a
      configurable number of steps)

    Parameters
    ----------
    game_name : str
        Identifier used only for logging / directory structure (e.g. "1942").
    observation_mode : str
        One of {"vision", "text", "both"}. Determines what the agent receives.
    agent_observations_base_dir : str
        Where per‑step screenshots are written (if vision mode is enabled).
    env_type : str, default "retro"
        Currently only "retro" is supported.
    render_mode : str, default "rgb_array"
        Passed to `retro.make`. Use "human" to see the live window.
    log_root_dir : str, default "runs_output"
        Root directory for episode‑level JSONL logs.
    config_path : str | None, default None
        Optional JSON config that can override the default action mapping &
        termination parameters. Same shape as the 2048 wrapper's config:
        {
            "env_id": "1942-Nes",
            "env_init_kwargs": { ... },
            "action_mapping": { "left": 0, "right": 1, ... },
            "max_unchanged_steps_for_termination": 50
        }
    """

    # A conservative default – works for the standard *1942 (NES)* Retro bundle
    _DEFAULT_RETRO_ENV_ID = "1942-Nes"
    _SKIP_ACTION_IDX = -1  # Only used in logs – no env action at this index

    def __init__(
        self,
        game_name: str,
        observation_mode: str,
        agent_observations_base_dir: str,
        env_type: str = "retro",
        render_mode: str = "rgb_array",
        log_root_dir: str = "runs_output",
        config_path: Optional[str] = None,
    ):
        # Public bookkeeping
        self.game_name = game_name
        self.observation_mode = observation_mode.lower()
        self.env_type = env_type.lower()
        self.render_mode = render_mode

        # Paths
        self.agent_observations_base_dir = agent_observations_base_dir
        os.makedirs(self.agent_observations_base_dir, exist_ok=True)
        self.log_root_dir = log_root_dir
        os.makedirs(self.log_root_dir, exist_ok=True)

        # Episode‑level state
        self.current_episode_id: int = 0
        self.current_step_num: int = 0
        self.episode_log_file_path: Optional[str] = None
        self.episode_log_file_handle = None

        # Underlying env handle & cached results
        self.env: Optional[gym.Env] = None
        self.current_raw_observation: Any = None  # np.ndarray (H×W×3)
        self.current_info: Dict[str, Any] = {}

        # Action mapping -> default is NES controller (A,B,SELECT,START,UP,DOWN,LEFT,RIGHT)
        # The agent provides *strings*; this dict maps them → *indices* in Retro's binary list.
        # "A" --> dodge
        # "B" --> shoot
        self.action_mapping: Dict[str, int] = {
            "a": 0,
            "b": 1,
            "up": 2,
            "down": 3,
            "left": 4,
            "right": 5,
            "select": 6,
            "start": 7,
        }
        self._max_unchanged_steps: int = 200  # ~5s at 60 fps
        self.env_id: str = self._DEFAULT_RETRO_ENV_ID
        self.env_init_kwargs: Dict[str, Any] = {}

        # Allow external JSON config overrides (mirrors 2048 wrapper semantics)
        if config_path and os.path.exists(config_path):
            self._load_json_config(config_path)

        # ─── Stuck‑state detection ─────────────────────────────────────────────
        self._last_frame_hash: Optional[str] = None
        self._unchanged_frame_count: int = 0

    # ────────────────────────────────────────────────────────────────────────────
    #  SET CONFIG
    # ────────────────────────────────────────────────────────────────────────────

    def _load_json_config(self, path: str) -> None:
        try:
            with open(path, "r") as f:
                cfg = json.load(f)
        except json.JSONDecodeError as e:
            print(f"[NineteenFortyTwoEnvWrapper] JSON decode error for {path}: {e}; using defaults.")
            return
        except Exception as e:
            print(f"[NineteenFortyTwoEnvWrapper] Could not read config {path}: {e}; using defaults.")
            return

        self.env_id = cfg.get("env_id", self.env_id)
        self.env_init_kwargs = cfg.get("env_init_kwargs", self.env_init_kwargs)

        # Action mapping override – values must be *int*
        if "action_mapping" in cfg:
            raw_map = cfg["action_mapping"]
            if isinstance(raw_map, dict):
                try:
                    self.action_mapping = {str(k).lower(): int(v) for k, v in raw_map.items()}
                    print(
                        f"[NineteenFortyTwoEnvWrapper] Loaded custom action mapping from {path}."
                    )
                except ValueError:
                    print(
                        f"[NineteenFortyTwoEnvWrapper] Invalid action_mapping values in {path}; must be int."
                    )
        self._max_unchanged_steps = int(
            cfg.get("max_unchanged_steps_for_termination", self._max_unchanged_steps)
        )

    def _initialize_env(self) -> None:
        if self.env_type != "retro":
            raise ValueError(
                "NineteenFortyTwoEnvWrapper currently supports only env_type='retro'."
            )
        print(
            f"[NineteenFortyTwoEnvWrapper] Creating Retro env: id='{self.env_id}', render_mode='{self.render_mode}', kwargs={self.env_init_kwargs}"
        )
        self.env = retro.make(
            game=self.env_id, render_mode=self.render_mode, **self.env_init_kwargs
        )

    # ────────────────────────────────────────────────────────────────────────────
    # ❷  EPISODE LIFECYCLE
    # ────────────────────────────────────────────────────────────────────────────

    def reset(
        self, seed: Optional[int] = None, episode_id: int = 1
    ) -> Tuple[Observation, Dict[str, Any]]:
        if self.env is None:
            self._initialize_env()

        self.current_raw_observation, self.current_info = self.env.reset(seed=seed)
        self.current_episode_id = episode_id
        self.current_step_num = 0

        # Stuck‑state counters
        self._last_frame_hash = None
        self._unchanged_frame_count = 0

        # (Re)open episode log
        if self.episode_log_file_handle:
            self.episode_log_file_handle.close()
        self.episode_log_file_path = os.path.join(
            self.log_root_dir, f"episode_{episode_id:03d}_log.jsonl"
        )
        self.episode_log_file_handle = open(self.episode_log_file_path, "a")

        return self._extract_observation(self.current_raw_observation, self.current_info), self.current_info

    # ────────────────────────────────────────────────────────────────────────────
    # ❸  OBSERVATION HELPERS
    # ────────────────────────────────────────────────────────────────────────────

    def _create_agent_observation_path(self, step_num: int) -> str:
        """Generates a path like `obs_e001_s0001.png`."""
        return os.path.join(
            self.agent_observations_base_dir,
            f"env_obs_e{self.current_episode_id:03d}_s{step_num:04d}.png",
        )

    def _save_frame_image(self, frame: np.ndarray, save_path: str) -> None:
        try:
            Image.fromarray(frame).save(save_path)
        except Exception as e:
            print(f"[NineteenFortyTwoEnvWrapper] Could not save frame to {save_path}: {e}")

    def _extract_observation(
        self, raw_observation: Any, info: Dict[str, Any]
    ) -> Observation:
        """Convert Retro's RGB frame & info dict → Observation."""

        img_path = None
        text_repr = None

        # Vision: save screenshot
        if self.observation_mode in {"vision", "both"}:
            img_path = self._create_agent_observation_path(self.current_step_num + 1)
            self._save_frame_image(raw_observation, img_path)

        # Text: simple key metadata (score & lives). Extend as needed.
        if self.observation_mode in {"text", "both"}:
            score = info.get("score", 0)
            lives = info.get("lives", 0)
            level = info.get("level", 0)
            text_repr = f"score:{score}, lives:{lives}, level:{level}"

        return Observation(img_path=img_path, textual_representation=text_repr)

    # ────────────────────────────────────────────────────────────────────────────
    # ❹  STEP & TERMINATION LOGIC
    # ────────────────────────────────────────────────────────────────────────────

    def _agent_action_to_env_action(self, action_str: Optional[str]):
        """Maps an **agent‑provided string** → actual Retro button vector.

        The NES controller in Retro is an 8‑element binary vector (A, B, Select,
        Start, Up, Down, Left, Right). We create a *one‑hot* action for single
        button presses; invalid / "skip" strings map to *all‑zeros* (no‑op).
        """

        if action_str is None or not action_str.strip():
            return [0] * 8  # Skip / no‑op

        idx = self.action_mapping.get(action_str.lower())
        if idx is None:
            print(
                f"[NineteenFortyTwoEnvWrapper] Unknown action '{action_str}'. Executing NO‑OP."
            )
            return [0] * 8  # Unknown → no‑op

        vec = [0] * 8
        vec[idx] = 1
        return vec

    def step(
        self,
        agent_action_str: Optional[str],
        thought_process: str = "",
        time_taken_s: float = 0.0,
    ) -> Tuple[Observation, float, bool, bool, Dict[str, Any]]:
        if self.env is None:
            raise RuntimeError("Environment not initialised – call reset() first.")

        self.current_step_num += 1
        env_action = self._agent_action_to_env_action(agent_action_str)

        (
            self.current_raw_observation,
            reward,
            terminated,
            truncated,
            self.current_info,
        ) = self.env.step(env_action)

        obs = self._extract_observation(self.current_raw_observation, self.current_info)
        terminated, truncated = self._verify_termination(obs, terminated, truncated)
        self._log_step(agent_action_str, thought_process, reward, terminated, truncated, time_taken_s)

        return obs, float(reward), terminated, truncated, self.current_info

    # ────────────────────────────────────────────────────────────────────────────
    # ❺  STUCK‑DETECTION / TERMINATION OVERRIDE
    # ────────────────────────────────────────────────────────────────────────────

    def _verify_termination(
        self, observation: Observation, terminated: bool, truncated: bool
    ) -> Tuple[bool, bool]:
        """Terminate if the *visual frame* is unchanged for `_max_unchanged_steps`."""

        if terminated or truncated:
            return terminated, truncated  # Honour env's own signals first

        # Hash current RGB frame bytes for cheap comparison
        cur_hash = hashlib.md5(self.current_raw_observation.tobytes()).hexdigest()

        if cur_hash == self._last_frame_hash:
            self._unchanged_frame_count += 1
        else:
            self._unchanged_frame_count = 0
        self._last_frame_hash = cur_hash

        if self._unchanged_frame_count >= self._max_unchanged_steps:
            print(
                f"[NineteenFortyTwoEnvWrapper] Terminating – frame unchanged for {self._max_unchanged_steps} steps."
            )
            terminated = True

        return terminated, truncated

    # ────────────────────────────────────────────────────────────────────────────
    # ❻  LOGGING HELPERS
    # ────────────────────────────────────────────────────────────────────────────

    def _log_step(
        self,
        agent_action_str: Optional[str],
        thought: str,
        reward: float,
        terminated: bool,
        truncated: bool,
        time_taken_s: float,
    ) -> None:
        if not self.episode_log_file_handle:
            return
        entry = {
            "episode_id": self.current_episode_id,
            "step": self.current_step_num,
            "agent_action": agent_action_str,
            "thought": thought,
            "reward": float(reward),
            "terminated": terminated,
            "truncated": truncated,
            "time_taken_s": time_taken_s,
        }
        try:
            self.episode_log_file_handle.write(json.dumps(entry) + "\n")
            self.episode_log_file_handle.flush()
        except Exception as e:
            print(f"[NineteenFortyTwoEnvWrapper] Could not write log entry: {e}")

    # ────────────────────────────────────────────────────────────────────────────
    # ❼  CLEAN‑UP
    # ────────────────────────────────────────────────────────────────────────────

    def render_human(self) -> None:
        """Show live window if `render_mode` == 'human'."""
        if self.env and self.render_mode == "human":
            self.env.render()

    def close(self) -> None:
        if self.env:
            self.env.close()
            self.env = None
        if self.episode_log_file_handle:
            try:
                self.episode_log_file_handle.close()
            except Exception:
                pass
            self.episode_log_file_handle = None
