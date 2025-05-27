from __future__ import annotations

import os, json, hashlib
from typing import Any, Dict, List, Tuple, Optional

import retro
import numpy as np
from PIL import Image
import gymnasium as gym

from gamingagent.envs.gym_env_adapter import GymEnvAdapter
from gamingagent.modules.core_module import Observation

__all__ = ["NineteenFortyTwoEnvWrapper"]

# helper functions
def _save_frame(img: np.ndarray, path: str):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        Image.fromarray(img).save(path)
    except Exception as e:
        print(f"[1942] Warning: could not save {path}: {e}")

# main wrapper
class NineteenFortyTwoEnvWrapper(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    _DEFAULT_ENV_ID = "1942-Nes"
    _DEFAULT_ACTION_MAP = {
        "a": 0,   # loop‑the‑loop / dodge
        "b": 1,   # shoot
        "up": 2,
        "down": 3,
        "left": 4,
        "right": 5,
    }

    def __init__(
        self,
        game_name: str, # e.g., "super_mario_bros"
        config_dir_path: str = "gamingagent/envs/retro_03_1942",
        observation_mode: str = "vision",
        base_log_dir: str = "cache/nineteen_forty_two/default_run",
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
        self.action_map: Dict[str, int] = {
            **self._DEFAULT_ACTION_MAP,
            **{k.lower(): int(v) for k, v in _cfg.get("action_mapping", {}).items()},
        }
        self._max_stuck_steps = _cfg.get("max_unchanged_steps_for_termination", 200)
        self.render_mode_human = _cfg.get("render_mode_human", False)

        self.base_log_dir = base_log_dir

        # ── adapter ──
        self.adapter = GymEnvAdapter(
            game_name=self.game_name,
            observation_mode=observation_mode,
            agent_cache_dir=self.base_log_dir,
        )

        # ── retro env created lazily ──
        self._raw_env: Optional[retro.Retro] = None
        self.current_frame: Optional[np.ndarray] = None
        self.current_info: Dict[str, Any] = {}

        # ── stuck‑frame vars ──
        self._last_hash: Optional[str] = None
        self._unchanged = 0

    def _initialize_env(self):
        if self._raw_env is None:
            record_dir = os.path.join(self.adapter.agent_cache_dir, "bk2_recordings")
            os.makedirs(record_dir, exist_ok=True)
            self._raw_env = retro.make(
                game=self.env_id,
                render_mode="human" if self.render_mode_human else None,
                record=record_dir,
                **self.env_init_kwargs,
            )

    def _buttons_from_str(self, s: Optional[str]) -> List[int]:
        if not s:
            return [0] * 8
        idx = self.action_map.get(s.lower())
        vec = [0] * 8
        if idx is not None:
            vec[idx] = 1
        return vec

    def _frame_hash(self, arr: np.ndarray) -> str:
        return hashlib.md5(arr.tobytes()).hexdigest()

    # ───────────────────── Gym API ──────────────────────
    def reset(self, *, seed: int | None = None, episode_id: int = 1, **kwargs):
        self._initialize_env()
        self.adapter.reset_episode(episode_id)
        self.current_frame, _ = self._raw_env.reset(seed=seed)
        self.current_info = self._extract_info()

        img_path = None
        if self.adapter.observation_mode in ("vision", "both"):
            img_path = self.adapter._create_agent_observation_path(
                self.adapter.current_episode_id, self.adapter.current_step_num
            )
            _save_frame(self.current_frame, img_path)

        obs = self.adapter.create_agent_observation(img_path=img_path, text_representation=self._text_repr())
        return obs, self.current_info.copy()

    def step(
        self,
        agent_action_str: Optional[str],
        thought_process: str = "",
        time_taken_s: float = 0.0,
    ):
        if self._raw_env is None:
            raise RuntimeError("Call reset() first")

        self.adapter.increment_step()
        act = self._buttons_from_str(agent_action_str)
        self.current_frame, reward, term, trunc, retro_info = self._raw_env.step(act)

        self.current_info = {**self._extract_info(), **retro_info}

        # perf score = Δ‑score
        perf = self._calculate_perf()

        img_path = None
        if self.adapter.observation_mode in ("vision", "both"):
            img_path = self.adapter._create_agent_observation_path(
                self.adapter.current_episode_id, self.adapter.current_step_num
            )
            _save_frame(self.current_frame, img_path)

        obs = self.adapter.create_agent_observation(img_path=img_path, text_representation=self._text_repr())

        term, trunc = self.adapter.verify_termination(obs, term, trunc)

        self.adapter.log_step_data(
            agent_action_str=agent_action_str,
            thought_process=thought_process,
            reward=float(reward),
            info=self.current_info,
            terminated=term,
            truncated=trunc,
            time_taken_s=time_taken_s,
            perf_score=perf,
            agent_observation=obs,
        )
        return obs, float(reward), term, trunc, self.current_info.copy(), perf

    # ───────────────────── info / perf helpers ─────────────────────
    def _extract_info(self) -> Dict[str, Any]:
        # 1942’s default core exposes score & lives in the info dict already.
        info = self._raw_env.get_info() if hasattr(self._raw_env, "get_info") else {}
        score = info.get("score", 0)
        lives = info.get("lives", 0)
        return {"score": score, "lives": lives, **info}

    def _calculate_perf(self) -> float:
        score = self.current_info.get("score", 0)
        best = getattr(self, "_best_score", 0)
        delta = max(0, score - best)
        if delta:
            self._best_score = score
        return float(delta)

    def _text_repr(self) -> str:
        return f"score:{self.current_info.get('score', 0)}, lives:{self.current_info.get('lives', 0)}"

    # render / close
    def render(self, *_, **__):
        if self.render_mode_human:
            self._raw_env.render()
        elif self.current_frame is not None:
            return self.current_frame.copy()
        return None

    def close(self):
        if self._raw_env:
            self._raw_env.close()
            self._raw_env = None
        self.adapter.close_log_file()
