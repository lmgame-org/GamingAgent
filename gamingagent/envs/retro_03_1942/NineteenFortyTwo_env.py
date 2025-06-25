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

# main wrapper
class NineteenFortyTwoEnvWrapper(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    _DEFAULT_ENV_ID = "1942-Nes"

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

        self._max_stuck_steps = _cfg.get("max_unchanged_steps_for_termination", 200)
        self.render_mode_human = _cfg.get("render_mode_human", False)

        self.base_log_dir = base_log_dir

        # ── adapter ──
        self.adapter = GymEnvAdapter(
            game_name=self.game_name,
            observation_mode=observation_mode,
            agent_cache_dir=self.base_log_dir,
            game_specific_config_path=cfg_file,
            max_steps_for_stuck=self._max_stuck_steps
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

            self.num_buttons = len(self._raw_env.buttons)

    def _buttons_from_str(self, token: Optional[str]) -> List[int]:
        """
        Convert ONE action token (e.g. 'up') to a boolean button vector.
        The adapter is the single source of the mapping.
        """
        n = getattr(self, "num_buttons", 8)
        vec = np.zeros(n, dtype=int)

        if token is None or not token.strip():
            return vec.tolist()

        env_act = self.adapter.map_agent_action_to_env_action(token.strip())
        # numpy / list → full vector
        if isinstance(env_act, (list, np.ndarray)):
            tmp = np.array(env_act, dtype=bool)
            if tmp.size != n:
                tmp = np.resize(tmp, n)
            return tmp.astype(int).tolist()

        # int index → one‑hot
        if isinstance(env_act, (int, np.integer)) and 0 <= env_act < n:
            vec[env_act] = 1
            return vec.tolist()

        print(f"[1942] Warning: unknown action token '{token}'")
        return vec.tolist()

    def _frame_hash(self, arr: np.ndarray) -> str:
        return hashlib.md5(arr.tobytes()).hexdigest()

    # ───────────────────── Gym API ──────────────────────
    def reset(self, *, seed: int | None = None, max_memory: Optional[int] = 10, episode_id: int = 1, **kwargs):
        self._initialize_env()
        self.adapter.reset_episode(episode_id)
        # Remove max_memory from kwargs before passing to env.reset()
        env_kwargs = {k: v for k, v in kwargs.items() if k != 'max_memory'}
        self.current_frame, _ = self._raw_env.reset(seed=seed, **env_kwargs)
        self.current_info = self._extract_info()
        self.current_info['total_score'] = self.current_info['score']

        img_path = None
        if self.adapter.observation_mode in ("vision", "both"):
            img_path = self.adapter.save_frame_and_get_path(self.current_frame)

        obs = self.adapter.create_agent_observation(img_path=img_path, text_representation=self._text_repr(), max_memory=max_memory)
        return obs, self.current_info.copy()

    def step(
        self,
        agent_action_str: Optional[str],
        thought_process: str = "",
        time_taken_s: float = 0.0,
    ):
        """
        Execute one or more actions.  If `agent_action_str` contains the
        separator  “||”, each token is run in a *separate* frame in order.

        Example
        -------
        agent_action_str == "up||b"   →
            frame 1 : UP
            frame 2 : B (shoot)

        Rewards and perf‑scores are accumulated; termination can occur
        after any sub‑frame.
        """
        if self._raw_env is None:
            raise RuntimeError("Call reset() first")

        # ── split into sequential tokens ───────────────────────────────
        tokens: List[Optional[str]] = [
            t.strip() for t in (agent_action_str or "").split("||") if t.strip()
        ] or [None]                                  # None → no‑op

        total_reward: float = 0.0
        combined_perf: float = 0.0
        term = trunc = False
        retro_info: Dict[str, Any] = {}

        for i, tok in enumerate(tokens):
            self.adapter.increment_step()            # one increment *per* sub‑action
            act_vec = self._buttons_from_str(tok)
            self.current_frame, r, term, trunc, retro_info = self._raw_env.step(
                act_vec
            )

            total_reward += float(r)
            self.current_info = {**self._extract_info(), **retro_info}
            combined_perf += self._calculate_perf()

            if term or trunc:                        # stop if game ended
                break

        # ── save screenshot of the LAST frame, if needed ───────────────
        img_path = None
        if self.adapter.observation_mode in ("vision", "both") and self.current_frame is not None:
            img_path = self.adapter.save_frame_and_get_path(self.current_frame)

        # ── build observation & post‑processing ────────────────────────
        obs = self.adapter.create_agent_observation(
            img_path=img_path,
            text_representation=self._text_repr(),
        )

        term, trunc = self.adapter.verify_termination(obs, term, trunc)

        self.adapter.log_step_data(
            agent_action_str=agent_action_str,
            thought_process=thought_process,
            reward=total_reward,
            info=self.current_info,
            terminated=term,
            truncated=trunc,
            time_taken_s=time_taken_s,
            perf_score=combined_perf,
            agent_observation=obs,
        )

        return obs, total_reward, term, trunc, self.current_info.copy(), combined_perf


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
