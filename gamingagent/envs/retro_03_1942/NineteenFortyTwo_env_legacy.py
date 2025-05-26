from __future__ import annotations

import os
import json
from typing import Any, Dict, Tuple, Optional, List, Union

import retro
import numpy as np
from PIL import Image

from gamingagent.envs.gym_env_adapter import GymEnvAdapter
from gamingagent.modules.core_module import Observation

# ─────────────────────────────────────────────────────────────────────────────
#  Helper functions                       
# ─────────────────────────────────────────────────────────────────────────────

def _read_ram_value(ram: bytes, address_str: str) -> int:
    return int(ram[int(address_str, 16)])

def _read_ram_fields(ram: bytes, addresses_config: Dict[str, Any]) -> Dict[str, Any]:
    if ram is None:
        return {k: None for k in addresses_config}
    data: Dict[str, Any] = {}
    for key, addr in addresses_config.items():
        try:
            if addr is None:
                data[key] = None
            elif isinstance(addr, str):  # single byte
                data[key] = _read_ram_value(ram, addr)
            elif isinstance(addr, list):  # multi‑byte field
                data[key] = [_read_ram_value(ram, a) for a in addr]
            elif isinstance(addr, dict):  # composite field
                data[key] = {sub: _read_ram_value(ram, a) for sub, a in addr.items()}
            else:
                print(f"[RAM] Unknown address format for {key}: {addr}")
                data[key] = None
        except Exception as e:
            print(f"[RAM] Error reading {key}: {e}")
            data[key] = None
    return data

def _convert_bcd_bytes_to_int_string(bcd: Union[List[int], bytes]) -> str:
    if not bcd:
        return "0"
    vals = list(bcd) if isinstance(bcd, bytes) else bcd
    return "".join(f"{v:02x}" for v in vals).lstrip("0") or "0"

# ─────────────────────────────────────────────────────────────────────────────
#  Main wrapper                                                               │
# ─────────────────────────────────────────────────────────────────────────────

class NineteenFortyTwoEnvWrapper:
    """Retro wrapper for Capcom’s **1942 (NES)** with SMB‑style ergonomics."""

    def __init__(
        self,
        game_name: str,            # logical name e.g. "nineteen_forty_two"
        model_name: str,           # used only for adapter cache path
        config_dir_path: str,      # dir hosting game_env_config.json
        observation_mode: str,     # vision | text | both
        base_log_dir: str,         # root cache dir
    ) -> None:
        self.game_name = game_name
        self.model_name = model_name
        self.config_dir_path = config_dir_path
        self.game_specific_config_json_path = os.path.join(config_dir_path, "game_env_config.json")
        self.observation_mode = observation_mode.lower()
        self.base_log_dir = base_log_dir

        self._load_wrapper_config()
        self._initialize_env()

        self.adapter = GymEnvAdapter(
            game_name=self.game_name,
            observation_mode=self.observation_mode,
            agent_cache_dir=self.base_log_dir,
        )

        # Runtime bookkeeping
        self.current_game_info: Dict[str, Any] = {}
        self.current_episode_max_score: int = 0  # 1942 has no x‑pos; use score
        self.current_meta_episode_accumulated_reward: float = 0.0

    # ───────────────────────── Config & env init ────────────────────────── #

    def _load_wrapper_config(self):
        print(f"[1942Wrapper] Loading config from {self.game_specific_config_json_path}")
        with open(self.game_specific_config_json_path, "r") as f:
            cfg = json.load(f)
        self.env_id = cfg.get("env_id", "1942-Nes")
        self.env_init_kwargs = cfg.get("env_init_kwargs", {})
        am = cfg.get("action_mapping", {})
        self.action_mapping: Dict[str, List[int]] = {k.lower(): v for k, v in am.items()}

        custom = cfg.get("custom_game_specific_config", {})
        self.ram_addresses = {
            "lives": custom.get("lives_ram_address"),            # e.g. "0x0062"
            "score": custom.get("score_ram_address"),            # ["0x004E", "0x004F", "0x0050"] BCD
            "stage": custom.get("stage_ram_address"),            # current stage number
        }
        self.render_mode_human = cfg.get("render_mode_human", False)
        self.retro_obs_type_str = custom.get("observation_type", "IMAGE")

    def _initialize_env(self):
        obs_enum = {
            "IMAGE": retro.Observations.IMAGE,
            "RAM": retro.Observations.RAM,
            "RGB": retro.Observations.RGB_ARRAY,
        }.get(self.retro_obs_type_str.upper(), retro.Observations.IMAGE)

        record_path = os.path.join(self.base_log_dir, "bk2_recordings")
        os.makedirs(record_path, exist_ok=True)

        render_arg = "human" if self.render_mode_human else None

        print(f"[1942Wrapper] Starting Retro env id={self.env_id}, obs_type={obs_enum}, render={render_arg}, record={record_path}")
        self.env = retro.make(
            self.env_id,
            obs_type=obs_enum,
            render_mode=render_arg,
            record=record_path,
            **self.env_init_kwargs,
        )
        print(f"[1942Wrapper] Buttons: {self.env.buttons}")

    # ───────────────────────────── Helpers ──────────────────────────────── #

    def _save_frame_get_path(self, frame: np.ndarray, episode_id: int, step: int) -> Optional[str]:
        if frame is None or not isinstance(frame, np.ndarray):
            return None
        path = self.adapter._create_agent_observation_path(episode_id, step)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        Image.fromarray(frame).save(path)
        return path

    def _extract_game_info(self, ram: Optional[bytes]) -> Dict[str, Any]:
        raw = _read_ram_fields(ram, self.ram_addresses)
        score_bcd = raw.get("score")
        score_int = int(_convert_bcd_bytes_to_int_string(score_bcd)) if score_bcd else 0
        lives_raw = raw.get("lives", 0)
        is_game_over = lives_raw == 0
        return {
            "score": score_int,
            "lives": lives_raw,
            "stage": raw.get("stage", 0),
            "is_game_over": is_game_over,
        }

    def _build_text_repr(self, info: Dict[str, Any]) -> str:
        return f"Score:{info['score']} Lives:{info['lives']} Stage:{info['stage']}" + (" GAME OVER" if info.get("is_game_over") else "")

    def _calculate_perf_score(self, score_now: int) -> float:
        delta = max(0, score_now - self.current_episode_max_score)
        if delta:
            self.current_episode_max_score = score_now
        return float(delta)

    # ───────────────────────────── Gym‑like API ─────────────────────────── #

    def reset(self, episode_id: int, **kwargs) -> Tuple[Observation, Dict[str, Any]]:
        self.adapter.reset_episode(episode_id)

        frame, _ = self.env.reset(**kwargs)
        ram = self.env.get_ram()
        self.current_game_info = self._extract_game_info(ram)
        self.current_episode_max_score = self.current_game_info["score"]
        self.current_meta_episode_accumulated_reward = 0.0

        img_path = None
        if self.observation_mode in ["vision", "both"]:
            img_path = self._save_frame_get_path(frame, episode_id, self.adapter.current_step_num)

        obs = self.adapter.create_agent_observation(img_path, self._build_text_repr(self.current_game_info))
        return obs, self.current_game_info.copy()

    def step(
        self,
        agent_action_str: Optional[str],
        thought_process: str = "",
        time_taken_s: float = 0.0,
    ) -> Tuple[Observation, float, bool, bool, Dict[str, Any], float]:
        import re
        base_action = "noop"
        frames = 1
        if agent_action_str:
            m = re.match(r"\(?'?(\w+)'?,\s*(\d+)\)?", agent_action_str)
            if m:
                base_action, frames = m.group(1), int(m.group(2))
            else:
                base_action = agent_action_str.strip("()' ")
        buttons = self.action_mapping.get(base_action.lower(), [0] * len(self.env.buttons))

        perf_accum = 0.0
        reward_accum = 0.0
        meta_terminated = meta_truncated = False
        last_obs: Optional[Observation] = None

        for f in range(frames):
            self.adapter.increment_step()
            frame_img, r, term, trunc, retro_info = self.env.step(buttons)
            ram = self.env.get_ram()
            game_info = self._extract_game_info(ram)
            game_info.update(retro_info)

            reward_accum += float(r)
            perf = self._calculate_perf_score(game_info["score"])
            perf_accum += perf

            img_path = None
            if self.observation_mode in ["vision", "both"]:
                img_path = self._save_frame_get_path(frame_img, self.adapter.current_episode_id, self.adapter.current_step_num)

            last_obs = self.adapter.create_agent_observation(img_path, self._build_text_repr(game_info))

            term_stuck, trunc_stuck = self.adapter.verify_termination(last_obs, term, trunc)
            meta_terminated = meta_terminated or game_info["is_game_over"] or term_stuck
            meta_truncated = meta_truncated or trunc_stuck

            self.adapter.log_step_data(
                agent_action_str=base_action,
                thought_process=thought_process,
                reward=float(r),
                info=game_info,
                terminated=meta_terminated,
                truncated=meta_truncated,
                time_taken_s=time_taken_s if f == 0 else 0.0,
                perf_score=perf,
                agent_observation=last_obs,
            )
            if meta_terminated or meta_truncated:
                break

        self.current_game_info = game_info
        self.current_meta_episode_accumulated_reward += reward_accum
        return (
            last_obs,
            self.current_meta_episode_accumulated_reward,
            meta_terminated,
            meta_truncated,
            self.current_game_info.copy(),
            perf_accum,
        )

    def render(self):
        if self.render_mode_human:
            self.env.render()

    def close(self):
        print("[1942Wrapper] Closing env…")
        self.env.close()
        self.adapter.close_log_file()
