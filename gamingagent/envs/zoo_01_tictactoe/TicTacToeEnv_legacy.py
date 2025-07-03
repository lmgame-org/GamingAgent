"""
TicTacToeEnv — Gymnasium‑style environment that wraps PettingZoo’s
`tictactoe_v3` AEC environment.  Env types:

- single (default) – classic RL setting.  The agent controls
  **player_1** (X); **player_2** (O) is driven by a lightweight opponent
  policy (random by default, but pluggable).
- self_play – single agent controls *both* players.  Each call to
  `step()` applies the action to whichever PettingZoo agent currently has
  the turn, allowing standard self‑play training loops without a second
  process.  No opponent policy is triggered.


MultiTicTacToeEnv – single-controller environment with two agents, it controls whichever PettingZoo
  player has the turn.

Env design
----------------------------------------------------
- Action space – Discrete(10): 
    0 = no‑op
    1‑9 = place mark in cells 0‑8 (see `ACTION_LOOKUP`).
- Observation space – 3 × 3 integer grid (0 empty, 1 agent, 2 opponent)
  converted from PettingZoo’s two‑plane observation.
- Rendering – Pillow raster with X/O glyphs, reused for on‑disk vision
  observations and pygame human display.
- Performance score – accumulative (+1 win, −1 loss, 0 draw) so the
  runner’s dashboards stay meaningful.
"""
from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import gymnasium as gym
import numpy as np
import pygame
from PIL import Image, ImageDraw, ImageFont
from gymnasium.core import RenderFrame
from gymnasium.spaces import Box, Discrete
from pettingzoo.classic import tictactoe_v3

from gamingagent.envs.gym_env_adapter import GymEnvAdapter
from gamingagent.modules.core_module import Observation

# Action lookup table & geometry
ACTION_LOOKUP: Dict[int, str] = {0: "no operation"}
ACTION_LOOKUP.update({i + 1: f"place {i}" for i in range(9)})  # 1‑9 → cells 0‑8

# Board layout (row, col) coordinates for each cell index
CELL_INDEX_TO_COORD = {
    0: (0, 0), 1: (0, 1), 2: (0, 2),
    3: (1, 0), 4: (1, 1), 5: (1, 2),
    6: (2, 0), 7: (2, 1), 8: (2, 2),
}

# Utility: convert PettingZoo observation → 3×3 int grid
def _convert_obs(pz_obs: Dict[str, np.ndarray]) -> np.ndarray:
    """PettingZoo gives 2 planes; we squish to a single 3×3 with values 0/1/2."""
    planes = pz_obs["observation"]  # shape (3,3,2)
    board = np.zeros((3, 3), dtype=np.uint8)
    board[planes[:, :, 0] == 1] = 1  # agent (X)
    board[planes[:, :, 1] == 1] = 2  # opponent (O)
    return board

# Utility: render board to PNG
def create_board_image_tictactoe(
    board_state: np.ndarray,
    save_path: str | None,
    tile_size: int = 64,
    perf_score: Optional[float] = None,
    action_taken_str: Optional[str] = None,
):
    """Save a 256‑colour PNG of the board (or return silently if no path)."""
    img_w = img_h = tile_size * 3
    img = Image.new("RGB", (img_w, img_h), (240, 240, 240))
    draw = ImageDraw.Draw(img)

    # Grid lines
    for i in range(1, 3):
        draw.line((0, i * tile_size, img_w, i * tile_size), fill="black", width=3)
        draw.line((i * tile_size, 0, i * tile_size, img_h), fill="black", width=3)

    # Font for glyphs
    try:
        font = ImageFont.truetype("arial.ttf", int(tile_size * 0.6))
    except IOError:
        font = ImageFont.load_default()

    def _draw_centered(text: str, row: int, col: int):
        bbox = draw.textbbox((0, 0), text, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = col * tile_size + (tile_size - w) / 2
        y = row * tile_size + (tile_size - h) / 2
        draw.text((x, y), text, fill="black", font=font)

    for idx, (r, c) in CELL_INDEX_TO_COORD.items():
        val = board_state[r, c]
        if val == 1:
            _draw_centered("X", r, c)
        elif val == 2:
            _draw_centered("O", r, c)

    # Perf & action annotations
    if perf_score is not None:
        draw.text((5, 5), f"Perf: {perf_score:+.0f}", fill="blue", font=font)
    if action_taken_str is not None:
        draw.text((5, img_h - 20), f"Action: {action_taken_str}", fill="blue", font=font)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        img.save(save_path)

# ────────────────────────────────────────────────────────────────────────────────
class SingleTicTacToeEnv(gym.Env):
    """Gym wrapper around PettingZoo Tic‑Tac‑Toe with configurable modes."""

    metadata = {"render_modes": ["human", "rgb_array", "raw"], "render_fps": 4}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        env_type: str = "single",  # "single" | "self_play"
        opponent_policy: str | Callable | None = "random",
        tile_size_for_render: int = 64,
        # Adapter plumbing
        game_name_for_adapter: str = "tictactoe",
        observation_mode_for_adapter: str = "vision",
        agent_cache_dir_for_adapter: str = "cache/tictactoe/default_run",
        game_specific_config_path_for_adapter: str = "gamingagent/envs/zoo_01_tictactoe/game_env_config.json",
        max_stuck_steps_for_adapter: Optional[int] = 10,
    ):
        super().__init__()

        assert env_type in {"single", "self_play"}, "env_type must be 'single' or 'self_play'"
        self.env_type = env_type
        self.render_mode = render_mode
        self.tile_size_for_render_size_for_render = tile_size_for_render
        self.opponent_policy = opponent_policy  # used only in "single" mode

        # Underlying PettingZoo env
        self.pz_env = tictactoe_v3.env(render_mode=None)

        # Gym‑style spaces (extra no‑op slot)
        self.action_space = Discrete(10)
        self.observation_space = Box(low=0, high=2, shape=(3, 3), dtype=np.uint8)

        # Adapter initialisation
        self.adapter = GymEnvAdapter(
            game_name=game_name_for_adapter,
            observation_mode=observation_mode_for_adapter,
            agent_cache_dir=agent_cache_dir_for_adapter,
            game_specific_config_path=game_specific_config_path_for_adapter,
            max_steps_for_stuck=max_stuck_steps_for_adapter,
        )

        # Rendering helpers
        self.window = None
        self.clock = None

        # Episode bookkeeping
        self.num_env_steps: int = 0
        self.current_reward_last_step: float = 0.0
        self.cumulative_perf_score: float = 0.0  # +1 win, −1 loss

    # helpers
    # TODO: support other types of opponent policies
    def _default_opponent_move(self, opponent_policy):
        """Random legal move – only used in 'single' mode."""
        if opponent_policy == "random":
            obs_p2 = self.pz_env.observe("player_2")
            legal = np.where(obs_p2["action_mask"] == 1)[0]
            act = random.choice(legal) if legal.size else None
            self.pz_env.step(act)
        else:
            raise NotImplementedError(
                f"Opponent policy '{opponent_policy}' not implemented. "
                "Use 'random' or implement your own."
            )

    def _current_board_state(self) -> np.ndarray:
        return _convert_obs(self.pz_env.observe("player_1"))

    def _get_info(self) -> Dict[str, Any]:
        return {
            "num_env_steps": self.num_env_steps,
            "reward_last_step": self.current_reward_last_step,
            "terminations": self.pz_env.terminations.copy(),
        }

    # Gym API
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
        max_memory: int = 10,
        episode_id: int = 1,
    ) -> Tuple[Observation, Dict[str, Any]]:
        super().reset(seed=seed)
        self.num_env_steps = 0
        self.cumulative_perf_score = 0.0
        self.current_reward_last_step = 0.0

        self.pz_env.reset(seed=seed)
        self.adapter.reset_episode(episode_id)

        board = self._current_board_state()
        info = self._get_info()
        perf = self.cumulative_perf_score

        img_path, text_repr = None, None
        if self.adapter.observation_mode in {"vision", "both"}:
            img_path = self.adapter._create_agent_observation_path(
                episode_id, self.adapter.current_step_num
            )
            create_board_image_tictactoe(board, img_path, self.tile_size_for_render_size_for_render, perf)
        if self.adapter.observation_mode in {"text", "both"}:
            text_repr = "\n".join(" ".join(str(cell) for cell in row) for row in board)

        agent_obs = self.adapter.create_agent_observation(
            img_path=img_path, text_representation=text_repr, max_memory=max_memory
        )

        if self.render_mode == "human":
            self._render_frame()
        return agent_obs, info

    def _apply_action(self, env_act_idx: Optional[int]):
        """Helper to apply a mapped Gym action to whichever agent is active."""
        if env_act_idx is not None and 1 <= env_act_idx <= 9:
            cell = env_act_idx - 1
            self.pz_env.step(cell)
        else:
            self.pz_env.step(None)  # no‑op or illegal

    def step(
        self,
        agent_action_str: Optional[str],
        thought_process: str = "",
        time_taken_s: float = 0.0,
    ) -> Tuple[Observation, float, bool, bool, Dict[str, Any], float]:
        self.adapter.increment_step()

        env_act_idx = self.adapter.map_agent_action_to_env_action(agent_action_str)

        if self.env_type == "single":
            # Ensure it's player_1's turn first
            while self.pz_env.agent_selection != "player_1":
                self.pz_env.step(None)
            self._apply_action(env_act_idx)
            # Opponent move if game not over
            if not self.pz_env.terminations["player_1"]:
                self._default_opponent_move(self.opponent_policy)
        else:  # self_play
            self._apply_action(env_act_idx)

        # Reward & termination info (perspective of player_1)
        reward = float(self.pz_env.rewards["player_1"])
        self.current_reward_last_step = reward

        if self.pz_env.terminations["player_1"]:
            if reward > 0:
                self.cumulative_perf_score += 1
            elif reward < 0:
                self.cumulative_perf_score -= 1
        perf = self.cumulative_perf_score

        terminated = self.pz_env.terminations["player_1"]
        truncated = self.pz_env.truncations["player_1"]
        self.num_env_steps += 1

        board = self._current_board_state()
        info = self._get_info()

        img_path, text_repr = None, None
        if self.adapter.observation_mode in {"vision", "both"}:
            img_path = self.adapter._create_agent_observation_path(
                self.adapter.current_episode_id, self.adapter.current_step_num
            )
            create_board_image_tictactoe(
                board, img_path, self.tile_size_for_render_size_for_render, perf, agent_action_str
            )
        if self.adapter.observation_mode in {"text", "both"}:
            text_repr = "\n".join(" ".join(str(cell) for cell in row) for row in board)

        agent_obs = self.adapter.create_agent_observation(
            img_path=img_path, text_representation=text_repr
        )

        final_terminated, final_truncated = self.adapter.verify_termination(
            agent_obs, terminated, truncated
        )

        self.adapter.log_step_data(
            agent_action_str=agent_action_str,
            thought_process=thought_process,
            reward=reward,
            info=info,
            terminated=final_terminated,
            truncated=final_truncated,
            time_taken_s=time_taken_s,
            perf_score=perf,
            agent_observation=agent_obs,
        )

        if self.render_mode == "human":
            self._render_frame()
        return agent_obs, reward, final_terminated, final_truncated, info, perf

    # Rendering helpers
    def _render_frame_rgb(self) -> Optional[np.ndarray]:
        board = self._current_board_state()
        temp_path = os.path.join(self.adapter.agent_cache_dir, "_temp_render.png")
        create_board_image_tictactoe(board, temp_path, self.tile_size_for_render_size_for_render)
        if os.path.exists(temp_path):
            arr = np.array(Image.open(temp_path).convert("RGB"))
            os.remove(temp_path)
            return arr
        return None

    def _render_frame(self):
        rgb = self._render_frame_rgb()
        if rgb is None:
            return
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((rgb.shape[1], rgb.shape[0]))
            pygame.display.set_caption("Tic‑Tac‑Toe")
        if self.clock is None:
            self.clock = pygame.time.Clock()
        surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
        self.window.blit(surf, (0, 0))
        pygame.event.pump()
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    # Gym render API
    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        if self.render_mode == "rgb_array":
            return self._render_frame_rgb()
        elif self.render_mode == "human":
            self._render_frame()
            return None
        elif self.render_mode == "raw":
            return self._current_board_state()
        return None

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
        self.adapter.close_log_file()

class MultiTicTacToeEnv(gym.Env):
    """Two models; step once per turn with turn‑enforced `current_player`."""
    
    metadata = {"render_modes": ["human", "rgb_array", "raw"], "render_fps": 4}
    
    def __init__(
        self,
        *,
        render_mode: Optional[str] = None,
        tile_size_for_render: int = 64,
        p1_cache: str = "cache/tictactoe/p1",
        p2_cache: str = "cache/tictactoe/p2",
        game_specific_config_path_for_adapter: str = "gamingagent/envs/tictactoe/multiagent_game_env_config.json",
        max_stuck_steps_for_adapter: int | None = 10,
        game_name_for_adapter: str = "tictactoe",
        observation_mode_for_adapter: str = "vision",
        agent_cache_dir_for_adapter: str = "cache/tictactoe/default_run",
    ):
        super().__init__()
        self.render_mode = render_mode
        self.tile_size_for_render = tile_size_for_render
        self._cache_dir = os.path.commonpath([p1_cache, p2_cache])

        self.pz_env = tictactoe_v3.env(render_mode=None)
        self.adapter_p1 = GymEnvAdapter("p1_tictactoe", "vision", p1_cache, game_specific_config_path_for_adapter, max_stuck_steps_for_adapter)
        self.adapter_p2 = GymEnvAdapter("p2_tictactoe", "vision", p2_cache, game_specific_config_path_for_adapter, max_stuck_steps_for_adapter)
        self._adapters = [self.adapter_p1, self.adapter_p2]

        self.window = self.clock = None
        self.perf_p1 = self.perf_p2 = 0.0
        self.step_id = 0
        self.current_player: str = "player_1"  # will be reset properly in reset()

        self.adapter = GymEnvAdapter(
            game_name=game_name_for_adapter,
            observation_mode=observation_mode_for_adapter,
            agent_cache_dir=agent_cache_dir_for_adapter,
            game_specific_config_path=game_specific_config_path_for_adapter,
            max_steps_for_stuck=max_stuck_steps_for_adapter,
        )

    # Rendering helpers
    def _rgb(self):
        temp = os.path.join(self._cache_dir, "_tmp.png")
        create_board_image_tictactoe(self._board(), temp, self.tile_size_for_render)
        if os.path.exists(temp):
            arr = np.array(Image.open(temp).convert("RGB")); os.remove(temp); return arr
        return None
    def _show(self):
        arr = self._rgb();
        if arr is None: return
        if self.window is None:
            pygame.init(); pygame.display.init(); self.window = pygame.display.set_mode((arr.shape[1], arr.shape[0]))
        if self.clock is None: self.clock = pygame.time.Clock()
        surf = pygame.surfarray.make_surface(arr.swapaxes(0,1)); self.window.blit(surf,(0,0)); pygame.event.pump(); pygame.display.flip(); self.clock.tick(self.metadata["render_fps"])
    def render(self, mode=None):
        if self.render_mode=="human": self._show()
        elif self.render_mode=="rgb_array": return self._rgb()
        elif self.render_mode=="raw": return self._board()

    def _update_current_board_state(self):
        """Update board states for both players, for access in agent-specific obs."""
        self._current_board_state = {
            "player_1": _convert_obs(self.pz_env.observe("player_1")),
            "player_2": _convert_obs(self.pz_env.observe("player_2")),
        }

    def _board(self):
        """Convenience: get current player's board view (for rendering)."""
        # Always up-to-date thanks to step/reset calling _update_current_board_state.
        return self._current_board_state[self.current_player]


    # Gym API
    def reset(self, *, seed: int | None = None, **kw):
        super().reset(seed=seed)
        self.pz_env.reset(seed=seed)
        self.current_player = self.pz_env.agent_selection  # typically "player_1"
        self.perf_p1 = self.perf_p2 = 0.0; self.step_id = 0
        self.adapter_p1.reset_episode(1); self.adapter_p2.reset_episode(1)
        self._update_current_board_state()

        board = self._board()
        
        def mk(adapter, board):
            img_path, text_repr = None, None
            if self.adapter.observation_mode in {"text", "both"}:
                text_repr = "\n".join(" ".join(str(cell) for cell in row) for row in board)
            img = adapter._create_agent_observation_path(1,0); create_board_image_tictactoe(board, img, self.tile_size_for_render)
            return adapter.create_agent_observation(img_path=img, text_representation=text_repr)
        obs = {"player_1": mk(self.adapter_p1, board), "player_2": mk(self.adapter_p2, board)}
        if self.render_mode=="human": self._show()
        return obs, {}

    def step(self, agent_name: str, action_str: Optional[str]):
        assert agent_name == self.current_player, f"It is {self.current_player}'s turn, not {agent_name}"
        adapter = self.adapter_p1 if agent_name=="player_1" else self.adapter_p2
        idx = adapter.map_agent_action_to_env_action(action_str)
        self.pz_env.step(idx-1 if idx and 1<=idx<=9 else None)
        self.current_player = self.pz_env.agent_selection  # update to next turn
        self.step_id += 1

        self._update_current_board_state()

        r1 = float(self.pz_env.rewards["player_1"]); r2 = float(self.pz_env.rewards["player_2"])
        if self.pz_env.terminations["player_1"]:
            self.perf_p1 += 1 if r1 > 0 else -1 if r1 < 0 else 0
            self.perf_p2 += 1 if r2 > 0 else -1 if r2 < 0 else 0
        term = any(self.pz_env.terminations.values()); trunc = any(self.pz_env.truncations.values())
        
        board = self._board()
        
        def mk(adapter, board):
            img_path, text_repr = None, None
            if self.adapter.observation_mode in {"text", "both"}:
                text_repr = "\n".join(" ".join(str(cell) for cell in row) for row in board)
            img = adapter._create_agent_observation_path(1,0); create_board_image_tictactoe(board, img, self.tile_size_for_render)
            return adapter.create_agent_observation(img_path=img, text_representation=text_repr)
        next_obs = {"player_1": mk(self.adapter_p1, board), "player_2": mk(self.adapter_p2, board)}
        
        rewards = {"player_1": r1, "player_2": r2}
        perf = {"player_1": self.perf_p1, "player_2": self.perf_p2}
        if self.render_mode=="human": self._show()
        return next_obs, rewards, term, trunc, {}, perf

    def close(self):
        if self.window: pygame.display.quit(); pygame.quit(); self.window=None
        self.adapter_p1.close_log_file(); self.adapter_p2.close_log_file()
