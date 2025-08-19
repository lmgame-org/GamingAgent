"""Unified Tic‑Tac‑Toe Gymnasium environments
================================================
This module provides **SingleTicTacToeEnv** and **MultiTicTacToeEnv** that
share the same helper utilities and rendering code.  `MultiTicTacToeEnv`
sub‑classes `SingleTicTacToeEnv`, eliminating the code duplication that
previously existed between the two standalone implementations while still
exposing the multi‑agent interface required by training loops that control
_X_ and _O_ with separate models.

Key points
----------
* Shared helpers – action look‑up table, geometry, observation conversion
  and Pillow board rendering now live at module level so both envs reuse them.
* Bug fix – attribute typo `tile_size_for_render_size_for_render` is now
  the correct `self.tile_size_for_render` everywhere.
* Adapters everywhere – both single‑ and multi‑agent variants use
  `GymEnvAdapter` exclusively for agent I/O, episode bookkeeping and on‑disk
  vision observations.
* Minimal surface‑area change – public APIs remain exactly the same:
    * `SingleTicTacToeEnv` construction parameters are unchanged.
    * `MultiTicTacToeEnv.step(agent_name, action_str)` signature is preserved.

This makes it straightforward to swap out the old classes by importing from
this module instead.
"""
from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import gymnasium as gym
import numpy as np
import pygame
from PIL import Image, ImageDraw, ImageFont
<<<<<<< HEAD
import imageio
=======
>>>>>>> origin/main
from gymnasium.core import RenderFrame
from gymnasium.spaces import Box, Discrete
from pettingzoo.classic import tictactoe_v3

from gamingagent.envs.gym_env_adapter import GymEnvAdapter
from gamingagent.modules.core_module import Observation

# Action lookup table & geometry
ACTION_LOOKUP: Dict[int, str] = {i: f"place {i}" for i in range(9)}  # 0-8 → cells 0‑8

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

# Helper: create informative text representation including action mask
def _create_text_representation(board: np.ndarray, action_mask: np.ndarray, current_player: str) -> str:
    """Create a comprehensive text representation including legal moves."""
    # Create a visual board representation
    board_lines = []
    board_lines.append("Current Board:")
    board_lines.append("  0 1 2")
    for i in range(3):
        row_str = f"{i} "
        for j in range(3):
            cell_val = board[i, j]
            if cell_val == 0:
                row_str += ". "
            elif cell_val == 1:
                row_str += "X "
            else:  # cell_val == 2
                row_str += "O "
        board_lines.append(row_str)
    
    # Add legend
    board_lines.append("")
    board_lines.append("Legend: X=Player 1, O=Player 2, .=Empty")
    
    # Add legal moves information
    legal_moves = []
    for i in range(len(action_mask)):
        if action_mask[i] == 1:
                row, col = i // 3, i % 3
                legal_moves.append(f"place {i} (row {row}, col {col})")
    
    board_lines.append("")
    board_lines.append(f"Current Player: {current_player}")
    board_lines.append(f"Legal Actions: {', '.join(legal_moves)}")
    board_lines.append("CHOOSING ANY OTHER ACTION WILL CAUSE IMMEDIATE GAME TERMINATION")
    board_lines.append("")
    board_lines.append("Action Format: Use 'place X' where X is the cell number (0-8)")
    board_lines.append("Cell numbering: 0=top-left, 1=top-center, 2=top-right, 3=middle-left, etc.")
    
    return "\n".join(board_lines)

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

<<<<<<< HEAD
def _generate_video_from_rgb_array(frames: List[np.ndarray], output_path: str, frame_rate: int = 2):
    if not frames:
        return
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    try:
        writer = imageio.get_writer(
            output_path,
            fps=frame_rate,
            codec='libx264',
            pixelformat='yuv420p',
            macro_block_size=1
        )
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        print(f"GUI video saved to {output_path}")
    except Exception as e:
        print(f"Error generating GUI video: {e}")

=======
>>>>>>> origin/main
# ──────────────────────────────────────────────────────────────────────────────
# Single‑agent environment
# ----------------------------------------------------------------------------

class SingleTicTacToeEnv(gym.Env):
    """Gym wrapper around PettingZoo Tic‑Tac‑Toe with configurable modes.

    Modes
    -----
    * **single** – agent controls *player_1* (X); *player_2* is a pluggable
      opponent policy (random by default).
    """

    metadata = {"render_modes": ["human", "rgb_array", "raw"], "render_fps": 4}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        env_type: str = "single",  # "single"
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

        assert env_type in {"single"}, "env_type must be 'single'"
        self.env_type = env_type
        self.render_mode = render_mode
        self.tile_size_for_render = tile_size_for_render
        self.opponent_policy = opponent_policy  # used only in "single" mode

        # Underlying PettingZoo env
        self.pz_env = tictactoe_v3.env(render_mode=None)

        self.current_player = "player_1" # default

        # Gym‑style spaces
        self.action_space = Discrete(9)
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
        self.window: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None

        # Episode bookkeeping
        self.num_env_steps: int = 0
        self.current_reward_last_step: float = 0.0
        self.cumulative_perf_score: float = 0.0  # +1 win, −1 loss

    # helpers
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

    def _current_board_state(self, player: str = "player_1") -> np.ndarray:
        return _convert_obs(self.pz_env.observe(player))

    def _get_info(self) -> Dict[str, Any]:
        return {
            "num_env_steps": self.num_env_steps,
            "reward_last_step": self.current_reward_last_step,
            "terminations": self.pz_env.terminations.copy(),
        }

    # ---------------------------------------------------------------------
    # Gym API
    # ---------------------------------------------------------------------

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
        self.current_player = self.pz_env.agent_selection

        board = self._current_board_state(self.current_player)
        obs_pz = self.pz_env.observe(self.current_player)

        self.adapter.reset_episode(episode_id)

        info = self._get_info()

        img_path = text_repr = None
        if self.adapter.observation_mode in {"vision", "both"}:
            img_path = self.adapter._create_agent_observation_path(
                episode_id, self.adapter.current_step_num
            )
            create_board_image_tictactoe(board, img_path, self.tile_size_for_render, self.cumulative_perf_score)
        
        if self.adapter.observation_mode in {"text", "both"}:
            text_repr = _create_text_representation(
                board, obs_pz["action_mask"], self.current_player
            )

        agent_obs = self.adapter.create_agent_observation(
            img_path=img_path, text_representation=text_repr, max_memory=max_memory
        )

        if self.render_mode == "human":
            self._render_frame()
        return agent_obs, info

    def _apply_action(self, env_act_idx: Optional[int]):
        """Helper to apply a mapped Gym action to whichever agent is active."""
        self.pz_env.step(env_act_idx)

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
            self.current_player = self.pz_env.agent_selection
        else:
            raise NotImplementedError(
                f"Environment type '{self.env_type}' not implemented. "
            )

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

        board = self._current_board_state(self.current_player)
        obs_pz = self.pz_env.observe(self.current_player)
        
        info = self._get_info()
        img_path = text_repr = None
        if self.adapter.observation_mode in {"vision", "both"}:
            img_path = self.adapter._create_agent_observation_path(
                self.adapter.current_episode_id, self.adapter.current_step_num
            )
            create_board_image_tictactoe(
                board, img_path, self.tile_size_for_render, perf, agent_action_str
            )
        
        if self.adapter.observation_mode in {"text", "both"}:
            text_repr = _create_text_representation(
                board, obs_pz["action_mask"], self.current_player
            )

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

    # Rendering
    def _render_frame_rgb(self) -> Optional[np.ndarray]:
        board = self._current_board_state(self.current_player)
        temp_path = os.path.join(
            self.adapter.agent_cache_dir, "_temp_render.png"
        )
        create_board_image_tictactoe(board, temp_path, self.tile_size_for_render)
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
            return self._current_board_state(self.current_player)
        return None

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
        self.adapter.close_log_file()

# ──────────────────────────────────────────────────────────────────────────────
# Multi‑agent environment (inherits from the single‑agent implementation)
# ----------------------------------------------------------------------------

class MultiTicTacToeEnv(SingleTicTacToeEnv):
    """Two‑model controller that reuses the single‑agent core."""

    def __init__(
        self,
        *,
        render_mode: Optional[str] = None,
        tile_size_for_render: int = 64,
        p1_cache: str = "cache/tictactoe/p1",
        p2_cache: str = "cache/tictactoe/p2",
        game_name_for_adapter: str = "tictactoe",
        observation_mode_for_adapter: str = "vision",
        agent_cache_dir_for_adapter: str = "cache/tictactoe/default_run",
        game_specific_config_path_for_adapter: str = "gamingagent/envs/zoo_01_tictactoe/game_env_config.json",
        max_stuck_steps_for_adapter: Optional[int] = 10,
<<<<<<< HEAD
        record_video: bool = True,
        video_frame_rate: int = 2,
=======
>>>>>>> origin/main
    ):
        super().__init__(
            render_mode=render_mode,
            env_type="single",
            opponent_policy=None,
            tile_size_for_render=tile_size_for_render,
            game_name_for_adapter=game_name_for_adapter,
            observation_mode_for_adapter=observation_mode_for_adapter,
            agent_cache_dir_for_adapter=p1_cache,
            game_specific_config_path_for_adapter=game_specific_config_path_for_adapter,
            max_stuck_steps_for_adapter=max_stuck_steps_for_adapter,
        )
        self.adapter_p1 = GymEnvAdapter(
            game_name="p1_tictactoe",
            observation_mode=observation_mode_for_adapter,
            agent_cache_dir=p1_cache,
            game_specific_config_path=game_specific_config_path_for_adapter,
            max_steps_for_stuck=max_stuck_steps_for_adapter,
        )
        self.adapter_p2 = GymEnvAdapter(
            game_name="p2_tictactoe",
            observation_mode=observation_mode_for_adapter,
            agent_cache_dir=p2_cache,
            game_specific_config_path=game_specific_config_path_for_adapter,
            max_steps_for_stuck=max_stuck_steps_for_adapter,
        )
        self._adapters = {"player_1": self.adapter_p1, "player_2": self.adapter_p2}

        self.perf_scores = {"player_1": 0.0, "player_2": 0.0}
        self.current_player = "player_1" # default
        self.step_id = 0

        # Ensure board state is always present, even before reset
        self._board_state_dict = {
            "player_1": np.zeros((3, 3), dtype=np.uint8),
            "player_2": np.zeros((3, 3), dtype=np.uint8)
        }

<<<<<<< HEAD
        self.record_video = record_video
        self.video_frame_rate = video_frame_rate
        self.gui_frames: List[np.ndarray] = []
        try:
            self.base_cache_dir = os.path.dirname(self._adapters["player_1"].agent_cache_dir)
        except Exception:
            self.base_cache_dir = "cache/tictactoe"

=======
>>>>>>> origin/main
    def _update_current_board_state(self):
        # Get the actual board state from PettingZoo
        p1_obs = self.pz_env.observe("player_1")
        p2_obs = self.pz_env.observe("player_2")
        
        # For player_1: use standard conversion (1=X, 2=O)
        self._board_state_dict["player_1"] = _convert_obs(p1_obs)
        
        # For player_2: we need to swap the perspective
        # In PettingZoo, player_2's plane 0 is their own pieces, plane 1 is opponent's pieces
        # But _convert_obs assumes plane 0 = agent, plane 1 = opponent
        # So we need to swap the planes for player_2
        planes = p2_obs["observation"]
        player_2_board = np.zeros((3, 3), dtype=np.uint8)
        player_2_board[planes[:, :, 0] == 1] = 1  # player_2's pieces (O)
        player_2_board[planes[:, :, 1] == 1] = 2  # player_1's pieces (X)
        self._board_state_dict["player_2"] = player_2_board

    def _board_for(self, agent_name: str) -> np.ndarray:
        return self._board_state_dict[agent_name]

    def reset(self, *, seed: int | None = None, episode_id: int = 1, **kwargs):
<<<<<<< HEAD
        if self.record_video and getattr(self, "gui_frames", None):
            if len(self.gui_frames) > 0 and hasattr(self, "current_episode_id"):
                video_path = os.path.join(self.base_cache_dir, "videos", f"episode_{self.current_episode_id}_gui.mp4")
                _generate_video_from_rgb_array(self.gui_frames, video_path, self.video_frame_rate)
        self.gui_frames = []
=======
>>>>>>> origin/main
        for adap in self._adapters.values():
            adap.reset_episode(episode_id)

        super().reset(seed=seed)

        self.step_id = 0
        self.current_episode_id = episode_id
        self.perf_scores = {"player_1": 0.0, "player_2": 0.0}

        # Must update board state right after env reset
        self._update_current_board_state()

        obs_dict = {}
<<<<<<< HEAD
        p1_frame_path = None
=======
>>>>>>> origin/main
        for agent_name, adap in self._adapters.items():
            board = self._board_for(agent_name)
            img_path = adap._create_agent_observation_path(episode_id, 0)
            create_board_image_tictactoe(board, img_path, self.tile_size_for_render)
<<<<<<< HEAD
            if agent_name == "player_1":
                p1_frame_path = img_path
=======
>>>>>>> origin/main
            obs_pz = self.pz_env.observe(agent_name)
            is_active = (self.pz_env.agent_selection == agent_name)
            if adap.observation_mode in {"text", "both"}:
                if is_active:
                    action_mask = obs_pz["action_mask"]
                else:
                    action_mask = np.zeros_like(obs_pz["action_mask"])
                # Use the actual current player (whose turn it is) for the text representation
                current_player_turn = self.pz_env.agent_selection
                text_repr = _create_text_representation(board, action_mask, current_player_turn)
            else:
                text_repr = None
            obs_dict[agent_name] = adap.create_agent_observation(
                img_path=img_path, text_representation=text_repr
            )

        if self.render_mode == "human":
            self._render_frame_multi()
<<<<<<< HEAD
        if self.record_video:
            # Prefer the persisted player_1 image to avoid empty frames
            try:
                if p1_frame_path and os.path.exists(p1_frame_path):
                    arr = np.array(Image.open(p1_frame_path).convert("RGB"))
                    self.gui_frames.append(arr)
                    print(f"[TicTacToe] Appended reset frame from {p1_frame_path}")
                else:
                    rgb = self._render_frame_rgb()
                    if rgb is not None:
                        self.gui_frames.append(rgb)
            except Exception as e:
                print(f"[TicTacToe] Failed to append reset frame: {e}")
=======
>>>>>>> origin/main
        return obs_dict, {}


    def step(self, agent_name: str, action_str: Optional[str]):
        assert agent_name == self.current_player, (
            f"It is {self.current_player}'s turn, not {agent_name}")
        adap = self._adapters[agent_name]
        
<<<<<<< HEAD
        # increment adapter step counter for acting agent
        adap.increment_step()
        
=======
>>>>>>> origin/main
        env_act_idx = adap.map_agent_action_to_env_action(action_str)
        # Try to apply the action, catching illegal move errors
        try:
            self._apply_action(env_act_idx)
        except Exception as e:
            # Log error and treat as forced termination (PettingZoo usually terminates here)
            print(f"[ERROR] Step failed or illegal move for agent {agent_name}: {e}")
            
            # Set rewards: illegal move maker loses (-1), other player gets 0
            if agent_name == "player_1":
                rewards = {"player_1": -1.0, "player_2": 0.0}
                self.perf_scores["player_1"] -= 1  # penalty for illegal move
            else:  # agent_name == "player_2"
                rewards = {"player_1": 0.0, "player_2": -1.0}
                self.perf_scores["player_2"] -= 1  # penalty for illegal move
            
<<<<<<< HEAD
            # log the illegal step for the acting agent
            adap.log_step_data(
                agent_action_str=action_str,
                thought_process="",
                reward=rewards.get(agent_name, 0.0),
                info={"illegal": True, "error": str(e)},
                terminated=True,
                truncated=True,
                time_taken_s=0.0,
                perf_score=self.perf_scores.get(agent_name, 0.0),
                agent_observation=None,
            )

            # append a video frame for illegal move so the video isn't empty
            if getattr(self, "record_video", False):
                try:
                    self._update_current_board_state()
                    p1_board = self._board_for("player_1")
                    img_path_vid = self._adapters["player_1"]._create_agent_observation_path(
                        getattr(self, "current_episode_id", 0), getattr(self, "step_id", 0)
                    )
                    create_board_image_tictactoe(p1_board, img_path_vid, self.tile_size_for_render)
                    arr_illegal = np.array(Image.open(img_path_vid).convert("RGB"))
                    self.gui_frames.append(arr_illegal)
                    print(f"[TicTacToe] Appended illegal-move frame from {img_path_vid}")
                except Exception as e2:
                    print(f"[TicTacToe] Failed to append illegal-move frame: {e2}")
                
=======
>>>>>>> origin/main
            return (
                {},  # obs
                rewards,
                True,  # terminations
                True,  # truncations
                {},    # info
                self.perf_scores.copy(),
            )
<<<<<<< HEAD
        
        self.current_player = self.pz_env.agent_selection
        self.step_id += 1
        
        self._update_current_board_state()
        
=======

        self.current_player = self.pz_env.agent_selection
        self.step_id += 1

        self._update_current_board_state()

>>>>>>> origin/main
        rewards = {
            "player_1": float(self.pz_env.rewards["player_1"]),
            "player_2": float(self.pz_env.rewards["player_2"]),
        }
        terminations = any(self.pz_env.terminations.values())
        truncations = any(self.pz_env.truncations.values())
<<<<<<< HEAD
        
=======

>>>>>>> origin/main
        if terminations:
            for p in ("player_1", "player_2"):
                r = rewards[p]
                if r > 0:
                    self.perf_scores[p] += 1
                elif r < 0:
                    self.perf_scores[p] -= 1
<<<<<<< HEAD
        
        next_obs = {}
        p1_frame_path = None
=======

        next_obs = {}
>>>>>>> origin/main
        for agent_name, adap in self._adapters.items():
            board = self._board_for(agent_name)
            img_path = adap._create_agent_observation_path(self.current_episode_id, self.step_id)
            create_board_image_tictactoe(board, img_path, self.tile_size_for_render)
<<<<<<< HEAD
            if agent_name == "player_1":
                p1_frame_path = img_path
=======
>>>>>>> origin/main
            text_repr = None
            if adap.observation_mode in {"text", "both"}:
                # Use the actual current player (whose turn it is) for the text representation
                current_player_turn = self.pz_env.agent_selection
                text_repr = _create_text_representation(board, self.pz_env.observe(agent_name)["action_mask"], current_player_turn)
            next_obs[agent_name] = adap.create_agent_observation(
                img_path=img_path, text_representation=text_repr
            )
<<<<<<< HEAD
        
        if self.record_video:
            try:
                if p1_frame_path and os.path.exists(p1_frame_path):
                    arr = np.array(Image.open(p1_frame_path).convert("RGB"))
                    self.gui_frames.append(arr)
                    print(f"[TicTacToe] Appended step frame from {p1_frame_path}")
                else:
                    rgb = self._render_frame_rgb()
                    if rgb is not None:
                        self.gui_frames.append(rgb)
            except Exception as e:
                print(f"[TicTacToe] Failed to append step frame: {e}")
            # If episode ended, write out the video immediately
            if terminations or truncations:
                try:
                    video_path = os.path.join(self.base_cache_dir, "videos", f"episode_{self.current_episode_id}_gui.mp4")
                    _generate_video_from_rgb_array(self.gui_frames, video_path, self.video_frame_rate)
                    print(f"[TicTacToe] Saved episode video with {len(self.gui_frames)} frames -> {video_path}")
                    self.gui_frames = []
                except Exception as e3:
                    print(f"[TicTacToe] Failed to save episode video: {e3}")
        
        # log step data for the acting agent
        if agent_name in self._adapters and agent_name in next_obs:
            acting_adap = self._adapters[agent_name]
            acting_adap.log_step_data(
                agent_action_str=action_str,
                thought_process="",
                reward=rewards.get(agent_name, 0.0),
                info={"step_id": self.step_id, "current_player": agent_name},
                terminated=terminations,
                truncated=truncations,
                time_taken_s=0.0,
                perf_score=self.perf_scores.get(agent_name, 0.0),
                agent_observation=next_obs[agent_name],
            )
        if self.render_mode == "human":
            self._render_frame_multi()
        
=======

        if self.render_mode == "human":
            self._render_frame_multi()

>>>>>>> origin/main
        return next_obs, rewards, terminations, truncations, {}, self.perf_scores.copy()

    def _render_frame_multi(self):
        # Always update before rendering
        self._update_current_board_state()
        rgb = self._render_frame_rgb()
        if rgb is None:
            return
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((rgb.shape[1], rgb.shape[0]))
            pygame.display.set_caption("Tic‑Tac‑Toe – Multi")
        if self.clock is None:
            self.clock = pygame.time.Clock()
        surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
        self.window.blit(surf, (0, 0))
        pygame.event.pump()
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def _render_frame_rgb(self) -> Optional[np.ndarray]:
        # Always update before rendering
        self._update_current_board_state()
        # Use player_1's perspective for consistent rendering (1=X, 2=O)
        board = self._board_for("player_1")
        temp_path = os.path.join(
            self._adapters["player_1"].agent_cache_dir, "_temp_render.png"
        )
        create_board_image_tictactoe(board, temp_path, self.tile_size_for_render)
        if os.path.exists(temp_path):
            arr = np.array(Image.open(temp_path).convert("RGB"))
            os.remove(temp_path)
            return arr
        return None

    def close(self):
        super().close()
<<<<<<< HEAD
        if self.record_video and getattr(self, "gui_frames", None):
            if len(self.gui_frames) > 0 and hasattr(self, "current_episode_id"):
                video_path = os.path.join(self.base_cache_dir, "videos", f"episode_{self.current_episode_id}_gui_final.mp4")
                _generate_video_from_rgb_array(self.gui_frames, video_path, self.video_frame_rate)
        self.gui_frames = []
=======
>>>>>>> origin/main
        for adap in self._adapters.values():
            adap.close_log_file()
