"""Unified Texas Hold'em Gymnasium environments (clean + vision images)

Uses PettingZoo's built-in human renderer. Vision observations are generated
as PNG files (ASCII text rendered into an image) so that modules expecting
observation.img_path keep working.
"""

from __future__ import annotations
import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete
from pettingzoo.classic import texas_holdem_v4
from PIL import Image, ImageDraw, ImageFont
import imageio
import glob
from natsort import natsorted
import json

from gamingagent.envs.gym_env_adapter import GymEnvAdapter
from gamingagent.modules.core_module import Observation
# ---------------------------------------------------------------------------
# Helper utilities for optional text observation
# ---------------------------------------------------------------------------

def _get_street_from_community_cards(community_cards: List[Any]) -> str:
    n = len(community_cards)
    if n == 0: return "pre-flop"
    if n == 3: return "flop"
    if n == 4: return "turn"
    if n == 5: return "river"
    return "unknown"

def _card_block_simple(rank: str, suit: str) -> List[str]:
    """Simplified card representation using ASCII characters for better font compatibility."""
    suits = {"Hearts": "H", "Diamonds": "D", "Clubs": "C", "Spades": "S",
             "H": "H", "D": "D", "C": "C", "S": "S"}
    symbol = suits.get(suit, suit)
    return [
        "+-------+",
        f"| {rank:<2}    |",
        "|       |",
        f"|   {symbol}   |",
        "|       |",
        f"|    {rank:>2} |",
        "+-------+",
    ]


def _render_card_row(cards: List[Tuple[str, str]]) -> List[str]:
    lines = [""] * 7
    for rank, suit in cards:
        if rank == "?" or suit == "?":
            block = [
                "+-------+",
                "| ? ? ? |",
                "|       |",
                "|   ?   |",
                "|       |",
                "| ? ? ? |",
                "+-------+",
            ]
        else:
            block = _card_block_simple(rank, suit)  # Use simple ASCII version for better compatibility
        for i in range(7):
            lines[i] += f"{block[i]} "
    return lines


def _create_simplified_text_representation(game, current_player: str, action_mask: np.ndarray,
                                          moves_history: Optional[List[str]] = None, env_ref=None) -> str:
    """Create a simplified text representation suitable for small images."""
    try:
        player_idx = int(str(current_player).split('_')[-1])
    except Exception:
        player_idx = 0
    
    community_cards = getattr(game, "public_cards", [])
    hole_cards = getattr(game.players[player_idx], "hand", [])
    
    # Use tournament chip stacks if available, otherwise fall back to pot contributions
    if env_ref and hasattr(env_ref, 'chip_stacks') and hasattr(env_ref, 'tournament_mode') and env_ref.tournament_mode:
        # Show tournament chip stacks
        chip_info = {f"player_{i}": env_ref.chip_stacks.get(f"player_{i}", 0) for i in range(len(game.players))}
        pot_contributions = {f"player_{i}": int(getattr(p, "in_chips", 0)) for i, p in enumerate(game.players)}
        total_pot = sum(pot_contributions.values())
        info_type = "Tournament Chips"
    else:
        # Fall back to PettingZoo's pot contributions
        chip_info = {f"player_{i}": int(getattr(p, "in_chips", 0)) for i, p in enumerate(game.players)}
        total_pot = sum(chip_info.values())
        info_type = "Pot Contributions"

    folded_players = set()
    for i, p in enumerate(game.players):
        player_name = f"player_{i}"
        
        # Check player status - PettingZoo uses "folded" string for folded players
        player_status = getattr(p, 'status', None)
        if player_status:
            status_str = str(player_status).lower()
            if "folded" in status_str:
                folded_players.add(player_name)

    street = _get_street_from_community_cards(community_cards)
    moves_history = moves_history or []

    out: List[str] = []
    out.append("=== TEXAS HOLD'EM ===")
    out.append("")
    out.append(f"Street: {street.upper()}")
    out.append(f"Current Player: {current_player}")
    out.append(f"Pot: ${total_pot}")
    out.append("")
    
    # Community cards (simple format)
    out.append("Community Cards:")
    if community_cards:
        # Use full suit names for clarity, e.g., "10 Spades"
        cards_str = " ".join([f"{c.rank} {c.suit}" for c in community_cards])
        out.append(f"  {cards_str}")
    else:
        out.append("  (none yet)")
    out.append("")
    
    # Your hand
    out.append("Your Hand:")
    if hole_cards:
        # Use full suit names for clarity, e.g., "3 Hearts"
        hand_str = " ".join([f"{c.rank} {c.suit}" for c in hole_cards])
        out.append(f"  {hand_str}")
    else:
        out.append("  (hidden)")
    out.append("")
    
    # Players info with correct chip information
    out.append(f"Players ({info_type}):")
    for i in range(len(game.players)):
        pn = f"player_{i}"
        chips = chip_info.get(pn, 0)
        if pn in folded_players:
            status = "FOLDED"
        elif pn == current_player:
            status = "ACTING"
        else:
            status = "waiting"
        
        # Show both tournament chips and current bet if in tournament mode
        if env_ref and hasattr(env_ref, 'chip_stacks') and hasattr(env_ref, 'tournament_mode') and env_ref.tournament_mode:
            current_bet = pot_contributions.get(pn, 0)
            total_chips = chips
            # Calculate available chips (total minus what's currently bet)
            available_chips = total_chips - current_bet
            
            # Format chips as integers for cleaner display
            total_display = int(round(total_chips))
            available_display = int(round(available_chips))
            
            if current_bet > 0:
                out.append(f"  {pn}: ${available_display} available (${total_display} total, bet: ${current_bet}) ({status})")
            else:
                out.append(f"  {pn}: ${total_display} total ({status})")
        else:
            out.append(f"  {pn}: ${int(round(chips))} ({status})")
    out.append("")
    
    # Available actions
    out.append("Available Actions:")
    action_names = ["CALL", "RAISE", "FOLD", "CHECK"]
    legal = [action_names[i].lower() for i, ok in enumerate(action_mask) if ok == 1]
    if legal:
        out.append(f"  {' | '.join(legal)}")
    else:
        out.append("  (none)")
    out.append("")
    
    # Recent moves
    if moves_history:
        out.append("Recent Moves:")
        for mv in moves_history[-3:]:
            out.append(f"  {mv}")
    
    return "\n".join(out)

def _generate_video_from_rgb_array(
    frames: List[np.ndarray],
    output_path: str,
    frame_rate: int = 2,
):
    """Generates a video from a list of RGB frames."""
    if not frames:
        return
    
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        # Use specific codec and pixel format for better compatibility
        writer = imageio.get_writer(
            output_path, 
            fps=frame_rate, 
            codec='libx264',         # Standard codec for mp4
            pixelformat='yuv420p',   # Handles odd frame dimensions
            macro_block_size=1       # Suppresses macroblock size warning
        )
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        print(f"GUI video saved to {output_path}")
    except Exception as e:
        print(f"Error generating GUI video: {e}")


def create_poker_table_image(
    game,
    current_player: str,
    action_mask: np.ndarray,
    save_path: str | None,
    table_size: tuple = (1200, 800),
    moves_history: Optional[List[str]] = None,
    env_ref=None,
):
    """Render text representation onto a PNG for vision observations."""
    if save_path is None:
        return
    
    text_content = _create_simplified_text_representation(
        game, current_player, action_mask, moves_history=moves_history, env_ref=env_ref
    )


    img = Image.new("RGB", table_size, (240, 248, 255))
    draw = ImageDraw.Draw(img)
    
    # Better font handling with multiple fallbacks and appropriate sizing
    font = None
    font_size = max(8, min(14, table_size[1] // 50))  # Scale font size based on image height
    
    # Try multiple font options
    font_options = [
        "DejaVuSansMono.ttf",
        "Courier New.ttf", 
        "Courier.ttf",
        "Monaco.ttf",  # macOS
        "Consolas.ttf",  # Windows
        "LiberationMono-Regular.ttf",  # Linux
    ]
    
    for font_name in font_options:
        try:
            font = ImageFont.truetype(font_name, font_size)
            break
        except (IOError, OSError):
            continue
    
    # Final fallback to default font
    if font is None:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()

    # Adjust line spacing based on image size
    y_offset = max(10, table_size[1] // 50)
    line_height = max(10, font_size + 2)
    x_offset = max(10, table_size[0] // 100)
    
    # Split text into lines and render
    lines = text_content.split("\n")
    max_lines = (table_size[1] - 2 * y_offset) // line_height
    
    for i, line in enumerate(lines[:max_lines]):
        if y_offset > table_size[1] - line_height:
            break
        
        # Truncate line if too long for image width
        max_chars = (table_size[0] - 2 * x_offset) // (font_size // 2)
        if len(line) > max_chars:
            line = line[:max_chars-3] + "..."
            
        try:
            draw.text((x_offset, y_offset), line, fill=(0, 0, 0), font=font)
        except Exception:
            # Fallback: try to encode problematic characters
            safe_line = line.encode('ascii', 'replace').decode('ascii')
            draw.text((x_offset, y_offset), safe_line, fill=(0, 0, 0), font=font)
        
        y_offset += line_height

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img.save(save_path)

# ---------------------------------------------------------------------------
# Single-agent environment
# ---------------------------------------------------------------------------

class SingleTexasHoldemEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        env_type: str = "single",
        opponent_policy: str | Callable | None = "random",
        num_players: int = 2,
        table_size_for_render: tuple = (1000, 700),
        game_name_for_adapter: str = "texasholdem",
        observation_mode_for_adapter: str = "text",
        agent_cache_dir_for_adapter: str = "cache/texasholdem/default_run",
        game_specific_config_path_for_adapter: str = "gamingagent/envs/zoo_02_texasholdem/game_env_config.json",
        max_stuck_steps_for_adapter: Optional[int] = 10,
    ):
        super().__init__()
        assert env_type == "single"
        assert num_players >= 2

        self.render_mode = render_mode
        self.num_players = num_players
        self.opponent_policy = opponent_policy
        self.table_size_for_render = table_size_for_render

        self.pz_env = texas_holdem_v4.env(num_players=num_players, render_mode=render_mode)

        self.player_names = [f"player_{i}" for i in range(num_players)]
        self.agent_player = "player_0"
        self.current_player = self.agent_player

        self.action_space = Discrete(4)
        self.observation_space = Box(low=0, high=1, shape=(72,), dtype=np.uint8)

        self.adapter = GymEnvAdapter(
            game_name=game_name_for_adapter,
            observation_mode=observation_mode_for_adapter,
            agent_cache_dir=agent_cache_dir_for_adapter,
            game_specific_config_path=game_specific_config_path_for_adapter,
            max_steps_for_stuck=max_stuck_steps_for_adapter,
        )

        self.num_env_steps = 0
        self.current_reward_last_step = 0.0
        self.cumulative_perf_score = 0.0

    def _default_opponent_move(self, opponent_player: str, opponent_policy):
        if opponent_policy == "random":
            obs = self.pz_env.observe(opponent_player)
            legal_actions = np.where(obs["action_mask"] == 1)[0]
            action = random.choice(legal_actions) if len(legal_actions) > 0 else None
            self.pz_env.step(action)
        else:
            raise NotImplementedError("Only 'random' opponent_policy implemented.")

    def _get_info(self) -> Dict[str, Any]:
        return {
            "num_env_steps": self.num_env_steps,
            "reward_last_step": self.current_reward_last_step,
            "terminations": self.pz_env.terminations.copy(),
            "current_player": self.current_player,
        }

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
        self.adapter.reset_episode(episode_id)

        obs_pz = self.pz_env.observe(self.agent_player)
        action_mask = obs_pz["action_mask"]
        info = self._get_info()

        img_path = text_repr = None
        if self.adapter.observation_mode in {"vision", "both"}:
            img_path = self.adapter._create_agent_observation_path(
                episode_id, self.adapter.current_step_num
            )
            create_poker_table_image(
                self.pz_env.unwrapped.env.game,
                self.current_player,
                action_mask,
                img_path,
                self.table_size_for_render,
                moves_history=[],
                env_ref=self,
            )

        if self.adapter.observation_mode in {"text", "both"}:
            text_repr = _create_simplified_text_representation(
                self.pz_env.unwrapped.env.game,
                self.current_player,
                action_mask,
                moves_history=[],
                env_ref=self,
            )

        agent_obs = self.adapter.create_agent_observation(
            img_path=img_path, text_representation=text_repr, max_memory=max_memory
        )

        if self.render_mode == "human":
            self.render()
        return agent_obs, info

    def _apply_action(self, env_act_idx: Optional[int]):
        self.pz_env.step(env_act_idx)

    def step(
        self,
        agent_action_str: Optional[str],
        thought_process: str = "",
        time_taken_s: float = 0.0,
    ) -> Tuple[Observation, float, bool, bool, Dict[str, Any], float]:
        self.adapter.increment_step()
        env_act_idx = self.adapter.map_agent_action_to_env_action(agent_action_str)

        while (self.current_player != self.agent_player and
               not self.pz_env.terminations[self.agent_player]):
            self._default_opponent_move(self.current_player, self.opponent_policy)
            self.current_player = self.pz_env.agent_selection

        if self.current_player == self.agent_player:
            self._apply_action(env_act_idx)
            self.current_player = self.pz_env.agent_selection

        reward = float(self.pz_env.rewards[self.agent_player])
        self.current_reward_last_step = reward
        self.cumulative_perf_score += reward

        terminated = self.pz_env.terminations[self.agent_player]
        truncated = self.pz_env.truncations[self.agent_player]
        self.num_env_steps += 1

        obs_pz = self.pz_env.observe(self.agent_player)
        action_mask = obs_pz["action_mask"]
        info = self._get_info()

        img_path = text_repr = None
        if self.adapter.observation_mode in {"vision", "both"}:
            img_path = self.adapter._create_agent_observation_path(
                self.adapter.current_episode_id, self.adapter.current_step_num
            )
            create_poker_table_image(
                self.pz_env.unwrapped.env.game,
                self.current_player,
                action_mask,
                img_path,
                self.table_size_for_render,
                moves_history=[f"Last action: {agent_action_str}"] if agent_action_str else [],
                env_ref=self,
            )

        if self.adapter.observation_mode in {"text", "both"}:
            text_repr = _create_simplified_text_representation(
                self.pz_env.unwrapped.env.game,
                self.current_player,
                action_mask,
                moves_history=[f"Last action: {agent_action_str}"] if agent_action_str else [],
                env_ref=self,
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
            perf_score=self.cumulative_perf_score,
            agent_observation=agent_obs,
        )

        if self.render_mode == "human":
            self.render()
        return agent_obs, reward, final_terminated, final_truncated, info, self.cumulative_perf_score

    def render(self) -> Optional[Union[Any, List[Any]]]:
        return self.pz_env.render()

    def close(self):
        self.pz_env.close()
        self.adapter.close_log_file()

# ---------------------------------------------------------------------------
# Multi-agent environment
# ---------------------------------------------------------------------------

class MultiTexasHoldemEnv(SingleTexasHoldemEnv):
    def __init__(
         self,
         *,
         render_mode: Optional[str] = None,
         num_players: int = 2,
         table_size_for_render: tuple = (1200, 800),
         base_cache_dir: str = "cache/texasholdem",
         game_name_for_adapter: str = "texasholdem",
         observation_mode_for_adapter: str = "text",
         game_specific_config_path_for_adapter: str = "gamingagent/envs/zoo_02_texasholdem/game_env_config.json",
         max_stuck_steps_for_adapter: Optional[int] = 10,
         enable_player_elimination: bool = True,
         starting_chips: int = 100,
         max_tournament_hands: Optional[int] = None,
        record_video: bool = True,
        video_frame_rate: int = 2,
     ):
        if not 2 <= num_players <= 10:
            raise ValueError("num_players must be between 2 and 10")

        self.record_video = record_video
        self.video_frame_rate = video_frame_rate
        self.gui_frames: List[np.ndarray] = []
        
        pz_render_mode = render_mode
        if self.record_video:
            pz_render_mode = "rgb_array"
            if render_mode == "human":
                print("Warning: video recording is enabled, so the interactive GUI window will not be displayed. Frames will be saved to a video file instead.")

        super().__init__(
            render_mode=pz_render_mode,
            env_type="single",
            opponent_policy=None,
            num_players=num_players,
            table_size_for_render=table_size_for_render,
            game_name_for_adapter=game_name_for_adapter,
            observation_mode_for_adapter=observation_mode_for_adapter,
            agent_cache_dir_for_adapter=f"{base_cache_dir}/player_0",
            game_specific_config_path_for_adapter=game_specific_config_path_for_adapter,
            max_stuck_steps_for_adapter=max_stuck_steps_for_adapter,
        )

        self.base_cache_dir = base_cache_dir
        self.enable_player_elimination = enable_player_elimination
        self.starting_chips = starting_chips
        # Always enable tournament mode - players keep their chips across hands
        self.tournament_mode = True
        self.max_tournament_hands = int(max_tournament_hands) if max_tournament_hands is not None else None
        self.hands_played = 0
        self._adapters: Dict[str, GymEnvAdapter] = {}
        self.agent_players = set()
        self.eliminated_players = set()
        # Optional mapping from env player -> stable identity/model name
        self.player_identities: Dict[str, str] = {}
        # Track hand number at which each player is eliminated (1-based hand index)
        self.eliminated_on_hand: Dict[str, int] = {}

        for i in range(num_players):
            player_name = f"player_{i}"
            cache_dir = f"{base_cache_dir}/{player_name}"
            self._adapters[player_name] = GymEnvAdapter(
                game_name=f"{player_name}_{game_name_for_adapter}",
                observation_mode=observation_mode_for_adapter,
                agent_cache_dir=cache_dir,
                game_specific_config_path=game_specific_config_path_for_adapter,
                max_steps_for_stuck=max_stuck_steps_for_adapter,
            )
            self.agent_players.add(player_name)

        self.perf_scores = {p: 0.0 for p in self.player_names}
        self.chip_stacks = {p: starting_chips for p in self.player_names}
        self.current_player = "player_0"
        self.step_id = 0
        self.dealer_position = 0
        self.round_number = 1
        self.moves_history: List[str] = []
        # Aggression Factor tracking (tournament-wide)
        # bets = total chips put into pot (sum of bet/raise amounts), raises = count, calls = count
        self.af_stats = {
            p: {"bets_chips": 0, "raises": 0, "calls": 0, "folds": 0}
            for p in self.player_names
        }
        # Track per-hand prev contribution to compute deltas
        self._prev_in_chips = {p: 0 for p in self.player_names}

    def set_player_identities(self, mapping: Dict[str, str]):
        """Provide a mapping of env player name -> stable identity/model name."""
        try:
            self.player_identities = dict(mapping) if mapping else {}
        except Exception:
            self.player_identities = {}

    def tournament_over(self) -> bool:
        if not getattr(self, "tournament_mode", False):
            return False
        if self.max_tournament_hands is not None and self.hands_played >= self.max_tournament_hands:
            return True
        return len(self.get_active_players()) <= 1

    def get_active_players(self) -> List[str]:
        if not self.enable_player_elimination:
            return self.player_names.copy()
        return [p for p in self.player_names if p not in self.eliminated_players]

    def get_agent_players(self) -> List[str]:
        active = self.get_active_players()
        return [p for p in active if p in self.agent_players]

    def _record_move(self, player_name: str, action_str: str, chips_bet: int = 0):
        text = f"{player_name}: {action_str.upper()}"
        if chips_bet > 0:
            text += f" (${chips_bet})"
        self.moves_history.append(text)
        if len(self.moves_history) > 10:
            self.moves_history = self.moves_history[-10:]
        # Update AF counters when explicit chips_bet provided by caller
        # Note: we also compute via in_chips deltas after stepping to be robust
        if hasattr(self, "af_stats") and player_name in self.af_stats:
            a = action_str.lower()
            if "raise" in a:
                self.af_stats[player_name]["raises"] += 1
                if chips_bet > 0:
                    self.af_stats[player_name]["bets_chips"] += int(chips_bet)
            elif "bet" in a:
                if chips_bet > 0:
                    self.af_stats[player_name]["bets_chips"] += int(chips_bet)
            elif "call" in a:
                self.af_stats[player_name]["calls"] += 1
            elif "fold" in a:
                self.af_stats[player_name]["folds"] += 1

    def _get_action_description(self, action_str: Optional[str]) -> str:
        if not action_str: return "UNKNOWN"
        a = action_str.lower()
        if "call" in a: return "CALL"
        if "raise" in a: return "RAISE"
        if "fold" in a: return "FOLD"
        if "check" in a: return "CHECK"
        return action_str.upper()

    def _get_game_info(self) -> Dict[str, Any]:
        return {
            "round_number": self.round_number,
            "active_players": self.get_active_players(),
            "agent_players": self.get_agent_players(),
            "eliminated_players": list(self.eliminated_players),
            "chip_stacks": self.chip_stacks.copy(),
            "dealer_position": self.dealer_position,
            "blinds": {"controlled_by": "pettingzoo"},
            "tournament_mode": bool(self.tournament_mode),
            "hands_played": int(self.hands_played),
            "max_tournament_hands": self.max_tournament_hands,
        }

    def reset(self, *, seed: int | None = None, episode_id: int = 1, **kwargs):
        if self.record_video and self.gui_frames:
            video_path = os.path.join(self.base_cache_dir, "videos", f"episode_{self.current_episode_id}_gui.mp4")
            _generate_video_from_rgb_array(
                frames=self.gui_frames,
                output_path=video_path,
                frame_rate=self.video_frame_rate
            )
        self.gui_frames = []

        for adap in self._adapters.values():
            adap.reset_episode(episode_id)
        super().reset(seed=seed)

        self.step_id = 0
        self.current_episode_id = episode_id
        self.moves_history = []
        # Reset per-hand baseline contributions
        self._prev_in_chips = {p: 0 for p in self.player_names}

        if self.tournament_mode and episode_id > 1:
            # Continue tournament: keep stacks/elims, rotate dealer, next round
            self.perf_scores = {p: 0.0 for p in self.player_names}
            self.round_number += 1
            active = self.get_active_players()
            if active:
                self.dealer_position = (self.dealer_position + 1) % len(active)
        else:
            # Fresh tournament (or non-tournament single hand)
            self.eliminated_players.clear()
            self.eliminated_on_hand.clear()
            self.chip_stacks = {p: self.starting_chips for p in self.player_names}
            self.perf_scores = {p: 0.0 for p in self.player_names}
            self.round_number = 1
            self.dealer_position = 0
            self.hands_played = 0
            # Reset tournament AF stats at start of a new tournament
            self.af_stats = {
                p: {"bets_chips": 0, "raises": 0, "calls": 0, "folds": 0}
                for p in self.player_names
            }

        obs_dict: Dict[str, Observation] = {}
        agent_players = self.get_agent_players()

        for agent_name in agent_players:
            adap = self._adapters[agent_name]
            obs_pz = self.pz_env.observe(agent_name)
            action_mask = obs_pz["action_mask"]
            if self.pz_env.agent_selection != agent_name:
                action_mask = np.zeros_like(action_mask)

            img_path = text_repr = None
            if adap.observation_mode in {"vision", "both"}:
                img_path = adap._create_agent_observation_path(episode_id, 0)
                create_poker_table_image(
                    self.pz_env.unwrapped.env.game,
                    self.pz_env.agent_selection,
                    action_mask,
                    img_path,
                    self.table_size_for_render,
                    moves_history=self.moves_history,
                    env_ref=self,
                )

            if adap.observation_mode in {"text", "both"}:
                text_repr = _create_simplified_text_representation(
                    self.pz_env.unwrapped.env.game,
                    self.pz_env.agent_selection,
                    action_mask,
                    moves_history=self.moves_history,
                    env_ref=self,
                )

            obs_dict[agent_name] = adap.create_agent_observation(
                img_path=img_path, text_representation=text_repr
            )

        if self.record_video:
            frame = self.pz_env.render()
            if frame is not None:
                self.gui_frames.append(frame)

        if self.render_mode == "human":
            self.render()
        return obs_dict, self._get_game_info()

    def step(self, agent_name: str, action_str: Optional[str],
             thought_process: str = "", time_taken_s: float = 0.0):
        if agent_name not in self.agent_players:
            raise ValueError(f"{agent_name} is not an agent player")
        if agent_name not in self.get_active_players():
            raise ValueError(f"{agent_name} is eliminated")

        action_desc = self._get_action_description(action_str)
        self._record_move(agent_name, action_desc)

        game_ended = False
        while not game_ended:
            current_player = self.pz_env.agent_selection
            if not current_player:
                game_ended = True
                break

            active = self.get_active_players()

            # Auto-fold eliminated seats so the hand can progress
            if current_player not in active:
                # Try to play a legal FOLD (2). If action mask disagrees, pick any legal action with "fold" semantics.
                try:
                    # Most implementations: 0=CALL,1=RAISE,2=FOLD,3=CHECK
                    self.pz_env.step(2)
                    if hasattr(self, "af_stats") and current_player in self.af_stats:
                        self.af_stats[current_player]["folds"] += 1
                except Exception:
                    obs_tmp = self.pz_env.observe(current_player)
                    legal = np.where(obs_tmp["action_mask"] == 1)[0]
                    # Choose the last legal as a fallback (often FOLD is legal late in a betting round)
                    chosen = int(legal[-1]) if len(legal) else None
                    self.pz_env.step(chosen)
                    # Best-effort: if chosen likely fold, count it
                    if hasattr(self, "af_stats") and current_player in self.af_stats and chosen == 2:
                        self.af_stats[current_player]["folds"] += 1
                continue  # keep advancing until an active seat or termination
                
            # If it's the acting agent's turn, apply their action
            if current_player == agent_name:
                adap = self._adapters[agent_name]
                adap.increment_step()
                env_act_idx = adap.map_agent_action_to_env_action(action_str)
                try:
                    # Capture pre-step contributions to compute deltas after step
                    try:
                        pre_in_chips = {f"player_{i}": int(getattr(p, "in_chips", 0)) for i, p in enumerate(self.pz_env.unwrapped.env.game.players)}
                    except Exception:
                        pre_in_chips = self._prev_in_chips.copy()
                    self.pz_env.step(env_act_idx)
                except Exception as e:
                    print(f"[ERROR] step failed for {agent_name}: {e}")
                    return self._handle_illegal_action(agent_name)
                # Update AF stats based on in_chips delta and action string
                try:
                    post_in_chips = {f"player_{i}": int(getattr(p, "in_chips", 0)) for i, p in enumerate(self.pz_env.unwrapped.env.game.players)}
                except Exception:
                    post_in_chips = pre_in_chips
                # Save baseline for all players for next step as a fallback
                self._prev_in_chips.update(post_in_chips)
                # Compute how much this agent just contributed
                pre_val = pre_in_chips.get(agent_name, 0)
                post_val = post_in_chips.get(agent_name, pre_val)
                delta = max(0, int(post_val - pre_val))
                # Classify basic action types for AF
                a = (action_str or "").lower()
                if hasattr(self, "af_stats") and agent_name in self.af_stats:
                    if "raise" in a or (delta > 0 and "call" not in a and "check" not in a):
                        # Treat any positive delta without explicit call/check as bet/raise chips
                        self.af_stats[agent_name]["bets_chips"] += int(delta)
                        if "raise" in a:
                            self.af_stats[agent_name]["raises"] += 1
                    if "call" in a:
                        self.af_stats[agent_name]["calls"] += 1
                continue  # performed the agent's move; advance until next active agent or termination

            # If it's another agent's turn, stop here so that caller can let that agent act
            if current_player in self.agent_players:
                break

        self.current_player = self.pz_env.agent_selection
        self.step_id += 1

        rewards = {pn: float(self.pz_env.rewards.get(pn, 0)) for pn in self.player_names}
        terminations = any(self.pz_env.terminations.values())
        truncations = any(self.pz_env.truncations.values())

        for pn in self.player_names:
            self.perf_scores[pn] += rewards[pn]

        # If a hand ended, and we're in tournament mode, settle stacks & eliminations
        if (terminations or truncations) and self.tournament_mode:
            # Settle stacks for this hand (zero-sum across players)
            for pn in self.player_names:
                reward = self.perf_scores.get(pn, 0.0)
            
                chip_change = reward * 2
            
                # Round to be safe, though chip changes should now be integers
                self.chip_stacks[pn] += round(chip_change)
            if self.enable_player_elimination:
                for pn in self.player_names:
                    if self.chip_stacks[pn] <= 0 and pn not in self.eliminated_players:
                        self.eliminated_players.add(pn)
                        # Record hand number at elimination
                        self.eliminated_on_hand[pn] = int(self.hands_played)
                        print(f"[MultiTexasHoldem] {pn} eliminated on hand {self.hands_played}")
            self.hands_played += 1
            if self.tournament_over():
                terminations = True
                truncations = True
                winner_list = self.get_active_players()
                winner = winner_list[0] if len(winner_list) == 1 else "None"
                print(f"[MultiTexasHoldem] Tournament ended. Winner: {winner}")
                # Persist AF metrics at tournament end
                try:
                    os.makedirs(os.path.join(self.base_cache_dir, "metrics"), exist_ok=True)
                    af_out = {}
                    for pn, s in self.af_stats.items():
                        calls = max(0, int(s.get("calls", 0)))
                        bets_chips = max(0, int(s.get("bets_chips", 0)))
                        raises = max(0, int(s.get("raises", 0)))
                        folds = max(0, int(s.get("folds", 0)))
                        af_value = float((bets_chips + raises) / calls) if calls > 0 else float("inf")
                        model_name = self.player_identities.get(pn, "unknown") if hasattr(self, "player_identities") else "unknown"
                        final_chips = int(self.chip_stacks.get(pn, 0))
                        eliminated_on = int(self.eliminated_on_hand.get(pn, 0)) if hasattr(self, "eliminated_on_hand") else 0
                        # Fold percentage: tournament end -> folds/hands_played; early exit -> folds/eliminated_on_hand (if available) else folds/hands_played
                        denom = int(self.hands_played) if self.tournament_over() else (eliminated_on if eliminated_on > 0 else int(self.hands_played))
                        fold_pct = (float(folds) / denom) if denom > 0 else None
                        af_out[pn] = {
                            "bets_chips": bets_chips,
                            "raises": raises,
                            "calls": calls,
                            "folds": folds,
                            "fold_percentage": fold_pct,
                            "aggression_factor": af_value,
                            "model_name": model_name,
                            "final_chips": final_chips,
                            "eliminated_on_hand": eliminated_on
                        }
                        # Also save per-player metric in their cache dir
                        player_metrics_dir = os.path.join(self.base_cache_dir, pn)
                        os.makedirs(player_metrics_dir, exist_ok=True)
                        with open(os.path.join(player_metrics_dir, "af_summary.json"), "w") as fpm:
                            json.dump(af_out[pn], fpm, indent=2)
                    with open(os.path.join(self.base_cache_dir, "metrics", "af_summary.json"), "w") as f:
                        json.dump(af_out, f, indent=2)
                    print("[MultiTexasHoldem] Saved AF metrics to cache.")
                except Exception as e:
                    print(f"[MultiTexasHoldem] Failed to save AF metrics: {e}")

        next_obs: Dict[str, Observation] = {}
        info_for_acting_agent = self._get_game_info()
        for current_agent in self.get_agent_players():
            adap = self._adapters[current_agent]
            obs_pz = self.pz_env.observe(current_agent)
            action_mask = obs_pz["action_mask"]
            if self.pz_env.agent_selection != current_agent:
                action_mask = np.zeros_like(action_mask)

            img_path = text_repr = None
            if adap.observation_mode in {"vision", "both"}:
                img_path = adap._create_agent_observation_path(self.current_episode_id, self.step_id)
                create_poker_table_image(
                    self.pz_env.unwrapped.env.game,
                    self.pz_env.agent_selection,
                    action_mask,
                    img_path,
                    self.table_size_for_render,
                    moves_history=self.moves_history,
                )

            if adap.observation_mode in {"text", "both"}:
                text_repr = _create_simplified_text_representation(
                    self.pz_env.unwrapped.env.game,
                    self.pz_env.agent_selection,
                    action_mask,
                    moves_history=self.moves_history,
                    env_ref=self,
                )

            agent_obs = adap.create_agent_observation(
                img_path=img_path, text_representation=text_repr
            )
            next_obs[current_agent] = agent_obs

            if current_agent == agent_name:
                final_term, final_trunc = adap.verify_termination(
                    agent_obs, terminations, truncations
                )
                adap.log_step_data(
                    agent_action_str=action_str,
                    thought_process=thought_process,
                    reward=rewards[agent_name],
                    info=info_for_acting_agent,
                    terminated=final_term,
                    truncated=final_trunc,
                    time_taken_s=time_taken_s,
                    perf_score=self.perf_scores[agent_name],
                    agent_observation=agent_obs,
                )

        if self.record_video:
            frame = self.pz_env.render()
            if frame is not None:
                self.gui_frames.append(frame)

        if self.render_mode == "human":
            self.render()

        # Auto-end tournament early if only one player remains
        if self.tournament_mode and len(self.get_active_players()) <= 1:
            terminations = True
            truncations = True

        return next_obs, rewards, terminations, truncations, info_for_acting_agent, self.perf_scores.copy()

    def _handle_illegal_action(self, agent_name: str):
        rewards = {pn: 0.0 for pn in self.player_names}
        rewards[agent_name] = -10.0
        self.perf_scores[agent_name] -= 10
        return (
            {agent_name: None},
            rewards,
            True,
            True,
            self._get_game_info(),
            self.perf_scores.copy(),
        )

    def record_episode_results(self, episode_id: int, final_scores: Dict[str, float], total_steps: int):
        for pn in self.get_agent_players():
            if pn in self._adapters:
                adap = self._adapters[pn]
                final_score = final_scores.get(pn, 0.0)
                total_reward = self.perf_scores.get(pn, 0.0)
                adap.record_episode_result(
                    episode_id=episode_id,
                    score=final_score,
                    steps=total_steps,
                    total_reward=total_reward,
                    total_perf_score=total_reward,
                )

    def finalize_run_summaries(self, run_settings: Dict):
        # Attach AF metrics into each agent's summary file under extras.af
        summaries = {}
        for pn in self.get_agent_players():
            if pn in self._adapters:
                summary = self._adapters[pn].finalize_and_save_summary(run_settings)
                # Persist AF per player already saved in step() end; also embed into summary JSON file
                try:
                    agent_summary_path = os.path.join(self._adapters[pn].agent_cache_dir, "gym_run_summary.json")
                    with open(agent_summary_path, "r") as f:
                        data = json.load(f)
                    data.setdefault("extras", {})["aggression_factor"] = {
                        "bets_chips": int(self.af_stats.get(pn, {}).get("bets_chips", 0)),
                        "raises": int(self.af_stats.get(pn, {}).get("raises", 0)),
                        "calls": int(self.af_stats.get(pn, {}).get("calls", 0)),
                        "folds": int(self.af_stats.get(pn, {}).get("folds", 0)),
                        "AF": (float((self.af_stats[pn]["bets_chips"] + self.af_stats[pn]["raises"]) / self.af_stats[pn]["calls"]) if self.af_stats[pn]["calls"] > 0 else float("inf")) if pn in self.af_stats else None,
                        "model_name": self.player_identities.get(pn, "unknown"),
                        "eliminated_on_hand": int(self.eliminated_on_hand.get(pn, 0)) if hasattr(self, "eliminated_on_hand") else 0,
                        "fold_percentage": (float(int(self.af_stats.get(pn, {}).get("folds", 0))) / (int(self.hands_played) if self.tournament_over() else (int(self.eliminated_on_hand.get(pn, 0)) if hasattr(self, "eliminated_on_hand") and int(self.eliminated_on_hand.get(pn, 0)) > 0 else int(self.hands_played)))) if (int(self.hands_played) > 0 or (hasattr(self, "eliminated_on_hand") and int(self.eliminated_on_hand.get(pn, 0)) > 0)) else None
                    }
                    with open(agent_summary_path, "w") as f:
                        json.dump(data, f, indent=2)
                except Exception:
                    pass
                summaries[pn] = summary
        return summaries

    def close(self):
        if self.record_video and self.gui_frames:
            video_path = os.path.join(self.base_cache_dir, "videos", f"episode_{self.current_episode_id}_gui_final.mp4")
            _generate_video_from_rgb_array(
                frames=self.gui_frames,
                output_path=video_path,
                frame_rate=self.video_frame_rate
            )
        self.gui_frames = []
        # Fallback: ensure AF metrics are saved even if tournament end hook was missed
        try:
            if getattr(self, "tournament_mode", False) and hasattr(self, "af_stats") and self.af_stats:
                metrics_dir = os.path.join(self.base_cache_dir, "metrics")
                os.makedirs(metrics_dir, exist_ok=True)
                metrics_path = os.path.join(metrics_dir, "af_summary.json")
                if not os.path.isfile(metrics_path):
                    af_out = {}
                    for pn, s in self.af_stats.items():
                        calls = max(0, int(s.get("calls", 0)))
                        bets_chips = max(0, int(s.get("bets_chips", 0)))
                        raises = max(0, int(s.get("raises", 0)))
                        folds = max(0, int(s.get("folds", 0)))
                        af_value = float((bets_chips + raises) / calls) if calls > 0 else float("inf")
                        model_name = self.player_identities.get(pn, "unknown") if hasattr(self, "player_identities") else "unknown"
                        final_chips = int(self.chip_stacks.get(pn, 0)) if hasattr(self, "chip_stacks") else 0
                        af_out[pn] = {
                            "bets_chips": bets_chips,
                            "raises": raises,
                            "calls": calls,
                            "folds": folds,
                            "aggression_factor": af_value,
                            "model_name": model_name,
                            "final_chips": final_chips
                        }
                        # per-player file
                        player_metrics_dir = os.path.join(self.base_cache_dir, pn)
                        os.makedirs(player_metrics_dir, exist_ok=True)
                        with open(os.path.join(player_metrics_dir, "af_summary.json"), "w") as fpm:
                            json.dump(af_out[pn], fpm, indent=2)
                    with open(metrics_path, "w") as f:
                        json.dump(af_out, f, indent=2)
                    print("[MultiTexasHoldem] Fallback saved AF metrics to cache.")
        except Exception as e:
            print(f"[MultiTexasHoldem] Fallback AF metrics save failed: {e}")
        super().close()
        for adap in self._adapters.values():
            adap.close_log_file()