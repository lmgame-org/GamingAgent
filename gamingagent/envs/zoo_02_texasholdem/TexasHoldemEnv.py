"""Unified Texas Hold'em Gymnasium environments (clean + vision images)

Uses PettingZoo's built-in human renderer. Vision observations are generated
as PNG files (ASCII text rendered into an image) so that modules expecting
`observation.img_path` keep working.
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

def _card_block(rank: str, suit: str) -> List[str]:
    suits = {"Hearts": "♥", "Diamonds": "♦", "Clubs": "♣", "Spades": "♠",
             "H": "♥", "D": "♦", "C": "♣", "S": "♠"}
    symbol = suits.get(suit, suit)
    return [
        "┌─────────┐",
        f"│ {rank:<2}      │",
        "│         │",
        f"│    {symbol}    │",
        "│         │",
        f"│      {rank:>2} │",
        "└─────────┘",
    ]

def _render_card_row(cards: List[Tuple[str, str]]) -> List[str]:
    lines = [""] * 7
    for rank, suit in cards:
        if rank == "?" or suit == "?":
            block = [
                "┌─────────┐",
                "│ ░░░░░░░ │",
                "│ ░░░░░░░ │",
                "│ ░░░?░░░ │",
                "│ ░░░░░░░ │",
                "│ ░░░░░░░ │",
                "└─────────┘",
            ]
        else:
            block = _card_block(rank, suit)
        for i in range(7):
            lines[i] += f"{block[i]} "
    return lines

def _create_text_representation(game, current_player: str, action_mask: np.ndarray,
                                moves_history: Optional[List[str]] = None) -> str:
    player_idx = int(current_player.split('_')[-1])
    community_cards = getattr(game, "public_cards", [])
    hole_cards = getattr(game.players[player_idx], "hand", [])
    chip_stacks = {f"player_{i}": int(p.in_chips) for i, p in enumerate(game.players)}
    total_pot = sum(chip_stacks.values())

    eliminated_players = set()
    for i, p in enumerate(game.players):
        status_name = getattr(p.status, "name", str(p.status))
        if status_name == "FOLDED":
            eliminated_players.add(f"player_{i}")

    street = _get_street_from_community_cards(community_cards)
    moves_history = moves_history or []

    header_width = 122
    out: List[str] = []
    out.append("┌" + "─" * header_width + "┐")
    out.append("│" + "🃏 TEXAS HOLD'EM POKER".center(header_width) + "│")
    out.append("└" + "─" * header_width + "┘\n")

    out.append("┌─ GAME STATUS " + "─" * (header_width - 14) + "┐")
    out.append(
        f"│ Street: {street.upper():<20} Current Player: {current_player:<15} "
        f"💰 Pot: ${total_pot:<10}" + " " * (header_width - 66) + "│"
    )
    out.append("└" + "─" * header_width + "┘\n")

    out.append("┌─ RECENT MOVES " + "─" * (header_width - 15) + "┐")
    recent = moves_history[-3:] if moves_history else ["No moves yet - game starting..."]
    for mv in recent:
        out.append(f"│ {mv:<{header_width - 2}} │")
    for _ in range(3 - len(recent)):
        out.append("│" + " " * header_width + "│")
    out.append("└" + "─" * header_width + "┘\n")

    # Left column
    left: List[str] = []
    left.append("┌─ COMMUNITY CARDS " + "─" * 49 + "┐")
    left.append(f"│   Stage: {street.upper()} ({len(community_cards)}/5 cards)" + " " *
                (68 - 25 - len(street)) + "│")
    left.append("│" + " " * 68 + "│")
    card_slots = [(c.rank, c.suit) for c in community_cards]
    while len(card_slots) < 5:
        card_slots.append(("?", "?"))
    for line in _render_card_row(card_slots):
        left.append(f"│  {line:<66}│")
    left.append("│" + " " * 68 + "│")
    left.append("└" + "─" * 68 + "┘\n")

    left.append("┌─ YOUR HAND " + "─" * 55 + "┐")
    left.append(f"│ Player: {current_player}" + " " * (68 - 10 - len(current_player)) + "│")
    current_chips = chip_stacks.get(current_player, 0)
    left.append(f"│ 💰 Chips: ${current_chips}" + " " * (68 - 12 - len(str(current_chips))) + "│")
    left.append("│" + " " * 68 + "│")
    player_cards = [(c.rank, c.suit) for c in hole_cards]
    while len(player_cards) < 2:
        player_cards.append(("?", "?"))
    for line in _render_card_row(player_cards):
        left.append(f"│  {line:<66}│")
    left.append("│" + " " * 68 + "│")
    left.append("└" + "─" * 68 + "┘")

    # Right column
    right_w = 50
    right: List[str] = []
    num_players = getattr(game, "num_players", len(game.players))
    right.append("┌─ ALL PLAYERS " + "─" * (right_w - 16) + "┐")
    for i in range(num_players):
        pn = f"player_{i}"
        chips = chip_stacks.get(pn, 0)
        status = "💀 ELIMINATED" if pn in eliminated_players else \
                 ("🎯 ACTING" if pn == current_player else "⏳ waiting")
        line = f"│ {pn:<9} | Chips: ${chips:<7} | {status:<10}"
        right.append(f"{line:<{right_w-1}}│")
    right.append("└" + "─" * (right_w - 2) + "┘\n")

    right.append("┌─ AVAILABLE ACTIONS " + "─" * (right_w - 20) + "┐")
    action_names = ["CALL", "RAISE", "FOLD", "CHECK"]
    legal = [action_names[i].lower() for i, ok in enumerate(action_mask) if ok == 1]
    if legal:
        right.append(f"│ ⚡ {' | '.join(legal):<{right_w-5}} │")
    else:
        right.append(f"│ {'No legal actions available':<{right_w-3}} │")
    right.append("│" + " " * (right_w - 2) + "│")
    right.append("│ 📋 RULES: Best 5-card hand wins.              │")
    right.append("└" + "─" * (right_w - 2) + "┘")

    max_len = max(len(left), len(right))
    left += [' ' * 70] * (max_len - len(left))
    right += [' ' * right_w] * (max_len - len(right))
    for l, r in zip(left, right):
        out.append(f"{l}  {r}")

    return "\n".join(out)

def create_poker_table_image(
    game,
    current_player: str,
    action_mask: np.ndarray,
    save_path: str | None,
    table_size: tuple = (1200, 800),
    moves_history: Optional[List[str]] = None,
):
    """Render text representation onto a PNG for vision observations."""
    if save_path is None:
        return
    text_content = _create_text_representation(
        game, current_player, action_mask, moves_history=moves_history
    )

    img = Image.new("RGB", table_size, (240, 248, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", 12)
    except IOError:
        font = ImageFont.load_default()

    y_offset = 20
    line_height = 16
    for line in text_content.split("\n"):
        draw.text((20, y_offset), line, fill=(0, 0, 0), font=font)
        y_offset += line_height
        if y_offset > table_size[1] - 40:
            break

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
            )

        if self.adapter.observation_mode in {"text", "both"}:
            text_repr = _create_text_representation(
                self.pz_env.unwrapped.env.game,
                self.current_player,
                action_mask,
                moves_history=[],
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
            )

        if self.adapter.observation_mode in {"text", "both"}:
            text_repr = _create_text_representation(
                self.pz_env.unwrapped.env.game,
                self.current_player,
                action_mask,
                moves_history=[f"Last action: {agent_action_str}"] if agent_action_str else [],
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
        starting_chips: int = 1000,
        big_blind: int = 20,
        small_blind: int = 10,
    ):
        if not 2 <= num_players <= 10:
            raise ValueError("num_players must be between 2 and 10")

        super().__init__(
            render_mode=render_mode,
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
        self.big_blind = big_blind
        self.small_blind = small_blind

        self._adapters: Dict[str, GymEnvAdapter] = {}
        self.agent_players = set()
        self.eliminated_players = set()

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

    def get_active_players(self) -> List[str]:
        if not self.enable_player_elimination:
            return self.player_names.copy()
        return [p for p in self.player_names if p not in self.eliminated_players]

    def get_agent_players(self) -> List[str]:
        active = self.get_active_players()
        return [p for p in active if p in self.agent_players]

    def _update_chip_stacks(self):
        if hasattr(self.pz_env, "env") and hasattr(self.pz_env.env, "chips"):
            for i, pn in enumerate(self.player_names):
                if i < len(self.pz_env.env.chips):
                    self.chip_stacks[pn] = self.pz_env.env.chips[i]
        else:
            for pn in self.player_names:
                if pn in self.pz_env.rewards:
                    r = self.pz_env.rewards[pn]
                    if r != 0:
                        self.chip_stacks[pn] += r
                        self.perf_scores[pn] += r
        if self.enable_player_elimination:
            for pn in self.player_names:
                if self.chip_stacks[pn] <= 0 and pn not in self.eliminated_players:
                    self.eliminated_players.add(pn)
                    print(f"[MultiTexasHoldem] {pn} eliminated")

    def _record_move(self, player_name: str, action_str: str, chips_bet: int = 0):
        text = f"{player_name}: {action_str.upper()}"
        if chips_bet > 0:
            text += f" (${chips_bet})"
        self.moves_history.append(text)
        if len(self.moves_history) > 10:
            self.moves_history = self.moves_history[-10:]

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
            "blinds": {"small": self.small_blind, "big": self.big_blind},
        }

    def reset(self, *, seed: int | None = None, episode_id: int = 1, **kwargs):
        for adap in self._adapters.values():
            adap.reset_episode(episode_id)
        super().reset(seed=seed)

        self.step_id = 0
        self.current_episode_id = episode_id
        self.eliminated_players.clear()
        self.chip_stacks = {p: self.starting_chips for p in self.player_names}
        self.perf_scores = {p: 0.0 for p in self.player_names}
        self.round_number = 1
        self.dealer_position = 0
        self.moves_history = []

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
                )

            if adap.observation_mode in {"text", "both"}:
                text_repr = _create_text_representation(
                    self.pz_env.unwrapped.env.game,
                    self.pz_env.agent_selection,
                    action_mask,
                    moves_history=self.moves_history,
                )

            obs_dict[agent_name] = adap.create_agent_observation(
                img_path=img_path, text_representation=text_repr
            )

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
            if not current_player or current_player not in self.get_active_players():
                game_ended = True
                break
            if current_player == agent_name:
                adap = self._adapters[agent_name]
                adap.increment_step()
                env_act_idx = adap.map_agent_action_to_env_action(action_str)
                try:
                    self.pz_env.step(env_act_idx)
                except Exception as e:
                    print(f"[ERROR] step failed for {agent_name}: {e}")
                    return self._handle_illegal_action(agent_name)
                break
            elif current_player in self.agent_players:
                break
            else:
                self.pz_env.step(None)

        self.current_player = self.pz_env.agent_selection
        self.step_id += 1
        self._update_chip_stacks()

        rewards = {pn: float(self.pz_env.rewards.get(pn, 0)) for pn in self.player_names}
        terminations = any(self.pz_env.terminations.values())
        truncations = any(self.pz_env.truncations.values())

        active_players = self.get_active_players()
        if len(active_players) <= 1:
            terminations = True
            print(f"[MultiTexasHoldem] Tournament ended. Winner: {active_players[0] if active_players else 'None'}")

        for pn in self.player_names:
            self.perf_scores[pn] += rewards[pn]

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
                text_repr = _create_text_representation(
                    self.pz_env.unwrapped.env.game,
                    self.pz_env.agent_selection,
                    action_mask,
                    moves_history=self.moves_history,
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

        if self.render_mode == "human":
            self.render()

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
        return {
            pn: self._adapters[pn].finalize_and_save_summary(run_settings)
            for pn in self.get_agent_players()
            if pn in self._adapters
        }

    def close(self):
        super().close()
        for adap in self._adapters.values():
            adap.close_log_file()
