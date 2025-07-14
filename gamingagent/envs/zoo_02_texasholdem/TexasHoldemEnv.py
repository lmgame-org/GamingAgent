"""Unified Texas Hold'em Gymnasium environments
==============================================
This module provides **SingleTexasHoldemEnv** and **MultiTexasHoldemEnv** that
share the same helper utilities and rendering code. `MultiTexasHoldemEnv`
sub-classes `SingleTexasHoldemEnv`, following the same pattern as TicTacToe.

Key points
----------
* Shared helpers – action lookup table, observation conversion, and text
  representation now live at module level so both envs reuse them.
* Adapters everywhere – both single- and multi-agent variants use
  `GymEnvAdapter` exclusively for agent I/O, episode bookkeeping and logging.
* Minimal surface-area change – public APIs remain exactly the same structure
  as TicTacToe implementation.
"""
from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import gymnasium as gym
import numpy as np
import pygame
from PIL import Image, ImageDraw, ImageFont
from gymnasium.spaces import Box, Discrete
from pettingzoo.classic import texas_holdem_v4

from gamingagent.envs.gym_env_adapter import GymEnvAdapter
from gamingagent.modules.core_module import Observation

# Action lookup table for Texas Hold'em
ACTION_LOOKUP: Dict[int, str] = {
    0: "call",
    1: "raise", 
    2: "fold",
    3: "check"
}

# Card mapping for observation parsing
CARD_SUITS = ["Spades", "Hearts", "Diamonds", "Clubs"]
CARD_RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]

def _parse_cards_from_observation(obs_vector: np.ndarray) -> Tuple[List[str], List[str]]:
    """Parse hole cards and community cards from observation vector.
    
    According to PettingZoo docs, the first 52 bits represent cards that the 
    current player can see (both hole cards and community cards).
    We cannot easily distinguish between hole and community cards from 
    observation alone, so we make reasonable assumptions.
    
    Returns:
        Tuple of (hole_cards, community_cards) as lists of card strings
    """
    visible_cards = []
    
    # Parse first 52 bits for all visible cards
    for i in range(52):
        if obs_vector[i] == 1:
            suit_idx = i // 13
            rank_idx = i % 13
            card_str = f"{CARD_RANKS[rank_idx]} {CARD_SUITS[suit_idx]}"
            visible_cards.append(card_str)
    
    # In Texas Hold'em, players see exactly 2 hole cards + 0-5 community cards
    # Total visible should be 2 (pre-flop) to 7 (river)
    if len(visible_cards) <= 2:
        # Pre-flop: only hole cards visible
        hole_cards = visible_cards
        community_cards = []
    elif len(visible_cards) <= 7:
        # Post-flop: assume first 2 are hole cards, rest are community
        hole_cards = visible_cards[:2]
        community_cards = visible_cards[2:]
    else:
        # Fallback: something is wrong, but try to make sense of it
        hole_cards = visible_cards[:2] if len(visible_cards) >= 2 else visible_cards
        community_cards = visible_cards[2:] if len(visible_cards) > 2 else []
    
    return hole_cards, community_cards

def _get_street_from_community_cards(community_cards: List[str]) -> str:
    """Determine current street based on number of community cards."""
    num_community = len(community_cards)
    if num_community == 0:
        return "pre-flop"
    elif num_community == 3:
        return "flop"
    elif num_community == 4:
        return "turn"
    elif num_community == 5:
        return "river"
    else:
        # Fallback for unexpected states
        if num_community < 3:
            return "pre-flop"
        elif num_community > 5:
            return "river"
        else:
            return "unknown"

def _parse_betting_info(obs_vector: np.ndarray) -> Dict[str, int]:
    """Parse betting information from observation vector.
    
    According to PettingZoo docs: bits 52-71 encode chips raised this street,
    with 4 bits per round (5 bits each for rounds 1-4).
    """
    betting_info = {}
    
    # Chips raised in each round (bits 52-71, 5 bits per round as per docs)
    for round_idx in range(4):
        start_bit = 52 + round_idx * 5
        
        # Convert binary to decimal for chips raised (5 bits can represent 0-31)
        chips_raised = 0
        for i in range(5):
            bit_idx = start_bit + i
            if bit_idx < len(obs_vector) and obs_vector[bit_idx] == 1:
                chips_raised += 2 ** i
        
        street_names = ["pre-flop", "flop", "turn", "river"]
        street_name = street_names[round_idx] if round_idx < len(street_names) else f"round_{round_idx + 1}"
        betting_info[f"{street_name}_chips"] = chips_raised
    
    return betting_info

def _create_text_representation(obs_vector: np.ndarray, action_mask: np.ndarray, current_player: str) -> str:
    """Create comprehensive text representation of Texas Hold'em state."""
    hole_cards, community_cards = _parse_cards_from_observation(obs_vector)
    street = _get_street_from_community_cards(community_cards)
    betting_info = _parse_betting_info(obs_vector)
    
    # Create board representation
    lines = []
    lines.append("=== TEXAS HOLD'EM GAME STATE ===")
    lines.append("")
    
    # Game phase
    lines.append(f"Current Street: {street.upper()}")
    lines.append(f"Current Player: {current_player}")
    lines.append("")
    
    # Hole cards
    if hole_cards:
        lines.append(f"Your Hole Cards: {', '.join(hole_cards)}")
    else:
        lines.append("Your Hole Cards: [Not visible]")
    
    # Community cards with proper formatting
    if community_cards:
        lines.append(f"Community Cards ({len(community_cards)}/5): {', '.join(community_cards)}")
        # Add visual spacing for community cards
        if len(community_cards) == 3:
            lines.append("                   [FLOP]")
        elif len(community_cards) == 4:
            lines.append("                   [TURN]")
        elif len(community_cards) == 5:
            lines.append("                   [RIVER]")
    else:
        lines.append("Community Cards: [None dealt yet - PRE-FLOP]")
    
    lines.append("")
    
    # Betting information with better formatting
    lines.append("Betting Information:")
    total_pot = 0
    for round_name, chips in betting_info.items():
        total_pot += chips
        if chips > 0:
            lines.append(f"  {round_name.replace('_', ' ').title()}: {chips} chips")
    
    lines.append(f"  Current Pot Size: {total_pot} chips")
    lines.append("")
    
    # Texas Hold'em rules reminder
    lines.append("TEXAS HOLD'EM RULES REMINDER:")
    lines.append("• Make the best 5-card hand from your 2 hole cards + 5 community cards")
    lines.append("• You can use 0, 1, or 2 of your hole cards")
    lines.append("• CALL = match current bet, RAISE = increase bet, FOLD = give up hand, CHECK = pass (if no bet)")
    lines.append("")
    
    # Legal actions with clearer formatting
    legal_actions = []
    action_names = ["CALL", "RAISE", "FOLD", "CHECK"]
    action_descriptions = [
        "match the current bet",
        "increase the current bet", 
        "give up your hand",
        "pass action (no bet required)"
    ]
    
    for i, is_legal in enumerate(action_mask):
        if is_legal == 1:
            legal_actions.append(f"{action_names[i].lower()} (ID: {i}) - {action_descriptions[i]}")
    
    lines.append("LEGAL ACTIONS:")
    for action in legal_actions:
        lines.append(f"  • {action}")
    
    lines.append("")
    lines.append("CRITICAL: Only use the legal actions listed above!")
    lines.append("Taking an illegal action will cause immediate game termination with penalty!")
    lines.append("")
    lines.append("Response format: Use action names like 'call', 'raise', 'fold', or 'check'")
    
    return "\n".join(lines)

def create_poker_table_image(
    obs_vector: np.ndarray,
    action_mask: np.ndarray,
    current_player: str,
    save_path: str | None,
    table_size: tuple = (800, 600),
    perf_score: Optional[float] = None,
    action_taken_str: Optional[str] = None,
):
    """Create a visual representation of the poker table."""
    if save_path is None:
        return
    
    # Parse game state
    hole_cards, community_cards = _parse_cards_from_observation(obs_vector)
    street = _get_street_from_community_cards(community_cards)
    betting_info = _parse_betting_info(obs_vector)
    
    # Create image
    img = Image.new("RGB", table_size, (0, 100, 0))  # Green table
    draw = ImageDraw.Draw(img)
    
    # Load fonts
    try:
        title_font = ImageFont.truetype("arial.ttf", 24)
        card_font = ImageFont.truetype("arial.ttf", 18)
        info_font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        title_font = ImageFont.load_default()
        card_font = ImageFont.load_default()
        info_font = ImageFont.load_default()
    
    # Draw table ellipse
    table_margin = 50
    table_rect = (table_margin, table_margin + 50, table_size[0] - table_margin, table_size[1] - table_margin)
    draw.ellipse(table_rect, fill=(0, 150, 0), outline=(139, 69, 19), width=8)
    
    # Title
    draw.text((table_size[0]//2 - 100, 10), "TEXAS HOLD'EM", fill="white", font=title_font)
    
    # Current player indicator
    draw.text((10, 40), f"Current Player: {current_player}", fill="yellow", font=info_font)
    
    # Street indicator
    draw.text((10, 60), f"Street: {street.upper()}", fill="white", font=info_font)
    
    # Community cards area (center of table)
    community_y = table_size[1] // 2 - 40
    community_x_start = table_size[0] // 2 - 150
    
    # Draw community cards
    for i, card in enumerate(community_cards):
        card_x = community_x_start + i * 65
        card_rect = (card_x, community_y, card_x + 60, community_y + 80)
        draw.rectangle(card_rect, fill="white", outline="black", width=2)
        
        # Draw card text
        card_text = card.replace(" ", "\n")
        draw.text((card_x + 5, community_y + 5), card_text, fill="black", font=card_font)
    
    # Draw empty slots for remaining community cards
    for i in range(len(community_cards), 5):
        card_x = community_x_start + i * 65
        card_rect = (card_x, community_y, card_x + 60, community_y + 80)
        draw.rectangle(card_rect, fill="gray", outline="black", width=2)
    
    # Player positions (simplified - just show 2 players)
    player_positions = [
        (table_size[0] // 2 - 100, table_size[1] - 150),  # Player 0 (bottom)
        (table_size[0] // 2 - 100, 100),  # Player 1 (top)
    ]
    
    # Draw players
    for i, (px, py) in enumerate(player_positions):
        player_name = f"player_{i}"
        is_current = (current_player == player_name)
        
        # Player area
        player_rect = (px, py, px + 200, py + 60)
        color = "yellow" if is_current else "lightgray"
        draw.rectangle(player_rect, fill=color, outline="black", width=2)
        
        # Player name
        draw.text((px + 5, py + 5), f"Player {i}", fill="black", font=info_font)
        
        # Hole cards (only show for current player in actual game)
        if i == 0 and hole_cards:  # Show hole cards for player 0
            for j, card in enumerate(hole_cards[:2]):
                card_x = px + 10 + j * 45
                card_y = py + 25
                card_rect = (card_x, card_y, card_x + 40, card_y + 30)
                draw.rectangle(card_rect, fill="white", outline="black", width=1)
                draw.text((card_x + 2, card_y + 2), card[:2], fill="black", font=card_font)
        else:
            # Show card backs for other players
            for j in range(2):
                card_x = px + 10 + j * 45
                card_y = py + 25
                card_rect = (card_x, card_y, card_x + 40, card_y + 30)
                draw.rectangle(card_rect, fill="red", outline="black", width=1)
                draw.text((card_x + 15, card_y + 10), "?", fill="white", font=card_font)
    
    # Pot information
    total_pot = sum(betting_info.values())
    pot_text = f"Pot: {total_pot} chips"
    draw.text((table_size[0] // 2 - 50, community_y + 100), pot_text, fill="white", font=info_font)
    
    # Legal actions
    legal_actions = []
    action_names = ["Call", "Raise", "Fold", "Check"]
    for i, is_legal in enumerate(action_mask):
        if is_legal == 1:
            legal_actions.append(action_names[i])
    
    actions_text = f"Legal Actions: {', '.join(legal_actions)}"
    draw.text((10, table_size[1] - 40), actions_text, fill="white", font=info_font)
    
    # Performance and action annotations
    if perf_score is not None:
        draw.text((table_size[0] - 150, 10), f"Performance: {perf_score:+.1f}", fill="cyan", font=info_font)
    if action_taken_str is not None:
        draw.text((table_size[0] - 200, table_size[1] - 20), f"Last Action: {action_taken_str}", fill="cyan", font=info_font)
    
    # Save the image
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img.save(save_path)

# ──────────────────────────────────────────────────────────────────────────────
# Single-agent environment
# ──────────────────────────────────────────────────────────────────────────────

class SingleTexasHoldemEnv(gym.Env):
    """Gym wrapper around PettingZoo Texas Hold'em with configurable modes.

    Modes
    -----
    * **single** – agent controls *player_0*; *player_1* is a pluggable
      opponent policy (random by default).
    """

    metadata = {"render_modes": ["human", "rgb_array", "raw"], "render_fps": 4}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        env_type: str = "single",
        opponent_policy: str | Callable | None = "random",
        num_players: int = 2,
        table_size_for_render: tuple = (800, 600),
        # Adapter plumbing
        game_name_for_adapter: str = "texasholdem",
        observation_mode_for_adapter: str = "text",
        agent_cache_dir_for_adapter: str = "cache/texasholdem/default_run",
        game_specific_config_path_for_adapter: str = "gamingagent/envs/zoo_02_texasholdem/game_env_config.json",
        max_stuck_steps_for_adapter: Optional[int] = 10,
    ):
        super().__init__()

        assert env_type in {"single"}, "env_type must be 'single'"
        assert num_players >= 2, "num_players must be at least 2"
        
        self.env_type = env_type
        self.render_mode = render_mode
        self.num_players = num_players
        self.opponent_policy = opponent_policy
        self.table_size_for_render = table_size_for_render

        # Underlying PettingZoo env
        self.pz_env = texas_holdem_v4.env(num_players=num_players)
        
        # Player names (PettingZoo uses player_0, player_1, etc.)
        self.player_names = [f"player_{i}" for i in range(num_players)]
        self.agent_player = "player_0"  # Our agent controls player_0
        
        self.current_player = self.agent_player  # default

        # Rendering helpers
        self.window: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None

        # Gym-style spaces
        self.action_space = Discrete(4)  # Call, Raise, Fold, Check
        self.observation_space = Box(low=0, high=1, shape=(72,), dtype=np.uint8)

        # Adapter initialisation
        self.adapter = GymEnvAdapter(
            game_name=game_name_for_adapter,
            observation_mode=observation_mode_for_adapter,
            agent_cache_dir=agent_cache_dir_for_adapter,
            game_specific_config_path=game_specific_config_path_for_adapter,
            max_steps_for_stuck=max_stuck_steps_for_adapter,
        )

        # Episode bookkeeping
        self.num_env_steps: int = 0
        self.current_reward_last_step: float = 0.0
        self.cumulative_perf_score: float = 0.0  # +chips won, -chips lost

    def _default_opponent_move(self, opponent_player: str, opponent_policy):
        """Execute opponent move based on policy."""
        if opponent_policy == "random":
            obs = self.pz_env.observe(opponent_player)
            legal_actions = np.where(obs["action_mask"] == 1)[0]
            if len(legal_actions) > 0:
                action = random.choice(legal_actions)
                self.pz_env.step(action)
            else:
                self.pz_env.step(None)
        else:
            raise NotImplementedError(
                f"Opponent policy '{opponent_policy}' not implemented. "
                "Use 'random' or implement your own."
            )

    def _get_info(self) -> Dict[str, Any]:
        return {
            "num_env_steps": self.num_env_steps,
            "reward_last_step": self.current_reward_last_step,
            "terminations": self.pz_env.terminations.copy(),
            "current_player": self.current_player,
        }

    # -------------------------------------------------------------------------
    # Gym API
    # -------------------------------------------------------------------------

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

        # Get initial observation
        obs_pz = self.pz_env.observe(self.agent_player)
        obs_vector = obs_pz["observation"]
        action_mask = obs_pz["action_mask"]

        info = self._get_info()

        # Create observations
        img_path = text_repr = None
        if self.adapter.observation_mode in {"vision", "both"}:
            img_path = self.adapter._create_agent_observation_path(
                episode_id, self.adapter.current_step_num
            )
            create_poker_table_image(
                obs_vector, action_mask, self.current_player, img_path, 
                self.table_size_for_render, self.cumulative_perf_score
            )
            
        if self.adapter.observation_mode in {"text", "both"}:
            text_repr = _create_text_representation(
                obs_vector, action_mask, self.current_player
            )

        agent_obs = self.adapter.create_agent_observation(
            img_path=img_path, text_representation=text_repr, max_memory=max_memory
        )

        if self.render_mode == "human":
            self.render()
        return agent_obs, info

    def _apply_action(self, env_act_idx: Optional[int]):
        """Helper to apply a mapped Gym action to the current agent."""
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
            # Process turns until it's our agent's turn or game ends
            while (self.current_player != self.agent_player and 
                   not self.pz_env.terminations[self.agent_player]):
                # Opponent's turn
                self._default_opponent_move(self.current_player, self.opponent_policy)
                self.current_player = self.pz_env.agent_selection

            # Apply our agent's action if it's their turn
            if self.current_player == self.agent_player:
                self._apply_action(env_act_idx)
                self.current_player = self.pz_env.agent_selection
        else:
            raise NotImplementedError(f"Environment type '{self.env_type}' not implemented.")

        # Get reward and termination info from our agent's perspective
        reward = float(self.pz_env.rewards[self.agent_player])
        self.current_reward_last_step = reward
        self.cumulative_perf_score += reward

        terminated = self.pz_env.terminations[self.agent_player]
        truncated = self.pz_env.truncations[self.agent_player]
        self.num_env_steps += 1

        # Get next observation
        obs_pz = self.pz_env.observe(self.agent_player)
        obs_vector = obs_pz["observation"]
        action_mask = obs_pz["action_mask"]

        info = self._get_info()

        # Create observations
        img_path = text_repr = None
        if self.adapter.observation_mode in {"vision", "both"}:
            img_path = self.adapter._create_agent_observation_path(
                self.adapter.current_episode_id, self.adapter.current_step_num
            )
            create_poker_table_image(
                obs_vector, action_mask, self.current_player, img_path, 
                self.table_size_for_render, self.cumulative_perf_score, agent_action_str
            )
            
        if self.adapter.observation_mode in {"text", "both"}:
            text_repr = _create_text_representation(
                obs_vector, action_mask, self.current_player
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

    # Rendering methods
    def _render_frame_rgb(self) -> Optional[np.ndarray]:
        """Render the current game state as an RGB array."""
        obs_pz = self.pz_env.observe(self.agent_player)
        obs_vector = obs_pz["observation"]
        action_mask = obs_pz["action_mask"]
        
        temp_path = os.path.join(
            self.adapter.agent_cache_dir, "_temp_render.png"
        )
        create_poker_table_image(
            obs_vector, action_mask, self.current_player, temp_path, 
            self.table_size_for_render, self.cumulative_perf_score
        )
        
        if os.path.exists(temp_path):
            arr = np.array(Image.open(temp_path).convert("RGB"))
            os.remove(temp_path)
            return arr
        return None

    def _render_frame(self):
        """Render the current game state in a pygame window."""
        rgb = self._render_frame_rgb()
        if rgb is None:
            return
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.table_size_for_render)
            pygame.display.set_caption("Texas Hold'em")
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
        self.window.blit(surf, (0, 0))
        pygame.event.pump()
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def render(self) -> Optional[Union[Any, List[Any]]]:
        if self.render_mode == "human":
            self._render_frame()
            return None
        elif self.render_mode == "rgb_array":
            return self._render_frame_rgb()
        elif self.render_mode == "raw":
            obs_pz = self.pz_env.observe(self.agent_player)
            return obs_pz["observation"]
        return None

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
        self.adapter.close_log_file()

# ──────────────────────────────────────────────────────────────────────────────
# Multi-agent environment (inherits from the single-agent implementation)
# ──────────────────────────────────────────────────────────────────────────────

class MultiTexasHoldemEnv(SingleTexasHoldemEnv):
    """Two-model controller that reuses the single-agent core."""

    def __init__(
        self,
        *,
        render_mode: Optional[str] = None,
        num_players: int = 2,
        table_size_for_render: tuple = (800, 600),
        p1_cache: str = "cache/texasholdem/p1",
        p2_cache: str = "cache/texasholdem/p2",
        game_name_for_adapter: str = "texasholdem",
        observation_mode_for_adapter: str = "text",
        game_specific_config_path_for_adapter: str = "gamingagent/envs/zoo_02_texasholdem/game_env_config.json",
        max_stuck_steps_for_adapter: Optional[int] = 10,
    ):
        super().__init__(
            render_mode=render_mode,
            env_type="single",
            opponent_policy=None,
            num_players=num_players,
            table_size_for_render=table_size_for_render,
            game_name_for_adapter=game_name_for_adapter,
            observation_mode_for_adapter=observation_mode_for_adapter,
            agent_cache_dir_for_adapter=p1_cache,
            game_specific_config_path_for_adapter=game_specific_config_path_for_adapter,
            max_stuck_steps_for_adapter=max_stuck_steps_for_adapter,
        )

        # Create separate adapters for each player
        self.adapter_p1 = GymEnvAdapter(
            game_name="p1_texasholdem",
            observation_mode=observation_mode_for_adapter,
            agent_cache_dir=p1_cache,
            game_specific_config_path=game_specific_config_path_for_adapter,
            max_steps_for_stuck=max_stuck_steps_for_adapter,
        )
        self.adapter_p2 = GymEnvAdapter(
            game_name="p2_texasholdem", 
            observation_mode=observation_mode_for_adapter,
            agent_cache_dir=p2_cache,
            game_specific_config_path=game_specific_config_path_for_adapter,
            max_steps_for_stuck=max_stuck_steps_for_adapter,
        )

        self._adapters = {"player_0": self.adapter_p1, "player_1": self.adapter_p2}
        self.perf_scores = {"player_0": 0.0, "player_1": 0.0}
        self.current_player = "player_0"  # default
        self.step_id = 0

    def reset(self, *, seed: int | None = None, episode_id: int = 1, **kwargs):
        for adap in self._adapters.values():
            adap.reset_episode(episode_id)

        super().reset(seed=seed)

        self.step_id = 0
        self.current_episode_id = episode_id
        self.perf_scores = {"player_0": 0.0, "player_1": 0.0}

        obs_dict = {}
        for agent_name, adap in self._adapters.items():
            obs_pz = self.pz_env.observe(agent_name)
            obs_vector = obs_pz["observation"]
            action_mask = obs_pz["action_mask"]

            # Only show legal actions for the current player
            if self.pz_env.agent_selection != agent_name:
                action_mask = np.zeros_like(action_mask)

            # Create observations
            img_path = text_repr = None
            if adap.observation_mode in {"vision", "both"}:
                img_path = adap._create_agent_observation_path(episode_id, 0)
                create_poker_table_image(
                    obs_vector, action_mask, self.pz_env.agent_selection, img_path, 
                    self.table_size_for_render
                )
                
            if adap.observation_mode in {"text", "both"}:
                text_repr = _create_text_representation(
                    obs_vector, action_mask, self.pz_env.agent_selection
                )

            obs_dict[agent_name] = adap.create_agent_observation(
                img_path=img_path, text_representation=text_repr
            )

        if self.render_mode == "human":
            self.render()
        return obs_dict, {}

    def step(self, agent_name: str, action_str: Optional[str]):
        assert agent_name == self.pz_env.agent_selection, (
            f"It is {self.pz_env.agent_selection}'s turn, not {agent_name}")
        
        adap = self._adapters[agent_name]
        env_act_idx = adap.map_agent_action_to_env_action(action_str)
        
        try:
            self._apply_action(env_act_idx)
        except Exception as e:
            print(f"[ERROR] Step failed for agent {agent_name}: {e}")
            # Set punishment for illegal move
            rewards = {name: 0.0 for name in self.player_names}
            rewards[agent_name] = -1.0
            self.perf_scores[agent_name] -= 1
            
            return (
                {},  # obs
                rewards,
                True,  # terminations
                True,  # truncations
                {},    # info
                self.perf_scores.copy(),
            )

        self.current_player = self.pz_env.agent_selection
        self.step_id += 1

        rewards = {name: float(self.pz_env.rewards[name]) for name in self.player_names}
        terminations = any(self.pz_env.terminations.values())
        truncations = any(self.pz_env.truncations.values())

        # Update performance scores
        for player_name in self.player_names:
            self.perf_scores[player_name] += rewards[player_name]

        next_obs = {}
        for agent_name, adap in self._adapters.items():
            obs_pz = self.pz_env.observe(agent_name)
            obs_vector = obs_pz["observation"]
            action_mask = obs_pz["action_mask"]

            # Only show legal actions for the current player
            if self.pz_env.agent_selection != agent_name:
                action_mask = np.zeros_like(action_mask)

            # Create observations
            img_path = text_repr = None
            if adap.observation_mode in {"vision", "both"}:
                img_path = adap._create_agent_observation_path(self.current_episode_id, self.step_id)
                create_poker_table_image(
                    obs_vector, action_mask, self.pz_env.agent_selection, img_path, 
                    self.table_size_for_render, self.perf_scores.get(agent_name, 0), action_str
                )
                
            if adap.observation_mode in {"text", "both"}:
                text_repr = _create_text_representation(
                    obs_vector, action_mask, self.pz_env.agent_selection
                )

            next_obs[agent_name] = adap.create_agent_observation(
                img_path=img_path, text_representation=text_repr
            )

        if self.render_mode == "human":
            self.render()
        return next_obs, rewards, terminations, truncations, {}, self.perf_scores.copy()

    def close(self):
        super().close()
        for adap in self._adapters.values():
            adap.close_log_file()
