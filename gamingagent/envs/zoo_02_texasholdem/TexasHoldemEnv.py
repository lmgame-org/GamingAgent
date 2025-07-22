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
    table_size: tuple = (1000, 700),  # Increased default size
    perf_score: Optional[float] = None,
    action_taken_str: Optional[str] = None,
):
    """Create a comprehensive visual representation of the poker table."""
    if save_path is None:
        return
    
    # Parse game state
    hole_cards, community_cards = _parse_cards_from_observation(obs_vector)
    street = _get_street_from_community_cards(community_cards)
    betting_info = _parse_betting_info(obs_vector)
    
    # Create image with larger canvas
    img = Image.new("RGB", table_size, (20, 50, 20))  # Dark green background
    draw = ImageDraw.Draw(img)
    
    # Load fonts with fallback
    try:
        title_font = ImageFont.truetype("arial.ttf", 28)
        card_font = ImageFont.truetype("arial.ttf", 16)
        info_font = ImageFont.truetype("arial.ttf", 14)
        small_font = ImageFont.truetype("arial.ttf", 12)
        large_font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        title_font = ImageFont.load_default()
        card_font = ImageFont.load_default()
        info_font = ImageFont.load_default()
        small_font = ImageFont.load_default()
        large_font = ImageFont.load_default()
    
    # Enhanced table design
    table_margin = 80
    table_center_x, table_center_y = table_size[0] // 2, table_size[1] // 2
    table_width, table_height = table_size[0] - 2 * table_margin, table_size[1] - 2 * table_margin
    
    # Draw outer table ring (felt border)
    outer_table = (table_margin - 20, table_margin - 20, 
                   table_size[0] - table_margin + 20, table_size[1] - table_margin + 20)
    draw.ellipse(outer_table, fill=(139, 69, 19), outline=(101, 67, 33), width=8)
    
    # Draw main table surface
    main_table = (table_margin, table_margin, 
                  table_size[0] - table_margin, table_size[1] - table_margin)
    draw.ellipse(main_table, fill=(0, 120, 0), outline=(0, 80, 0), width=4)
    
    # Inner table accent
    inner_margin = 20
    inner_table = (table_margin + inner_margin, table_margin + inner_margin,
                   table_size[0] - table_margin - inner_margin, table_size[1] - table_margin - inner_margin)
    draw.ellipse(inner_table, fill=None, outline=(0, 150, 0), width=2)
    
    # Title and game info header
    header_y = 15
    draw.text((table_center_x - 120, header_y), "TEXAS HOLD'EM POKER", 
              fill="gold", font=title_font, anchor="mm")
    
    # Street indicator with visual progress
    street_y = header_y + 35
    street_display = f"CURRENT STREET: {street.upper()}"
    draw.text((table_center_x, street_y), street_display, 
              fill="white", font=large_font, anchor="mm")
    
    # Street progress bar
    progress_y = street_y + 25
    progress_width = 300
    progress_x = table_center_x - progress_width // 2
    streets = ["PRE-FLOP", "FLOP", "TURN", "RIVER"]
    current_street_idx = streets.index(street.upper()) if street.upper() in streets else 0
    
    for i, street_name in enumerate(streets):
        segment_width = progress_width // 4
        segment_x = progress_x + i * segment_width
        color = "yellow" if i <= current_street_idx else "gray"
        draw.rectangle((segment_x, progress_y, segment_x + segment_width - 2, progress_y + 20), 
                      fill=color, outline="black")
        draw.text((segment_x + segment_width // 2, progress_y + 10), street_name[:4], 
                 fill="black", font=small_font, anchor="mm")
    
    # Enhanced community cards area
    community_y = table_center_y - 50
    community_x_start = table_center_x - 175
    
    # Community cards background
    comm_bg = (community_x_start - 15, community_y - 15, 
               community_x_start + 350, community_y + 95)
    draw.rectangle(comm_bg, fill=(0, 80, 0), outline="gold", width=2)
    draw.text((table_center_x, community_y - 25), "COMMUNITY CARDS", 
              fill="gold", font=info_font, anchor="mm")
    
    # Draw community cards with enhanced design
    card_suits_symbols = {"Spades": "♠", "Hearts": "♥", "Diamonds": "♦", "Clubs": "♣"}
    card_suit_colors = {"Spades": "black", "Hearts": "red", "Diamonds": "red", "Clubs": "black"}
    
    for i, card in enumerate(community_cards):
        card_x = community_x_start + i * 70
        card_rect = (card_x, community_y, card_x + 65, community_y + 85)
        
        # Card shadow
        shadow_rect = (card_x + 3, community_y + 3, card_x + 68, community_y + 88)
        draw.rectangle(shadow_rect, fill="gray", outline=None)
        
        # Card background
        draw.rectangle(card_rect, fill="white", outline="black", width=2)
        
        # Parse card rank and suit
        card_parts = card.split(" ")
        if len(card_parts) == 2:
            rank, suit = card_parts
            symbol = card_suits_symbols.get(suit, suit[0])
            color = card_suit_colors.get(suit, "black")
            
            # Draw rank (top-left)
            draw.text((card_x + 5, community_y + 5), rank, fill=color, font=card_font)
            # Draw suit symbol (top-left, below rank)
            draw.text((card_x + 5, community_y + 20), symbol, fill=color, font=card_font)
            # Draw large centered symbol
            draw.text((card_x + 32, community_y + 42), symbol, fill=color, font=large_font, anchor="mm")
            # Draw rank (bottom-right, rotated)
            draw.text((card_x + 60, community_y + 65), rank, fill=color, font=card_font, anchor="rb")
    
    # Draw placeholder slots for remaining community cards
    for i in range(len(community_cards), 5):
        card_x = community_x_start + i * 70
        card_rect = (card_x, community_y, card_x + 65, community_y + 85)
        draw.rectangle(card_rect, fill="lightgray", outline="darkgray", width=2)
        draw.text((card_x + 32, community_y + 42), "?", fill="gray", font=large_font, anchor="mm")
    
    # Enhanced pot information
    total_pot = sum(betting_info.values())
    pot_y = community_y + 110
    
    # Pot background
    pot_bg = (table_center_x - 80, pot_y - 10, table_center_x + 80, pot_y + 30)
    draw.rectangle(pot_bg, fill="darkgreen", outline="gold", width=2)
    draw.text((table_center_x, pot_y + 10), f"POT: ${total_pot}", 
              fill="gold", font=large_font, anchor="mm")
    
    # Dynamic player positions around the table (supports 2-10 players)
    # Try to determine number of players from observation or default to 2
    num_players = 2  # Default fallback
    
    # Calculate player positions dynamically based on number of players
    player_positions = []
    import math
    
    if num_players == 2:
        # Special case for heads-up
        player_positions = [
            (table_center_x - 120, table_size[1] - 160, "bottom"),    # Player 0 (bottom)
            (table_center_x - 120, 140, "top")                        # Player 1 (top)
        ]
    else:
        # Circular arrangement for 3-10 players
        for i in range(min(num_players, 10)):  # Cap at 10 for visual reasons
            # Start from top and go clockwise
            angle = (2 * math.pi * i) / num_players - math.pi / 2
            
            # Adjust radius based on number of players for better spacing
            if num_players <= 6:
                radius_x, radius_y = table_width // 3, table_height // 3
            else:
                radius_x, radius_y = table_width // 2.5, table_height // 2.5
            
            x = table_center_x + radius_x * math.cos(angle) - 100
            y = table_center_y + radius_y * math.sin(angle) - 30
            
            # Ensure positions stay within bounds
            x = max(20, min(table_size[0] - 220, x))
            y = max(20, min(table_size[1] - 140, y))
            
            position = f"player_{i}"
            player_positions.append((x, y, position))
    
    # Draw players with enhanced information
    for i, (px, py, position) in enumerate(player_positions):
        player_name = f"player_{i}"
        is_current = (current_player == player_name)
        is_agent = (i == 0)  # Player 0 is our agent
        
        # Player area background
        player_width, player_height = 240, 120
        player_rect = (px, py, px + player_width, py + player_height)
        
        # Highlight current player
        border_color = "gold" if is_current else "white"
        bg_color = (50, 100, 50) if is_current else (30, 80, 30)
        
        # Player background with shadow
        shadow_rect = (px + 3, py + 3, px + player_width + 3, py + player_height + 3)
        draw.rectangle(shadow_rect, fill="black", outline=None)
        draw.rectangle(player_rect, fill=bg_color, outline=border_color, width=3)
        
        # Player name and info
        name_color = "yellow" if is_current else "white"
        draw.text((px + 10, py + 5), f"Player {i}", fill=name_color, font=info_font)
        
        if is_agent:
            draw.text((px + 10, py + 20), "(YOU)", fill="cyan", font=small_font)
        
        # Simulated chip stack (based on performance score)
        base_chips = 1000
        current_chips = base_chips + (perf_score if is_agent and perf_score else 0)
        chip_text = f"Chips: ${max(0, int(current_chips))}"
        draw.text((px + 10, py + 95), chip_text, fill="white", font=small_font)
        
        # Draw chip stack visualization
        chip_colors = ["red", "blue", "green", "yellow", "purple"]
        chip_x = px + player_width - 60
        for j in range(min(5, max(1, int(current_chips // 200)))):
            chip_y = py + 70 - j * 3
            color_idx = j % len(chip_colors)
            draw.ellipse((chip_x, chip_y, chip_x + 20, chip_y + 8), 
                        fill=chip_colors[color_idx], outline="black")
        
        # Enhanced hole cards
        if is_agent and hole_cards:
            # Show actual cards for our agent
            for j, card in enumerate(hole_cards[:2]):
                card_x = px + 15 + j * 50
                card_y = py + 35
                card_rect = (card_x, card_y, card_x + 45, card_y + 55)
                
                # Card shadow
                draw.rectangle((card_x + 2, card_y + 2, card_x + 47, card_y + 57), 
                              fill="gray")
                draw.rectangle(card_rect, fill="white", outline="black", width=1)
                
                # Parse and draw card
                card_parts = card.split(" ")
                if len(card_parts) == 2:
                    rank, suit = card_parts
                    symbol = card_suits_symbols.get(suit, suit[0])
                    color = card_suit_colors.get(suit, "black")
                    
                    draw.text((card_x + 3, card_y + 3), rank, fill=color, font=small_font)
                    draw.text((card_x + 3, card_y + 15), symbol, fill=color, font=small_font)
        else:
            # Show card backs for other players
            for j in range(2):
                card_x = px + 15 + j * 50
                card_y = py + 35
                card_rect = (card_x, card_y, card_x + 45, card_y + 55)
                
                draw.rectangle((card_x + 2, card_y + 2, card_x + 47, card_y + 57), 
                              fill="gray")
                draw.rectangle(card_rect, fill="darkred", outline="black", width=1)
                
                # Card back pattern
                draw.line((card_x + 5, card_y + 5, card_x + 40, card_y + 50), fill="red", width=1)
                draw.line((card_x + 40, card_y + 5, card_x + 5, card_y + 50), fill="red", width=1)
                draw.text((card_x + 22, card_y + 27), "♠", fill="red", font=small_font, anchor="mm")
        
        # Player action status
        if is_current:
            draw.text((px + 10, py + 75), "ACTING NOW", fill="yellow", font=small_font)
        
        # Dealer button (simplified - assume player 0 has it)
        if i == 0:
            button_x, button_y = px + player_width - 30, py + 10
            draw.ellipse((button_x, button_y, button_x + 20, button_y + 20), 
                        fill="white", outline="black", width=2)
            draw.text((button_x + 10, button_y + 10), "D", fill="black", font=small_font, anchor="mm")
    
    # Enhanced legal actions display
    actions_y = table_size[1] - 80
    legal_actions = []
    action_names = ["CALL", "RAISE", "FOLD", "CHECK"]
    action_colors = ["green", "orange", "red", "blue"]
    
    for i, is_legal in enumerate(action_mask):
        if is_legal == 1:
            legal_actions.append((action_names[i], action_colors[i], i))
    
    if legal_actions:
        actions_title = "AVAILABLE ACTIONS:"
        draw.text((20, actions_y - 25), actions_title, fill="gold", font=info_font)
        
        action_x = 20
        for action_name, color, action_id in legal_actions:
            # Action button background
            button_width = 80
            button_rect = (action_x, actions_y, action_x + button_width, actions_y + 25)
            draw.rectangle(button_rect, fill=color, outline="white", width=2)
            draw.text((action_x + button_width//2, actions_y + 12), f"{action_name}", 
                     fill="white", font=small_font, anchor="mm")
            draw.text((action_x + button_width//2, actions_y + 30), f"({action_id})", 
                     fill="white", font=small_font, anchor="mm")
            action_x += button_width + 10
    
    # Performance and game statistics
    stats_x = table_size[0] - 200
    stats_y = 80
    
    # Stats background
    stats_bg = (stats_x - 10, stats_y - 10, table_size[0] - 10, stats_y + 100)
    draw.rectangle(stats_bg, fill=(20, 40, 20), outline="white", width=1)
    
    draw.text((stats_x, stats_y), "GAME STATS", fill="cyan", font=info_font)
    
    if perf_score is not None:
        perf_color = "green" if perf_score >= 0 else "red"
        perf_sign = "+" if perf_score >= 0 else ""
        draw.text((stats_x, stats_y + 20), f"Performance: {perf_sign}{perf_score:.1f}", 
                 fill=perf_color, font=small_font)
    
    if action_taken_str:
        draw.text((stats_x, stats_y + 35), f"Last Action: {action_taken_str}", 
                 fill="white", font=small_font)
    
    # Betting round information
    draw.text((stats_x, stats_y + 50), "Betting Rounds:", fill="white", font=small_font)
    for i, (round_name, chips) in enumerate(betting_info.items()):
        if chips > 0:
            round_display = round_name.replace('_chips', '').replace('_', ' ').title()
            draw.text((stats_x, stats_y + 65 + i * 12), f"{round_display}: ${chips}", 
                     fill="yellow", font=small_font)
    
    # Game rules reminder (bottom right)
    rules_y = table_size[1] - 60
    draw.text((table_size[0] - 250, rules_y), "Best 5-card hand wins!", 
             fill="white", font=small_font)
    draw.text((table_size[0] - 250, rules_y + 15), "Use 0-2 hole cards + community cards", 
             fill="white", font=small_font)
    
    # Save the enhanced image
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
        table_size_for_render: tuple = (1000, 700),
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
    """Multi-player Texas Hold'em environment supporting 2-10 players with dynamic agent management."""

    def __init__(
        self,
        *,
        render_mode: Optional[str] = None,
        num_players: int = 2,
        table_size_for_render: tuple = (1000, 700),
        base_cache_dir: str = "cache/texasholdem",
        game_name_for_adapter: str = "texasholdem",
        observation_mode_for_adapter: str = "text",
        game_specific_config_path_for_adapter: str = "gamingagent/envs/zoo_02_texasholdem/game_env_config.json",
        max_stuck_steps_for_adapter: Optional[int] = 10,
        # New tournament-style options
        enable_player_elimination: bool = True,
        starting_chips: int = 1000,
        big_blind: int = 20,
        small_blind: int = 10,
        # AI opponent policies for non-agent players
        opponent_policies: Optional[Dict[str, str]] = None,
    ):
        # Validate player count
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

        self.num_players = num_players
        self.base_cache_dir = base_cache_dir
        self.enable_player_elimination = enable_player_elimination
        self.starting_chips = starting_chips
        self.big_blind = big_blind
        self.small_blind = small_blind
        
        # Create dynamic adapters for each player
        self._adapters = {}
        self.agent_players = set()  # Track which players are AI agents
        self.eliminated_players = set()  # Track eliminated players
        
        # Default opponent policies
        default_policies = {
            "random": "random",
            "conservative": "conservative", 
            "aggressive": "aggressive",
            "tight": "tight"
        }
        self.opponent_policies = opponent_policies or {}
        
        # Create adapters for all players
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
            
            # By default, all players are agents unless specified as AI opponents
            self.agent_players.add(player_name)

        # Game state tracking
        self.perf_scores = {f"player_{i}": 0.0 for i in range(num_players)}
        self.chip_stacks = {f"player_{i}": starting_chips for i in range(num_players)}
        self.current_player = "player_0"
        self.step_id = 0
        self.dealer_position = 0  # Track dealer button position
        self.round_number = 1

    def set_ai_opponents(self, ai_opponents: Dict[str, str]):
        """Set specific players as AI opponents with policies.
        
        Args:
            ai_opponents: Dict mapping player names to policy types
                         e.g., {"player_2": "random", "player_3": "aggressive"}
        """
        for player_name, policy in ai_opponents.items():
            if player_name in self._adapters:
                self.opponent_policies[player_name] = policy
                self.agent_players.discard(player_name)  # Remove from agent players
                print(f"[MultiTexasHoldem] Set {player_name} as AI opponent with '{policy}' policy")

    def get_active_players(self) -> List[str]:
        """Get list of players still in the game (not eliminated)."""
        if self.enable_player_elimination:
            return [p for p in self.player_names if p not in self.eliminated_players]
        else:
            return self.player_names.copy()

    def get_agent_players(self) -> List[str]:
        """Get list of players controlled by AI agents (not AI opponents)."""
        active_players = self.get_active_players()
        return [p for p in active_players if p in self.agent_players]

    def _execute_ai_opponent_action(self, player_name: str):
        """Execute an action for an AI opponent based on their policy."""
        policy = self.opponent_policies.get(player_name, "random")
        obs_pz = self.pz_env.observe(player_name)
        
        if obs_pz is None:
            return
            
        legal_actions = np.where(obs_pz["action_mask"] == 1)[0]
        
        if len(legal_actions) == 0:
            self.pz_env.step(None)
            return
            
        if policy == "random":
            action = random.choice(legal_actions)
        elif policy == "conservative":
            # Prefer check/call over raise, fold if no good options
            if 3 in legal_actions:  # check
                action = 3
            elif 0 in legal_actions:  # call
                action = 0
            elif 2 in legal_actions:  # fold
                action = 2
            else:
                action = random.choice(legal_actions)
        elif policy == "aggressive":
            # Prefer raise, then call, avoid fold
            if 1 in legal_actions:  # raise
                action = 1
            elif 0 in legal_actions:  # call
                action = 0
            elif 3 in legal_actions:  # check
                action = 3
            else:
                action = random.choice(legal_actions)
        elif policy == "tight":
            # Very conservative, fold often
            if 2 in legal_actions and random.random() < 0.4:  # fold 40% of time
                action = 2
            elif 3 in legal_actions:  # check
                action = 3
            elif 0 in legal_actions:  # call
                action = 0
            else:
                action = random.choice(legal_actions)
        else:
            action = random.choice(legal_actions)
            
        self.pz_env.step(action)

    def _update_chip_stacks(self):
        """Update chip stack tracking based on game rewards."""
        for player_name in self.player_names:
            if hasattr(self.pz_env, 'rewards') and player_name in self.pz_env.rewards:
                reward = self.pz_env.rewards[player_name]
                self.chip_stacks[player_name] += reward
                
                # Handle elimination
                if self.enable_player_elimination and self.chip_stacks[player_name] <= 0:
                    self.eliminated_players.add(player_name)
                    print(f"[MultiTexasHoldem] {player_name} eliminated (no chips remaining)")

    def _advance_dealer_button(self):
        """Move dealer button to next active player."""
        active_players = self.get_active_players()
        if len(active_players) < 2:
            return
            
        current_dealer = f"player_{self.dealer_position}"
        if current_dealer not in active_players:
            # Find next active player
            for i in range(self.num_players):
                potential_dealer = f"player_{i}"
                if potential_dealer in active_players:
                    self.dealer_position = i
                    break

    def reset(self, *, seed: int | None = None, episode_id: int = 1, **kwargs):
        """Reset environment for multi-player game."""
        # Reset all adapters
        for adap in self._adapters.values():
            adap.reset_episode(episode_id)

        super().reset(seed=seed)

        # Reset game state
        self.step_id = 0
        self.current_episode_id = episode_id
        self.eliminated_players.clear()
        self.chip_stacks = {f"player_{i}": self.starting_chips for i in range(self.num_players)}
        self.perf_scores = {f"player_{i}": 0.0 for i in range(self.num_players)}
        self.round_number = 1
        self.dealer_position = 0

        # Create observations for all agent players
        obs_dict = {}
        agent_players = self.get_agent_players()
        
        for agent_name in agent_players:
            adap = self._adapters[agent_name]
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
                    self.table_size_for_render, perf_score=self.perf_scores.get(agent_name, 0)
                )
                
            if adap.observation_mode in {"text", "both"}:
                # Enhanced text with tournament info
                base_text = _create_text_representation(
                    obs_vector, action_mask, self.pz_env.agent_selection
                )
                tournament_info = self._get_tournament_status_text()
                text_repr = f"{base_text}\n\n{tournament_info}"

            obs_dict[agent_name] = adap.create_agent_observation(
                img_path=img_path, text_representation=text_repr
            )

        if self.render_mode == "human":
            self.render()
            
        return obs_dict, self._get_game_info()

    def _get_tournament_status_text(self) -> str:
        """Get tournament status information as text."""
        active_players = self.get_active_players()
        lines = []
        lines.append("=== TOURNAMENT STATUS ===")
        lines.append(f"Round: {self.round_number}")
        lines.append(f"Active Players: {len(active_players)}/{self.num_players}")
        lines.append(f"Dealer Button: player_{self.dealer_position}")
        lines.append(f"Blinds: ${self.small_blind}/{self.big_blind}")
        lines.append("")
        lines.append("CHIP STACKS:")
        for player_name in self.player_names:
            status = "ACTIVE" if player_name in active_players else "ELIMINATED"
            agent_type = "AGENT" if player_name in self.agent_players else f"AI({self.opponent_policies.get(player_name, 'random')})"
            chips = self.chip_stacks[player_name]
            lines.append(f"  {player_name}: ${chips:,} ({status}, {agent_type})")
        return "\n".join(lines)

    def _get_game_info(self) -> Dict[str, Any]:
        """Get comprehensive game information."""
        return {
            "round_number": self.round_number,
            "active_players": self.get_active_players(),
            "agent_players": self.get_agent_players(),
            "eliminated_players": list(self.eliminated_players),
            "chip_stacks": self.chip_stacks.copy(),
            "dealer_position": self.dealer_position,
            "blinds": {"small": self.small_blind, "big": self.big_blind}
        }

    def step(self, agent_name: str, action_str: Optional[str], thought_process: str = "", time_taken_s: float = 0.0):
        """Execute a step for an agent player, handling AI opponents automatically."""
        
        # Verify this is an agent player's turn
        if agent_name not in self.agent_players:
            raise ValueError(f"{agent_name} is not an agent player")
            
        if agent_name not in self.get_active_players():
            raise ValueError(f"{agent_name} is eliminated")

        # Process turns until it's an agent's turn or game ends
        game_ended = False
        while not game_ended:
            current_player = self.pz_env.agent_selection
            
            if not current_player or current_player not in self.get_active_players():
                game_ended = True
                break
                
            if current_player == agent_name:
                # Agent's turn - execute their action
                adap = self._adapters[agent_name]
                adap.increment_step()
                
                env_act_idx = adap.map_agent_action_to_env_action(action_str)
                
                try:
                    self._apply_action(env_act_idx)
                    break  # Agent acted, exit the loop
                except Exception as e:
                    print(f"[ERROR] Step failed for agent {agent_name}: {e}")
                    return self._handle_illegal_action(agent_name)
                    
            elif current_player in self.agent_players:
                # Another agent's turn - this shouldn't happen in single-step mode
                break
            else:
                # AI opponent's turn
                self._execute_ai_opponent_action(current_player)

        # Update game state
        self.current_player = self.pz_env.agent_selection
        self.step_id += 1
        self._update_chip_stacks()

        # Get rewards and game state
        rewards = {name: float(self.pz_env.rewards.get(name, 0)) for name in self.player_names}
        terminations = any(self.pz_env.terminations.values())
        truncations = any(self.pz_env.truncations.values())
        
        # Check for tournament end conditions
        active_players = self.get_active_players()
        if len(active_players) <= 1:
            terminations = True
            print(f"[MultiTexasHoldem] Tournament ended. Winner: {active_players[0] if active_players else 'None'}")

        # Update performance scores
        for player_name in self.player_names:
            self.perf_scores[player_name] += rewards[player_name]

        # Create observations for all active agent players
        next_obs = {}
        info_for_acting_agent = self._get_game_info()
        agent_players = self.get_agent_players()
        
        for current_agent_name in agent_players:
            adap = self._adapters[current_agent_name]
            obs_pz = self.pz_env.observe(current_agent_name)
            obs_vector = obs_pz["observation"]
            action_mask = obs_pz["action_mask"]

            # Only show legal actions for the current player
            if self.pz_env.agent_selection != current_agent_name:
                action_mask = np.zeros_like(action_mask)

            # Create observations
            img_path = text_repr = None
            if adap.observation_mode in {"vision", "both"}:
                img_path = adap._create_agent_observation_path(self.current_episode_id, self.step_id)
                create_poker_table_image(
                    obs_vector, action_mask, self.pz_env.agent_selection, img_path, 
                    self.table_size_for_render, 
                    perf_score=self.perf_scores.get(current_agent_name, 0), 
                    action_taken_str=action_str if current_agent_name == agent_name else None
                )
                
            if adap.observation_mode in {"text", "both"}:
                base_text = _create_text_representation(
                    obs_vector, action_mask, self.pz_env.agent_selection
                )
                tournament_info = self._get_tournament_status_text()
                text_repr = f"{base_text}\n\n{tournament_info}"

            agent_obs = adap.create_agent_observation(
                img_path=img_path, text_representation=text_repr
            )
            next_obs[current_agent_name] = agent_obs

            # Log step data only for the agent who acted
            if current_agent_name == agent_name:
                final_terminated, final_truncated = adap.verify_termination(
                    agent_obs, terminations, truncations
                )
                
                adap.log_step_data(
                    agent_action_str=action_str,
                    thought_process=thought_process,
                    reward=rewards[agent_name],
                    info=info_for_acting_agent,
                    terminated=final_terminated,
                    truncated=final_truncated,
                    time_taken_s=time_taken_s,
                    perf_score=self.perf_scores[agent_name],
                    agent_observation=agent_obs,
                )

        if self.render_mode == "human":
            self.render()
            
        return next_obs, rewards, terminations, truncations, info_for_acting_agent, self.perf_scores.copy()

    def _handle_illegal_action(self, agent_name: str):
        """Handle illegal action by an agent."""
        rewards = {name: 0.0 for name in self.player_names}
        rewards[agent_name] = -10.0  # Penalty for illegal action
        self.perf_scores[agent_name] -= 10
        
        return (
            {agent_name: None},  # Empty observation
            rewards,
            True,  # terminated
            True,  # truncated
            self._get_game_info(),
            self.perf_scores.copy(),
        )

    def record_episode_results(self, episode_id: int, final_scores: Dict[str, float], total_steps: int):
        """Record final episode results for all agent players."""
        for player_name in self.get_agent_players():
            if player_name in self._adapters:
                adap = self._adapters[player_name]
                final_score = final_scores.get(player_name, 0.0)
                total_reward = self.perf_scores.get(player_name, 0.0)
                adap.record_episode_result(
                    episode_id=episode_id,
                    score=final_score,
                    steps=total_steps,
                    total_reward=total_reward,
                    total_perf_score=total_reward
                )

    def finalize_run_summaries(self, run_settings: Dict):
        """Generate final run summaries for all agent players."""
        summaries = {}
        for player_name in self.get_agent_players():
            if player_name in self._adapters:
                adap = self._adapters[player_name]
                summaries[player_name] = adap.finalize_and_save_summary(run_settings)
        return summaries

    def close(self):
        super().close()
        for adap in self._adapters.values():
            adap.close_log_file()
