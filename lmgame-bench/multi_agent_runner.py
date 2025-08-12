from __future__ import annotations
"""
multiagent_tictactoe_runner.py – final complete version
======================================================
Multi‑model Tic‑Tac‑Toe runner aligned with single_agent_runner.py.
"""

import argparse
import datetime as _dt
import os
import time
from typing import Any, Dict, Optional

import yaml
import trueskill
import sys

import json
from pathlib import Path

from gamingagent.agents.base_agent import BaseAgent
from gamingagent.envs.zoo_01_tictactoe.TicTacToeEnv import MultiTicTacToeEnv
from gamingagent.envs.zoo_02_texasholdem.TexasHoldemEnv import MultiTexasHoldemEnv
from gamingagent.modules import PerceptionModule, ReasoningModule
from tools.utils import draw_grid_on_image

# Map game_name to environment class and config directory
GAME_ENV_CLASS_MAPPING = {
    "tictactoe": MultiTicTacToeEnv,  # For multi-agent mode
    "texasholdem": MultiTexasHoldemEnv,  # For multi-agent mode
    # ...
}

GAME_CONFIG_MAPPING = {
    "tictactoe": "zoo_01_tictactoe",
    "texasholdem": "zoo_02_texasholdem",
    }

###############################################################################
# Helpers
###############################################################################

def str_to_bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).lower()
    if v in {"yes", "true", "t", "y", "1"}:
        return True
    if v in {"no", "false", "f", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected")


def load_yaml(path):
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception as e:
        print(f"[WARN] YAML load error: {e}")
        return {}


def _is_default(args, key, defaults):
    return getattr(args, key) == getattr(defaults, key)

def ranks_from_values(values):
    """Largest value => rank 0; equal values share the same rank."""
    order = sorted(values, reverse=True)
    rmap, rank, last = {}, 0, None
    for v in order:
        if v != last:
            rmap[v] = rank
            rank += 1
            last = v
    return [rmap[v] for v in values]

def ensure_dir(p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)

def append_jsonl(path, obj):
    ensure_dir(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def load_ratings(path, ts_env):
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {k: ts_env.Rating(mu=v["mu"], sigma=v["sigma"]) for k, v in raw.items()}
    return {}

def save_ratings(path, ratings):
    ensure_dir(path)
    data = {k: {"mu": float(r.mu), "sigma": float(r.sigma)} for k, r in ratings.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def merge_cli_yaml(args, cfg, defaults):
    """
    Merge CLI arguments with YAML config, ensuring CLI values take precedence.
    Special handling for model names to always use CLI values.
    """
    # First parse with just command line values
    args._cli_values = {}
    for key in vars(args):
        args._cli_values[key] = getattr(args, key)

    # Store YAML config for reference
    args._yaml_defaults = cfg if cfg else {}

    # List of parameters that should be checked for YAML values
    keys = [
        "num_runs",
        "max_steps",
        "observation_mode",
        "seed",
        "use_reflection",
        "use_perception",
        "use_summary",
        "max_memory",
        "scaffolding",
        "vllm_url",
        "modal_url",
        "tournament_hands",
    ]

    # Only apply YAML values for parameters that:
    # 1. Weren't explicitly set on command line (i.e., are still at their default)
    # 2. Have a different value in YAML
    for k in keys:
        if k in cfg:
            cli_value = getattr(args, k)
            default_value = getattr(defaults, k)
            yaml_value = cfg[k]
            
            # If the CLI value is the default, consider using the YAML value.
            if cli_value == default_value:
                if yaml_value is not None and cli_value != yaml_value:
                    print(f"INFO: Using YAML value for '{k}': {yaml_value} (CLI default was: {cli_value})")
                    setattr(args, k, yaml_value)
            # If a non-default CLI value was provided, it takes precedence.
            elif cli_value != yaml_value:
                 print(f"INFO: Using CLI value for '{k}': {cli_value} (overriding YAML value: {yaml_value})")
    
    # Special handling for model names - always use CLI values
    if "model_x" in cfg and getattr(args, "model_x") != cfg["model_x"]:
        print(f"INFO: Using CLI value for model_x: {args.model_x} (YAML value ignored: {cfg['model_x']})")
    if "model_o" in cfg and getattr(args, "model_o") != cfg["model_o"]:
        print(f"INFO: Using CLI value for model_o: {args.model_o} (YAML value ignored: {cfg['model_o']})")

    # Handle boolean flags separately
    bool_flags = ['harness', 'use_custom_prompt']
    for flag in bool_flags:
        if flag in cfg and cfg[flag] is True and not getattr(args, flag):
            # Only set to True if not explicitly set on command line
            if f"--{flag.replace('_', '-')}" not in sys.argv:
                print(f"INFO: Using YAML value for boolean flag '{flag}': True")
                setattr(args, flag, True)

    # Store agent-specific configurations
    args._agent_defaults = cfg.get("agent_defaults", {})
    args._agent_x_cfg = cfg.get("agent_x", {})
    args._agent_o_cfg = cfg.get("agent_o", {})

    return args

###############################################################################
# Agent factory
###############################################################################

def _parse_scaff(raw):
    if not raw:
        return None
    if isinstance(raw, dict):
        if raw.get("funcname") == "draw_grid_on_image":
            return {"func": draw_grid_on_image, "funcArgs": raw.get("funcArgs", {})}
        return None
    try:
        r, c = map(int, raw.strip("() ").split(","))
        return {"func": draw_grid_on_image, "funcArgs": {"grid_dim": (r, c)}}
    except Exception:
        return None

def flatten_agent_cfg(cfg: dict) -> dict:
    """Flatten modules in agent_defaults/agent_x/agent_o to top-level keys."""
    flat = dict(cfg) if cfg else {}
    modules = flat.pop("modules", {})
    if modules:
        for module_name, module_cfg in modules.items():
            if isinstance(module_cfg, dict):
                for k, v in module_cfg.items():
                    # E.g., memory_module.max_memory -> max_memory
                    if k not in flat:
                        flat[k] = v
    return flat

def make_agent(
    role, model_name, args, root, prompt, agent_default_cfg, custom_agent_cfg
):
    cache = os.path.join(root, role)
    os.makedirs(cache, exist_ok=True)
    kw = dict(
        game_name=args.game_name,
        model_name=model_name,
        config_path=prompt,
        harness=args.harness,
        use_custom_prompt=args.use_custom_prompt,
        max_memory=args.max_memory,
        use_reflection=args.use_reflection,
        use_perception=args.use_perception,
        use_summary=args.use_summary,
        custom_modules=(
            {"perception_module": PerceptionModule, "reasoning_module": ReasoningModule}
            if args.harness else None
        ),
        observation_mode=args.observation_mode,
        scaffolding=_parse_scaff(args.scaffolding),
        cache_dir=cache,
        vllm_url=args.vllm_url,
        modal_url=args.modal_url
    )
    # Explicitly flatten agent defaults and agent-specific configs
    kw.update(flatten_agent_cfg(agent_default_cfg))
    kw.update(flatten_agent_cfg(custom_agent_cfg))
    return BaseAgent(**kw)

###############################################################################
# Environment
###############################################################################

def create_environment(
    game_name: str,
    observation_mode: str,
    config_dir_name: str,
    cache_dir: str,
    harness: bool = False,
    multiagent_arg: str = "multi",
    record_video: bool = False,
):
    """
    Creates and returns a game environment instance based on the game name and config.
    - Loads game-specific config (JSON) and pulls out env_init_kwargs.
    - Dynamically instantiates the correct Env class.
    """
    # single‑agent
    assert multiagent_arg == "multi", "This script only supports multi-agent games."

    # Default to tictactoe if only that is implemented
    env_class = GAME_ENV_CLASS_MAPPING.get(game_name.lower())
    if not env_class:
        raise ValueError(f"Environment for '{game_name}' is not implemented.")

    config_json_path = os.path.join(
        "gamingagent", "envs", config_dir_name, "game_env_config.json"
    )
    if not os.path.isfile(config_json_path):
        raise FileNotFoundError(f"Config file not found at {config_json_path}")

    with open(config_json_path, "r", encoding="utf-8") as f:
        cfg_json = json.load(f)

    env_init_kwargs = cfg_json.get("env_init_kwargs", {})
    render_mode = env_init_kwargs.get("render_mode", "human")
    tile_size_for_render = env_init_kwargs.get("tile_size_for_render", 64)
    max_stuck_steps = cfg_json.get("max_unchanged_steps_for_termination", 10)

    # Support both multi-agent and single-agent interface
    if game_name.lower() == "tictactoe":
        env = MultiTicTacToeEnv(
            render_mode=render_mode,
            tile_size_for_render=tile_size_for_render,
            p1_cache=os.path.join(cache_dir, "p1_cache"),
            p2_cache=os.path.join(cache_dir, "p2_cache"),
            game_name_for_adapter="multi_tictactoe",
            observation_mode_for_adapter=observation_mode,
            agent_cache_dir_for_adapter=cache_dir,
            game_specific_config_path_for_adapter=config_json_path,
            max_stuck_steps_for_adapter=max_stuck_steps,
        )
        return env
    elif game_name.lower() == "texasholdem":
        # Enhanced Texas Hold'em with new parameters
        env = MultiTexasHoldemEnv(
             render_mode=render_mode,
             num_players=env_init_kwargs.get("num_players", 2),
             table_size_for_render=tuple(env_init_kwargs.get("table_size_for_render", [1000, 700])),
             base_cache_dir=cache_dir,
             game_name_for_adapter="multi_texasholdem",
             observation_mode_for_adapter=observation_mode,
             game_specific_config_path_for_adapter=config_json_path,
             max_stuck_steps_for_adapter=max_stuck_steps,
             enable_player_elimination=env_init_kwargs.get("enable_player_elimination", False),
             starting_chips=env_init_kwargs.get("starting_chips", 1000),
             big_blind=env_init_kwargs.get("big_blind", 20),
             small_blind=env_init_kwargs.get("small_blind", 10),
             max_tournament_hands=env_init_kwargs.get("max_tournament_hands"),
            record_video=record_video,
         )
        
        # Set up AI opponents if specified in config
        ai_opponents = env_init_kwargs.get("ai_opponents", {})
        if ai_opponents:
            env.set_ai_opponents(ai_opponents)
            print(f"[Runner] Set AI opponents: {ai_opponents}")
        
        return env
    else:
        print(f"ERROR: Game '{game_name.lower()}' is not defined or implemented in multi_agent_runner.py's create_environment function.")
        return None

###############################################################################
# Episode
###############################################################################

def play_episode(env, agents, eid, max_turns, seed):
    obs, _ = env.reset(seed=seed, episode_id=eid)
    
    # Handle different player naming conventions
    if hasattr(env, 'pz_env') and hasattr(env.pz_env, 'agents'):
        env_player_names = env.pz_env.agents
    else:
        env_player_names = list(agents.keys())
    
    # Create mapping between environment players and agent keys
    player_mapping = {}
    reverse_mapping = {}
    
    # For Texas Hold'em: env uses player_0, player_1... agents use same
    if hasattr(env, 'get_agent_players'):
        agent_players = env.get_agent_players()
        for env_player in env_player_names:
            if env_player in agent_players and env_player in agents:
                player_mapping[env_player] = env_player
                reverse_mapping[env_player] = env_player
    else:
        # Legacy mapping for other games
        if "player_0" in env_player_names:
            player_mapping = {"player_0": "player_1", "player_1": "player_2"}
            reverse_mapping = {"player_1": "player_0", "player_2": "player_1"}
        else:
            player_mapping = {"player_1": "player_1", "player_2": "player_2"}
            reverse_mapping = {"player_1": "player_1", "player_2": "player_2"}
    
    # Initialize totals for all agents
    totals = {agent_key: 0.0 for agent_key in agents.keys()}
    moves_log = []

    for t in range(max_turns):
        # Get current player from environment
        if hasattr(env, 'current_player'):
            env_cur = env.current_player
        elif hasattr(env, 'pz_env') and hasattr(env.pz_env, 'agent_selection'):
            env_cur = env.pz_env.agent_selection
        else:
            print("ERROR: Could not determine current player from environment")
            break

        # Map environment player name to agent name
        agent_cur = player_mapping.get(env_cur, env_cur)

        if agent_cur not in agents:
            print(f"ERROR: Agent '{agent_cur}' not found in agents dict")
            break

        ad, _ = agents[agent_cur].get_action(obs[env_cur])
        act = None if ad is None else ad.get("action")
        moves_log.append(f"Turn {t+1}: {agent_cur} ({env_cur}) -> {act}")

        obs, rew, term, trunc, *_ = env.step(env_cur, act)

        # Handle different reward structures
        if isinstance(rew, dict):
            for env_player, reward in rew.items():
                agent_player = player_mapping.get(env_player, env_player)
                if agent_player in totals:
                    totals[agent_player] += reward
        
        env.render()
        if term or trunc:
            break

    # Print detailed game summary
    print(f"\n{'='*60}")
    print(f"GAME {eid} SUMMARY")
    print(f"{'='*60}")
    print(f"Total turns: {t+1}")
    for agent_key, reward in totals.items():
        print(f"{agent_key} total reward: {reward:.1f}")
    
    # Determine game result
    illegal = False
    illegal_player = None
    
    # Check for illegal moves - only flag if reward is significantly worse than normal loss
    # In TicTacToe, normal loss = -1.0, so only flag if much worse (like -10 or similar)
    for agent_key, reward in totals.items():
        if reward <= -5.0 and all(r >= 0 for k, r in totals.items() if k != agent_key):
            result = f"ILLEGAL MOVE by {agent_key}"
            illegal = True
            illegal_player = agent_key
            break
    
    if not illegal:
        # Find winner (highest reward)
        max_reward = max(totals.values())
        winners = [k for k, v in totals.items() if v == max_reward]
        
        if len(winners) == 1 and max_reward > 0:
            result = f"{winners[0]} WINS!"
        elif all(v == 0 for v in totals.values()):
            result = "DRAW/TIE (no rewards)"
        else:
            result = f"DRAW/TIE ({len(winners)} players tied with {max_reward:.1f})"
    
    print(f"Result: {result}")

    # Show move history
    print(f"\nMove History:")
    for move in moves_log:
        print(f"  {move}")

    print(f"{'='*60}\n")

    # Record episode results using the appropriate method based on environment type
    if hasattr(env, 'record_episode_results'):
        # Enhanced multi-agent environment (e.g., Texas Hold'em)
        # Map agent keys back to environment player names for recording
        final_scores = {}
        for agent_key, reward in totals.items():
            # For Texas Hold'em, agent keys are the same as env player names
            env_player = reverse_mapping.get(agent_key, agent_key)
            final_scores[env_player] = reward
        env.record_episode_results(episode_id=eid, final_scores=final_scores, total_steps=t+1)
    elif (hasattr(env, 'adapter_p1') and hasattr(env, 'adapter_p2')) and (env.adapter_p1 and env.adapter_p2):
        # Legacy multi-agent environment (e.g., TicTacToe)
        agent_keys = list(totals.keys())
        if len(agent_keys) >= 2:
            env.adapter_p1.record_episode_result(
                episode_id=eid,
                score=totals[agent_keys[0]],
                steps=t+1,
                total_reward=totals[agent_keys[0]],
                total_perf_score=totals[agent_keys[0]]
            )
            env.adapter_p2.record_episode_result(
                episode_id=eid,
                score=totals[agent_keys[1]],
                steps=t+1,
                total_reward=totals[agent_keys[1]],
                total_perf_score=totals[agent_keys[1]]
            )
    
    #result_dict = dict(totals)  # Copy all player totals
    result_dict = dict(totals)
    result_dict["illegal"] = illegal
    result_dict["illegal_player"] = illegal_player
    return result_dict

###############################################################################
# CLI
###############################################################################

def build_parser():
    p = argparse.ArgumentParser("Multi‑Agent TicTacToe Runner")
    p.add_argument("--game_name", type=str, default=None,
                        help="Name of the game (e.g., tictactoe). Set by prelim parser.")
    p.add_argument("--num_runs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=30)
    p.add_argument("--observation_mode", choices=["vision", "text", "both"], default="both")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--model_x", type=str, default="gemini-2.5-flash")
    p.add_argument("--model_o", type=str, default="claude-3-5-sonnet-latest")
    p.add_argument("--player_models", type=str, nargs="+", default=None,
                   help="List of models for multiple players (e.g., --player_models gpt-4o-mini claude-3-5-sonnet gemini-2.5-flash)")
    p.add_argument("--harness", action="store_true")
    p.add_argument("--multiagent_arg", type=str, default="multi",
                        choices=["single", "multi"], help="Multi-agent mode configuration.")
    p.add_argument("--use_custom_prompt", action="store_true")
    p.add_argument("--use_reflection", type=str_to_bool, default=True)
    p.add_argument("--use_perception", type=str_to_bool, default=True)
    p.add_argument("--use_summary", type=str_to_bool, default=False)
    p.add_argument("--max_memory", type=int, default=20)
    p.add_argument("--scaffolding", type=str, default=None)
    p.add_argument("--vllm_url", type=str, default=None)
    p.add_argument("--modal_url", type=str, default=None)
    p.add_argument("--tournament_hands", type=int, default=None)
    p.add_argument("--record_video", type=str_to_bool, default=False, help="Enable video recording of the GUI")
    return p

###############################################################################
# Main
###############################################################################

def main(argv: Optional[list[str]] = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    game_config_name = GAME_CONFIG_MAPPING.get(args.game_name)
    assert game_config_name is not None, f"Game '{args.game_name}' not found in GAME_CONFIG_MAPPING. Please check your --game_name argument."
    args.config_path = os.path.join(
        "gamingagent", "configs", game_config_name, "multiagent_config.yaml"
    )
    defaults = parser.parse_args([])
    args = merge_cli_yaml(args, load_yaml(args.config_path), defaults)

    print("argument configuration:")
    print(args)

    run_root = os.path.join(
        "cache", args.game_name,
        f"multi_{args.model_x[:10]}_{args.model_o[:10]}",
        _dt.datetime.now().strftime("%Y%m%d_%H%M%S"),
    )

    prompt_path = os.path.join(
        "gamingagent", "configs", game_config_name, "module_prompts.json"
    )
    if not os.path.isfile(prompt_path):
        prompt_path = None

    # Load and (optionally) patch num_players BEFORE creating env
    config_json_path = os.path.join("gamingagent", "envs", game_config_name, "game_env_config.json")
    with open(config_json_path, "r", encoding="utf-8") as f:
        cfg_json = json.load(f)

    env_num_players = cfg_json.get("env_init_kwargs", {}).get("num_players", 2)

    if args.player_models and len(args.player_models) > 1:
        num_agents = len(args.player_models)
        if num_agents != env_num_players:
            print(f"[Runner] Updating num_players from {env_num_players} to {num_agents} to match player_models")
            cfg_json.setdefault("env_init_kwargs", {})["num_players"] = num_agents
            with open(config_json_path, "w", encoding="utf-8") as f:
                json.dump(cfg_json, f, indent=2)
            env_num_players = num_agents  # keep local in sync

    # Create agents based on player models or fallback to model_x/model_o
    agents = {}
    
    if args.player_models and len(args.player_models) > 1:
        # Multi-model mode: create agents for each specified model
        num_agents = len(args.player_models)
        print(f"[Runner] Creating {num_agents} agents with models: {args.player_models}")
        
        # Update environment config to match number of models
        if num_agents != env_num_players:
            print(f"[Runner] Updating num_players from {env_num_players} to {num_agents} to match player_models")
            cfg_json["env_init_kwargs"]["num_players"] = num_agents
            with open(config_json_path, "w", encoding="utf-8") as f:
                json.dump(cfg_json, f, indent=2)
        
        for i, model_name in enumerate(args.player_models):
            player_key = f"player_{i}" if args.game_name.lower() == "texasholdem" else f"player_{i+1}"
            cache_name = f"p{i+1}_cache"
            agents[player_key] = make_agent(
                cache_name, model_name, args, run_root, prompt_path, 
                args._agent_defaults, args._agent_x_cfg if i == 0 else args._agent_o_cfg
            )
            print(f"  {player_key}: {model_name}")
    else:
        # Legacy 2-player mode: use model_x and model_o
        if args.game_name.lower() == "texasholdem":
            agents = {
                "player_0": make_agent(
                    "p1_cache", args.model_x, args, run_root, prompt_path, args._agent_defaults, args._agent_x_cfg
                ),
                "player_1": make_agent(
                    "p2_cache", args.model_o, args, run_root, prompt_path, args._agent_defaults, args._agent_o_cfg
                ),
            }
        else:
            agents = {
                "player_1": make_agent(
                    "p1_cache", args.model_x, args, run_root, prompt_path, args._agent_defaults, args._agent_x_cfg
                ),
                "player_2": make_agent(
                    "p2_cache", args.model_o, args, run_root, prompt_path, args._agent_defaults, args._agent_o_cfg
                ),
            }
    if args.game_name.lower() == "texasholdem" and args.tournament_hands:
        cfg_json.setdefault("env_init_kwargs", {})["max_tournament_hands"] = int(args.tournament_hands)
        # recommend elimination in tournaments
        cfg_json["env_init_kwargs"]["enable_player_elimination"] = True

    with open(config_json_path, "w", encoding="utf-8") as f:
        json.dump(cfg_json, f, indent=2)

    env = create_environment(
        game_name=args.game_name,
        observation_mode=args.observation_mode,
        config_dir_name=game_config_name,
        cache_dir=run_root,
        harness=args.harness,
        multiagent_arg=args.multiagent_arg,
        record_video=args.record_video,
    )

    # One persistent log per game
    base_dir = os.path.join("cache", args.game_name)
    match_log_path = os.path.join(base_dir, "trueskill_matches.jsonl")
    ratings_path   = os.path.join(base_dir, "trueskill_ratings.json")

    # Configure TrueSkill; set draw_probability=0.0 since we handle ties explicitly
    # Note: TrueSkill 2 style updates (using chip performance) are automatically enabled for Texas Hold'em
    # TicTacToe: non-zero draw probability; no dynamics to avoid sigma blowing up on repeated draws
    ts = trueskill.TrueSkill(draw_probability=0.45, tau=0.0)

    # Map env player -> stable identity (prefer model_name)
    agent_ids = {k: getattr(agents[k], "model_name", k) for k in agents.keys()}

    # Load or initialize ratings
    ratings = load_ratings(ratings_path, ts)
    for pid, stable_id in agent_ids.items():
        ratings.setdefault(stable_id, ts.Rating())


    print("\n--- INITIAL TRUESKILL RATINGS (BEFORE FIRST HAND) ---")
    for name, r in ratings.items():
        print(f"  {name:30s} μ={r.mu:.4f}, σ={r.sigma:.4f}")


    cseed = args.seed
    agent_keys = list(agents.keys())  
    game_results = {f"{k}_wins": 0 for k in agent_keys} 
    game_results.update({"draws": 0, "illegal_moves": 0}) 

    total_hands = args.tournament_hands if (args.game_name.lower()=="texasholdem" and args.tournament_hands) else args.num_runs
    for eid in range(1, total_hands + 1):
        print(f"\n{'='*20} STARTING HAND {eid}/{total_hands} {'='*20}")
        result = play_episode(env, agents, eid, args.max_steps, cseed)

        # --- TrueSkill 2 Tournament Mode: Accumulate performance, update only at end ---
        if not result:
            cseed = None if cseed is None else cseed + 1
            time.sleep(1)
            continue

        illegal_game = bool(result.get("illegal", False))
        illegal_player = result.get("illegal_player")

        # Build per-hand chip deltas for Texas Hold'em; fallback to reward totals
        if args.game_name.lower() == "texasholdem" and hasattr(env, "perf_scores"):
            # Per-hand delta = sum of rewards this hand (robust and public API)
            chip_deltas = {p: float(env.perf_scores.get(p, 0.0)) for p in env.player_names}
        else:
            chip_deltas = {k: float(v) for k, v in result.items() if k not in {"illegal", "illegal_player"}}
        
        # TrueSkill 2 Tournament Mode: Always use cumulative chip stacks for Texas Hold'em
        # Only update TrueSkill ratings at the end of the tournament
        is_tournament_end = (args.game_name.lower() == "texasholdem" and 
                           hasattr(env, "chip_stacks") and 
                           (eid == total_hands or 
                            (hasattr(env, "tournament_over") and env.tournament_over())))
        
        if args.game_name.lower() == "texasholdem" and hasattr(env, "chip_stacks"):
            # Always use cumulative chip stacks for ordering/logging in Texas Hold'em
            final_chip_counts = {p: float(env.chip_stacks.get(p, 0)) for p in env.player_names}
            ordered = sorted(final_chip_counts.items(), key=lambda kv: kv[1], reverse=True)
            
            if is_tournament_end:
                print(f"\n🏆 TOURNAMENT FINAL RANKING - Updating TrueSkill based on final chip stacks")
                print(f"Final chip standings:")
                for i, (player, chips) in enumerate(ordered, 1):
                    print(f"  #{i}: {player} - {chips:,.0f} chips")
        else:
            # Non-poker games: use per-game results
            ordered = sorted(chip_deltas.items(), key=lambda kv: kv[1], reverse=True)
        
        values  = [v for _, v in ordered]
        ranks   = ranks_from_values(values)  # 0-based; ties share rank

        # Prepare participant records (aligned order) and pre-ratings
        participants = []
        pre_ratings = []
        for (env_player, performance), rank in zip(ordered, ranks):
            sid = agent_ids.get(env_player, env_player)
            pre = ratings.get(sid, ts.Rating())
            pre_ratings.append((pre,))
            participants.append({
                "env_player": env_player,
                "id": sid,
                "chip_delta": chip_deltas.get(env_player, 0.0),  # Per-hand delta for logging
                "final_chips": performance if args.game_name.lower() == "texasholdem" else None,
                "rank": int(rank),
                "illegal": illegal_game and (illegal_player == env_player),
                "tournament_final": is_tournament_end,
            })

        # TrueSkill 2: Only update ratings at tournament end for Texas Hold'em
        should_update_ratings = (not illegal_game and 
                               (args.game_name.lower() != "texasholdem" or is_tournament_end))
        
        if should_update_ratings:
            # TrueSkill 2 style updates for Texas Hold'em using cumulative chip performance
            if args.game_name.lower() == "texasholdem" and len(values) > 1:
                # Use cumulative chip performance for more accurate skill assessment
                max_perf = max(values) if values else 1
                min_perf = min(values) if values else 0
                perf_range = max_perf - min_perf if max_perf > min_perf else 1
                
                # Create performance-weighted rankings based on chip differentials
                weighted_ranks = []
                for i, (_, performance) in enumerate(ordered):
                    if perf_range > 0:
                        # Normalize performance to 0-1 scale
                        normalized_perf = (performance - min_perf) / perf_range
                        # Convert to weighted rank (better performance = lower rank)
                        weighted_rank = len(ordered) * (1 - normalized_perf)
                    else:
                        # If all performances are equal, use regular ranking
                        weighted_rank = float(ranks[i])
                    weighted_ranks.append(weighted_rank)
                
                post_ratings = ts.rate(pre_ratings, ranks=weighted_ranks)
                print(f"Applied TrueSkill 2 updates based on cumulative tournament performance")
            else:
                # Standard TrueSkill for non-poker games
                post_ratings = ts.rate(pre_ratings, ranks=ranks)
        else:
            # No rating update - keep current ratings
            post_ratings = pre_ratings
            if args.game_name.lower() == "texasholdem" and not is_tournament_end:
                print(f"Hand {eid}/{total_hands}: Accumulating performance, TrueSkill ratings unchanged")

        # Attach pre/post and persist new ratings in memory
        for part, (pre,), (post,) in zip(participants, pre_ratings, post_ratings):
            part["pre"]  = {"mu": float(pre.mu),  "sigma": float(pre.sigma)}
            part["post"] = {"mu": float(post.mu), "sigma": float(post.sigma)}
            if should_update_ratings:
                ratings[part["id"]] = post

        # Append one JSON line for this hand
        append_jsonl(match_log_path, {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "game": args.game_name,
            "episode_id": eid,
            "seed": cseed,
            "illegal": illegal_game,
            "illegal_player": illegal_player,
            "config": {
                "starting_chips": getattr(env, "starting_chips", None),
                "small_blind": getattr(env, "small_blind", None),
                "big_blind": getattr(env, "big_blind", None),
                "num_players": len(chip_deltas),
                "observation_mode": args.observation_mode,
                "harness": args.harness,
            },
            "participants": participants,
        })

        # Optional: update OVERALL SUMMARY counters
        if illegal_game:
            game_results["illegal_moves"] += 1
        else:
            if chip_deltas:
                max_delta = max(chip_deltas.values())
                winners = [p for p, d in chip_deltas.items() if d == max_delta and max_delta > 0]
                if len(winners) == 1:
                    wk = f"{winners[0]}_wins"
                    if wk in game_results:
                        game_results[wk] += 1
                else:
                    game_results["draws"] += 1

        # Optional: quick leaderboard print
        sorted_lb = sorted(ratings.items(), key=lambda kv: ts.expose(kv[1]), reverse=True)
        print(f"\n--- TrueSkill after hand {eid} ---")
        for name, r in sorted_lb[:10]:
            print(f"  {name:30s} {ts.expose(r):6.2f}  (μ={r.mu:.2f}, σ={r.sigma:.3f})")

        # advance seed each episode
        cseed = None if cseed is None else cseed + 1
    # Generate run summaries
    run_settings = {
        "game_name": args.game_name,
        "num_runs": args.num_runs,
        "max_steps": args.max_steps,
        "observation_mode": args.observation_mode,
        "model_x": args.model_x,
        "model_o": args.model_o,
        "harness": args.harness,
        "seed": args.seed
    }

    if hasattr(env, 'finalize_run_summaries'):
        # Enhanced multi-agent environment
        summaries = env.finalize_run_summaries(run_settings)
        print(f"\n📊 Generated run summaries for {len(summaries)} agents")

    # Print overall summary
    if args.num_runs > 1:
        print(f"\n{'#'*70}")
        print(f"OVERALL SUMMARY ({args.num_runs} games)")
        print(f"{'#'*70}")
        
        # Show wins for each agent
        for agent in agent_keys:
            win_key = f"{agent}_wins"
            if win_key in game_results:
                wins = game_results[win_key]
                win_rate = wins / args.num_runs * 100
                print(f"{agent} wins: {wins} ({win_rate:.1f}%)")
        
        print(f"Draws: {game_results['draws']} ({game_results['draws']/args.num_runs*100:.1f}%)")
        print(f"Games ended by illegal moves: {game_results['illegal_moves']} ({game_results['illegal_moves']/args.num_runs*100:.1f}%)")
        
        # Show model assignments if using player_models
        if args.player_models and len(args.player_models) > 1:
            print(f"\nModel Assignments:")
            for i, model in enumerate(args.player_models):
                player_key = f"player_{i}" if args.game_name.lower() == "texasholdem" else f"player_{i+1}"
                print(f"  {player_key}: {model}")
        
        # Print final TrueSkill leaderboard
        print(f"\n--- FINAL TRUESKILL RANKING ---")
        sorted_ratings = sorted(ratings.items(), key=lambda item: trueskill.expose(item[1]), reverse=True)
        for rank, (agent_key, rating) in enumerate(sorted_ratings, 1):
            exposed_rating = trueskill.expose(rating)
            print(f"  #{rank}: {agent_key:<15} Skill = {exposed_rating:.2f} (μ={rating.mu:.2f}, σ={rating.sigma:.3f})")

        print(f"{'#'*70}")

    save_ratings(ratings_path, ratings)

    env.close()

if __name__ == "__main__":
    main()
