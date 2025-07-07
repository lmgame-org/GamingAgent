from __future__ import annotations
"""
multiagent_tictactoe_runner.py – final complete version
======================================================
Multi‑model Tic‑Tac‑Toe runner aligned with `single_agent_runner.py`.
"""

import argparse
import datetime as _dt
import os
import time
from typing import Any, Dict, Optional

import yaml

from gamingagent.agents.base_agent import BaseAgent
from gamingagent.envs.zoo_01_tictactoe.TicTacToeEnv import MultiTicTacToeEnv
from gamingagent.modules import PerceptionModule, ReasoningModule
from tools.utils import draw_grid_on_image

# Map game_name to environment class and config directory
GAME_ENV_CLASS_MAPPING = {
    "tictactoe": MultiTicTacToeEnv,  # For multi-agent mode
    # ...
}

GAME_CONFIG_MAPPING = {
    "tictactoe": "zoo_01_tictactoe",
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


def merge_cli_yaml(args, cfg, defaults):
    keys = [
        "model_x", 
        "model_o", 
        "num_runs", 
        "max_steps", 
        "observation_mode", 
        "seed",
        "use_custom_prompt", 
        "use_reflection", 
        "use_perception", 
        "use_summary",
        "max_memory", 
        "scaffolding", 
        "vllm_url", 
        "modal_url"
    ]

    for k in keys:
        if k in cfg and _is_default(args, k, defaults):
            setattr(args, k, cfg[k])
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
):
    """
    Creates and returns a game environment instance based on the game name and config.
    - Loads game-specific config (JSON) and pulls out env_init_kwargs.
    - Dynamically instantiates the correct Env class.
    """
    import json
    import os

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
    else:
        print(f"ERROR: Game '{game_name.lower()}' is not defined or implemented in single_agent_runner.py's create_environment function.")
        return None

###############################################################################
# Episode
###############################################################################

def play_episode(env, agents, eid, max_turns, seed):
    obs, _ = env.reset(seed=seed, episode_id=eid)
    totals = {"player_1": 0.0, "player_2": 0.0}
    moves_log = []
    
    for t in range(max_turns):
        cur = env.current_player
        ad, _ = agents[cur].get_action(obs[cur])
        act = None if ad is None else ad.get("action")
        moves_log.append(f"Turn {t+1}: {cur} -> {act}")
        obs, rew, term, trunc, *_ = env.step(cur, act)
        totals["player_1"] += rew["player_1"]
        totals["player_2"] += rew["player_2"]
        env.render()
        if term or trunc:
            break
    
    # Print detailed game summary
    print(f"\n{'='*60}")
    print(f"GAME {eid} SUMMARY")
    print(f"{'='*60}")
    print(f"Total turns: {t+1}")
    print(f"Player 1 (X) total reward: {totals['player_1']:.1f}")
    print(f"Player 2 (O) total reward: {totals['player_2']:.1f}")
    
    # Determine game result
    illegal = False
    illegal_player = None
    if totals['player_1'] == -1.0 and totals['player_2'] == 0.0:
        result = "ILLEGAL MOVE by Player 1 (X)"
        illegal = True
        illegal_player = "player_1"
    elif totals['player_2'] == -1.0 and totals['player_1'] == 0.0:
        result = "ILLEGAL MOVE by Player 2 (O)"
        illegal = True
        illegal_player = "player_2"
    elif totals['player_1'] > 0 and totals['player_2'] < 0:
        result = "Player 1 (X) WINS!"
    elif totals['player_2'] > 0 and totals['player_1'] < 0:
        result = "Player 2 (O) WINS!"
    else:
        result = "DRAW/TIE"
    print(f"Result: {result}")
    
    # Show move history
    print(f"\nMove History:")
    for move in moves_log:
        print(f"  {move}")
    
    print(f"{'='*60}\n")

    if (hasattr(env, 'adapter_p1') and hasattr(env, 'adapter_p2')) and (env.adapter_p1 and env.adapter_p2):
        env.adapter_p1.record_episode_result(
            episode_id=eid,
            score=totals["player_1"],
            steps=t+1,
            total_reward=totals["player_1"],
            total_perf_score=totals["player_1"] # Potentially overwritten
        )

        env.adapter_p2.record_episode_result(
            episode_id=eid,
            score=totals["player_2"],
            steps=t+1,
            total_reward=totals["player_2"],
            total_perf_score=totals["player_2"] # Potentially overwritten
        )
    
    # Return both totals and illegal info for main summary
    return {"player_1": totals["player_1"], "player_2": totals["player_2"], "illegal": illegal, "illegal_player": illegal_player}

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

    env = create_environment(
        game_name=args.game_name,
        observation_mode=args.observation_mode,
        config_dir_name=game_config_name,
        cache_dir=run_root,
        harness=args.harness,
        multiagent_arg=args.multiagent_arg,
    )

    agents = {
        "player_1": make_agent(
            "p1_cache", args.model_x, args, run_root, prompt_path, args._agent_defaults, args._agent_x_cfg
        ),
        "player_2": make_agent(
            "p2_cache", args.model_o, args, run_root, prompt_path, args._agent_defaults, args._agent_o_cfg
        ),
    }

    cseed = args.seed
    game_results = {"player_1_wins": 0, "player_2_wins": 0, "draws": 0, "illegal_moves": 0}
    
    for eid in range(1, args.num_runs + 1):
        result = play_episode(env, agents, eid, args.max_steps, cseed)
        if result:
            if result["illegal"]:
                game_results["illegal_moves"] += 1
            elif result["player_1"] > 0 and result["player_2"] < 0:
                game_results["player_1_wins"] += 1
            elif result["player_2"] > 0 and result["player_1"] < 0:
                game_results["player_2_wins"] += 1
            else:
                game_results["draws"] += 1
        cseed = None if cseed is None else cseed + 1
        time.sleep(1)
    
    # Print overall summary 
    if args.num_runs > 1:
        print(f"\n{'#'*70}")
        print(f"OVERALL SUMMARY ({args.num_runs} games)")
        print(f"{'#'*70}")
        print(f"Player 1 (X) wins: {game_results['player_1_wins']}")
        print(f"Player 2 (O) wins: {game_results['player_2_wins']}")
        print(f"Draws: {game_results['draws']}")
        print(f"Games ended by illegal moves: {game_results['illegal_moves']}")
        print(f"Player 1 win rate: {game_results['player_1_wins']/args.num_runs*100:.1f}%")
        print(f"Player 2 win rate: {game_results['player_2_wins']/args.num_runs*100:.1f}%")
        print(f"Illegal move rate: {game_results['illegal_moves']/args.num_runs*100:.1f}%")
        print(f"{'#'*70}")
    
    env.close()

if __name__ == "__main__":
    main()
