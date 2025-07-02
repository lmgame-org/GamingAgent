"""multiagent_tictactoe_runner.py
================================
A minimal runner patterned after *custom_runner.py* but specialised for the
**multi‑model** Tic‑Tac‑Toe setup.  It wires two independent `BaseAgent`s to
`MultiTicTacToeEnv`, stepping the environment one PettingZoo turn at a time.

Usage (CLI)
-----------
```
python multiagent_tictactoe_runner.py \
    --model_name_x gpt-4o \
    --model_name_o claude-3-haiku \
    --num_runs 5 \
    --observation_mode vision
```

Only the arguments relevant to this single game are exposed.  All prompt/config
paths are resolved relative to `gamingagent/configs/zoo_01_tictactoe/`.
"""
from __future__ import annotations

import argparse, os, json, time, datetime, sys
from typing import Dict, Any, Optional

import gymnasium as gym
import numpy as np

from gamingagent.agents.base_agent import BaseAgent
from gamingagent.envs.zoo_01_tictactoe.tictactoe_env import MultiTicTacToeEnv

# ─── CLI ---------------------------------------------------------------------

def str_to_bool(v: str | bool):
    if isinstance(v, bool):
        return v
    if v.lower() in {"yes", "true", "t", "y", "1"}:
        return True
    if v.lower() in {"no", "false", "f", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected")

def get_args(argv: Optional[list[str]] = None):
    p = argparse.ArgumentParser("Multi‑TicTacToe Runner")
    p.add_argument("--num_runs", type=int, default=1, help="Number of episodes")
    p.add_argument("--max_steps", type=int, default=30, help="Max PettingZoo turns per episode")
    p.add_argument("--observation_mode", choices=["vision", "text", "both"], default="vision")
    p.add_argument("--model_name_x", type=str, default="gpt-4o", help="Model for player_1 (X)")
    p.add_argument("--model_name_o", type=str, default="claude-3-haiku", help="Model for player_2 (O)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--use_custom_prompt", action="store_true")
    return p.parse_args(argv)

# ─── Helper: create BaseAgent ------------------------------------------------

def make_agent(model_name: str, role_tag: str, cache_root: str, obs_mode: str, prompt_path: str | None):
    cache_dir = os.path.join(cache_root, role_tag)
    os.makedirs(cache_dir, exist_ok=True)
    return BaseAgent(
        game_name="tictactoe",
        model_name=model_name,
        config_path=prompt_path,
        harness=False,
        use_custom_prompt=True,
        max_memory=10,
        observation_mode=obs_mode,
        cache_dir=cache_dir,
    )

# ─── Episode loop ------------------------------------------------------------

def play_episode(env: MultiTicTacToeEnv, agents: Dict[str, BaseAgent], ep_id: int, max_turns: int):
    obs, _ = env.reset(seed=None, episode_id=ep_id)
    total_reward = {"player_1": 0.0, "player_2": 0.0}
    for turn in range(max_turns):
        current = env.current_player  # "player_1" or "player_2"
        agent = agents[current]
        act_dict, _ = agent.get_action(obs[current])
        act_str = None if act_dict is None else act_dict.get("action")
        obs, rewards, term, trunc, _, _ = env.step(current, act_str)
        total_reward["player_1"] += rewards["player_1"]
        total_reward["player_2"] += rewards["player_2"]
        env.render()
        if term or trunc:
            break
    print(f"Episode {ep_id} finished after {turn+1} turns. Rewards: X={total_reward['player_1']}, O={total_reward['player_2']}")

# ─── Main --------------------------------------------------------------------

def main(argv: Optional[list[str]] = None):
    args = get_args(argv)

    run_root = os.path.join(
        "cache",
        "tictactoe",
        f"multi_{args.model_name_x[:10]}_{args.model_name_o[:10]}",
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    os.makedirs(run_root, exist_ok=True)

    # Config path (prompts etc.)
    cfg_dir = os.path.join("gamingagent", "configs", "zoo_01_tictactoe")
    prompt_path = os.path.join(cfg_dir, "module_prompts.json")
    if not os.path.isfile(prompt_path):
        prompt_path = None

    # Create environment
    env = MultiTicTacToeEnv(
        render_mode="human",
        p1_cache=os.path.join(run_root, "p1_cache"),
        p2_cache=os.path.join(run_root, "p2_cache"),
        game_specific_config_path_for_adapter=os.path.join("gamingagent", "envs", "zoo_01_tictactoe", "game_env_config.json"),
    )

    # Create agents
    agents = {
        "player_1": make_agent(args.model_name_x, "player_1", run_root, args.observation_mode, prompt_path),
        "player_2": make_agent(args.model_name_o, "player_2", run_root, args.observation_mode, prompt_path),
    }

    for ep in range(1, args.num_runs + 1):
        play_episode(env, agents, ep, args.max_steps)
        if ep < args.num_runs:
            time.sleep(1)

    env.close()

if __name__ == "__main__":
    main()
