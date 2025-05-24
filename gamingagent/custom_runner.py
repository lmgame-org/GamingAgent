import argparse
import os
import json
import datetime
import time
import numpy as np
import yaml

from gamingagent.agents.base_agent import BaseAgent
from gamingagent.modules import PerceptionModule, ReasoningModule # Observation is imported by Env
# Directly import the specific environment we are using
from gamingagent.envs.custom_01_2048.twentyFortyEightEnv import TwentyFortyEightEnv

game_config_mapping = {"twenty_forty_eight": "custom_01_2048",
                       "sokoban": "custom_02_sokoban",
                       "tetris": "custom_03_tetris",
                       "candy_crush": "custom_04_candy_crush",
                       "super_mario_bros":"retro_01_super_mario_bros",
                       "ace_attorney":"retro_02_ace_attorney"}

def parse_arguments(defaults_map=None, argv_to_parse=None):
    parser = argparse.ArgumentParser(description="Run GamingAgent for the 2048 Gym Environment.")
    # Game name is fixed for this runner, but kept for config loading structure
    parser.add_argument("--game_name", type=str, default="twenty_forty_eight", 
                        help="Name of the game (fixed to twenty_forty_eight for this runner).")
    parser.add_argument("--model_name", type=str, default="claude-3-haiku-20240307",
                        help="Name of the model for the agent.")
    parser.add_argument("--config_root_dir", type=str, default="configs",
                        help="Root directory for agent configurations.")
    parser.add_argument("--harness", action="store_true",
                        help="Use perception-memory-reasoning pipeline (harness mode). Default is False.")
    parser.add_argument("--num_runs", type=int, default=1, help="Number of game episodes.")
    parser.add_argument("--observation_mode", type=str, default="vision",
                        choices=["vision", "text", "both"], help="Agent's observation mode.")
    parser.add_argument("--max_memory", type=int, default=20, help="Agent's max memory entries.")
    parser.add_argument("--max_steps_per_episode", type=int, default=1000, help="Max steps per episode.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for environment.")
    # Env type is fixed to custom gym for this runner

    if defaults_map:
        parser.set_defaults(**defaults_map)
        
    if argv_to_parse:
        return parser.parse_args(argv_to_parse)
    return parser.parse_args()

def create_environment(game_name_arg: str, 
                       obs_mode_arg: str, 
                       config_dir_name_for_env_cfg: str, # For loading game_env_config.json
                       cache_dir_for_adapter: str):
    """Creates and returns a game environment instance based on the game name."""
    
    env_specific_config_path = os.path.join("gamingagent/envs", config_dir_name_for_env_cfg, "game_env_config.json")
    env_init_params = {} # Will be populated based on the specific game

    if game_name_arg == "twenty_forty_eight":
        # Load params specific to 2048
        if os.path.exists(env_specific_config_path):
            with open(env_specific_config_path, 'r') as f:
                env_specific_config = json.load(f)
                env_init_params['size'] = env_specific_config.get('env_init_kwargs', {}).get('size', 4)
                env_init_params['max_pow'] = env_specific_config.get('env_init_kwargs', {}).get('max_pow', 16)
                env_init_params['render_mode'] = env_specific_config.get('render_mode_gym_make', 'human')
                env_init_params['max_stuck_steps_for_adapter'] = env_specific_config.get('max_unchanged_steps_for_termination', 10)
        else:
            print(f"Warning: {env_specific_config_path} for {game_name_arg} not found. Using default env parameters.")
            env_init_params['size'] = 4
            env_init_params['max_pow'] = 16
            env_init_params['render_mode'] = 'human'
            env_init_params['max_stuck_steps_for_adapter'] = 10

        print(f"Initializing environment: {game_name_arg} with params: {env_init_params}")
        env = TwentyFortyEightEnv(
            render_mode=env_init_params.get('render_mode'),
            size=env_init_params.get('size'),
            max_pow=env_init_params.get('max_pow'),
            game_name_for_adapter=game_name_arg,
            observation_mode_for_adapter=obs_mode_arg, 
            agent_cache_dir_for_adapter=cache_dir_for_adapter, 
            game_specific_config_path_for_adapter=env_specific_config_path, # This is path to its own config
            max_stuck_steps_for_adapter=env_init_params.get('max_stuck_steps_for_adapter')
        )
        return env
    # Example for adding another game:
    # elif game_name_arg == "sokoban":
    #     # Load params specific to Sokoban (example, adjust as needed)
    #     if os.path.exists(env_specific_config_path):
    #         with open(env_specific_config_path, 'r') as f:
    #             env_specific_config = json.load(f)
    #             env_init_params['dim_room'] = env_specific_config.get('env_init_kwargs', {}).get('dim_room', (10,10))
    #             env_init_params['num_boxes'] = env_specific_config.get('env_init_kwargs', {}).get('num_boxes', 3)
    #             # ... other sokoban params
    #     else:
    #         print(f"Warning: {env_specific_config_path} for {game_name_arg} not found. Using default env parameters.")
    #         # ... set sokoban defaults ...
    #     from gamingagent.envs.custom_02_sokoban.sokobanEnv import SokobanEnv # Assuming this exists
    #     print(f"Initializing environment: {game_name_arg} with params: {env_init_params}")
    #     env = SokobanEnv(
    #         # ... pass sokoban specific params ...
    #         dim_room=env_init_params.get('dim_room'),
    #         num_boxes=env_init_params.get('num_boxes'),
    #         game_name_for_adapter=game_name_arg, 
    #         observation_mode_for_adapter=obs_mode_arg, 
    #         agent_cache_dir_for_adapter=cache_dir_for_adapter, 
    #         game_specific_config_path_for_adapter=env_specific_config_path 
    #     )
    #     return env
    else:
        print(f"ERROR: Game '{game_name_arg}' is not defined or implemented in custom_runner.py's create_environment function.")
        return None

def run_game_episode(agent: BaseAgent, game_env: TwentyFortyEightEnv, episode_id: int, args: argparse.Namespace):
    print(f"Starting Episode {episode_id} for {args.game_name} with seed {args.seed if args.seed is not None else 'default'}...")

    # Pass episode_id to env.reset
    agent_observation, last_info = game_env.reset(seed=args.seed, episode_id=episode_id)
    if args.seed is not None: args.seed += 1 # Increment seed for next potential run

    total_reward_for_episode = 0.0
    total_perf_score_for_episode = 0.0
    final_step_num = 0

    for step_num in range(args.max_steps_per_episode):
        final_step_num = step_num + 1
        if game_env.render_mode == 'human':
            game_env.render() # Call env's render method directly

        start_time = time.time()
        action_dict = agent.get_action(agent_observation)
        end_time = time.time()
        time_taken_s = end_time - start_time

        action_str_agent = action_dict.get("action", "None").strip().lower()
        thought_process = action_dict.get("thought", "")

        # Step the environment using the new signature, including agent action details
        agent_observation, reward, terminated, truncated, last_info, current_step_perf_score = game_env.step(
            agent_action_str=action_str_agent, 
            thought_process=thought_process, 
            time_taken_s=time_taken_s
        )
            
        total_reward_for_episode += reward
        total_perf_score_for_episode += current_step_perf_score

        if terminated or truncated:
            break
            
    # game_env.close() is called after all runs are complete in main

    final_score_from_env = float(last_info.get('total_score', 0.0)) 

    print(f"Episode {episode_id} finished after {final_step_num} steps. Final Env Score: {final_score_from_env}, Total Reward: {total_reward_for_episode:.2f}, Total Perf Score: {total_perf_score_for_episode:.2f}")
    
    # Record results with the adapter
    if hasattr(game_env, 'adapter') and game_env.adapter:
        game_env.adapter.record_episode_result(
            episode_id=episode_id,
            score=final_score_from_env,
            steps=final_step_num,
            total_reward=total_reward_for_episode,
            total_perf_score=total_perf_score_for_episode
        )
    else:
        print("Warning: game_env.adapter not found. Cannot record episode result for summary.")

    return # No need to return individual run results from here, adapter handles them

def main():
    prelim_parser = argparse.ArgumentParser(add_help=False)
    prelim_parser.add_argument("--game_name", type=str, default="twenty_forty_eight")
    prelim_parser.add_argument("--config_root_dir", type=str, default="configs")
    pre_args, remaining_argv = prelim_parser.parse_known_args()

    config_dir_name = game_config_mapping.get(pre_args.game_name.lower())
    if not config_dir_name:
        print(f"Warning: Game name '{pre_args.game_name}' not found in game_config_mapping. Using game name directly for config path.")
        config_dir_name = pre_args.game_name

    defaults_from_yaml = {}
    config_file_path = os.path.join(pre_args.config_root_dir, config_dir_name, "config.yaml")

    if os.path.exists(config_file_path):
        try:
            with open(config_file_path, 'r') as f:
                loaded_yaml = yaml.safe_load(f)
                if loaded_yaml:
                    if loaded_yaml.get('game_env'):
                        game_env_config_yaml = loaded_yaml['game_env']
                        defaults_from_yaml['num_runs'] = game_env_config_yaml.get('num_runs')
                        defaults_from_yaml['max_steps_per_episode'] = game_env_config_yaml.get('max_steps')
                        defaults_from_yaml['seed'] = game_env_config_yaml.get('seed')

                    if loaded_yaml.get('agent'):
                        agent_config_yaml = loaded_yaml['agent']
                        defaults_from_yaml['model_name'] = agent_config_yaml.get('model_name')
                        defaults_from_yaml['harness'] = agent_config_yaml.get('harness')
                        defaults_from_yaml['observation_mode'] = agent_config_yaml.get('observation_mode')
                        
                        # Still load max_memory from its specific module config if present
                        if agent_config_yaml.get('modules'):
                            if agent_config_yaml['modules'].get('memory_module'):
                                defaults_from_yaml['max_memory'] = agent_config_yaml['modules']['memory_module'].get('max_memory')
                    defaults_from_yaml = {k: v for k, v in defaults_from_yaml.items() if v is not None}
        except Exception as e:
            print(f"Warning: Could not load or process defaults from {config_file_path}: {e}")
    else:
        print(f"Info: Main config file {config_file_path} not found. Using command-line args and hardcoded defaults.")

    args = parse_arguments(defaults_map=defaults_from_yaml, argv_to_parse=remaining_argv)

    agent_prompts_config_path = os.path.join(args.config_root_dir, config_dir_name, "module_prompts.json")
    if not os.path.isfile(agent_prompts_config_path):
        print(f"Warning: Agent prompts file {agent_prompts_config_path} not found. Agent will use default prompts.")
        agent_prompts_config_path = None

    custom_modules_for_agent = None
    if args.harness:
        print("Initializing agent in HARNESS mode.")
        custom_modules_for_agent = {"perception_module": PerceptionModule, "reasoning_module": ReasoningModule}
    else:
        print("Initializing agent in NON-HARNESS (BaseModule direct) mode.")

    agent = BaseAgent(
        game_name=args.game_name,
        model_name=args.model_name,
        config_path=agent_prompts_config_path,
        harness=args.harness,
        max_memory=args.max_memory, custom_modules=custom_modules_for_agent,
        observation_mode=args.observation_mode
    )
    
    runner_log_dir = agent.cache_dir
    os.makedirs(runner_log_dir, exist_ok=True)
    print(f"Agent cache directory (contains episode logs and summary): {runner_log_dir}")

    # Env params are now loaded inside create_environment
    game_env = create_environment(
        game_name_arg=args.game_name,
        obs_mode_arg=args.observation_mode,
        config_dir_name_for_env_cfg=config_dir_name, # Pass the mapped dir name
        cache_dir_for_adapter=runner_log_dir
    )

    if game_env is None:
        print("Failed to create game environment. Exiting.")
        return

    for i in range(args.num_runs):
        run_id = i + 1
        # run_game_episode now doesn't return values, results are stored in adapter
        run_game_episode(agent, game_env, run_id, args)
        if i < args.num_runs - 1:
            print("Cooldown for 1 second before next run...")
            time.sleep(1)
    
    # Finalize and save summary using the adapter
    overall_stat_summary = {}
    if hasattr(game_env, 'adapter') and game_env.adapter:
        overall_stat_summary = game_env.adapter.finalize_and_save_summary(vars(args))
    else:
        print("Warning: game_env.adapter not found. Cannot finalize and save summary.")

    game_env.close() # Close environment after all runs

    print("\n" + "="*30 + " Overall Summary " + "="*30)
    print(f"Game: {args.game_name}, Model: {args.model_name}, Mode: {'Harness' if args.harness else 'BaseOnly'}, ObsMode: {args.observation_mode}")
    print(f"Number of runs: {args.num_runs}")
    
    if args.num_runs > 0 and overall_stat_summary:
        for key_snake, stats in overall_stat_summary.items():
            # Convert snake_case key back to Title Case for printing
            key_title = key_snake.replace("_", " ").title()
            if stats["mean"] is not None:
                print(f"Average {key_title}: {stats['mean']:.2f} (Std: {stats['std']:.2f}, Min: {stats['min']:.2f}, Max: {stats['max']:.2f})")
            else:
                print(f"Average {key_title}: N/A (no data)")
    else:
        print("No runs were completed or summary data is unavailable.")

if __name__ == "__main__":
    main() 