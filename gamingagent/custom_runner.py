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

    # Get final score from the info dict provided by the env
    # In our modified TwentyFortyEightEnv, 'total_score' is in the info dict.
    final_score_from_env = float(last_info.get('total_score', 0.0)) 

    print(f"Episode {episode_id} finished after {final_step_num} steps. Final Env Score: {final_score_from_env}, Total Reward: {total_reward_for_episode:.2f}, Total Perf Score: {total_perf_score_for_episode:.2f}")
    return final_score_from_env, final_step_num, total_reward_for_episode, total_perf_score_for_episode

def main():
    prelim_parser = argparse.ArgumentParser(add_help=False)
    prelim_parser.add_argument("--game_name", type=str, default="twenty_forty_eight")
    prelim_parser.add_argument("--config_root_dir", type=str, default="configs")
    pre_args, remaining_argv = prelim_parser.parse_known_args()

    # Resolve the actual config directory name using the mapping
    config_dir_name = game_config_mapping.get(pre_args.game_name.lower())
    if not config_dir_name:
        print(f"Warning: Game name '{pre_args.game_name}' not found in game_config_mapping. Using game name directly for config path.")
        config_dir_name = pre_args.game_name # Fallback

    defaults_from_yaml = {}
    # Use the resolved config_dir_name for the path
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
                        # render_mode for env is now an init param of TwentyFortyEightEnv, can be in game_env_config.json

                    if loaded_yaml.get('agent'):
                        agent_config_yaml = loaded_yaml['agent']
                        defaults_from_yaml['model_name'] = agent_config_yaml.get('model_name')
                        defaults_from_yaml['harness'] = agent_config_yaml.get('harness')
                        if agent_config_yaml.get('modules'):
                            if agent_config_yaml['modules'].get('base_module'):
                                defaults_from_yaml['observation_mode'] = agent_config_yaml['modules']['base_module'].get('observation_mode')
                            if agent_config_yaml['modules'].get('memory_module'):
                                defaults_from_yaml['max_memory'] = agent_config_yaml['modules']['memory_module'].get('max_memory')
                    defaults_from_yaml = {k: v for k, v in defaults_from_yaml.items() if v is not None}
        except Exception as e:
            print(f"Warning: Could not load or process defaults from {config_file_path}: {e}")
    else:
        print(f"Info: Main config file {config_file_path} not found. Using command-line args and hardcoded defaults.")

    args = parse_arguments(defaults_map=defaults_from_yaml, argv_to_parse=remaining_argv)

    # agent_prompts_config_path should also use the mapped config_dir_name
    # args.game_name will be the user-friendly name (e.g., "twenty_forty_eight")
    # config_dir_name will be the mapped name (e.g., "custom_01_2048")
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
        game_name=args.game_name, # Agent can still use the user-friendly game_name
        model_name=args.model_name,
        config_path=agent_prompts_config_path, # This path is now correctly mapped
        harness=args.harness,
        max_memory=args.max_memory, custom_modules=custom_modules_for_agent,
        observation_mode=args.observation_mode
    )
    
    runner_log_dir = agent.cache_dir
    os.makedirs(runner_log_dir, exist_ok=True)
    print(f"Agent cache directory (contains episode logs and summary): {runner_log_dir}")

    # Load env parameters from game_env_config.json
    env_params = {}
    # Use the config_dir_name (e.g., "custom_01_2048") from the mapping 
    # to locate the environment's specific game_env_config.json.
    # This assumes game_env_config.json is inside a folder named like custom_01_2048 within gamingagent/envs/
    env_specific_config_path = os.path.join("gamingagent/envs", config_dir_name, "game_env_config.json")
    
    if os.path.exists(env_specific_config_path):
        with open(env_specific_config_path, 'r') as f:
            env_specific_config = json.load(f)
            env_params['size'] = env_specific_config.get('env_init_kwargs', {}).get('size', 4)
            env_params['max_pow'] = env_specific_config.get('env_init_kwargs', {}).get('max_pow', 16)
            env_params['render_mode'] = env_specific_config.get('render_mode_gym_make', 'human')
            # max_stuck_steps for adapter is also in this config
            env_params['max_stuck_steps_for_adapter'] = env_specific_config.get('max_unchanged_steps_for_termination', 10)
    else:
        print(f"Warning: {env_specific_config_path} not found. Using default env parameters.")
        # Set defaults if file not found, to match TwentyFortyEightEnv defaults
        env_params['size'] = 4
        env_params['max_pow'] = 16
        env_params['render_mode'] = 'human'
        env_params['max_stuck_steps_for_adapter'] = 10

    game_env = TwentyFortyEightEnv(
        render_mode=env_params['render_mode'],
        size=env_params['size'],
        max_pow=env_params['max_pow'],
        game_name_for_adapter=args.game_name, # User-friendly name for adapter's internal game_name
        observation_mode_for_adapter=args.observation_mode, 
        agent_cache_dir_for_adapter=runner_log_dir, 
        # This path needs to be for the adapter to load its OWN specific settings (like action_mapping)
        # This should use the mapped config_dir_name if the adapter's settings are in game_env_config.json within that mapped dir.
        # OR if game_env_config.json is always at the env's own dir, it should be that.
        # The TwentyFortyEightEnv constructor has `game_specific_config_path_for_adapter` which is used by GymEnvAdapter
        # This adapter config should indeed come from where the game_env_config.json is located.
        game_specific_config_path_for_adapter=env_specific_config_path, 
        max_stuck_steps_for_adapter=env_params['max_stuck_steps_for_adapter']
    )

    all_run_results = []
    for i in range(args.num_runs):
        run_id = i + 1
        score, steps, total_reward, total_episode_perf_score = run_game_episode(agent, game_env, run_id, args)
        all_run_results.append({
            "run_id": run_id,
            "score": score, # This is final_score_from_env
            "steps": steps,
            "total_reward_for_episode": total_reward,
            "total_perf_score_for_episode": total_episode_perf_score
        })
        if i < args.num_runs - 1:
            print("Cooldown for 1 second before next run...") # Shorter cooldown
            time.sleep(1)
    
    game_env.close() # Close environment after all runs

    print("\n" + "="*30 + " Overall Summary " + "="*30)
    print(f"Game: {args.game_name}, Model: {args.model_name}, Mode: {'Harness' if args.harness else 'BaseOnly'}, ObsMode: {args.observation_mode}")
    print(f"Number of runs: {args.num_runs}")
    
    summary_data = {"settings": vars(args), "individual_run_results": all_run_results, "overall_stat_summary": {}}

    if args.num_runs > 0 and all_run_results:
        scores = [r['score'] for r in all_run_results]
        steps_list = [r['steps'] for r in all_run_results]
        total_rewards_list = [r['total_reward_for_episode'] for r in all_run_results]
        total_perf_scores_list = [r['total_perf_score_for_episode'] for r in all_run_results]

        stats_map = {
            "Final Env Scores": scores,
            "Steps Taken": steps_list,
            "Total Rewards": total_rewards_list,
            "Total Performance Scores": total_perf_scores_list
        }

        for key, values in stats_map.items():
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                min_val = np.min(values)
                max_val = np.max(values)
                print(f"Average {key}: {mean_val:.2f} (Std: {std_val:.2f}, Min: {min_val:.2f}, Max: {max_val:.2f})")
                # Convert NumPy types to Python native types for JSON serialization
                summary_data["overall_stat_summary"][key.lower().replace(" ", "_")] = {
                    "mean": float(mean_val),
                    "std": float(std_val),
                    "min": int(min_val) if isinstance(min_val, np.integer) else float(min_val),
                    "max": int(max_val) if isinstance(max_val, np.integer) else float(max_val),
                    "values": [int(v) if isinstance(v, np.integer) else float(v) for v in values]
                }
            else:
                print(f"Average {key}: N/A (no data)")
                summary_data["overall_stat_summary"][key.lower().replace(" ", "_")] = {"mean": None, "std": None, "min":None, "max":None, "values": []}
    
    summary_file_path = os.path.join(runner_log_dir, "gym_run_summary.json")
    with open(summary_file_path, 'w') as f: json.dump(summary_data, f, indent=2)
    print(f"Run summary saved to: {summary_file_path}")

if __name__ == "__main__":
    main() 