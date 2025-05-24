import argparse
import os
import json
import datetime
import time
import numpy as np
import yaml # Added for loading config.yaml
# No longer directly using gym here, it's handled by the env wrapper
# from PIL import Image, ImageDraw, ImageFont # Handled by specific env wrapper

from gamingagent.agents.base_agent import BaseAgent
from gamingagent.modules import PerceptionModule, ReasoningModule # Observation is imported by BaseGameEnv
from gamingagent.envs import get_game_env_wrapper, BaseGameEnv # Import the factory and base class

# --- Game Specifics are now in their respective env wrapper files ---

def parse_arguments(defaults_map=None, argv_to_parse=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run GamingAgent for various games.")
    parser.add_argument("--game_name", type=str, default="twenty_forty_eight",
                        help="Name of the game to run (e.g., twenty_forty_eight). This also determines which config.yaml is loaded for defaults.")
    parser.add_argument("--model_name", type=str, default="claude-3-5-sonnet-20241022",
                        help="Name of the model for the agent.")
    parser.add_argument("--config_root_dir", type=str, default="configs",
                        help="Root directory for game and agent configurations.")
    parser.add_argument("--harness", action="store_true",
                        help="Use perception-memory-reasoning pipeline (harness mode).")
    parser.add_argument("--num_runs", type=int, default=1, help="Number of game episodes.")
    parser.add_argument("--observation_mode", type=str, default="vision",
                        choices=["vision", "text", "both"], help="Agent's observation mode.")
    parser.add_argument("--max_memory", type=int, default=20, help="Agent's max memory entries.")
    parser.add_argument("--max_steps_per_episode", type=int, default=1000, help="Max steps per episode.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for environment.")
    parser.add_argument("--env_type", type=str, default="custom", choices=["custom", "retro"],
                        help="Type of game environment framework to use ('custom' for envs in gamingagent/envs, 'retro' for gym-retro).")
    # New arguments for token_limit and reasoning_effort
    parser.add_argument("--token_limit", type=int, default=8000, help="Token limit for the agent's model.")
    parser.add_argument("--reasoning_effort", type=str, default="medium",
                        choices=["low", "medium", "high"], help="Reasoning effort for the agent's model.")
    parser.add_argument("--replay_mode", type=str, default="none", 
                        choices=["none", "frame_by_frame", "seed_and_actions", "both"],
                        help="Generate replay video(s) after each episode. 'none' for no replay, 'frame_by_frame' for visual log, 'seed_and_actions' for re-simulation, 'both' for both.")
    parser.add_argument("--replay_frame_duration", type=float, default=0.5, 
                        help="Duration of each frame in the replay GIF.")
    
    # Set defaults from YAML if provided
    # These will be overridden by command-line arguments if specified
    if defaults_map:
        parser.set_defaults(**defaults_map)
        
    if argv_to_parse:
        return parser.parse_args(argv_to_parse)
    return parser.parse_args()

def run_game_episode(agent: BaseAgent, game_env: BaseGameEnv, episode_id: int, args: argparse.Namespace):
    print(f"Starting Episode {episode_id} for {args.game_name} with seed {args.seed if args.seed is not None else 'default'}...")

    # Store initial seed for this episode for replay purposes
    initial_seed_for_episode = args.seed

    agent_observation, last_info = game_env.reset(seed=args.seed, episode_id=episode_id)
    if args.seed is not None: args.seed += 1 # Increment for next episode if running multiple

    total_reward = 0
    total_episode_perf_score = 0.0

    # Lists to store data for replay logging
    episode_raw_observations_for_replay = []
    episode_agent_actions_str_for_replay = []
    episode_infos_for_replay = []
    episode_executed_action_ints_for_replay = []

    for step_num in range(args.max_steps_per_episode):
        game_env.render_human()

        start_time = time.time()
        action_dict = agent.get_action(agent_observation)
        end_time = time.time()
        time_taken_s = end_time - start_time

        # Ensure action_str_agent is always a string, defaulting to "None" if action is not found or is None
        agent_action_value = action_dict.get("action")
        action_str_agent = str(agent_action_value).strip().lower() if agent_action_value is not None else "none"
        thought_process = action_dict.get("thought", "")

        # Unpack perf_score from the game_env.step() return values
        agent_observation, reward, terminated, truncated, last_info, current_step_perf_score = game_env.step(
            action_str_agent, thought_process, time_taken_s
        )
            
        total_reward += reward
        total_episode_perf_score += current_step_perf_score

        # Log data for replay
        if last_info:
            episode_raw_observations_for_replay.append(last_info.get('raw_env_observation_for_replay'))
            episode_agent_actions_str_for_replay.append(action_str_agent)
            episode_infos_for_replay.append(last_info.copy()) # Store a copy
            episode_executed_action_ints_for_replay.append(last_info.get('executed_action_int_for_replay'))

        if terminated or truncated: break
            
    game_env.close()

    final_score = float(last_info.get('score', 0.0)) # This usually comes from the game's info dict

    print(f"Episode {episode_id} finished after {step_num+1} steps. Final Score: {final_score}, Total Reward: {total_reward}, Total Perf Score: {total_episode_perf_score}")

    # Save replay log for the episode
    episode_replay_data = {
        "initial_seed": initial_seed_for_episode,
        "raw_observations": episode_raw_observations_for_replay,
        "agent_actions_str": episode_agent_actions_str_for_replay,
        "infos": episode_infos_for_replay,
        "executed_action_ints": episode_executed_action_ints_for_replay,
        "final_score": final_score,
        "total_reward": total_reward,
        "steps_taken": step_num + 1
    }
    
    runner_log_dir = agent.cache_dir # Defined later, but needed here
    replay_log_filename = os.path.join(runner_log_dir, f"episode_{episode_id:03d}_replay_log.json")
    try:
        with open(replay_log_filename, 'w') as f:
            # Custom encoder for numpy types if they sneak in (e.g. in raw_obs board)
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    if isinstance(obj, np.floating):
                        return float(obj)
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return super(NumpyEncoder, self).default(obj)
            json.dump(episode_replay_data, f, indent=2, cls=NumpyEncoder)
        print(f"Saved episode replay log to {replay_log_filename}")
    except Exception as e:
        print(f"Error saving episode replay log: {e}")

    # Trigger replays if requested
    if args.replay_mode in ["frame_by_frame", "both"]:
        print(f"Generating frame-by-frame replay for episode {episode_id}...")
        # Ensure game_env is available (it was closed, might need re-init or a specific replay instance)
        # For now, assuming game_env methods are callable or we have a new instance for replay.
        # This might require passing the runner_log_dir to game_env.game_replay for output path construction.
        try:
            replay_video_path = os.path.join(runner_log_dir, f"episode_{episode_id:03d}_frame_replay.gif")
            game_env.game_replay(
                replay_data=episode_replay_data, 
                output_video_path=replay_video_path, 
                frame_duration=args.replay_frame_duration
            )
        except Exception as e:
            print(f"Error generating frame-by-frame replay: {e}")

    if args.replay_mode in ["seed_and_actions", "both"]:
        if hasattr(game_env, 'replay_from_seed_and_actions'):
            print(f"Generating seed-and-actions replay for episode {episode_id}...")
            try:
                replay_video_path = os.path.join(runner_log_dir, f"episode_{episode_id:03d}_seed_action_replay.gif")
                # Ensure the tile_size parameter for rendering in this replay is available or sensible.
                # For now, assuming a default or that it can be fetched from game_env config.
                tile_size = game_env._game_env_config.get("tile_size_for_render", 32) if hasattr(game_env, '_game_env_config') else 32

                game_env.replay_from_seed_and_actions(
                    initial_seed=initial_seed_for_episode,
                    executed_action_ints=episode_executed_action_ints_for_replay,
                    output_video_path=replay_video_path,
                    frame_duration=args.replay_frame_duration,
                    tile_size_for_replay_render=tile_size
                )
            except Exception as e:
                print(f"Error generating seed-and-actions replay: {e}")
        else:
            print("Warning: game_env does not have 'replay_from_seed_and_actions' method.")

    return final_score, step_num + 1, total_reward, total_episode_perf_score

def main():
    # 1. Initial parse for config location to load YAML defaults
    prelim_parser = argparse.ArgumentParser(add_help=False) # Avoids help conflict with main parser
    prelim_parser.add_argument("--game_name", type=str, default="twenty_forty_eight")
    prelim_parser.add_argument("--config_root_dir", type=str, default="configs")
    pre_args, remaining_argv = prelim_parser.parse_known_args()

    # 2. Load YAML and extract defaults
    defaults_from_yaml = {}
    config_file_path = os.path.join(pre_args.config_root_dir, pre_args.game_name, "config.yaml")

    if os.path.exists(config_file_path):
        try:
            with open(config_file_path, 'r') as f:
                loaded_yaml = yaml.safe_load(f)
                if loaded_yaml:
                    # Map YAML keys to argparse dest names
                    if loaded_yaml.get('game_env'):
                        game_env_config = loaded_yaml['game_env']
                        defaults_from_yaml['num_runs'] = game_env_config.get('num_runs')
                        defaults_from_yaml['max_steps_per_episode'] = game_env_config.get('max_steps')
                        defaults_from_yaml['seed'] = game_env_config.get('seed')
                        defaults_from_yaml['env_type'] = game_env_config.get('env_type') # Read env_type from YAML
                        # 'game_name' from yaml could be game_env_config.get('name'), but pre_args.game_name already defines it for loading.

                    if loaded_yaml.get('agent'):
                        agent_config = loaded_yaml['agent']
                        defaults_from_yaml['model_name'] = agent_config.get('model_name')
                        defaults_from_yaml['harness'] = agent_config.get('harness')
                        defaults_from_yaml['token_limit'] = agent_config.get('token_limit') # Load token_limit
                        defaults_from_yaml['reasoning_effort'] = agent_config.get('reasoning_effort') # Load reasoning_effort
                        if agent_config.get('modules'):
                            if agent_config['modules'].get('base_module'):
                                defaults_from_yaml['observation_mode'] = agent_config['modules']['base_module'].get('observation_mode')
                            if agent_config['modules'].get('memory_module'):
                                defaults_from_yaml['max_memory'] = agent_config['modules']['memory_module'].get('max_memory')

                    # Filter out None values, so they don't mistakenly override hardcoded defaults in argparse
                    # if the key exists in YAML but its value is null.
                    defaults_from_yaml = {k: v for k, v in defaults_from_yaml.items() if v is not None}
        except yaml.YAMLError as e:
            print(f"Warning: Error parsing YAML file {config_file_path}: {e}. Using command-line args and hardcoded defaults.")
        except Exception as e:
            print(f"Warning: Could not load or process defaults from {config_file_path}: {e}")
    else:
        print(f"Info: Main config file {config_file_path} not found. Using command-line args and hardcoded defaults.")

    # 3. Parse all arguments, with YAML defaults applied if not overridden by command line
    args = parse_arguments(defaults_map=defaults_from_yaml, argv_to_parse=remaining_argv)

    # Timestamp and safe_model_name are now primarily for BaseAgent's internal cache dir naming.
    # Runner will use the agent.cache_dir directly.

    agent_prompts_config_path = os.path.join(args.config_root_dir, args.game_name, "module_prompts.json")
    if not os.path.isfile(agent_prompts_config_path):
        print(f"Warning: Agent prompts file {agent_prompts_config_path} not found. Agent will use default prompts.")
        agent_prompts_config_path = None

    custom_modules_for_agent = None
    if args.harness:
        print("Initializing agent in HARNESS mode.")
        custom_modules_for_agent = {"perception_module": PerceptionModule, "reasoning_module": ReasoningModule}
    else:
        print("Initializing agent in NON-HARNESS (BaseModule direct) mode.")

    # Initialize BaseAgent first to get its observations_dir
    agent = BaseAgent(
        game_name=args.game_name, model_name=args.model_name,
        config_path=agent_prompts_config_path, harness=args.harness,
        max_memory=args.max_memory, custom_modules=custom_modules_for_agent,
        observation_mode=args.observation_mode,
        token_limit=args.token_limit, # Pass token_limit
        reasoning_effort=args.reasoning_effort # Pass reasoning_effort
        # cache_dir for agent is handled internally by BaseAgent
    )

    # Use the agent's cache_dir for runner-specific logs
    runner_log_dir = agent.cache_dir 
    # Ensure it exists (though BaseAgent should have created it)
    os.makedirs(runner_log_dir, exist_ok=True) 
    print(f"Agent cache directory (contains episode logs and summary): {runner_log_dir}")

    # Initialize the game environment wrapper
    try:
        game_env = get_game_env_wrapper(
            game_name=args.game_name, 
            observation_mode=args.observation_mode, 
            agent_observations_base_dir=agent.observations_dir,
            env_type=args.env_type,
            config_root_dir=args.config_root_dir,
            log_root_dir=runner_log_dir # Pass agent.cache_dir as log_root_dir for BaseGameEnv
        )
    except ValueError as e:
        print(f"Error initializing game environment wrapper: {e}")
        return
    except FileNotFoundError as e:
        print(f"Error: Configuration file missing for game environment. {e}")
        return

    all_run_results = [] # Stores detailed results for each run
    # The `results` dict for aggregate stats will be built from all_run_results later

    for i in range(args.num_runs):
        run_id = i + 1
        # Unpack total_episode_perf_score from run_game_episode results
        score, steps, total_reward, total_episode_perf_score = run_game_episode(agent, game_env, run_id, args)
        all_run_results.append({
            "run_id": run_id,
            "score": score,
            "steps": steps,
            "total_reward": total_reward,
            "total_episode_perf_score": total_episode_perf_score # Store it
        })
        if i < args.num_runs - 1:
            print("Cooldown for 2 seconds before next run...")
            time.sleep(2)

    print("\n" + "="*30 + " Overall Summary " + "="*30)
    print(f"Game: {args.game_name}, Model: {args.model_name}, Mode: {'Harness' if args.harness else 'BaseOnly'}, ObsMode: {args.observation_mode}")
    print(f"Number of runs: {args.num_runs}")
    
    summary_data = {"settings": vars(args), "individual_run_results": all_run_results, "overall_stat_summary": {}}

    if args.num_runs > 0 and all_run_results:
        # Calculate overall statistics
        scores = [r['score'] for r in all_run_results]
        steps_list = [r['steps'] for r in all_run_results]
        total_rewards_list = [r['total_reward'] for r in all_run_results]
        total_perf_scores_list = [r['total_episode_perf_score'] for r in all_run_results] # Get list of perf scores

        stats_map = {
            "Scores": scores,
            "Steps": steps_list,
            "Total Rewards": total_rewards_list,
            "Total Performance Scores": total_perf_scores_list # Add to stats_map
        }

        for key, values in stats_map.items():
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                print(f"Average {key}: {mean_val:.2f} (Std: {std_val:.2f})")
                summary_data["overall_stat_summary"][key.lower().replace(" ", "_")] = {"mean": mean_val, "std": std_val, "values": values}
            else:
                print(f"Average {key}: N/A (no data)")
                summary_data["overall_stat_summary"][key.lower().replace(" ", "_")] = {"mean": None, "std": None, "values": []}
    
    summary_file_path = os.path.join(runner_log_dir, "run_summary.json") # Renamed file
    with open(summary_file_path, 'w') as f: json.dump(summary_data, f, indent=2)
    print(f"Run summary saved to: {summary_file_path}")

if __name__ == "__main__":
    main()
