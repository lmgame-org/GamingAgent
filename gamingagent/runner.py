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
    
    # Set defaults from YAML if provided
    # These will be overridden by command-line arguments if specified
    if defaults_map:
        parser.set_defaults(**defaults_map)
        
    if argv_to_parse:
        return parser.parse_args(argv_to_parse)
    return parser.parse_args()

def run_game_episode(agent: BaseAgent, game_env: BaseGameEnv, episode_id: int, args: argparse.Namespace, episode_log_dir: str):
    """Runs a single episode of the specified game using the provided game_env wrapper."""
    print(f"Starting Episode {episode_id} for {args.game_name} with seed {args.seed if args.seed is not None else 'default'}...")

    agent_observation, info = game_env.reset(seed=args.seed, episode_id=episode_id)
    if args.seed is not None: args.seed += 1

    total_reward = 0
    max_tile_value = 0
    score = 0

    episode_log_file = os.path.join(episode_log_dir, f"episode_{episode_id:03d}_log.jsonl")
    # Open file at the start of the episode in append mode to stream logs
    with open(episode_log_file, 'a') as f_log:
        for step_num in range(args.max_steps_per_episode):
            game_env.render_human()

            start_time = time.time()
            action_dict = agent.get_action(agent_observation)
            end_time = time.time()

            action_str_agent = action_dict.get("action", "None").strip().lower()
            thought_process = action_dict.get("thought", "")

            agent_observation, reward, terminated, truncated, info, executed_env_action_idx = game_env.step(action_str_agent)
            
            total_reward += reward
            executed_action_str = game_env.map_env_action_to_agent_action(executed_env_action_idx)

            if args.game_name == "twenty_forty_eight":
                score = info.get('score', 0)
                current_board_state_powers = game_env.get_board_state(game_env.current_raw_observation, info)
                if isinstance(current_board_state_powers, np.ndarray) and current_board_state_powers.size > 0:
                    current_max_power_on_board = int(np.max(current_board_state_powers))
                else:
                    current_max_power_on_board = 0
                current_max_tile_val = 2**current_max_power_on_board if current_max_power_on_board > 0 else 0
                if current_max_tile_val > max_tile_value: max_tile_value = current_max_tile_val
            else:
                score = info.get('score', total_reward)
                current_max_tile_val = info.get('max_value', 0)
                if current_max_tile_val > max_tile_value: max_tile_value = current_max_tile_val

            print(f"E{episode_id} S{step_num+1}: AgentAct='{action_str_agent}', ExecAct='{executed_action_str}', R={reward:.2f}, Score={score}, MaxVal={max_tile_value}, T={end_time-start_time:.2f}s")
            
            log_entry = {
                "episode_id": episode_id,
                "step": step_num + 1,
                "agent_action": action_str_agent,
                "executed_action": executed_action_str,
                "executed_action_idx": int(executed_env_action_idx),
                "thought": thought_process,
                "reward": float(reward),
                "total_reward_so_far": float(total_reward),
                "score": float(score),
                "max_value_in_step": float(current_max_tile_val),
                "time_taken_s": float(f"{end_time-start_time:.3f}")
            }
            if args.game_name == "twenty_forty_eight" and 'current_board_state_powers' in locals() and isinstance(current_board_state_powers, np.ndarray):
                log_entry["board_state_powers_2048"] = current_board_state_powers.tolist()
            
            # Stream log entry to file immediately
            try:
                f_log.write(json.dumps(log_entry) + '\n')
            except TypeError as te:
                print(f"ERROR: Failed to serialize log_entry for JSON. Details: {te}")
                print(f"Log entry causing error: {log_entry}")
                # Optionally, re-raise the error or handle it to allow the episode to continue without this log entry
                # For now, we just print and continue, which means this step won't be logged.
            except Exception as e:
                print(f"ERROR: Failed to write log_entry to file. Details: {e}")
                print(f"Log entry: {log_entry}")
                # Similar handling, printing and continuing.

            if terminated or truncated: break
            
    # File is automatically closed when 'with' block exits
    game_env.close()
    final_score = score
    print(f"Episode {episode_id} finished after {step_num+1} steps. Final Score: {final_score}, Max Value: {max_tile_value}")
    print(f"  Episode log streamed to {episode_log_file}") # Updated message
    return final_score, max_tile_value, step_num + 1, total_reward

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
        observation_mode=args.observation_mode
        # cache_dir for agent is handled internally by BaseAgent
    )

    # Use the agent's cache_dir for runner-specific logs
    runner_log_dir = agent.cache_dir 
    # Ensure it exists (though BaseAgent should have created it)
    os.makedirs(runner_log_dir, exist_ok=True) 
    print(f"Runner logs (episode details, summary) will be saved in agent's cache directory: {runner_log_dir}")

    # Define the path creator for agent's visual observations
    # This uses the agent's own observations_dir for storing images created by the env_wrapper
    def agent_obs_path_fn(episode_id, step_num):
        return os.path.join(agent.observations_dir, f"env_obs_e{episode_id:03d}_s{step_num+1:04d}.png")

    # Initialize the game environment wrapper
    try:
        game_env = get_game_env_wrapper(
            game_name=args.game_name, 
            observation_mode=args.observation_mode, 
            agent_obs_path_creator=agent_obs_path_fn, # Pass the function here
            env_type=args.env_type,
            config_root_dir=args.config_root_dir
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
        score, max_val, steps, total_reward = run_game_episode(agent, game_env, run_id, args, runner_log_dir)
        all_run_results.append({
            "run_id": run_id,
            "score": score,
            "max_value": max_val,
            "steps": steps,
            "total_reward": total_reward
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
        max_values = [r['max_value'] for r in all_run_results]
        steps_list = [r['steps'] for r in all_run_results]
        total_rewards_list = [r['total_reward'] for r in all_run_results]

        stats_map = {
            "Scores": scores,
            "Max Values": max_values,
            "Steps": steps_list,
            "Total Rewards": total_rewards_list
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
