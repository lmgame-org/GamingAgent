import argparse
import subprocess
import os
import sys
import multiprocessing
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# python run.py --model_name gemini-2.0-flash --game_names sokoban,tetris,candy_crush,twenty_forty_eight,super_mario_bros --harness_mode both

def run_single_game_config(game_name: str, model_name: str, use_harness: bool, custom_runner_script_path: str) -> Tuple[str, bool, bool, str, str]:
    """
    Constructs and runs a single configuration of single_agent_runner.py.
    Returns a tuple: (game_name, use_harness, success_status, stdout, stderr)
    """
    print(f"  Preparing to run: Game='{game_name}', Model='{model_name}', Harness='{use_harness}'")
    
    base_command = [
        sys.executable,
        custom_runner_script_path,
        "--game_name", game_name,
        "--model_name", model_name
    ]
    
    if use_harness:
        command = base_command + ["--harness"]
    else:
        command = base_command

    log_file_base = f"logs/run_log_{game_name}_{model_name}_{'harness_true' if use_harness else 'harness_false'}.txt"
    os.makedirs(os.path.dirname(log_file_base), exist_ok=True)

    try:
        # Using shell=False is generally safer, arguments are passed as a list.
        process = subprocess.run(command, check=True, text=True, capture_output=True)
        print(f"    SUCCESS: Game='{game_name}', Model='{model_name}', Harness='{use_harness}'. Log: {log_file_base}")
        with open(log_file_base, 'w') as f_log:
            f_log.write("--- STDOUT ---\n")
            f_log.write(process.stdout if process.stdout else "")
            f_log.write("\n--- STDERR ---\n")
            f_log.write(process.stderr if process.stderr else "")
        return game_name, use_harness, True, process.stdout, process.stderr
    except subprocess.CalledProcessError as e:
        error_message = f"    ERROR: Game='{game_name}', Model='{model_name}', Harness='{use_harness}'. Log: {log_file_base}\n"
        error_message += f"      Command: {' '.join(e.cmd)}\n"
        error_message += f"      Return code: {e.returncode}\n"
        error_message += f"      Output (stdout):\n{e.stdout}\n"
        error_message += f"      Output (stderr):\n{e.stderr}"
        print(error_message)
        with open(log_file_base, 'w') as f_log:
            f_log.write(error_message)
        return game_name, use_harness, False, e.stdout, e.stderr
    except Exception as ex:
        error_message = f"    UNEXPECTED ERROR: Game='{game_name}', Model='{model_name}', Harness='{use_harness}'. Log: {log_file_base}\n"
        error_message += f"      Exception: {str(ex)}\n"
        print(error_message)
        with open(log_file_base, 'w') as f_log:
            f_log.write(error_message)
        return game_name, use_harness, False, "", str(ex)


def main():
    parser = argparse.ArgumentParser(description="Run game simulations in parallel using evaluation script and then trigger evaluation.")
    parser.add_argument("--model_name", type=str, default="gemini-2.0-flash",
                        help="Name of the model for the agent.")
    parser.add_argument("--game_names", type=str, 
                        default="sokoban,tetris,candy_crush,twenty_forty_eight",
                        help="Comma-separated list of game names.")
    parser.add_argument("--harness_mode", type=str, default="both",
                        choices=["true", "false", "both"],
                        help="Which harness mode to run: 'true' for harness only, 'false' for base_module only, 'both' for both.")
    parser.add_argument("--max_parallel_procs", type=int, default=None, # Default to cpu_count
                        help="Maximum number of parallel game instances to run. Defaults to number of CPU cores.")


    args = parser.parse_args()

    game_names_list = [name.strip() for name in args.game_names.split(',') if name.strip()]
    model_name = args.model_name

    if not game_names_list:
        print("Error: No game names provided.")
        sys.exit(1)
    if not model_name:
        print("Error: Model name must be provided.")
        sys.exit(1)

    custom_runner_script_path = os.path.join("lmgame-bench", "single_agent_runner.py")
    if not os.path.exists(custom_runner_script_path):
        print(f"Error: script not found at {custom_runner_script_path}")
        sys.exit(1)

    tasks_to_run = []
    for game_name in game_names_list:
        if args.harness_mode == "true":
            tasks_to_run.append((game_name, model_name, True, custom_runner_script_path))
        elif args.harness_mode == "false":
            tasks_to_run.append((game_name, model_name, False, custom_runner_script_path))
        elif args.harness_mode == "both":
            tasks_to_run.append((game_name, model_name, True, custom_runner_script_path))
            tasks_to_run.append((game_name, model_name, False, custom_runner_script_path))

    if not tasks_to_run:
        print("No tasks to run based on harness_mode selection.")
        sys.exit(0)

    num_processes = args.max_parallel_procs if args.max_parallel_procs else multiprocessing.cpu_count()
    print(f"Starting {len(tasks_to_run)} tasks with up to {num_processes} parallel processes...")

    # Create a logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    results = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Submit all tasks and keep track of the futures
        future_to_task_args = {}
        for task_args in tasks_to_run:
            future = executor.submit(run_single_game_config, *task_args)
            future_to_task_args[future] = task_args
            # print(f"Submitted task: {task_args[:3]}...") # Optional: for seeing submission order
            time.sleep(2) # Add 1-second delay between submissions
        
        print(f"All {len(tasks_to_run)} tasks submitted. Waiting for completion...")
        for future in as_completed(future_to_task_args):
            task_args_for_future = future_to_task_args[future]
            try:
                result = future.result() # This is the tuple returned by run_single_game_config
                results.append(result)
            except Exception as exc:
                # This block would catch exceptions if run_single_game_config itself failed catastrophically
                # before its own try/except block, or if future.result() had other issues.
                # run_single_game_config is designed to catch its own subprocess errors.
                print(f"    CRITICAL ERROR processing task {task_args_for_future[:2]}: {exc}")
                # Append a synthetic failure result so the summary counts it
                # (game_name, use_harness, success_status, stdout, stderr)
                results.append((task_args_for_future[0], task_args_for_future[2], False, "", str(exc)))

    print("\n----- All game runs complete -----")
    successful_runs = 0
    failed_runs = 0
    for game, harness, success, _, _ in results:
        status_str = "succeeded" if success else "FAILED"
        print(f"  Run: Game='{game}', Harness='{harness}' - {status_str}")
        if success:
            successful_runs +=1
        else:
            failed_runs +=1
    
    print(f"Summary: {successful_runs} successful runs, {failed_runs} failed runs.")

    if failed_runs > 0:
        print("Some runs failed. Check the logs in the 'logs' directory.")

    print("\n----- Main script finished. Check 'logs/' for details. -----")

if __name__ == "__main__":
    main()
