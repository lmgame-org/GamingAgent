import cv2
import gymnasium as gym
import numpy as np
import os
import json
import argparse
import time
from datetime import datetime
import asyncio
from PIL import Image

from tetris_gymnasium.envs.tetris import Tetris
from tests.test_agents.modules.tetris_modules import (
    TetrisPerceptionModule, 
    TetrisMemoryModule, 
    TetrisReasoningModule, 
    TetrisBaseModule, 
    tetris_actions,
    DEFAULT_ACTION
)

# Create cache directory for this run
CACHE_DIR = os.path.join("cache", "tetris_experiments", datetime.now().strftime("%Y%m%d_%H%M%S"))
BOARD_IMG_PATH = os.path.join(CACHE_DIR, "board_latest.png")
GAME_LOG_FILE = os.path.join(CACHE_DIR, "game_log.jsonl")
DATA_LOG_FILE = os.path.join(CACHE_DIR, "data_log.jsonl")
MEMORY_FILE = os.path.join(CACHE_DIR, "memory.json")
os.makedirs(CACHE_DIR, exist_ok=True)

# Helper function to convert NumPy types to Python native types for JSON serialization
def convert_numpy_types(obj):
    """Convert NumPy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [convert_numpy_types(x) for x in obj.tolist()]
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

async def run_actions(env, actions):
    """
    Execute a list of actions on the environment sequentially.
    
    Args:
        env: The Tetris environment
        actions: List of actions to execute
        
    Returns:
        The final observation, reward, terminated, truncated, and info
    """
    obs = None
    reward = 0
    terminated = False
    truncated = False
    info = {}
    
    for action in actions:
        if terminated or truncated:
            break
        
        obs, step_reward, terminated, truncated, info = env.step(action)
        reward += step_reward
        
    return obs, reward, terminated, truncated, info

class TetrisAgent:
    """A Tetris agent that uses either base module or perception+memory+reasoning modules."""
    
    def __init__(self, model_name="claude-3-7-sonnet-latest", agent_mode="full"):
        """Initialize the agent based on the specified agent_mode."""
        self.model_name = model_name
        self.agent_mode = agent_mode
        self.last_action = None # Will store the last action *sequence*
        
        self.base_module = None
        self.perception_module = None # For Tetris, this might be instantiated per call if it needs obs
        self.memory_module = None
        self.reasoning_module = None

        print(f"Initializing TetrisAgent with mode: {self.agent_mode}, model: {self.model_name}")

        if self.agent_mode == "base":
            self.base_module = TetrisBaseModule(model_name=model_name)
            print("Base Module initialized.")
        elif self.agent_mode == "full":
            # TetrisPerceptionModule is initialized with observation, so not here.
            self.memory_module = TetrisMemoryModule(
                model_name=model_name,
                memory_file=MEMORY_FILE
            )
            self.reasoning_module = TetrisReasoningModule(model_name=model_name)
            print("Full pipeline (Memory, Reasoning; Perception per-step) initialized.")
        elif self.agent_mode == "memory_reasoning":
            self.base_module = TetrisBaseModule(model_name=model_name)
            self.memory_module = TetrisMemoryModule(
                model_name=model_name,
                memory_file=MEMORY_FILE
            )
            print("Base Module and Memory Module initialized for Memory+Reasoning (Base+Memory) mode.")
        elif self.agent_mode == "perception_reasoning":
            # TetrisPerceptionModule is initialized with observation, so not here.
            self.reasoning_module = TetrisReasoningModule(model_name=model_name)
            print("Reasoning module (Perception per-step) initialized.")
        else:
            raise ValueError(f"Unknown agent_mode: {self.agent_mode}")
    
    async def get_action(self, observation, info=None, max_retries_const=1):
        """Get the next action from the agent."""
        try:
            action_plan = None
            perception_data_for_log = None
            memory_summary_for_log = None

            if self.agent_mode == "base":
                # Observation for base is the frame, info is the enriched original obs dict + img_path
                print("\n" + "="*80)
                action_plan = await self.base_module.process_observation(observation, info) # base_module expects await
                print("\nACTION PLAN (Base):")
                print(f"Action sequence: {action_plan.get('action_sequence', DEFAULT_ACTION)}")
                print(f"Thought: {action_plan.get('thought', '')}")
                print("="*80 + "\n")
                self.last_action = action_plan.get("action_sequence", DEFAULT_ACTION)
                return action_plan, None, None
            
            elif self.agent_mode == "full":
                # Observation for full is the env obs dict, info is also the env obs dict (or None if not used by perception)
                # Initialize PerceptionModule here as it takes observation in __init__
                current_perception_module = TetrisPerceptionModule(model_name=self.model_name, observation=observation)
                perception_data = current_perception_module.get_perception_data()
                perception_data_for_log = perception_data
                
                print("\n" + "="*80)
                print("PERCEPTION DATA (Full):")
                print(f"""Board state (text): {perception_data.get('board', 'N/A')}""")
                print(f"Next pieces: {perception_data.get('next_pieces', 'N/A')}")
                print(f"Potential states: {len(perception_data.get('potential_states', []))} states")

                simplified_state_for_memory = {
                    'board_text': perception_data.get('board', 'No board data'),
                    'next_pieces_text': perception_data.get('next_pieces', 'No next pieces data')
                }
                
                # Step 2: Memory
                max_retries = max_retries_const
                retry_count = 0
                memory_summary = None
                await self.memory_module.add_game_state(simplified_state_for_memory, self.last_action)
                
                while memory_summary is None and retry_count < max_retries:
                    memory_summary = self.memory_module.get_memory_summary()
                    if memory_summary is None or len(memory_summary) == 0:
                        retry_count += 1
                        print(f"Memory retrieval attempt {retry_count}/{max_retries} failed. Retrying...")
                        await asyncio.sleep(1)
                memory_summary_for_log = memory_summary
 
                if memory_summary:
                    print("\nMEMORY SUMMARY (Full):")
                    print(f"Memory entries: {len(memory_summary)}")
                    if len(memory_summary) > 0 and 'reflection' in memory_summary[-1]:
                        print(f"Latest reflection: {memory_summary[-1].get('reflection','N/A')}")
                
                # Step 3: Reasoning
                max_retries = max_retries_const
                retry_count = 0
                action_plan = None
                while action_plan is None and retry_count < max_retries:
                    action_plan = await self.reasoning_module.plan_action(
                        current_perception=perception_data, # Pass full perception data
                        memory_summary=memory_summary,
                        img_path=info.get('img_path', BOARD_IMG_PATH) if isinstance(info, dict) else BOARD_IMG_PATH,
                        max_retries=3
                    )
                    if action_plan is None or 'action_sequence' not in action_plan:
                        retry_count += 1
                        print(f"Action planning attempt {retry_count}/{max_retries} failed. Retrying...")
                        await asyncio.sleep(1)

            elif self.agent_mode == "memory_reasoning":
                # Observation for M+R is the frame (image), info is enriched original obs dict + img_path
                print("\n" + "="*80)
                
                # Step 1: Retrieve Memory Summary (history before this current observation)
                memory_summary = self.memory_module.get_memory_summary()
                memory_summary_for_log = memory_summary # Log this history that was used for decision

                # Step 2: Decision Making using BaseModule + Memory
                # The `observation` is the frame_image, `info` is the enriched_env_obs_dict
                action_plan = await self.base_module.process_observation(
                    observation, 
                    info, 
                    memory_summary=memory_summary
                )
                # `action_plan` will contain {"action_sequence": ..., "thought": ...}

                # Step 3: Prepare current state for memory logging (to be stored *after* this turn)
                # This textual representation is for storing the state that *resulted* from `self.last_action`
                # and *is* the current `info` (enriched_env_obs_dict).
                if not isinstance(info, dict):
                    print("Critical Error: `info` dictionary is missing or not a dict for memory_reasoning state logging.")
                    current_state_textual_for_memory = {
                        'board_text': 'Error: board info missing for memory logging',
                        'next_pieces_text': 'Error: next pieces info missing for memory logging'
                    }
                else:
                    # Use a temporary TetrisPerceptionModule to convert current `info` (which is enriched `obs`)
                    # into the textual format required by TetrisMemoryModule.
                    # `info` here is the state *before* the action_plan is executed.
                    temp_perception_for_logging = TetrisPerceptionModule(model_name=self.model_name, observation=info)
                    current_state_textual_for_memory = {
                        'board_text': temp_perception_for_logging.get_board(),
                        'next_pieces_text': temp_perception_for_logging.get_next_pieces()
                        # We don't need potential_states for memory logging, only for reasoning input if it were different.
                    }
                perception_data_for_log = current_state_textual_for_memory # This is what gets logged for this step's "perception"
                
                # The actual `add_game_state` will happen in `run_agent` *after* the environment steps
                # with the *new* observation. Here, we've just prepared what the perception module *would* see for the current state.
                # For now, `perception_data_for_log` holds this textual version of the current state.
                # The `memory_summary_for_log` holds the memory *used for* this decision.

                print("DECISION MADE (Memory+Reasoning using Base Module):")
                print(f"""Board state (from info for logging): {current_state_textual_for_memory.get('board_text', 'N/A')}""")
                print(f"Next pieces (from info for logging): {current_state_textual_for_memory.get('next_pieces_text', 'N/A')}")
                if memory_summary:
                    print(f"Memory summary (used for decision) entries: {len(memory_summary)}")

            elif self.agent_mode == "perception_reasoning":
                # Observation for P+R is the env obs dict, info is also the env obs dict
                print("\n" + "="*80)
                # Initialize PerceptionModule here
                current_perception_module = TetrisPerceptionModule(model_name=self.model_name, observation=observation)
                perception_data = current_perception_module.get_perception_data()
                perception_data_for_log = perception_data

                print("PERCEPTION DATA (Perception+Reasoning):")
                print(f"Board state (text):{perception_data.get('board', 'N/A')}")
                print(f"Next pieces: {perception_data.get('next_pieces', 'N/A')}")
                print(f"Potential states: {len(perception_data.get('potential_states', []))} states")

                # Step 2: Reasoning (no memory)
                max_retries = max_retries_const
                retry_count = 0
                action_plan = None
                memory_summary_for_log = None # Explicitly no memory for this mode
                
                while action_plan is None and retry_count < max_retries:
                    action_plan = await self.reasoning_module.plan_action(
                        current_perception=perception_data,
                        memory_summary=None, # No memory for this mode
                        img_path=info.get('img_path', BOARD_IMG_PATH) if isinstance(info, dict) else BOARD_IMG_PATH,
                        max_retries=3
                    )
                    if action_plan is None or 'action_sequence' not in action_plan:
                        retry_count += 1
                        print(f"Action planning attempt {retry_count}/{max_retries} failed. Retrying...")
                        await asyncio.sleep(1)
            
            else:
                print(f"Error: Unknown agent mode {self.agent_mode}")
                self.last_action = DEFAULT_ACTION
                return {"action_sequence": DEFAULT_ACTION, "thought": f"Unknown agent mode: {self.agent_mode}"}, None, None

            # Common fallback and action printing for modes other than "base"
            if action_plan is None or 'action_sequence' not in action_plan or not action_plan.get('action_sequence'):
                print("All reasoning attempts failed or returned empty sequence. Using fallback action.")
                action_plan = {
                    "action_sequence": DEFAULT_ACTION,
                    "thought": "Fallback action after failed reasoning attempts or empty sequence"
                }
            
            print("\nACTION PLAN:")
            print(f"Action sequence: {action_plan.get('action_sequence', DEFAULT_ACTION)}")
            print(f"Thought: {action_plan.get('thought', '')}")
            print("="*80 + "\n")
            
            self.last_action = action_plan.get("action_sequence", DEFAULT_ACTION)
            return action_plan, perception_data_for_log, memory_summary_for_log
        
        except Exception as e:
            print(f"Error in get_action: {e}")
            # Return a fallback action on error
            self.last_action = [7]  # no_operation
            return {
                "action_sequence": [7], 
                "thought": f"Error occurred: {str(e)}"
            }, None, None

def log_step(step, action_plan, perception_data, memory_summary, action, reward, info, observation, run_id=1):
    """Log comprehensive step information to a JSON Lines file."""
    # Convert numpy arrays and other types for JSON serialization
    if isinstance(observation, dict) and 'board' in observation and isinstance(observation['board'], np.ndarray):
        observation_copy = observation.copy()
        observation_copy['board'] = observation['board'].tolist()
        observation = observation_copy
    elif isinstance(observation, np.ndarray):
        observation = observation.tolist()
    
    # Create comprehensive log entry
    log_entry = {
        "run_id": run_id,
        "step": step,
        "timestamp": datetime.now().isoformat(),
        "action_sequence": action_plan.get("action_sequence", []),
        "action_index": int(action) if action is not None else None,
        "thought": action_plan.get("thought", ""),
        "reward": float(reward),
        "score": info.get('score', 0),
        "info": {k: v for k, v in info.items() if isinstance(v, (int, float, str, bool, list, dict))},
        "perception_data": perception_data,
        "memory_summary": memory_summary[-1] if memory_summary and len(memory_summary) > 0 else None
    }
    
    # Convert any remaining NumPy types
    log_entry = convert_numpy_types(log_entry)
    
    # Write to log file
    with open(GAME_LOG_FILE, 'a') as f:
        json.dump(log_entry, f)
        f.write('\n')

def log_raw_data(step, action, reward, info, observation, run_id=1):
    """Log raw data from the environment to a separate JSON Lines file."""
    # Convert info dictionary for JSON serialization
    if isinstance(info, dict):
        # Make a copy to avoid modifying the original
        info_copy = {}
        for k, v in info.items():
            if isinstance(v, np.ndarray):
                info_copy[k] = v.tolist()
            elif isinstance(v, (int, float, str, bool, list, dict)):
                info_copy[k] = v
            else:
                info_copy[k] = str(v)
    else:
        info_copy = str(info)
    
    # Create simple data log entry
    data_entry = {
        "run_id": run_id,
        "step": step,
        "timestamp": datetime.now().isoformat(),
        "action": int(action) if action is not None else None,
        "reward": float(reward),
        "info": info_copy,
        "observation": observation.tolist() if isinstance(observation, np.ndarray) else observation
    }
    
    # Convert any remaining NumPy types
    data_entry = convert_numpy_types(data_entry)
    
    # Write to data log file
    with open(DATA_LOG_FILE, 'a') as f:
        json.dump(data_entry, f)
        f.write('\n')

async def run_agent(args):
    """Run the Tetris agent with the specified parameters."""
    # Write experiment info
    with open(os.path.join(CACHE_DIR, "experiment_info.json"), 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "model": args.model,
            "agent_mode": args.agent_mode,
            "use_random": args.random,
            "num_runs": args.num_runs,
            "cache_dir": CACHE_DIR
        }, f, indent=2)
    
    # Create a summary file for tracking scores
    summary_file = os.path.join(CACHE_DIR, "runs_summary.jsonl")
    all_scores = []
    
    # Initialize the agent (shared across all runs)
    agent = None
    if not args.random:
        agent = TetrisAgent(
            model_name=args.model,
            agent_mode=args.agent_mode
        )
        print(f"\n{'{' * 80}")
        if args.agent_mode == "base":
            print(f"RUNNING TETRIS IN BASE MODULE MODE - Using model: {args.model}")
        elif args.agent_mode == "full":
            print(f"RUNNING TETRIS IN FULL PIPELINE MODE - Using model: {args.model}")
        elif args.agent_mode == "memory_reasoning":
            print(f"RUNNING TETRIS IN MEMORY + REASONING MODE - Using model: {args.model}")
        elif args.agent_mode == "perception_reasoning":
            print(f"RUNNING TETRIS IN PERCEPTION + REASONING MODE - Using model: {args.model}")
        print(f"{'}' * 80}\n")
    else:
        print("\n" + "*" * 80)
        print("RUNNING TETRIS WITH RANDOM ACTIONS - No AI model used")
        print("*" * 80 + "\n")
    
    # Run for the specified number of games
    for run_id in range(1, args.num_runs + 1):
        print(f"\n========== STARTING TETRIS RUN {run_id}/{args.num_runs} ==========\n")
        
        # Create the Tetris environment with appropriate render mode
        if args.agent_mode == "base" or args.agent_mode == "memory_reasoning":
            render_mode = "rgb_array"
        else: # full, perception_reasoning, random
            render_mode = "human"
        
        env = gym.make("tetris_gymnasium/Tetris", render_mode=render_mode)
        
        obs, info = env.reset(seed=run_id) # Use run_id as seed for variability
        
        terminated = False
        truncated = False
        all_rewards = 0
        step_count = 0
        max_retries = 3
        
        while not terminated and not truncated:
            if render_mode == "human":
                env.render()
                key = cv2.waitKey(100) # For human visualization and potential quit
                if key == ord('q'): break

            frame_for_agent = None
            info_for_agent = obs # Default for full, P+R

            if args.agent_mode == "base" or args.agent_mode == "memory_reasoning":
                try:
                    rendered_frame = env.render() # This is an RGB array
                    if rendered_frame is not None and isinstance(rendered_frame, np.ndarray):
                        frame_for_agent = rendered_frame
                        os.makedirs(os.path.dirname(BOARD_IMG_PATH), exist_ok=True)
                        Image.fromarray(frame_for_agent).save(BOARD_IMG_PATH)
                        # Enrich info for these modes
                        info_for_agent = dict(obs) # Start with the current obs from env.reset/step
                        info_for_agent['img_path'] = BOARD_IMG_PATH
                        # Add piece queue text if perception module can provide it (for base/M+R prompts)
                        # Need a temporary perception module instance based on the *current env obs*
                        temp_perception_for_info = TetrisPerceptionModule(model_name=agent.model_name, observation=obs)
                        info_for_agent['piece_queue_symbols'] = temp_perception_for_info.get_next_pieces()
                        # Potentially add more details if TetrisPerceptionModule can provide them and they are useful for prompts
                        # For example, textual representation of the full queue, not just symbols
                    else:
                        print("Warning: env.render() did not return a valid frame for base/memory_reasoning mode.")
                        # Fallback: agent.get_action will have to handle None frame or info_for_agent will not have img_path
                except Exception as e:
                    print(f"Warning: Could not save frame or enrich info for {args.agent_mode}: {e}")
            
            if args.random:
                action_sequence = [env.action_space.sample()]
                action_plan = {"action_sequence": action_sequence, "thought": "Random action"}
                perception_data, memory_summary = None, None
            else:
                retry_count = 0
                valid_action_sequence_found = False
                while not valid_action_sequence_found and retry_count < max_retries:
                    current_obs_for_agent = obs
                    current_info_for_agent = info_for_agent # This is either original obs or enriched obs

                    if args.agent_mode == "base" or args.agent_mode == "memory_reasoning":
                        current_obs_for_agent = frame_for_agent # Image frame
                        # current_info_for_agent is already set up (enriched obs)
                    # Else (full, P+R), current_obs_for_agent is obs, current_info_for_agent is obs (raw from env)

                    action_plan, perception_data, memory_summary = await agent.get_action(
                        current_obs_for_agent, 
                        current_info_for_agent, 
                        max_retries_const=1 # Retries are handled in the loop here
                    )
                    
                    action_sequence_from_plan = action_plan.get("action_sequence", [])
                    if action_sequence_from_plan and isinstance(action_sequence_from_plan, list) and len(action_sequence_from_plan) > 0:
                        valid_action_sequence_found = True
                    else:
                        print(f"Agent returned invalid or empty action sequence. Attempt {retry_count + 1}/{max_retries}")
                        retry_count += 1
                        await asyncio.sleep(1)
                
                if not valid_action_sequence_found:
                    print("Exhausted retry attempts or agent consistently failed to provide valid sequence, using default action.")
                    action_plan = {"action_sequence": DEFAULT_ACTION, "thought": "Fallback (default) after retries"}
            
            action_sequence = action_plan.get("action_sequence", DEFAULT_ACTION)
            if not action_sequence: # Ensure it's never empty
                action_sequence = DEFAULT_ACTION
                action_plan["action_sequence"] = action_sequence # Update plan for logging
            
            # Store the observation *before* the step, to correctly associate prior_action with state_that_resulted_from_it for memory
            obs_before_step = obs # This is the state on which the action_plan was based
            info_before_step = info_for_agent # This contains the enriched data used by base/M+R modes

            new_obs, reward, terminated, truncated, info_after_step = await run_actions(env, action_sequence)
            
            # After the step, update memory for modes that use it
            if not args.random and (args.agent_mode == "full" or args.agent_mode == "memory_reasoning"):
                state_to_log_in_memory = None
                # For memory, we need to store the textual representation of the state *achieved* by action_sequence
                # This new state is `new_obs` (and `info_after_step` which is derived from `new_obs`)
                
                # Use TetrisPerceptionModule to convert new_obs into the textual format memory module expects
                # The `new_obs` from env.step() is the dict like {'board': ..., 'queue': ...}
                temp_perception_for_memory_update = TetrisPerceptionModule(model_name=agent.model_name, observation=new_obs)
                textual_new_state_for_memory = {
                    'board_text': temp_perception_for_memory_update.get_board(),
                    'next_pieces_text': temp_perception_for_memory_update.get_next_pieces()
                }
                state_to_log_in_memory = textual_new_state_for_memory

                # The `action_plan["action_sequence"]` is the sequence that *led* to `new_obs`
                await agent.memory_module.add_game_state(state_to_log_in_memory, action_plan.get("action_sequence"))

            # Logging the step: 
            # `perception_data` was prepared in get_action (it's the perception of obs_before_step)
            # `memory_summary` was prepared in get_action (it's the memory *used for deciding* on action_plan)
            log_step(step_count, action_plan, perception_data, memory_summary, action_sequence[0] if action_sequence else None, reward, info_after_step, new_obs, run_id)
            log_raw_data(step_count, action_sequence[0] if action_sequence else None, reward, info_after_step, new_obs, run_id)
            
            obs = new_obs # Update current observation for the next loop iteration
            # info variable for the next loop iteration will be derived from new_obs at the start of the loop if needed

            all_rewards += reward
            step_count += 1
            
            lines_cleared = info_after_step.get('score', 0) # Tetris env returns lines cleared in 'score'
            print(f"Run {run_id}, Step: {step_count}, Actions: {action_sequence}, Reward: {reward:.2f}, Total Reward: {all_rewards:.2f}, Lines Cleared: {lines_cleared}")
            
            if render_mode == "human":
                 cv2.waitKey(1) # Brief pause to see move if human rendering
        
        env.close()
        all_scores.append(float(all_rewards))
        
        run_summary = {
            "run_id": int(run_id),
            "score": float(all_rewards),
            "steps": int(step_count),
            "lines_cleared": int(info_after_step.get('score',0)), # Log final lines cleared
            "timestamp": datetime.now().isoformat(),
        }
        with open(summary_file, 'a') as f:
            json.dump(run_summary, f)
            f.write('\n')
        
        print(f"\nRun {run_id} finished after {step_count} steps")
        print(f"Total reward: {all_rewards:.2f}")
        print(f"Lines cleared: {info_after_step.get('score',0)}")
        print(f"========== COMPLETED TETRIS RUN {run_id}/{args.num_runs} ==========\n")
    
    # Print overall summary
    if args.num_runs > 1:
        print("\n===== EXPERIMENT SUMMARY =====")
        print(f"Total runs: {args.num_runs}")
        print(f"Average total reward: {sum(all_scores) / len(all_scores):.2f}")
        print(f"Highest total reward: {max(all_scores)}")
        print(f"Lowest total reward: {min(all_scores)}")
        print(f"All total rewards: {all_scores}")
    
    return all_scores

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run the Tetris AI Agent')
    
    parser.add_argument('--model', type=str, default="claude-3-5-sonnet-latest",
                        help='Model name to use (default: claude-3-7-sonnet-latest)')
    
    parser.add_argument('--agent_mode', type=str, default="full",
                        choices=["full", "base", "memory_reasoning", "perception_reasoning"],
                        help='Agent mode to use (default: full)')
    
    parser.add_argument('--random', action='store_true',
                        help='Use random actions instead of AI agent (default: False)')
    
    parser.add_argument('--num_runs', type=int, default=1,
                        help='Number of game runs to perform (default: 1)')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    scores = asyncio.run(run_agent(args))
    if len(scores) == 1:
        print(f"Final total reward: {scores[0]}")
    else:
        print(f"Average total reward: {sum(scores) / len(scores):.2f}")