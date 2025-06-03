from tile_match_gym.tile_match_env import TileMatchEnv
import numpy as np
from collections import OrderedDict
import os
import json
import argparse
import time
from datetime import datetime
import asyncio
from PIL import Image

# Import our custom modules for Candy Crush
from tests.test_agents.modules.candy_crush_modules import (
    CandyCrushPerceptionModule,
    CandyCrushMemoryModule,
    CandyCrushReasoningModule,
    CandyCrushBaseModule
)

# Create cache directory for this run
CACHE_DIR = os.path.join("cache", "candy_crush_experiments", datetime.now().strftime("%Y%m%d_%H%M%S"))
BOARD_IMG_PATH = os.path.join(CACHE_DIR, "board_latest.png")
GAME_LOG_FILE = os.path.join(CACHE_DIR, "game_log.jsonl")
DATA_LOG_FILE = os.path.join(CACHE_DIR, "data_log.jsonl")
MEMORY_FILE = os.path.join(CACHE_DIR, "memory.json")
os.makedirs(CACHE_DIR, exist_ok=True)

class CandyCrushObservationWrapper:
    """Observation wrapper for Candy Crush environment."""
    
    def __init__(self, env, num_colours=4):
        self.env = env
        self.num_rows = env.num_rows
        self.num_cols = env.num_cols
        self.num_colours = num_colours
        self.global_num_colourless_specials = 0  # Update as needed
        self.global_num_colour_specials = 0  # Update as needed
        self.num_type_slices = 0  # Update as needed
        self.type_slices = []  # Update as needed

    def observation(self, obs) -> dict:
        """Apply one-hot encoding to the observation."""
        board = obs["board"]
        ohe_board = self._one_hot_encode_board(board)
        return OrderedDict([("board", ohe_board), ("num_moves_left", obs["num_moves_left"])])
    
    def _one_hot_encode_board(self, board: np.ndarray) -> np.ndarray:
        """One-hot encode the board."""
        tile_colours = board[0]
        rows, cols = np.indices(tile_colours.shape)
        colour_ohe = np.zeros((1 + self.num_colours, self.num_rows, self.num_cols)) # Remove colourless slice after encoding
        colour_ohe[tile_colours.flatten(), rows.flatten(), cols.flatten()] = 1
        ohe_board = colour_ohe[1:]

        # Only keep the types for the specials that are in the environment (absence of any 1 means ordinary)
        if self.num_type_slices > 0:
            tile_types = board[1] + self.global_num_colourless_specials
            type_ohe = np.zeros((2 + self.global_num_colour_specials + self.global_num_colourless_specials, self.num_rows, self.num_cols)) # +1 for ordinary, +1 for empty
            type_ohe[tile_types.flatten(), rows.flatten(), cols.flatten()] = 1
            type_ohe = type_ohe[self.type_slices]
            ohe_board = np.concatenate([ohe_board, type_ohe], axis=0) # 1 + num_colours + num_colourless_specials + num_colour_specials.
        
        return ohe_board

def decode_ohe_board(ohe_board, num_colours=4):
    """
    Decode a one-hot encoded board back to a 2D array of color indices.
    
    Args:
        ohe_board: One-hot encoded board from the observation wrapper
        num_colours: Number of colors in the game
        
    Returns:
        2D array with color indices
    """
    # If ohe_board has shape [num_colours, rows, cols]
    if len(ohe_board.shape) == 3:
        # Get the indices of the maximum values along the color dimension
        # This will tell us which color is present at each position
        decoded_board = np.zeros((ohe_board.shape[1], ohe_board.shape[2]), dtype=int)
        
        # For each position, find which color channel has a 1
        for r in range(ohe_board.shape[1]):
            for c in range(ohe_board.shape[2]):
                for color in range(num_colours):
                    if color < ohe_board.shape[0] and ohe_board[color, r, c] == 1:
                        decoded_board[r, c] = color + 1  # +1 because colors start at 1
                        break
        
        return decoded_board
    
    # If the shape is not as expected, return the input
    return ohe_board

def convert_obs_to_2d_array(obs):
    """Convert observation to a more readable 2D array showing the tile colors."""
    # Extract the board from the observation
    board = obs["board"]
    
    # If it's already a 2D array (raw observation)
    if len(board.shape) == 3 and board.shape[0] <= 2:
        # The first channel contains tile colors
        return board[0].astype(int)
    
    # If it's a one-hot encoded board from the wrapper
    return decode_ohe_board(board)

def print_board(board):
    """Print the board in a grid format with column and row numbers."""
    rows, cols = board.shape
    
    # Print column numbers
    print("  ", end="")
    for c in range(cols):
        print(f"{c}", end=" ")
    print("\n  " + "-" * (cols * 2))
    
    # Create color representation
    color_names = {0: "E", 1: "G", 2: "C", 3: "P", 4: "R", 5: "T", 6: "B"}
    
    # Print rows with row numbers
    for r in range(rows):
        print(f"{r}|", end="")
        for c in range(cols):
            color_id = board[r, c]
            color_symbol = color_names.get(color_id, str(color_id))
            print(f"{color_symbol:2}", end="")
        print()
    print()

def coords_to_action_index(coord1, coord2, env):
    """
    Convert a pair of coordinates to the corresponding action index.
    
    Args:
        coord1: First coordinate tuple (row, col)
        coord2: Second coordinate tuple (row, col)
        env: Environment with _action_to_coords mapping
        
    Returns:
        Action index (integer) corresponding to the coordinates
    """
    # Make sure the coordinates are tuples
    coord1 = tuple(coord1)
    coord2 = tuple(coord2)
    
    # Sort the coordinates to match the order in the action space
    if coord1 > coord2:
        coord1, coord2 = coord2, coord1
        
    # Look up the action index
    for action, (c1, c2) in enumerate(env._action_to_coords):
        if (c1, c2) == (coord1, coord2):
            return action
    
    # If no match is found
    return None

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

def log_step(step, action_plan, perception_data, memory_summary, action, reward, info, observation, run_id=1):
    """Log comprehensive step information to a JSON Lines file."""
    # Convert numpy arrays and other types for JSON serialization
    if isinstance(observation, np.ndarray):
        observation = observation.tolist()
    
    # Create comprehensive log entry
    log_entry = {
        "run_id": run_id,
        "step": step,
        "timestamp": datetime.now().isoformat(),
        "move": action_plan.get("move", "unknown"),
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

class CandyCrushAgent:
    """A Candy Crush agent that uses either base module or perception+memory+reasoning modules."""
    
    def __init__(self, model_name="claude-3-7-sonnet-latest", agent_mode="full"):
        """Initialize the agent with either base module or full pipeline."""
        self.model_name = model_name
        self.agent_mode = agent_mode
        self.last_action = None
        
        self.base_module = None
        self.perception_module = None
        self.memory_module = None
        self.reasoning_module = None

        print(f"Initializing CandyCrushAgent with mode: {self.agent_mode}, model: {self.model_name}")

        if self.agent_mode == "base":
            self.base_module = CandyCrushBaseModule(model_name=model_name)
            print("Base Module initialized.")
        elif self.agent_mode == "memory_reasoning": # Base + Memory
            self.base_module = CandyCrushBaseModule(model_name=model_name)
            self.memory_module = CandyCrushMemoryModule(
                model_name=model_name,
                memory_file=MEMORY_FILE
            )
            print("Base Module and Memory Module initialized for Memory+Reasoning (Base+Memory) mode.")
        elif self.agent_mode == "full":
            self.perception_module = CandyCrushPerceptionModule(model_name=model_name)
            self.memory_module = CandyCrushMemoryModule(
                model_name=model_name,
                memory_file=MEMORY_FILE
            )
            self.reasoning_module = CandyCrushReasoningModule(model_name=model_name)
            print("Full pipeline (Perception, Memory, Reasoning) initialized.")
        elif self.agent_mode == "perception_reasoning":
            self.perception_module = CandyCrushPerceptionModule(model_name=model_name)
            self.reasoning_module = CandyCrushReasoningModule(model_name=model_name)
            print("Perception and Reasoning modules initialized.")
        else:
            raise ValueError(f"Unknown agent_mode: {self.agent_mode}")
    
    async def get_action(self, observation, info=None, max_retries_const=1):
        """Get the next action from the agent."""
        try:
            action_plan = None
            perception_data_for_log = None # What the agent perceived for the current state decision
            memory_summary_for_log = None  # What memory was used for the current state decision

            if self.agent_mode == "base":
                # Observation for base is the frame, info is the original obs dict + img_path
                print("\n" + "="*80)
                action_plan = self.base_module.process_observation(observation, info, memory_summary=None) # No memory for pure base
                print("\nACTION PLAN (Base):")
                print(f"Action: {action_plan.get('move')}")
                print(f"Thought: {action_plan.get('thought', '')}")
                print("="*80 + "\n")
                self.last_action = action_plan.get("move")
                # perception_data_for_log and memory_summary_for_log remain None for pure base
                return action_plan, None, None
            
            elif self.agent_mode == "memory_reasoning": # Now Base + Memory
                # Observation is frame (image), info is enriched_env_obs_dict (contains raw obs + img_path)
                print("\n" + "="*80)
                print(f"ENTERING AGENT MODE: {self.agent_mode}")

                # Step 1: Retrieve Memory Summary (history *before* this current observation)
                memory_summary = self.memory_module.get_memory_summary()
                memory_summary_for_log = memory_summary

                # Step 2: Decision Making using BaseModule + Memory
                action_plan = self.base_module.process_observation(
                    observation,  # This is the frame_image
                    info,         # This is the enriched_env_obs_dict
                    memory_summary=memory_summary
                )

                # Step 3: Prepare current state's textual representation for logging (perception_data_for_log)
                # This textual form represents the state upon which the vision-based decision was made.
                # `info` (enriched_env_obs_dict) contains the raw `obs` data needed for this.
                if not isinstance(info, dict) or 'board' not in info:
                    print("Critical Error: `info` dictionary or `info['board']` is missing for memory_reasoning state logging.")
                    current_state_textual_for_log = {"error": "Missing info for perception log"}
                else:
                    # Use CandyCrushPerceptionModule to get a structured textual representation of the current state from `info`
                    # (info is the enriched version of the raw obs dict)
                    temp_perception_module = CandyCrushPerceptionModule(model_name=self.model_name)
                    # analyze_board expects obs["board"] as first arg, and full obs dict as second for other info like num_moves
                    current_state_textual_for_log = temp_perception_module.analyze_board(info['board'], info)
                perception_data_for_log = current_state_textual_for_log
                
                print("ACTION PLAN (Memory+Reasoning - Base Module):")
                print(f"Action: {action_plan.get('move')}")
                print(f"Thought: {action_plan.get('thought', '')}")
                if memory_summary_for_log:
                    print(f"Memory entries used for decision: {len(memory_summary_for_log)}")
                print("="*80 + "\n")
                self.last_action = action_plan.get("move")
                return action_plan, perception_data_for_log, memory_summary_for_log

            # --- Modes using ReasoningModule --- 
            current_perception_data = None
            if self.agent_mode == "full" or self.agent_mode == "perception_reasoning":
                # For these modes, `observation` is obs["board"], `info` is the full obs dict
                print("\n" + "="*80)
                print(f"ENTERING AGENT MODE: {self.agent_mode}")
                current_perception_data = self.perception_module.analyze_board(observation, info)
                perception_data_for_log = current_perception_data
                print(f"PERCEPTION DATA ({self.agent_mode}):")
                print(f"  Potential matches: {current_perception_data.get('num_potential_matches', 0)}")
                print(f"  Highest color: {current_perception_data.get('highest_color', 'N/A')}")

            retrieved_memory_summary = None
            if self.agent_mode == "full": # Only full mode uses memory with ReasoningModule now
                retrieved_memory_summary = self.memory_module.get_memory_summary()
                memory_summary_for_log = retrieved_memory_summary
                if memory_summary_for_log:
                    print(f"MEMORY SUMMARY ({self.agent_mode}):")
                    print(f"  Memory entries: {len(memory_summary_for_log)}")
                    if len(memory_summary_for_log) > 0 and 'reflection' in memory_summary_for_log[-1]:
                        print(f"  Latest reflection: {memory_summary_for_log[-1].get('reflection','N/A')}")
            
            # Reasoning step for "full" and "perception_reasoning"
            if self.agent_mode == "full" or self.agent_mode == "perception_reasoning":
                max_retries = max_retries_const
                retry_count = 0
                action_plan = None
                while action_plan is None and retry_count < max_retries:
                    action_plan = await self.reasoning_module.plan_action(
                        current_perception=current_perception_data,
                        memory_summary=retrieved_memory_summary, # This will be None for perception_reasoning
                        img_path=info.get('img_path', BOARD_IMG_PATH) if isinstance(info, dict) else BOARD_IMG_PATH,
                        max_retries=3
                    )
                    if action_plan is None or 'move' not in action_plan:
                        retry_count += 1
                        print(f"Action planning attempt {retry_count}/{max_retries} failed for {self.agent_mode}. Retrying...")
                        await asyncio.sleep(1)
                
                print(f"ACTION PLAN ({self.agent_mode} - Reasoning Module):")
                print(f"Action: {action_plan.get('move') if action_plan else 'N/A'}")
                print(f"Thought: {action_plan.get('thought', '') if action_plan else 'N/A'}")
                print("="*80 + "\n")
                if action_plan: self.last_action = action_plan.get("move")
                return action_plan, perception_data_for_log, memory_summary_for_log
            
            # Fallback if mode not matched, though constructor should prevent this
            print(f"Error: Agent mode {self.agent_mode} not handled in get_action logic after initial checks.")
            self.last_action = ((0,0),(0,1)) # Fallback
            return {"move": ((0,0),(0,1)), "thought": f"Unknown agent mode error: {self.agent_mode}"}, None, None

        except Exception as e:
            print(f"Error in get_action for mode {self.agent_mode}: {e}")
            # Return a fallback action on error
            fallback_move = ((0, 0), (0, 1))
            self.last_action = fallback_move
            return {
                "move": fallback_move, 
                "thought": f"Error occurred in {self.agent_mode}: {str(e)}"
            }, None, None
    
    def parse_move_to_action(self, move, env):
        """Convert move coordinates to action index for the environment."""
        if not move or len(move) != 2:
            print(f"Invalid move format: {move}")
            return None
            
        try:
            coord1, coord2 = move
            # Call coords_to_action_index to get the action index
            action_idx = coords_to_action_index(coord1, coord2, env)
            
            if action_idx is None:
                print(f"Could not find action index for coordinates: {move}")
                # Fall back to a random valid action
                action_idx = env.action_space.sample()
            
            return action_idx
            
        except Exception as e:
            print(f"Error parsing move to action: {e}")
            return None

async def run_agent(args):
    """Run the Candy Crush agent with the specified parameters."""
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
        agent = CandyCrushAgent(
            model_name=args.model,
            agent_mode=args.agent_mode
        )
        print(f"\n{'*' * 80}")
        if args.agent_mode == "base":
            print(f"RUNNING IN BASE MODULE MODE - Using model: {args.model}")
            print("Base module uses vision API to analyze rendered game images")
        elif args.agent_mode == "full":
            print(f"RUNNING IN FULL PIPELINE MODE - Using model: {args.model}")
            print("Full pipeline uses Perception → Memory → Reasoning modules")
        elif args.agent_mode == "memory_reasoning":
            print(f"RUNNING IN MEMORY + REASONING MODE - Using model: {args.model}")
            print("Memory and Reasoning modules with simplified perception")
        elif args.agent_mode == "perception_reasoning":
            print(f"RUNNING IN PERCEPTION + REASONING MODE - Using model: {args.model}")
            print("Perception and Reasoning modules, no memory module")
        print(f"{'*' * 80}\n")
    else:
        print("\n" + "*" * 80)
        print("RUNNING WITH RANDOM ACTIONS - No AI model used")
        print("*" * 80 + "\n")
    
    # Run for the specified number of games
    for run_id in range(1, args.num_runs + 1):
        print(f"\n========== STARTING RUN {run_id}/{args.num_runs} ==========\n")
        
        # For Candy Crush, render_mode="human" is often used for visualization,
        # but base/memory_reasoning (Base+Memory) needs an array if it uses env.render() for the frame.
        # However, CandyCrushBaseModule is designed to take a frame passed to it.
        # The main loop in Candy Crush already has logic to generate a frame if args.agent_mode == "base".
        # We will adapt this for memory_reasoning (Base+Memory) as well.
        
        current_render_mode = "human"
        if not args.random and (args.agent_mode == "base" or args.agent_mode == "memory_reasoning"):
            current_render_mode = "rgb_array"
            print(f"Using render_mode='rgb_array' for {args.agent_mode} mode.")

        env = TileMatchEnv(
            num_rows=8, 
            num_cols=8, 
            num_colours=4, 
            num_moves=50, 
            colourless_specials=[], 
            colour_specials=[], 
            seed=run_id,
            render_mode=current_render_mode # Use determined render_mode
        )
        
        wrapped_env = CandyCrushObservationWrapper(env)
        obs, _ = env.reset()
        # wrapped_obs = wrapped_env.observation(obs) # Not used directly in agent.get_action logic
        
        env.render() # Initial render
        
        print("Initial board state:")
        board_2d = convert_obs_to_2d_array(obs)
        print_board(board_2d)
        
        done = False
        terminated = False
        truncated = False
        total_reward = 0
        step = 0
        max_retries = 3
        stuck_count = 0
        max_stuck_actions = 50
        
        while not done:
            frame_for_agent = None
            info_for_agent = obs # Default for full, P+R modes; obs is the raw dict from env

            if not args.random and (args.agent_mode == "base" or args.agent_mode == "memory_reasoning"):
                try:
                    rendered_frame = env.render() # Attempt to get frame
                    if rendered_frame is None or not isinstance(rendered_frame, np.ndarray):
                        board_2d_fallback = convert_obs_to_2d_array(obs)
                        h_fb, w_fb = board_2d_fallback.shape
                        frame_for_agent = np.zeros((h_fb*50, w_fb*50, 3), dtype=np.uint8)
                        colors_fb = {0:(200,200,200),1:(0,255,0),2:(0,255,255),3:(255,0,255),4:(255,0,0)}
                        for r_fb in range(h_fb):
                            for c_fb in range(w_fb):
                                color_idx_fb = int(board_2d_fallback[r_fb,c_fb])
                                frame_for_agent[r_fb*50:(r_fb+1)*50, c_fb*50:(c_fb+1)*50] = colors_fb.get(color_idx_fb, (128,128,128))
                        print("Warning: env.render() did not return valid frame, using fallback image.")
                    else:
                        frame_for_agent = rendered_frame
                    
                    os.makedirs(os.path.dirname(BOARD_IMG_PATH), exist_ok=True)
                    Image.fromarray(frame_for_agent).save(BOARD_IMG_PATH)
                    # print(f"Saved board image to {BOARD_IMG_PATH} for {args.agent_mode} mode")

                    info_for_agent = dict(obs) # Create enriched info for base/M+R
                    info_for_agent['img_path'] = BOARD_IMG_PATH
                    # Add any other textual info useful for BaseModule prompt if needed, e.g., from obs
                    # For Candy Crush, num_moves_left is in obs, which is already part of info_for_agent

                except Exception as e:
                    print(f"Error preparing frame/info for {args.agent_mode} mode: {e}")
                    if frame_for_agent is None: # Ensure a fallback frame exists if error occurred early
                        board_2d_fallback = convert_obs_to_2d_array(obs)
                        h_fb, w_fb = board_2d_fallback.shape
                        frame_for_agent = np.zeros((h_fb*50, w_fb*50, 3), dtype=np.uint8)
                    if not isinstance(info_for_agent, dict) or 'img_path' not in info_for_agent:
                        info_for_agent = dict(obs) # Fallback info
                        info_for_agent['img_path'] = BOARD_IMG_PATH

            if args.random:
                action = env.action_space.sample()
                action_plan = {"move": env._action_to_coords[action], "thought": "Random action"}
                perception_data, memory_summary_for_decision = None, None
            else:
                retry_count = 0
                valid_action_found = False
                current_obs_for_agent_call = None
                current_info_for_agent_call = None

                if args.agent_mode == "base" or args.agent_mode == "memory_reasoning":
                    current_obs_for_agent_call = frame_for_agent
                    current_info_for_agent_call = info_for_agent
                else: # full, perception_reasoning
                    current_obs_for_agent_call = obs["board"] # Pass the board numpy array
                    current_info_for_agent_call = obs       # Pass the full obs dict

                while not valid_action_found and retry_count < max_retries:
                    action_plan, perception_data, memory_summary_for_decision = await agent.get_action(
                        current_obs_for_agent_call, 
                        current_info_for_agent_call, 
                        max_retries_const=1 # Outer loop handles retries
                    )
                    
                    if action_plan and isinstance(action_plan.get("move"), tuple) and len(action_plan.get("move")) == 2:
                        valid_action_found = True
                    else:
                        print(f"Agent returned invalid action plan. Attempt {retry_count + 1}/{max_retries}")
                        retry_count += 1
                        await asyncio.sleep(1)
                
                if not valid_action_found:
                    print("Exhausted retry attempts for agent, using fallback random action.")
                    action = env.action_space.sample()
                    action_plan = {"move": env._action_to_coords[action], "thought": "Fallback (random) after retries"}
                    perception_data, memory_summary_for_decision = None, None # No specific perception/memory for this random fallback
            
            # Parse move and take step
            agent_move = action_plan.get("move")
            action = agent.parse_move_to_action(agent_move, env) if not args.random else env.action_space.sample()
            if action is None: # If parse_move_to_action failed for AI agent
                print(f"Failed to parse agent move {agent_move}, using random action as fallback.")
                action = env.action_space.sample()
                action_plan["move"] = env._action_to_coords[action] # Update plan for logging
                action_plan["thought"] += " (Used random due to parse failure)"

            current_board_for_stuck_check = obs['board'][0].copy() if 'board' in obs and obs['board'].ndim > 1 else None
            
            next_obs, reward, terminated, truncated, info_after_step = env.step(action)
            env.render()
            # wrapped_next_obs = wrapped_env.observation(next_obs) # Not directly used by agent.get_action

            # After the step, update memory for modes that use it
            if not args.random and (args.agent_mode == "full" or args.agent_mode == "memory_reasoning"):
                state_to_log_in_memory = None
                if args.agent_mode == "full":
                    # For full mode, perception module analyzes the new board state
                    state_to_log_in_memory = agent.perception_module.analyze_board(next_obs['board'], next_obs)
                elif args.agent_mode == "memory_reasoning":
                    # For Base+Memory, we need a textual representation of next_obs for memory storage.
                    # Use CandyCrushPerceptionModule to get this.
                    temp_perception_for_memory_update = CandyCrushPerceptionModule(model_name=agent.model_name)
                    state_to_log_in_memory = temp_perception_for_memory_update.analyze_board(next_obs['board'], next_obs)
                
                if state_to_log_in_memory and agent.memory_module:
                    agent.memory_module.add_game_state(state_to_log_in_memory, action_plan.get("move"))
            
            # Stuck check
            if current_board_for_stuck_check is not None and 'board' in next_obs and next_obs['board'].ndim > 1:
                new_board_for_stuck_check = next_obs['board'][0]
                if np.array_equal(current_board_for_stuck_check, new_board_for_stuck_check) and reward == 0:
                    stuck_count += 1
                    if stuck_count >= max_stuck_actions:
                        print(f"Board stuck for {max_stuck_actions} steps. Terminating.")
                        done = True
                else:
                    stuck_count = 0
            
            # `perception_data` here is from get_action (perception of state *before* action)
            # `memory_summary_for_decision` is from get_action (memory used *for* action)
            log_step(step, action_plan, perception_data, memory_summary_for_decision, action, reward, info_after_step, next_obs, run_id)
            log_raw_data(step, action, reward, info_after_step, next_obs, run_id)
            
            total_reward += reward
            step += 1
            
            # Print board state after action
            print(f"\nBoard after move (step {step}):")
            board_2d = convert_obs_to_2d_array(next_obs)
            print_board(board_2d)
            print(f"Action: {action}, Reward: {reward}, Total Reward: {total_reward}, Moves left: {next_obs['num_moves_left']}")
            
            # Check if game is done
            done = done or terminated or truncated
            
            # Update current observation
            obs = next_obs
        
        # Close the environment
        env.close()
        
        # Get final score
        all_scores.append(total_reward)  # Store total reward as the score
        
        # Log run summary
        run_summary = {
            "run_id": int(run_id),
            "score": float(total_reward),
            "steps": int(step),
            "timestamp": datetime.now().isoformat(),
            "termination_reason": "Normal completion" if stuck_count < max_stuck_actions else f"Terminated due to board being stuck for {max_stuck_actions} actions"
        }
        
        with open(summary_file, 'a') as f:
            json.dump(run_summary, f)
            f.write('\n')
        
        # Print final results for this run
        termination_reason = "Normal completion" if stuck_count < max_stuck_actions else f"Terminated due to board being stuck for {max_stuck_actions} actions"
        
        print(f"\nRun {run_id} finished after {step} steps")
        print(f"Total reward: {total_reward}")
        print(f"Termination reason: {termination_reason}")
        print(f"========== COMPLETED RUN {run_id}/{args.num_runs} ==========\n")
    
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
    parser = argparse.ArgumentParser(description='Run the Candy Crush AI Agent')
    
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