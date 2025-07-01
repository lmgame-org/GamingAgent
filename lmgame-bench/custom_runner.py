import argparse
import os
import json
import datetime
import time
import numpy as np
import yaml
from typing import Any
import sys
import re
import random

import gymnasium as gym

import retro
from retro.enums import Actions, Observations, State # retro.data will be used directly for Integrations

from gamingagent.agents.base_agent import BaseAgent
from gamingagent.modules import PerceptionModule, ReasoningModule # Observation is imported by Env
from tools.utils import draw_grid_on_image
# Directly import the specific environment we are using
from gamingagent.envs.custom_01_2048.twentyFortyEightEnv import TwentyFortyEightEnv
from gamingagent.envs.custom_02_sokoban.sokobanEnv import SokobanEnv
from gamingagent.envs.custom_03_candy_crush.candyCrushEnv import CandyCrushEnvWrapper
from gamingagent.envs.custom_04_tetris.tetrisEnv import TetrisEnv
from gamingagent.envs.custom_05_doom.doomEnv import DoomEnvWrapper
from gamingagent.envs.custom_06_pokemon_red.pokemonRedEnv import PokemonRedEnv

from gamingagent.envs.retro_01_super_mario_bros.superMarioBrosEnv import SuperMarioBrosEnvWrapper
from gamingagent.envs.retro_02_ace_attorney.aceAttorneyEnv import AceAttorneyEnv
from gamingagent.envs.retro_03_1942.NineteenFortyTwo_env import NineteenFortyTwoEnvWrapper

game_config_mapping = {"twenty_forty_eight": "custom_01_2048",
                       "sokoban": "custom_02_sokoban",
                       "candy_crush": "custom_03_candy_crush",
                       "tetris": "custom_04_tetris",
                       "doom": "custom_05_doom",
                       "pokemon_red": "custom_06_pokemon_red",
                       "super_mario_bros":"retro_01_super_mario_bros",
                       "ace_attorney":"retro_02_ace_attorney",
                       "nineteen_forty_two": "retro_03_1942"
                       }

def str_to_bool(v):
    """Convert string boolean values to actual booleans for argparse"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_arguments(defaults_map=None, argv_to_parse=None):
    parser = argparse.ArgumentParser(description="Run GamingAgent for a specified Gym Environment.")
    # Game name will be set by defaults_map from prelim_parser, so not strictly required here.
    # A check after parsing will ensure it has a value.
    parser.add_argument("--game_name", type=str, default=None, 
                        help="Name of the game (e.g., twenty_forty_eight, sokoban). Set by prelim parser.")
    parser.add_argument("--config_root_dir", type=str, default="gamingagent/configs",
                        help="Root directory for agent configurations.")
    parser.add_argument("--model_name", type=str, default="claude-3-haiku-20240307",
                        help="Name of the model for the agent.")
    parser.add_argument("--harness", action="store_true",
                        help="Use perception-memory-reasoning pipeline (harness mode). Default is False.")
    parser.add_argument("--num_runs", type=int, default=1, help="Number of game episodes.")
    parser.add_argument("--observation_mode", type=str, default="vision",
                        choices=["vision", "text", "both"], help="Agent's observation mode.")
    parser.add_argument("--max_memory", type=int, default=20, help="Agent's max memory entries.")
    parser.add_argument("--use_reflection", type=str_to_bool, default=True, help="Enable reflection in memory module. Default is True.")
    parser.add_argument("--use_perception", type=str_to_bool, default=True, help="Enable perception API calls for image processing. Default is True.")
    parser.add_argument("--use_summary", type=str_to_bool, default=False, help="Enable trajectory summarization in memory module. Default is False.")
    parser.add_argument("--max_steps_per_episode", type=int, default=1000, help="Max steps per episode.")
    parser.add_argument("--use_custom_prompt", action="store_true", help="If set, will use the custom prompt from module_prompts.json if present.")
    parser.add_argument("--scaffolding", type=str, default=None, help="Grid dimensions as '(rows,cols)' for coordinate grid on images, e.g., '(5,5)'. Default is None.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for environment.")
    # Env type is fixed to custom gym for this runner

    # Serving-related arguments
    parser.add_argument(
        "--modal_url",
        type=str,
        default=None,
        help="Optional URL for a Modal‑hosted inference endpoint passed to BaseAgent.",
    )
    parser.add_argument(
        "--vllm_url",
        type=str,
        default=None,
        help="Optional URL for a vLLM inference endpoint passed to BaseAgent.",
    )

    if defaults_map:
        parser.set_defaults(**defaults_map)
    
    if argv_to_parse:
        return parser.parse_args(argv_to_parse)
    return parser.parse_args()

def create_environment(game_name_arg: str, 
                       model_name_arg: str,
                       obs_mode_arg: str, 
                       config_dir_name_for_env_cfg: str, # For loading game_env_config.json
                       cache_dir_for_adapter: str,
                       harness: bool = False):
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
    elif game_name_arg == "sokoban":
        # Load params specific to Sokoban
        if os.path.exists(env_specific_config_path):
            with open(env_specific_config_path, 'r') as f:
                env_specific_config = json.load(f)
                env_init_kwargs = env_specific_config.get('env_init_kwargs', {})
                env_init_params['dim_room'] = env_init_kwargs.get('dim_room', (10,10))
                env_init_params['max_steps_episode'] = env_init_kwargs.get('max_steps_episode', 200)
                env_init_params['num_boxes'] = env_init_kwargs.get('num_boxes', 3)
                env_init_params['num_gen_steps'] = env_init_kwargs.get('num_gen_steps') # Can be None
                env_init_params['level_to_load'] = env_specific_config.get('level_to_load') # Can be None
                env_init_params['render_mode'] = env_specific_config.get('render_mode', 'human')
                env_init_params['tile_size_for_render'] = env_specific_config.get('tile_size_for_render', 32)
                env_init_params['max_stuck_steps_for_adapter'] = env_specific_config.get('max_unchanged_steps_for_termination', 20)
        else:
            print(f"Warning: {env_specific_config_path} for {game_name_arg} not found. Using default env parameters for Sokoban.")
            env_init_params['dim_room'] = (10,10)
            env_init_params['max_steps_episode'] = 200
            env_init_params['num_boxes'] = 3
            env_init_params['num_gen_steps'] = None
            env_init_params['level_to_load'] = None
            env_init_params['render_mode'] = 'human'
            env_init_params['tile_size_for_render'] = 32
            env_init_params['max_stuck_steps_for_adapter'] = 20

        
        print(f"Initializing environment: {game_name_arg} with params: {env_init_params}")
        env = SokobanEnv(
            render_mode=env_init_params.get('render_mode'),
            dim_room=tuple(env_init_params.get('dim_room')), # Ensure it's a tuple
            max_steps_episode=env_init_params.get('max_steps_episode'),
            num_boxes=env_init_params.get('num_boxes'),
            num_gen_steps=env_init_params.get('num_gen_steps'),
            level_to_load=env_init_params.get('level_to_load'),
            tile_size_for_render=env_init_params.get('tile_size_for_render'),
            game_name_for_adapter=game_name_arg, 
            observation_mode_for_adapter=obs_mode_arg, 
            agent_cache_dir_for_adapter=cache_dir_for_adapter, 
            game_specific_config_path_for_adapter=env_specific_config_path, 
            max_stuck_steps_for_adapter=env_init_params.get('max_stuck_steps_for_adapter')
        )
        return env
    elif game_name_arg == "candy_crush":
        # Load params specific to Candy Crush
        # The config_dir_name_for_env_cfg for candy_crush will be "custom_03_candy_crush"
        if os.path.exists(env_specific_config_path):
            with open(env_specific_config_path, 'r') as f:
                env_specific_config = json.load(f)
                env_init_kwargs = env_specific_config.get('env_init_kwargs', {})
                # Parameters for CandyCrushEnv's internal TileMatchEnv
                env_init_params['num_rows'] = env_init_kwargs.get('num_rows', 8)
                env_init_params['num_cols'] = env_init_kwargs.get('num_cols', 8)
                env_init_params['num_colours'] = env_init_kwargs.get('num_colours', 4)
                env_init_params['num_moves'] = env_init_kwargs.get('num_moves', 50)
                # render_mode is for the wrapper's internal renderer if used, not GymEnvAdapter
                env_init_params['render_mode_for_make'] = env_specific_config.get('render_mode_for_make', 'string') 
                env_init_params['tile_size_for_render'] = env_specific_config.get('tile_size_for_render', 32)
                # max_stuck_steps_for_adapter for GymEnvAdapter instance
                env_init_params['max_stuck_steps_for_adapter'] = env_specific_config.get('max_unchanged_steps_for_termination', 20) # Default from Sokoban
        else:
            print(f"Warning: {env_specific_config_path} for {game_name_arg} not found. Using default env parameters for Candy Crush.")
            env_init_params['num_rows'] = 8
            env_init_params['num_cols'] = 8
            env_init_params['num_colours'] = 4
            env_init_params['num_moves'] = 50
            env_init_params['render_mode_for_make'] = 'string'
            env_init_params['tile_size_for_render'] = 32
            env_init_params['max_stuck_steps_for_adapter'] = 20

        print(f"Initializing environment: {game_name_arg} with params: {env_init_params}")
        env = CandyCrushEnvWrapper(
            # Parameters for CandyCrushEnv -> TileMatchEnv core
            num_rows_override=env_init_params.get('num_rows'),
            num_cols_override=env_init_params.get('num_cols'),
            num_colours_override=env_init_params.get('num_colours'),
            num_moves_override=env_init_params.get('num_moves'),
            # Parameters for GymEnvAdapter instance within CandyCrushEnv
            game_name_for_adapter=game_name_arg, 
            observation_mode_for_adapter=obs_mode_arg, 
            agent_cache_dir_for_adapter=cache_dir_for_adapter, 
            # This is the path the env itself will use to load its own full config (including env_init_kwargs)
            game_specific_config_path_for_adapter=env_specific_config_path, 
            max_stuck_steps_for_adapter=env_init_params.get('max_stuck_steps_for_adapter'),
            # Other params potentially needed by CandyCrushEnv if not covered by game_specific_config_path_for_adapter
            # config_root_dir is already an arg to runner, CandyCrushEnv doesn't need it directly if path is absolute
        )
        return env
    elif game_name_arg == "tetris":
        # Load params specific to Tetris
        if os.path.exists(env_specific_config_path):
            with open(env_specific_config_path, 'r') as f:
                env_specific_config = json.load(f)
                env_init_kwargs = env_specific_config.get('env_init_kwargs', {})
                env_init_params['board_width'] = env_init_kwargs.get('board_width', 10)
                env_init_params['board_height'] = env_init_kwargs.get('board_height', 20)
                env_init_params['gravity'] = env_init_kwargs.get('gravity', True)
                env_init_params['render_upscale'] = env_init_kwargs.get('render_upscale', 25)
                env_init_params['queue_size'] = env_init_kwargs.get('queue_size', 4)
                env_init_params['render_mode_for_make'] = env_specific_config.get('render_mode_for_make', 'human') # Corresponds to TetrisEnv render_mode
                env_init_params['max_stuck_steps_for_adapter'] = env_specific_config.get('max_unchanged_steps_for_termination', 30)
        else:
            print(f"Warning: {env_specific_config_path} for {game_name_arg} not found. Using default env parameters for Tetris.")
            env_init_params['board_width'] = 10
            env_init_params['board_height'] = 20
            env_init_params['gravity'] = True
            env_init_params['render_upscale'] = 25
            env_init_params['queue_size'] = 4
            env_init_params['render_mode_for_make'] = 'human'
            env_init_params['max_stuck_steps_for_adapter'] = 30

        print(f"Initializing environment: {game_name_arg} with params: {env_init_params}")
        env = TetrisEnv(
            render_mode=env_init_params.get('render_mode_for_make'),
            board_width=env_init_params.get('board_width'),
            board_height=env_init_params.get('board_height'),
            gravity=env_init_params.get('gravity'),
            render_upscale=env_init_params.get('render_upscale'),
            queue_size=env_init_params.get('queue_size'),
            # Adapter related params
            game_name_for_adapter=game_name_arg,
            observation_mode_for_adapter=obs_mode_arg,
            agent_cache_dir_for_adapter=cache_dir_for_adapter,
            game_specific_config_path_for_adapter=env_specific_config_path,
            max_stuck_steps_for_adapter=env_init_params.get('max_stuck_steps_for_adapter')
            # seed will be passed during reset, not __init__ for TetrisEnv as per its definition
        )
        return env
    elif game_name_arg == "pokemon_red":
        # Load params specific to Pokemon Red
        if os.path.exists(env_specific_config_path):
            with open(env_specific_config_path, 'r') as f:
                env_specific_config = json.load(f)
                env_init_kwargs = env_specific_config.get('env_init_kwargs', {})
                env_init_params['rom_path'] = env_init_kwargs.get('rom_path')
                env_init_params['sound'] = env_init_kwargs.get('sound', False)
                env_init_params['render_mode_for_make'] = env_specific_config.get('render_mode', 'human')
                env_init_params['max_stuck_steps_for_adapter'] = env_specific_config.get('max_unchanged_steps_for_termination', 20)
        else:
            print(f"Warning: {env_specific_config_path} for {game_name_arg} not found. Using default env parameters for Pokemon Red.")
            env_init_params['rom_path'] = None
            env_init_params['sound'] = False
            env_init_params['render_mode_for_make'] = 'human'
            env_init_params['max_stuck_steps_for_adapter'] = 20

        print(f"Initializing environment: {game_name_arg} with params: {env_init_params}")
        env = PokemonRedEnv(
            render_mode=env_init_params.get('render_mode_for_make'),
            rom_path=env_init_params.get('rom_path'),
            sound=env_init_params.get('sound'),
            # Adapter related params
            game_name_for_adapter=game_name_arg,
            observation_mode_for_adapter=obs_mode_arg,
            agent_cache_dir_for_adapter=cache_dir_for_adapter,
            game_specific_config_path_for_adapter=env_specific_config_path,
            max_stuck_steps_for_adapter=env_init_params.get('max_stuck_steps_for_adapter'),
            harness=harness
        )
        return env
    elif game_name_arg == "super_mario_bros":
        # SuperMarioBrosEnvWrapper loads its specific configs internally.
        # The runner primarily needs to provide paths and agent/run-level settings.
        env_wrapper_config_dir = os.path.join("gamingagent/envs", config_dir_name_for_env_cfg)
        
        print(f"Initializing environment: {game_name_arg} using SuperMarioBrosEnvWrapper")
        print(f"  Wrapper config dir: {env_wrapper_config_dir}")
        print(f"  Model name for adapter: {model_name_arg}")
        print(f"  Observation mode for adapter: {obs_mode_arg}")
        print(f"  Base log dir for adapter: {cache_dir_for_adapter}")

        env = SuperMarioBrosEnvWrapper(
            game_name=game_name_arg,
            config_dir_path=env_wrapper_config_dir, # e.g., "gamingagent/envs/retro_01_super_mario_bros"
            observation_mode=obs_mode_arg,
            base_log_dir=cache_dir_for_adapter # This will be like "cache/super_mario_bros/{model_name}_agent_cache"
        )
        return env
    elif game_name_arg == "nineteen_forty_two":
        # NineteenFortyTwoEnvWrapper loads its specific configs internally.
        # The runner primarily needs to provide paths and agent/run-level settings.
        env_wrapper_config_dir = os.path.join("gamingagent/envs", config_dir_name_for_env_cfg)
        
        print(f"Initializing environment: {game_name_arg} using NineteenFortyTwoEnvWrapper")
        print(f"  Wrapper config dir: {env_wrapper_config_dir}")
        print(f"  Model name for adapter: {model_name_arg}")
        print(f"  Observation mode for adapter: {obs_mode_arg}")
        print(f"  Base log dir for adapter: {cache_dir_for_adapter}")

        env = NineteenFortyTwoEnvWrapper(
            game_name=game_name_arg,
            config_dir_path=env_wrapper_config_dir, # e.g., "gamingagent/envs/retro_03_1942"
            observation_mode=obs_mode_arg,
            base_log_dir=cache_dir_for_adapter # This will be like "cache/1942/{model_name}_agent_cache"
        )
        return env
    elif game_name_arg == "ace_attorney":
        # Parameters for AceAttorneyEnv which inherits from retro.Env
        # These will be passed to AceAttorneyEnv.__init__
        # Some will directly go to retro.Env.__init__ via super() call
        # Others are for the adapter or wrapper behavior
        env_params_for_constructor = {}
        if os.path.exists(env_specific_config_path):
            with open(env_specific_config_path, 'r') as f:
                env_cfg_json = json.load(f)
                # Params for retro.Env base class
                retro_init_kwargs = env_cfg_json.get('env_init_kwargs', {})
                env_params_for_constructor['game'] = retro_init_kwargs.get('retro_game_name', 'AceAttorney-GbAdvance')
                env_params_for_constructor['state'] = retro_init_kwargs.get('retro_state_name', State.DEFAULT) # From retro.enums
                env_params_for_constructor['scenario'] = retro_init_kwargs.get('scenario') 
                env_params_for_constructor['info'] = retro_init_kwargs.get('info') 
                
                use_restricted_val = retro_init_kwargs.get('use_restricted_actions', "FILTERED") # Get the value
                if isinstance(use_restricted_val, str):
                    use_restricted_str_upper = use_restricted_val.upper()
                    if use_restricted_str_upper == "DISCRETE":
                        env_params_for_constructor['use_restricted_actions'] = Actions.DISCRETE
                    elif use_restricted_str_upper == "MULTI_DISCRETE":
                        env_params_for_constructor['use_restricted_actions'] = Actions.MULTI_DISCRETE
                    elif use_restricted_str_upper == "ALL":
                        env_params_for_constructor['use_restricted_actions'] = Actions.ALL
                    # Default to FILTERED if string is "FILTERED" or unrecognized
                    else: 
                        env_params_for_constructor['use_restricted_actions'] = Actions.FILTERED
                elif isinstance(use_restricted_val, int):
                    # Pass integer directly, assuming it corresponds to retro.Actions enum values
                    env_params_for_constructor['use_restricted_actions'] = use_restricted_val
                else: 
                    # Fallback for unexpected types, default to FILTERED
                    print(f"Warning: Unexpected type for use_restricted_actions: {type(use_restricted_val)}. Defaulting to FILTERED.")
                    env_params_for_constructor['use_restricted_actions'] = Actions.FILTERED
                
                env_params_for_constructor['record'] = retro_init_kwargs.get('record', False)
                env_params_for_constructor['players'] = retro_init_kwargs.get('players', 1)
                
                inttype_str = retro_init_kwargs.get('inttype', "ALL").upper()
                if inttype_str == "CUSTOM":
                     env_params_for_constructor['inttype'] = retro.data.Integrations.CUSTOM
                elif inttype_str == "STABLE":
                     env_params_for_constructor['inttype'] = retro.data.Integrations.STABLE
                elif inttype_str == "EXPERIMENTAL":
                     env_params_for_constructor['inttype'] = retro.data.Integrations.EXPERIMENTAL
                elif inttype_str == "ALL":
                     env_params_for_constructor['inttype'] = retro.data.Integrations.ALL
                else:
                     env_params_for_constructor['inttype'] = retro.data.Integrations.ALL
                
                obs_type_str = retro_init_kwargs.get('obs_type', "IMAGE").upper()
                if obs_type_str == "RAM":
                    env_params_for_constructor['obs_type'] = Observations.RAM
                else: 
                    env_params_for_constructor['obs_type'] = Observations.IMAGE

                # Params for GymEnvAdapter instance within AceAttorneyEnv
                env_params_for_constructor['adapter_game_name'] = game_name_arg # Should be "ace_attorney"
                env_params_for_constructor['adapter_observation_mode'] = obs_mode_arg
                env_params_for_constructor['adapter_agent_cache_dir'] = cache_dir_for_adapter
                env_params_for_constructor['adapter_config_path'] = env_specific_config_path
                env_params_for_constructor['adapter_max_stuck_steps'] = env_cfg_json.get('max_unchanged_steps_for_termination', 50)
                
                # Parameter for AceAttorneyEnv wrapper itself (e.g. render mode)
                env_params_for_constructor['wrapper_render_mode'] = env_cfg_json.get('render_mode_gym_adapter', 'rgb_array')

        else:
            print(f"Warning: {env_specific_config_path} for {game_name_arg} not found. Using default parameters for AceAttorneyEnv.")
            # Fill with defaults if config is missing
            env_params_for_constructor['game'] = 'AceAttorney-GbAdvance'
            env_params_for_constructor['state'] = State.DEFAULT
            env_params_for_constructor['use_restricted_actions'] = Actions.FILTERED
            env_params_for_constructor['obs_type'] = Observations.IMAGE
            env_params_for_constructor['inttype'] = retro.data.Integrations.ALL
            env_params_for_constructor['adapter_game_name'] = game_name_arg
            env_params_for_constructor['adapter_observation_mode'] = obs_mode_arg
            env_params_for_constructor['adapter_agent_cache_dir'] = cache_dir_for_adapter
            env_params_for_constructor['adapter_config_path'] = env_specific_config_path
            env_params_for_constructor['adapter_max_stuck_steps'] = 50
            env_params_for_constructor['wrapper_render_mode'] = 'rgb_array'

        print(f"Initializing environment: AceAttorneyEnv with combined params: { {k:v for k,v in env_params_for_constructor.items() if k not in ['adapter_agent_cache_dir']} }")
        
        # Import retro here if not already at top level, for retro.STATE_DEFAULT etc.
        # import retro # No longer needed here as DefaultStates, etc. are imported above
        env = AceAttorneyEnv(**env_params_for_constructor)
        return env
    elif game_name_arg == "doom":
        # DoomEnvWrapper loads its specific configs internally.
        # The runner primarily needs to provide paths and agent/run-level settings.
        env_wrapper_config_dir = os.path.join("gamingagent/envs", config_dir_name_for_env_cfg)
        
        print(f"Initializing environment: {game_name_arg} using DoomEnvWrapper")
        print(f"  Wrapper config dir: {env_wrapper_config_dir}")
        print(f"  Model name for adapter: {model_name_arg}")
        print(f"  Observation mode for adapter: {obs_mode_arg}")
        print(f"  Base log dir for adapter: {cache_dir_for_adapter}")
        print(f"  Config path: {env_specific_config_path}")

        # Verify config file exists
        if not os.path.exists(env_specific_config_path):
            print(f"ERROR: Config file not found at {env_specific_config_path}")
            return None

        env = DoomEnvWrapper(
            game_name="doom",  # Match test file
            config_dir_path=os.path.dirname(env_specific_config_path),  # Use the directory containing the config file
            observation_mode=obs_mode_arg,
            base_log_dir=cache_dir_for_adapter,
            render_mode_human=True,  # Enable human rendering
            record_video=False,
            video_dir="videos/doom",
            model_name=model_name_arg,
            headless=False,  # Allow display
            debug=True  # Add debug mode to help track issues
        )
        return env
    else:
        print(f"ERROR: Game '{game_name_arg}' is not defined or implemented in custom_runner.py's create_environment function.")
        return None

def run_game_episode(agent: BaseAgent, game_env: gym.Env, episode_id: int, args: argparse.Namespace):
    """Run a single episode of the game."""
    # Pass episode_id to env.reset
    agent_observation, last_info = game_env.reset(max_memory=args.max_memory, seed=args.seed, episode_id=episode_id)
    if args.seed is not None: args.seed += 1 # Increment seed for next potential run

    total_reward_for_episode = 0.0
    total_perf_score_for_episode = 0.0
    final_step_num = 0

    for step_num in range(args.max_steps_per_episode):
        final_step_num = step_num + 1
        game_env.render() # Call env's render method directly

        start_time = time.time()
        action_dict, processed_agent_observation = agent.get_action(agent_observation)
        end_time = time.time()
        time_taken_s = end_time - start_time
        # Special handling for Doom game
        if isinstance(game_env, DoomEnvWrapper):
            # Handle action like test file
            if action_dict and action_dict.get("action") is not None:
                action_str = str(action_dict.get("action")).strip().lower()
                
                # For attack action, ensure it's always 8 frames
                if "attack" in action_str:
                    action_str = "(attack, 8)"
            else:
                action_str = "none"
        
            thought_process = action_dict.get("thought", "") if action_dict else "No thought process due to API failure."

            # Print action before step
            print(f"\nExecuting action: '{action_str}'")
            print(f"Thought process: {thought_process}")
            print(f"Time taken: {time_taken_s:.2f}s")
            print("-" * 50)

            # Step the environment with minimal parameters like test file
            agent_observation, reward, terminated, truncated, last_info, current_step_perf_score = game_env.step(action_str)
        else:
            # Ensure action_dict is not None and action is handled if None
            action_str = None
            if action_dict and action_dict.get("action") is not None:
                action_str = action_dict.get("action")
            
            action_str_agent = "None" # Default to "None" string if no valid action
            if action_str:
                action_str_agent = str(action_str).strip().lower()
            
            thought_process = action_dict.get("thought", "") if action_dict else "No thought process due to API failure."

            # --- MODIFIED: Extract raw LLM output to pass to env.step ---
            raw_llm_output_for_env = None

            if action_dict:
                if "raw_response_str" in action_dict and isinstance(action_dict["raw_response_str"], str):
                    raw_llm_output_for_env = action_dict["raw_response_str"]
            else:
                print("[Runner DEBUG] action_dict is None") # DEBUG
            
            # Conditionally pass raw_llm_output_for_next_obs
            step_args = {
                "agent_action_str": action_str_agent,
                "thought_process": thought_process,
                "time_taken_s": time_taken_s
            }
            if args.game_name == "ace_attorney":
                step_args["raw_llm_output_for_next_obs"] = raw_llm_output_for_env
            
            # Step the environment using the new signature, including agent action details
            agent_observation, reward, terminated, truncated, last_info, current_step_perf_score = game_env.step(**step_args)

        # Inherit game trajectory
        agent_observation.game_trajectory = processed_agent_observation.game_trajectory
            
        total_reward_for_episode += reward
        total_perf_score_for_episode += current_step_perf_score

        if terminated or truncated:
            break
            
    # game_env.close() is called after all runs are complete in main

    final_score_from_env = float(last_info.get('total_score', 0.0)) 

    # Updated print statement to show original values
    print(f"Episode {episode_id} finished after {final_step_num} steps. Original Final Env Score: {final_score_from_env}, Original Total Reward: {total_reward_for_episode:.2f}, Original Total Perf Score: {total_perf_score_for_episode:.2f}")
    
    # Overwrite scores for Ace Attorney episodes
    effective_total_reward = total_reward_for_episode
    effective_total_perf_score = total_perf_score_for_episode
    effective_final_score_from_env = final_score_from_env

    if isinstance(game_env, AceAttorneyEnv):
        current_checkpoint_score = 0
        if hasattr(game_env, 'calculate_final_performance_score'):
            try:
                # This will read the cumulative dialogue log up to this point in this execution of custom_runner.py
                current_checkpoint_score = game_env.calculate_final_performance_score()
                print(f"[Runner] Ace Attorney Episode {episode_id}: Checkpoint score calculated: {current_checkpoint_score}. Overwriting episode summary values.")
                effective_total_reward = float(current_checkpoint_score)
                effective_total_perf_score = float(current_checkpoint_score)
                effective_final_score_from_env = float(current_checkpoint_score) # Also use checkpoint score for the primary 'score' field
            except Exception as e:
                print(f"[Runner] Error calling calculate_final_performance_score for AceAttorneyEnv episode {episode_id}: {e}. Using original scores.")
        else:
            print("[Runner] AceAttorneyEnv instance does not have calculate_final_performance_score method. Using original scores.")

    # Record results with the adapter, using potentially overwritten values
    if hasattr(game_env, 'adapter') and game_env.adapter:
        game_env.adapter.record_episode_result(
            episode_id=episode_id,
            score=effective_final_score_from_env,       # Potentially overwritten
            steps=final_step_num,
            total_reward=effective_total_reward,        # Potentially overwritten
            total_perf_score=effective_total_perf_score # Potentially overwritten
        )
    else:
        print("Warning: game_env.adapter not found. Cannot record episode result for summary.")

    return

def main():
    prelim_parser = argparse.ArgumentParser(add_help=False)
    # No default for game_name here; it must be passed for prelim_parser to find the correct config.yaml
    prelim_parser.add_argument("--game_name", type=str, required=True, help="Game name needs to be passed to identify correct config.")
    prelim_parser.add_argument("--config_root_dir", type=str, default="gamingagent/configs", help="Root path config files.")
    pre_args, remaining_argv = prelim_parser.parse_known_args()

    if not pre_args.game_name:
        print("Warning: --game_name not provided or not parsed by prelim_parser. Game-specific defaults from config.yaml might not be loaded.")
        config_dir_name = None # No specific game config can be loaded
    else:
        config_dir_name = game_config_mapping.get(pre_args.game_name.lower())
    
    if not config_dir_name and pre_args.game_name: # game_name was provided, but not in mapping
        print(f"Warning: Game name '{pre_args.game_name}' not found in game_config_mapping. Using game name directly for config path.")
        config_dir_name = pre_args.game_name
    elif not config_dir_name and not pre_args.game_name: # game_name wasn't provided to prelim_parser
        # Defaults_from_yaml will be empty, main parser will use its own defaults or fail on required args
        pass

    defaults_from_yaml = {}
    config_file_path = None # Initialize config_file_path to ensure it's always defined

    # Add game_name from prelim_parser to defaults_from_yaml so it's passed to set_defaults for the main parser
    if pre_args.game_name:
        defaults_from_yaml['game_name'] = pre_args.game_name

    if config_dir_name: # Only try to load if we have a config_dir_name
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
                            defaults_from_yaml['observation_mode'] = agent_config_yaml.get('observation_mode')
                            defaults_from_yaml['use_custom_prompt'] = agent_config_yaml.get('use_custom_prompt')
                            defaults_from_yaml['use_reflection'] = agent_config_yaml.get('use_reflection')
                            defaults_from_yaml['use_perception'] = agent_config_yaml.get('use_perception')
                            defaults_from_yaml['use_summary'] = agent_config_yaml.get('use_summary')
                            defaults_from_yaml['scaffolding'] = agent_config_yaml.get('scaffolding')
                            
                            # Still load max_memory from its specific module config if present
                            if agent_config_yaml.get('modules'):
                                if agent_config_yaml['modules'].get('memory_module'):
                                    defaults_from_yaml['max_memory'] = agent_config_yaml['modules']['memory_module'].get('max_memory')
                        defaults_from_yaml = {k: v for k, v in defaults_from_yaml.items() if v is not None}
            except Exception as e:
                print(f"Warning: Could not load or process defaults from {config_file_path}: {e}")
        else:
            # This print is for when the specific game's config.yaml is not found
            print(f"Info: Game-specific config file {config_file_path} not found. Using command-line args and built-in defaults.")


    args = parse_arguments(defaults_map=defaults_from_yaml, argv_to_parse=remaining_argv)

    # Critical check: Ensure game_name has a value after all parsing.
    if not args.game_name:
        print("ERROR: game_name is missing after parsing. This should not happen if run.py provides it.")
        sys.exit(2) # Exit with a different code to distinguish from argparse error

    # --- Override specific args with YAML values if they existed in the config file ---
    # This makes config.yaml authoritative for these specific parameters over command-line args,
    # while game_name, model_name, and harness are primarily driven by command-line (from run.py).

    params_where_config_wins = {
        'num_runs', 
        'max_steps_per_episode',
        'seed',
        'max_memory',
        'use_reflection',
        'use_perception',
        'use_summary',
        'scaffolding'
    }

    if config_file_path and os.path.exists(config_file_path):
        for param_name in params_where_config_wins:
            if param_name in defaults_from_yaml: # If the param was indeed in the loaded YAML config
                yaml_value = defaults_from_yaml[param_name]
                current_arg_value = getattr(args, param_name, None)
                if current_arg_value != yaml_value:
                    print(f"INFO: Overriding '{param_name}' with value from {config_file_path}. Was: {current_arg_value}, Now: {yaml_value}")
                    setattr(args, param_name, yaml_value)
    # --- End of override logic ---

    # Ensure agent_prompts_config_path uses the potentially overridden args.config_root_dir and correct config_dir_name
    # config_dir_name determined earlier is correct for the game specified by command line.
    final_config_dir_name = config_dir_name 
    if not final_config_dir_name and args.game_name: # If prelim parsing didn't get game_name but main args did
        final_config_dir_name = game_config_mapping.get(args.game_name.lower(), args.game_name)

    agent_prompts_config_path = None
    if final_config_dir_name: # It might still be None if game_name was never resolved
        agent_prompts_config_path = os.path.join(args.config_root_dir, final_config_dir_name, "module_prompts.json")
        if not os.path.isfile(agent_prompts_config_path):
            print(f"Warning: Agent prompts file {agent_prompts_config_path} not found. Agent will use default prompts.")
            agent_prompts_config_path = None
    else:
        print("Warning: Could not determine config directory for prompts due to missing game name resolution.")

    # DEBUG PRINT
    # print(f"DEBUG: Value of args.harness before check: {args.harness} (type: {type(args.harness)})")

    custom_modules_for_agent = None
    if args.harness:
        print("Initializing agent in HARNESS mode.")
        custom_modules_for_agent = {"perception_module": PerceptionModule, "reasoning_module": ReasoningModule}
    else:
        print("Initializing agent in NON-HARNESS (BaseModule) mode.")

    # --- Create Environment FIRST ---
    runner_log_dir_base = os.path.join("cache", args.game_name, args.model_name.replace("-", "_")[:15], datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(runner_log_dir_base, exist_ok=True)
    print(f"Agent and Environment cache directory: {runner_log_dir_base}")

    # Parse scaffolding parameter
    scaffolding_dict = None
    if args.scaffolding:
        try:
            if isinstance(args.scaffolding, dict):
                # New dictionary format from config
                funcname = args.scaffolding.get('funcname')
                funcArgs = args.scaffolding.get('funcArgs', {})
                
                # Map function names to actual function objects
                function_mapping = {
                    'draw_grid_on_image': draw_grid_on_image
                }
                
                if funcname in function_mapping:
                    scaffolding_dict = {
                        'func': function_mapping[funcname],
                        'funcArgs': funcArgs
                    }
                    print(f"Using scaffolding function: {funcname} with args: {funcArgs}")
                else:
                    print(f"Warning: Unknown scaffolding function '{funcname}'. Using None.")
            else:
                # Legacy tuple format for backward compatibility
                scaffolding_str = str(args.scaffolding).strip()
                if scaffolding_str.startswith('(') and scaffolding_str.endswith(')'):
                    scaffolding_str = scaffolding_str[1:-1]  # Remove parentheses
                parts = [int(x.strip()) for x in scaffolding_str.split(',')]
                if len(parts) == 2:
                    scaffolding_dict = {
                        'func': draw_grid_on_image,
                        'funcArgs': {'grid_dim': tuple(parts)}
                    }
                    print(f"Using legacy scaffolding grid: {tuple(parts)}")
                else:
                    print(f"Warning: Invalid scaffolding format '{args.scaffolding}'. Expected '(rows,cols)'. Using None.")
        except (ValueError, AttributeError) as e:
            print(f"Warning: Could not parse scaffolding '{args.scaffolding}': {e}. Using None.")

    # --- Then Create Agent, passing the environment ---
    agent = BaseAgent(
        game_name=args.game_name,
        model_name=args.model_name,
        config_path=agent_prompts_config_path,
        harness=args.harness,
        use_custom_prompt=args.use_custom_prompt,
        max_memory=args.max_memory,
        use_reflection=args.use_reflection,
        use_perception=args.use_perception,
        use_summary=args.use_summary,
        custom_modules=custom_modules_for_agent,
        observation_mode=args.observation_mode,
        scaffolding=scaffolding_dict,
        cache_dir=runner_log_dir_base,
        vllm_url=args.vllm_url,
        modal_url=args.modal_url
    )
    
    # runner_log_dir = agent.cache_dir # Agent already sets its cache_dir, this can be removed or used for verification
    # os.makedirs(runner_log_dir, exist_ok=True) # Already created by agent or above
    # print(f"Agent cache directory (contains episode logs and summary): {runner_log_dir}")

    # Env params are now loaded inside create_environment
    game_env = create_environment(
        game_name_arg=args.game_name,
        model_name_arg=args.model_name,
        obs_mode_arg=args.observation_mode,
        config_dir_name_for_env_cfg=config_dir_name,
        cache_dir_for_adapter=runner_log_dir_base,
        harness=args.harness
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

    # --- Calculate and Print Ace Attorney Specific Final Score ---
    ace_attorney_checkpoint_score = None
    if isinstance(game_env, AceAttorneyEnv):
        if hasattr(game_env, 'calculate_final_performance_score'):
            try:
                ace_attorney_checkpoint_score = game_env.calculate_final_performance_score()
            except Exception as e:
                print(f"[Runner] Error calling calculate_final_performance_score for AceAttorneyEnv: {e}")
        else:
            print("[Runner] AceAttorneyEnv instance does not have calculate_final_performance_score method.")

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

    if ace_attorney_checkpoint_score is not None:
        print(f"Ace Attorney Checkpoint Score: {ace_attorney_checkpoint_score}")

if __name__ == "__main__":
    main() 
