import os
import json
from typing import Any, Dict, Tuple, Optional, List, Union

import retro
import numpy as np
from PIL import Image # For saving frames

from gamingagent.envs.gym_env_adapter import GymEnvAdapter # Changed from RetroEnvAdapter
from gamingagent.modules.core_module import Observation
# Removed: from tools.utils import convert_int_to_bcd_string, read_ram_fields

# Helper functions previously assumed to be in tools.utils

def _read_ram_value(ram: bytes, address_str: str) -> int:
    """Reads a single byte from RAM at the given hex address string."""
    return int(ram[int(address_str, 16)])

def _read_ram_fields(ram: bytes, addresses_config: Dict[str, Any]) -> Dict[str, Any]:
    """Reads multiple fields from RAM based on the addresses_config dictionary."""
    if ram is None:
        return {key: None for key in addresses_config}
    
    data = {}
    for key, addr_info in addresses_config.items():
        if addr_info is None: # Skip if address is not defined for a key
            data[key] = None
            continue
        try:
            if isinstance(addr_info, str): # Single address string
                data[key] = _read_ram_value(ram, addr_info)
            elif isinstance(addr_info, list): # List of address strings (e.g., for multi-byte score)
                data[key] = [_read_ram_value(ram, addr) for addr in addr_info]
            elif isinstance(addr_info, dict): # Nested dictionary of addresses (e.g., for x_position)
                data[key] = {sub_key: _read_ram_value(ram, sub_addr) for sub_key, sub_addr in addr_info.items()}
            else:
                print(f"[RAM Read] Warning: Unknown address format for key '{key}': {addr_info}")
                data[key] = None
        except IndexError:
            print(f"[RAM Read] Error: RAM address out of bounds for key '{key}' with address info '{addr_info}'. RAM size: {len(ram)}.")
            data[key] = 0 # Default to 0 or None if address is bad
        except ValueError:
            print(f"[RAM Read] Error: Invalid hex address string for key '{key}' with address info '{addr_info}'.")
            data[key] = 0
        except Exception as e:
            print(f"[RAM Read] Unexpected error reading RAM for key '{key}': {e}")
            data[key] = 0
    return data

def _convert_bcd_bytes_to_int_string(bcd_byte_values: Union[List[int], bytes]) -> str:
    """Converts a list/bytes of BCD byte values into a string of decimal digits."""
    if not bcd_byte_values:
        return "0"
    # Ensure we are working with integer values if input is bytes
    int_values = list(bcd_byte_values) if isinstance(bcd_byte_values, bytes) else bcd_byte_values
    
    hex_str = "".join([f'{val:02x}' for val in int_values])
    # Check if the hex_str is composed of valid BCD characters (0-9)
    # BCD means each nibble is a decimal digit. hex_str directly represents this.
    # Example: [0x12, 0x34] -> "1234"
    # No further conversion needed beyond formatting as hex, then it's a decimal string.
    return hex_str if hex_str else "0"


class SuperMarioBrosEnvWrapper:
    """
    A wrapper for the Super Mario Bros retro environment.
    This class handles direct retro environment interaction and uses GymEnvAdapter
    for logging, action mapping, and observation object creation from an image path.
    """

    def __init__(
        self,
        game_name: str, # e.g., "super_mario_bros"
        config_dir_path: str,  # Path to "gamingagent/envs/retro_01_super_mario_bros/"
        observation_mode: str,  # Should typically be "vision"
        base_log_dir: str,  # Base directory for all logs, e.g., "cache"
    ):
        self.game_name = game_name
        self.config_dir_path = config_dir_path
        self.game_specific_config_json_path = os.path.join(config_dir_path, "game_env_config.json")
        self.observation_mode = observation_mode # Used by GymEnvAdapter for hashing observation
        self.base_log_dir = base_log_dir # This is the directory GymEnvAdapter will use directly.

        self._load_wrapper_config() # Loads env_id, ram_addresses, etc.
        self._initialize_env()

        # GymEnvAdapter will use the provided base_log_dir directly as its agent_cache_dir
        self.adapter = GymEnvAdapter(
            game_name=self.game_name,
            observation_mode=self.observation_mode, 
            agent_cache_dir=self.base_log_dir # Pass base_log_dir directly
        )
        # print(f"[SuperMarioBrosEnvWrapper DEBUG __init__] Adapter's move_to_action_idx: {self.adapter.move_to_action_idx}") # This will be empty now
        self.current_game_info: Dict[str, Any] = {}
        self.current_episode_total_perf_score: float = 0.0
        self.current_episode_max_x_pos: int = 0
        self.current_meta_episode_accumulated_reward: float = 0.0
        
        # Removed explicit life counting attributes

    def _load_wrapper_config(self):
        """Loads configurations needed by this wrapper directly, like env_id and RAM addresses."""
        print(f"[SuperMarioBrosEnvWrapper] Loading wrapper config from: {self.game_specific_config_json_path}")
        try:
            with open(self.game_specific_config_json_path, 'r') as f:
                config = json.load(f)
            self.env_id = config.get("env_id", "SuperMarioBros-Nes")
            self.env_init_kwargs = config.get("env_init_kwargs", {})
            # Removed loading of initial_lives_per_meta_episode
            
            # Load action mapping directly here
            action_mapping_from_config = config.get("action_mapping", {})
            self.mario_action_mapping: Dict[str, List[int]] = {str(k).lower(): v for k, v in action_mapping_from_config.items()}
            print(f"[SuperMarioBrosEnvWrapper DEBUG _load_wrapper_config] Loaded mario_action_mapping: {self.mario_action_mapping}")

            # Extract custom_game_specific_config for RAM addresses and other retro specifics
            custom_config = config.get("custom_game_specific_config", {})
            self.ram_addresses = {
                "lives": custom_config.get("lives_ram_address"),
                "score": custom_config.get("score_ram_address"),
                "player_state": custom_config.get("player_state_ram_address"),
                "x_position": custom_config.get("x_position_ram_addresses"),
                "world": custom_config.get("world_ram_address"),
                "level": custom_config.get("level_ram_address"),
            }
            self.render_mode_human = config.get("render_mode_human", False)
            self.retro_obs_type_str = custom_config.get("observation_type", "IMAGE") # e.g. "IMAGE" or "RAM"

        except FileNotFoundError:
            print(f"[SuperMarioBrosEnvWrapper] ERROR: Config file not found at {self.game_specific_config_json_path}")
            raise
        except Exception as e:
            print(f"[SuperMarioBrosEnvWrapper] ERROR: Failed to load or parse config: {e}")
            raise

    def _initialize_env(self):
        """Initializes the retro environment."""
        obs_type_enum = retro.Observations.IMAGE # Default
        if self.retro_obs_type_str.upper() == "RAM":
            obs_type_enum = retro.Observations.RAM
        elif self.retro_obs_type_str.upper() == "RGB": # another common one
             obs_type_enum = retro.Observations.RGB_ARRAY 

        effective_env_init_kwargs = self.env_init_kwargs.copy()
        # Ensure obs_type from custom_game_specific_config is not passed if already handled
        if 'obs_type' in effective_env_init_kwargs: 
            print(f"[SuperMarioBrosEnvWrapper] Info: 'obs_type' found in env_init_kwargs from JSON ({effective_env_init_kwargs['obs_type']}), will be overridden by custom_game_specific_config.observation_type ({self.retro_obs_type_str}).")
            del effective_env_init_kwargs['obs_type']
        
        render_mode_arg = "human" if self.render_mode_human else None 
        
        # Define path for .bk2 recordings
        # self.adapter.agent_cache_dir should be set up by now if self.adapter is initialized before _initialize_env
        # Let's ensure self.adapter.agent_cache_dir is accessible or use self.base_log_dir directly
        # For now, using self.base_log_dir which is the main cache dir for this agent instance.
        record_path_base = self.base_log_dir # This is "cache/super_mario_bros/model_name/timestamp/"
        record_path_bk2 = os.path.join(record_path_base, "bk2_recordings")
        os.makedirs(record_path_bk2, exist_ok=True)
        print(f"[SuperMarioBrosEnvWrapper] Saving .bk2 recordings to: {record_path_bk2}")

        try:
            print(f"[SuperMarioBrosEnvWrapper] Initializing Retro env: id='{self.env_id}', obs_type='{obs_type_enum}', render_mode='{render_mode_arg}', record_path='{record_path_bk2}', kwargs={effective_env_init_kwargs}")
            self.env = retro.make(
                self.env_id, 
                obs_type=obs_type_enum, 
                render_mode=render_mode_arg, 
                record=record_path_bk2, # Added record argument
                **effective_env_init_kwargs
            )
            print(f"[SuperMarioBrosEnvWrapper] Underlying Retro buttons: {self.env.buttons}")
        except Exception as e:
            print(f"[SuperMarioBrosEnvWrapper] ERROR creating retro environment: {e}")
            raise

    def _save_frame_get_path(self, frame: np.ndarray, episode_id: int, step_num: int) -> Optional[str]:
        """Saves a raw frame (numpy array) to a PNG file and returns its path."""
        if frame is None or not isinstance(frame, np.ndarray):
            print("[SuperMarioBrosEnvWrapper] Warning: Attempted to save None or invalid frame.")
            return None
        try:
            # GymEnvAdapter provides the base path for observations
            img_path = self.adapter._create_agent_observation_path(episode_id, step_num) # Use adapter's path generation
            img_dir = os.path.dirname(img_path)
            if not os.path.exists(img_dir):
                os.makedirs(img_dir, exist_ok=True)
            
            img = Image.fromarray(frame)
            img.save(img_path)
            return img_path
        except Exception as e:
            print(f"[SuperMarioBrosEnvWrapper] ERROR: Failed to save frame to {img_path if 'img_path' in locals() else 'unknown path'}: {e}")
            return None

    def _extract_game_specific_info(self, ram: Optional[bytes]) -> Dict[str, Any]:
        """Extracts game-specific information from RAM using configured addresses."""
        if ram is None:
            print("[SuperMarioBrosEnvWrapper _extract_game_specific_info] RAM is None, returning default info indicating game over.")
            return {"score": 0, "lives": 0, "x_pos": 0, "world": 0, "level": 0, "player_state": 0, "is_game_over": True, "ram_lives_value": 255}

        info = {}
        # Use the local helper function _read_ram_fields
        raw_values = _read_ram_fields(ram, self.ram_addresses)
        
        score_bcd_bytes = raw_values.get("score")
        info["score"] = int(_convert_bcd_bytes_to_int_string(score_bcd_bytes)) if score_bcd_bytes is not None else 0
        
        # Lives from RAM (0x075A for SMB) is Player Lives - 1. 
        # e.g., 2 = 3 lives, 1 = 2 lives, 0 = 1 life. 0xFF (255) = Game Over.
        ram_lives_value = raw_values.get("lives", 255) # Default to 255 (Game Over) if not found
        info["ram_lives_value"] = ram_lives_value # Store the raw RAM value for inspection
        
        info["player_state"] = raw_values.get("player_state", 0)
        info["world"] = raw_values.get("world", 0)
        info["level"] = raw_values.get("level", 0)
        
        x_pos_data = raw_values.get("x_position")
        if isinstance(x_pos_data, dict):
             page_val = x_pos_data.get("page", 0)
             fine_val = x_pos_data.get("fine", 0)
             info["x_pos"] = page_val * 256 + fine_val
        elif isinstance(x_pos_data, (int, float)):
             info["x_pos"] = int(x_pos_data)
        else: 
            info["x_pos"] = 0
        
        # Determine if it's game over based on lives RAM value or specific player states
        is_game_over = (ram_lives_value == 255) # 0xFF typically means Game Over screen or no lives left.
        
        # Player state 0x06 is 'Dead', 0x0B is 'Dying'.
        # If player is in a 'Dead' or 'Dying' state AND the lives count (ram_lives_value) is 0 (meaning 1 life was left before this death),
        # it's effectively a game over situation for this attempt.
        # However, the game might transition through these states and then to a Game Over screen where ram_lives_value becomes 255.
        # Relying on ram_lives_value == 255 is often the most robust for "final game over".
        # Some games might also use player_state 0x0C for "Palette cycling, can't move" which might follow death.
        if info["player_state"] in [0x06, 0x0B] and ram_lives_value == 0: # Dying or Dead on last life (RAM 0)
             print(f"[SuperMarioBrosEnvWrapper _extract_game_specific_info] Mario state {info['player_state']} on last life (RAM lives 0). Considering game over.")
             is_game_over = True # This life is lost, and it was the last one.

        info["is_game_over"] = is_game_over
        return info

    def _build_textual_representation_for_log(self, game_info: Dict[str, Any]) -> Optional[str]:
        parts = []
        if "score" in game_info: parts.append(f"Score: {game_info['score']}")
        
        if "ram_lives_value" in game_info:
            if game_info["ram_lives_value"] == 255:
                parts.append(f"Lives (RAM): 0 (Game Over Screen)")
            else:
                parts.append(f"Lives (RAM): {game_info['ram_lives_value'] + 1}")

        if "world" in game_info and "level" in game_info: parts.append(f"World: {game_info.get('world', 'N/A')}-{game_info.get('level', 'N/A')}")
        if "x_pos" in game_info: parts.append(f"XPos: {game_info['x_pos']}")
        if game_info.get("is_game_over"): parts.append("RAM SAYS GAME OVER")
        return ", ".join(parts) if parts else None

    def reset(self, episode_id: int, **kwargs) -> Tuple[Observation, Dict[str, Any]]:
        self.adapter.reset_episode(episode_id) # GymEnvAdapter handles log file setup
        
        print(f"[SuperMarioBrosEnvWrapper reset] Starting meta-episode {episode_id}. Calling self.env.reset() ONCE.")
        # Removed initialization of explicit life counters
        
        observation_data, _ = self.env.reset(**kwargs) 
        raw_frame = observation_data # The actual image data
        
        initial_ram = self.env.get_ram()
        self.current_game_info = self._extract_game_specific_info(initial_ram)
        
        print(f"[SuperMarioBrosEnvWrapper reset] Initial game info: {self._build_textual_representation_for_log(self.current_game_info)}")

        img_path = None
        if self.observation_mode in ["vision", "both"]:
            img_path = self._save_frame_get_path(raw_frame, self.adapter.current_episode_id, self.adapter.current_step_num)

        agent_observation = self.adapter.create_agent_observation(
            img_path=img_path,
            text_representation="" 
        )
        
        self.current_episode_max_x_pos = self.current_game_info.get('x_pos', 0)
        self.current_episode_total_perf_score = 0.0
        self.current_meta_episode_accumulated_reward = 0.0

        info_to_return = self.current_game_info.copy()
        info_to_return['current_episode_max_x_pos'] = self.current_episode_max_x_pos
        # Removed current_lives_remaining_in_meta_episode from info
        
        return agent_observation, info_to_return

    def calculate_perf_score(self, current_x_pos_frame: int) -> float:
        step_perf_score = 0.0
        if current_x_pos_frame > self.current_episode_max_x_pos:
            step_perf_score = float(current_x_pos_frame - self.current_episode_max_x_pos)
        return step_perf_score

    def step(self, agent_action_str: Optional[str], thought_process: str = "", time_taken_s: float = 0.0) -> Tuple[Observation, float, bool, bool, Dict[str, Any], float]:
        import re
        base_action_name = "noop" 
        frame_count = 1       

        if agent_action_str:
            match = re.match(r"\(?'?(\w+(?:_\w+)*)'?,\s*(\d+)\)?", agent_action_str)
            if match:
                base_action_name = match.group(1)
                try:
                    parsed_frame_count = int(match.group(2))
                    if parsed_frame_count > 0: frame_count = parsed_frame_count
                except ValueError: 
                    # print(f"[SuperMarioBrosEnvWrapper] Warning: Could not parse frame_count for {base_action_name}. Defaulting to 1.")
                    pass 
            else: # Assume it's just the action name if no frame count
                base_action_name = agent_action_str.strip("()\\\' ")
                # print(f"[SuperMarioBrosEnvWrapper] Action string '{agent_action_str}' -> simple action '{base_action_name}', 1 frame.")
        
        env_action_buttons = self.mario_action_mapping.get(base_action_name.lower())
        if env_action_buttons is None:
            # print(f"[SuperMarioBrosEnvWrapper] Warning: Action '{base_action_name}' not found. Using NOOP.")
            base_action_name = "noop"
            env_action_buttons = self.mario_action_mapping.get("noop", [0]*len(self.env.buttons if hasattr(self, 'env') and self.env else 9))

        accumulated_reward_for_action_sequence = 0.0 
        accumulated_perf_score_for_action_sequence = 0.0
        
        meta_episode_terminated_this_step = False 
        meta_episode_truncated_this_step = False
        
        last_agent_observation_in_loop = None
        current_game_info_frame = self.current_game_info 

        for frame_num in range(frame_count):
            self.adapter.increment_step() 

            observation_data_frame, reward_frame, done_from_retro_frame, trunc_from_retro_frame, info_retro_frame = self.env.step(env_action_buttons)
            current_ram_frame = self.env.get_ram()
            
            current_game_info_frame = self._extract_game_specific_info(current_ram_frame) 
            current_game_info_frame.update(info_retro_frame) 

            # Removed explicit life decrementing logic and just_lost_life_flag

            current_x_pos_frame = current_game_info_frame.get('x_pos', self.current_episode_max_x_pos)
            current_step_perf_score_frame = self.calculate_perf_score(current_x_pos_frame)
            self.current_episode_max_x_pos = max(self.current_episode_max_x_pos, current_x_pos_frame)

            accumulated_reward_for_action_sequence += float(reward_frame)
            self.current_meta_episode_accumulated_reward += float(reward_frame) 
            accumulated_perf_score_for_action_sequence += current_step_perf_score_frame

            img_path_frame = None
            if self.observation_mode in ["vision", "both"]:
                img_path_frame = self._save_frame_get_path(observation_data_frame, self.adapter.current_episode_id, self.adapter.current_step_num)

            agent_observation_frame = self.adapter.create_agent_observation(
                img_path=img_path_frame, text_representation="" 
            )
            last_agent_observation_in_loop = agent_observation_frame
            
            # Stuck detection is not used for meta-episode termination anymore
            terminated_by_stuck_frame, _ = self.adapter.verify_termination(
                agent_observation=agent_observation_frame,
                current_terminated=done_from_retro_frame, 
                current_truncated=trunc_from_retro_frame 
            )

            self.adapter.log_step_data(
                agent_action_str=base_action_name, 
                thought_process=thought_process,  
                reward=float(reward_frame),
                info=current_game_info_frame, 
                terminated=(done_from_retro_frame or terminated_by_stuck_frame), 
                truncated=trunc_from_retro_frame,
                time_taken_s=time_taken_s if frame_num == 0 else 0.0,
                perf_score=current_step_perf_score_frame,
                agent_observation=agent_observation_frame
            )
            
            if current_game_info_frame.get("is_game_over", False):
                print(f"[SuperMarioBrosEnvWrapper step] Meta-episode terminated: RAM indicates Game Over. Info: {self._build_textual_representation_for_log(current_game_info_frame)}")
                meta_episode_terminated_this_step = True
            
            if trunc_from_retro_frame:
                print(f"[SuperMarioBrosEnvWrapper step] Meta-episode truncated by retro env.")
                meta_episode_truncated_this_step = True 

            if meta_episode_terminated_this_step or meta_episode_truncated_this_step:
                break 
        
        self.current_game_info = current_game_info_frame 
        # Removed current_lives_remaining_in_meta_episode from self.current_game_info update

        if last_agent_observation_in_loop is None:
            # print(\"[SuperMarioBrosEnvWrapper] Warning: Loop for frames did not produce an observation. Getting current state.\")
            current_ram_now = self.env.get_ram()
            self.current_game_info = self._extract_game_specific_info(current_ram_now)
            obs_data_now, _, _, _, _ = self.env.step(self.mario_action_mapping.get("noop", [0]*len(self.env.buttons if hasattr(self, 'env') and self.env else 9)))
            img_path_now = None
            if self.observation_mode in ["vision", "both"] and obs_data_now is not None:
                img_path_now = self._save_frame_get_path(obs_data_now, self.adapter.current_episode_id, self.adapter.current_step_num)
            last_agent_observation_in_loop = self.adapter.create_agent_observation(img_path=img_path_now, text_representation="")
            if obs_data_now is not None: 
                self.current_game_info = self._extract_game_specific_info(self.env.get_ram())
            # Removed current_lives_remaining_in_meta_episode from info in this block

        return (
            last_agent_observation_in_loop, 
            self.current_meta_episode_accumulated_reward, 
            meta_episode_terminated_this_step, 
            meta_episode_truncated_this_step, 
            self.current_game_info.copy(), 
            accumulated_perf_score_for_action_sequence 
        )

    def render(self) -> None:
        if self.render_mode_human:
            self.env.render()
        # else: GymEnvAdapter does not have a generic render method for non-human modes

    def close(self) -> None:
        print("[SuperMarioBrosEnvWrapper] Closing environment.")
        if hasattr(self, 'env') and self.env:
            self.env.close()
        if hasattr(self, 'adapter') and self.adapter:
            self.adapter.close_log_file() # GymEnvAdapter has this
