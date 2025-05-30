import os
import json
from typing import Any, Dict, Tuple, Optional, List, Union

import retro
import numpy as np
from PIL import Image # For saving frames

from gamingagent.envs.gym_env_adapter import GymEnvAdapter # Changed from RetroEnvAdapter
from gamingagent.modules.core_module import Observation

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
        self.current_episode_max_x_pos: int = 0
        #self.current_episode_total_perf_score: float = 0.0
        #self.accumulated_reward_for_action_sequence: float = 0.0
        #self.current_meta_episode_accumulated_reward: float = 0.0
        #self.accumulated_perf_score_for_action_sequence = 0.0
        
        # Removed explicit life counting attributes

    def _load_wrapper_config(self):
        """Loads configurations needed by this wrapper directly, like env_id and RAM addresses."""
        print(f"[SuperMarioBrosEnvWrapper] Loading wrapper config from: {self.game_specific_config_json_path}")
        try:
            with open(self.game_specific_config_json_path, 'r') as f:
                config = json.load(f)
            self.env_id = config.get("env_id", "SuperMarioBros-Nes")
            self.env_init_kwargs = config.get("env_init_kwargs", {})
            
            # Load action mapping directly here
            action_mapping_from_config = config.get("action_mapping", {})
            self.mario_action_mapping: Dict[str, List[int]] = {str(k).lower(): v for k, v in action_mapping_from_config.items()}
            print(f"[SuperMarioBrosEnvWrapper DEBUG _load_wrapper_config] Loaded mario_action_mapping: {self.mario_action_mapping}")

            # Extract custom_game_specific_config
            custom_config = config.get("custom_game_specific_config", {})
            self.retro_obs_type_str = custom_config.get("observation_type", "IMAGE") # e.g. "IMAGE" or "RAM"
            
            self.render_mode_human = config.get("render_mode_human", False)
            if self.render_mode_human and not os.environ.get("DISPLAY"):
                print("[SMB] DISPLAY is not set – switching to head‑less mode.")
                self.render_mode_human = False

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

        render_mode_arg = "human" if self.render_mode_human else "rgb_array"
        
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

    def _extract_game_specific_info(
        self,
        retro_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Return a dict with the fields we care about, preferring the values that the
        retro core already exposes through *info*.  We only touch RAM as a backup
        (for older cores that may lack a particular key).
        """
        info: Dict[str, Any] = {}

        # core values
        # These keys exist in the standard SMB core shipped with gym‑retro
        info["total_score"]        = retro_info.get("score", 0)
        info["lives"]        = retro_info.get("lives", 0)
        info["x_pos"]        = retro_info.get("xscrollLo", 0) + 255 * retro_info.get("xscrollHi", 0)
        info["world"]        = retro_info.get("levelHi", 0)
        info["level"]        = retro_info.get("levelLo", 0)

        # derive “game‑over” flag
        # In SMB the ‘lives’ counter shows 0 on the Game‑Over screen.
        info["is_game_over"] = (info["lives"] == 0)

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
        
        # after self.env.reset(...)
        observation_data, retro_info = self.env.reset(**kwargs)
        raw_frame = observation_data

        self.current_game_info = self._extract_game_specific_info(retro_info)
        
        print(f"[SuperMarioBrosEnvWrapper reset] Initial game info: {self._build_textual_representation_for_log(self.current_game_info)}")

        img_path = None
        if self.observation_mode in ["vision", "both"]:
            img_path = self._save_frame_get_path(raw_frame, self.adapter.current_episode_id, self.adapter.current_step_num)

        agent_observation = self.adapter.create_agent_observation(
            img_path=img_path,
            text_representation="" 
        )
        
        self.current_episode_max_x_pos = self.current_game_info.get('x_pos', 0)
        #self.current_episode_total_perf_score = 0.0

        info_to_return = self.current_game_info.copy()
        info_to_return['current_episode_max_x_pos'] = self.current_episode_max_x_pos
        # Removed current_lives_remaining_in_meta_episode from info
        
        return agent_observation, info_to_return

    def calculate_perf_score(self, current_x_pos_frame: int) -> float:
        step_perf_score = 0.0
        if current_x_pos_frame > self.current_episode_max_x_pos:
            step_perf_score = float(current_x_pos_frame - self.current_episode_max_x_pos)
        return step_perf_score

    def step(
        self,
        agent_action_str: Optional[str],
        thought_process: str = "",
        time_taken_s: float = 0.0,
    ) -> Tuple[Observation, float, bool, bool, Dict[str, Any], float]:
        """
        Execute `frame_count` physics steps for the given action, where the
        action string may be either:

            "move_left"               → 1 frame
            "(move_left,3)"           → 3 frames
            "hard_drop,1"             → 1 frame   (commas / spaces allowed)
        """

        import re  # local import to mirror Mario code

        # ---------------------------------------------------- #
        # 1)  Parse the incoming action string                 #
        # ---------------------------------------------------- #
        base_action_name = "noop"
        frame_count = 1

        if agent_action_str:
            # pattern matches:  action_name [,| )  integer]   with or w/out ()
            m = re.match(r"\(?'?(\w+(?:_\w+)*)'?[),\s]*([0-9]+)?\)?", agent_action_str)
            if m:
                base_action_name = m.group(1)
                if m.group(2):
                    try:
                        fc = int(m.group(2))
                        if fc > 0:
                            frame_count = fc
                    except ValueError:
                        pass
            else:
                base_action_name = agent_action_str.strip("()'\" ").lower()

        # Resolve to internal action index once (like Mario)
        env_action_idx = self.adapter.map_agent_action_to_env_action(base_action_name)
        if env_action_idx is None:
            base_action_name = "noop"
            env_action_idx = self.ACTION_NO_OP

        # ---------------------------------------------------- #
        # 2)  Run the original single‑frame body frame_count   #
        # ---------------------------------------------------- #
        total_reward = 0.0
        total_perf_score = 0.0
        terminated_final = False
        truncated_final = False
        last_agent_observation: Optional[Observation] = None

        for frame_num in range(frame_count):
            self.adapter.increment_step()

            # ---------------- SINGLE‑FRAME LOGIC ---------------
            current_step_reward = 0.0

            if not self.game_over:
                # --- handle player input ---
                if env_action_idx == self.ACTION_MOVE_LEFT:
                    if self.active_tetromino and not self._collision(
                        self.active_tetromino, self.x - 1, self.y
                    ):
                        self.x -= 1

                elif env_action_idx == self.ACTION_MOVE_RIGHT:
                    if self.active_tetromino and not self._collision(
                        self.active_tetromino, self.x + 1, self.y
                    ):
                        self.x += 1

                elif env_action_idx == self.ACTION_ROTATE_CLOCKWISE:
                    if self.active_tetromino:
                        rotated = self._rotate_tetromino(self.active_tetromino, True)
                        if not self._collision(rotated, self.x, self.y):
                            self.active_tetromino = rotated

                elif env_action_idx == self.ACTION_ROTATE_COUNTERCLOCKWISE:
                    if self.active_tetromino:
                        rotated = self._rotate_tetromino(self.active_tetromino, False)
                        if not self._collision(rotated, self.x, self.y):
                            self.active_tetromino = rotated

                elif env_action_idx == self.ACTION_SOFT_DROP:
                    if self.active_tetromino and not self._collision(
                        self.active_tetromino, self.x, self.y + 1
                    ):
                        self.y += 1
                    elif self.active_tetromino:
                        current_step_reward = self._commit_active_tetromino()

                elif env_action_idx == self.ACTION_HARD_DROP:
                    if self.active_tetromino:
                        while not self._collision(
                            self.active_tetromino, self.x, self.y + 1
                        ):
                            self.y += 1
                        current_step_reward = self._commit_active_tetromino()

                # --- gravity ---
                if (
                    self.gravity_enabled
                    and self.active_tetromino
                    and env_action_idx != self.ACTION_HARD_DROP
                    and not (
                        env_action_idx == self.ACTION_SOFT_DROP
                        and self.active_tetromino is None
                    )
                ):
                    if not self._collision(self.active_tetromino, self.x, self.y + 1):
                        self.y += 1
                    else:
                        if self.active_tetromino is not None:
                            current_step_reward = self._commit_active_tetromino()

            # ---------- bookkeeping & logging ----------
            terminated_flag = self.game_over
            obs_dict = self._get_obs()
            self.current_info_dict = self._get_info()

            current_step_perf_score = self.adapter.calculate_perf_score(
                current_step_reward, self.current_info_dict
            )
            self.total_perf_score_episode += current_step_perf_score

            img_path = txt_rep = None
            if self.adapter.observation_mode in ["vision", "both"]:
                img_path = self.adapter._create_agent_observation_path(
                    self.adapter.current_episode_id, self.adapter.current_step_num
                )
                create_board_image_tetris(
                    board=obs_dict["board"],
                    save_path=img_path,
                    pixel_color_mapping=self.pixel_id_to_color_map,
                    all_tetromino_objects=self.tetrominoes,
                    score=int(self.current_score),
                    lines=self.lines_cleared_total,
                    level=self.level,
                    next_pieces_ids=self.current_info_dict["next_piece_ids"],
                    perf_score=current_step_perf_score,
                )
            if self.adapter.observation_mode in ["text", "both"]:
                b_str = np.array2string(obs_dict["board"], separator=",")
                n_str = f"N:{self.current_info_dict['next_piece_ids']}"
                i_str = f"S:{self.current_score} L:{self.lines_cleared_total} V:{self.level}"
                txt_rep = f"{b_str}\n{n_str} {i_str}"

            agent_obs = self.adapter.create_agent_observation(img_path, txt_rep)

            final_term, final_trunc = self.adapter.verify_termination(
                agent_obs, terminated_flag, False
            )
            self.adapter.log_step_data(
                agent_action_str=base_action_name,
                thought_process=thought_process,
                reward=current_step_reward,
                info=self.current_info_dict,
                terminated=final_term,
                truncated=final_trunc,
                time_taken_s=time_taken_s if frame_num == 0 else 0.0,
                perf_score=current_step_perf_score,
                agent_observation=agent_obs,
            )
            if self.render_mode == "human":
                self.render()
            # -----------------------------------------------------

            total_reward += current_step_reward
            total_perf_score += current_step_perf_score
            last_agent_observation = agent_obs
            terminated_final, truncated_final = final_term, final_trunc

            if terminated_final or truncated_final:
                break  # stop executing further frames

        # ensure info dict reflects final state
        self.current_info_dict = self._get_info()

        return (
            last_agent_observation,
            total_reward,
            terminated_final,
            truncated_final,
            self.current_info_dict,
            total_perf_score,
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
