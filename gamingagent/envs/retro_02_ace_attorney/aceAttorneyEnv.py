import retro
from retro.retro_env import RetroEnv
from retro.enums import Actions, Observations # type: ignore
import gymnasium as gym # type: ignore
from gymnasium.core import SupportsFloat, RenderFrame # type: ignore
import numpy as np # type: ignore
from PIL import Image # type: ignore
import time
import json
import os
import hashlib
import re # For keyword mapping
from typing import Optional, Dict, Any, Tuple, List

from gamingagent.modules.core_module import Observation
from gamingagent.envs.gym_env_adapter import GymEnvAdapter
# from gamingagent.envs.env_utils import create_board_image_ace_attorney # If visual representation needed beyond raw pixels

# --- Constants ---
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
DEFAULT_GAME_SCRIPT_PATH = os.path.join(ASSETS_DIR, "mapping.json")
DEFAULT_SKIP_CONVERSATIONS_PATH = os.path.join(ASSETS_DIR, "ace_attorney_1_skip_conversations.json")
LIVES_RAM_VARIABLE_NAME = "lives" # Example RAM variable, confirm actual name from .json

# Ensure PIL is imported for image operations if any are done directly here
# from PIL import Image

class AceAttorneyEnv(RetroEnv):
    """
    Ace Attorney environment integrated with GamingAgent framework.
    Inherits from retro.retro_env.RetroEnv and uses GymEnvAdapter for agent interaction.
    Only 'lives' is reliably extracted from RAM. Dialogue and other game state
    elements are intended to be perceived by the agent from visual input.
    Level progression based on dialogue end statements is currently non-functional
    due to lack of direct dialogue extraction by the environment.
    """
    metadata = {
        # render_modes from retro.Env are typically ['human', 'rgb_array']
        # render_fps from retro.Env is usually based on the game's native FPS
    }

    def __init__(
        self,
        # retro.Env parameters
        game: str = "AceAttorney-GbAdvance",
        state: Optional[str] = "The_First_Turnabout", # This will be our primary level/scene identifier
        scenario: Optional[str] = None,
        info: Optional[str] = None,
        use_restricted_actions: int = Actions.FILTERED, # Defaulting to FILTERED
        record: bool = False,
        players: int = 1,
        inttype: int = retro.data.Integrations.ALL, # For accessing RAM variables if needed
        obs_type: int = Observations.RAM, # Essential for LIVES_RAM_VARIABLE_NAME
        # Adapter related parameters
        adapter_game_name: str = "ace_attorney",
        adapter_observation_mode: str = "vision", # MODIFIED: Default to "vision"
        adapter_agent_cache_dir: str = "cache/ace_attorney/default_run",
        adapter_config_path: str = "gamingagent/envs/retro_02_ace_attorney/game_env_config.json",
        adapter_max_stuck_steps: Optional[int] = 50,
        # Custom parameters
        wrapper_render_mode: Optional[str] = "rgb_array", # "human" or "rgb_array"
        game_script_path: str = DEFAULT_GAME_SCRIPT_PATH,
        skip_conversations_path: str = DEFAULT_SKIP_CONVERSATIONS_PATH,
        initial_lives: int = 5 # Default initial lives
    ):
        """
        Initializes the AceAttorneyEnv.

        Args:
            game: Name of the game integration in gym-retro.
            state: Initial game state/level to load.
            scenario: Specific scenario file for the game.
            info: Path to the game's info file (.json with RAM variable addresses).
            use_restricted_actions: Defines the action space type (e.g., FILTERED, DISCRETE, MULTI_DISCRETE).
                                    retro.Actions.FILTERED: uses "valid_actions" from scenario.json.
                                    retro.Actions.DISCRETE: maps to a discrete set of button combinations.
                                    retro.Actions.MULTI_BINARY: one button per action.
            record: Whether to record gameplay.
            players: Number of players.
            inttype: Integration type for gym-retro.
            obs_type: Observation type (RAM, IMAGE, etc.). RAM is needed for lives.
            adapter_game_name: Name of the game for the adapter.
            adapter_observation_mode: Observation mode for the GymEnvAdapter ("vision", "text", "both").
            adapter_agent_cache_dir: Cache directory for agent logs and observations.
            adapter_config_path: Path to the game_env_config.json for action mappings, etc.
            adapter_max_stuck_steps: Max steps for stuck detection by adapter.
            wrapper_render_mode: Render mode for the environment wrapper ('human', 'rgb_array').
            game_script_path: Path to the game script JSON file (e.g., mapping.json).
            skip_conversations_path: Path to JSON file for skipping known dialogue sequences.
            initial_lives: The starting number of lives/penalties.
        """
        # --- RetroEnv Initialization ---
        # ADDED: Register custom integration path
        custom_integration_path = os.path.dirname(os.path.abspath(__file__))
        retro.data.Integrations.add_custom_path(custom_integration_path)
        # print(f"[AceAttorneyEnv __init__] Added custom integration path: {custom_integration_path}")
        # print(f"[AceAttorneyEnv __init__] Available games after adding custom path: {retro.data.list_games(inttype=inttype)}")
        # print(f"[AceAttorneyEnv __init__] Checking for game: {game}")
        # if game not in retro.data.list_games(inttype=inttype):
        # print(f"[AceAttorneyEnv __init__] WARNING: Game '{game}' still not found after attempting to add custom path. Ensure integration files exist at the expected location.")

        # Determine num_buttons for the action array
        # This should align with how the game integration is set up (e.g., what env.buttons reports)
        # For GBA, it's typically 10 or 12 (B, SELECT, START, UP, DOWN, LEFT, RIGHT, A, L, R, + 2 Nones for 12)
        # We'll use a placeholder and confirm/adjust after first run or by inspecting retro game data.
        self.num_buttons = 12 # Assuming 12 for GBA based on previous findings (A=8, B=0, L=10, R=11)
        self.NO_OP_ACTION_ARRAY = np.zeros(self.num_buttons, dtype=bool)

        # Initialize RetroEnv (superclass)
        # obs_type is set to RAM to allow reading 'lives'. Visuals are handled by get_screen().
        super().__init__(
            game=game,
            state=state, # The initial .state file name
            scenario=scenario,
            info=info, # Should point to a data.json for RAM variables like 'lives'
            use_restricted_actions=use_restricted_actions,
            record=record,
            players=players,
            inttype=inttype,
            obs_type=obs_type
        )
        print(f"[AceAttorneyEnv __init__] RetroEnv initialized. Action space: {self.action_space}, Buttons: {self.buttons}")

        # --- Game Specific Variables & Configs ---
        self.initial_retro_state_name: str = state if state is not None else "The_First_Turnabout" # Default if None
        self.current_retro_state_name: str = self.initial_retro_state_name
        self.retro_inttype = inttype # Store for state loading

        self.game_script_data: Dict[str, Any] = {}
        self._load_game_script_data(game_script_path) # Loads mapping.json content

        self.skip_conversation_data: Dict[str, Any] = {}
        self._load_skip_conversation_data(skip_conversations_path)
        
        self.initial_lives = initial_lives
        self.current_lives = self.initial_lives
        self.current_raw_frame: Optional[np.ndarray] = None # Will store raw pixels from self.screen
        self.current_core_info: Dict[str, Any] = {} # Info from core retro env step/reset

        self.last_llm_dialogue_info: Optional[Dict] = None # ADDED: To store last LLM extracted dialogue
        self.dialogue_keyword_map: Dict = {} # ADDED: Loaded from mapping.json

        # Initialize level-specific data (dialogue log, skip map, end statements)
        self._initialize_level_specific_data() # Depends on self.game_script_data and self.current_retro_state_name

        # --- GymEnvAdapter Initialization ---
        self.adapter = GymEnvAdapter(
            game_name=adapter_game_name,
            observation_mode=adapter_observation_mode, # This should be "vision"
            agent_cache_dir=adapter_agent_cache_dir,
            game_specific_config_path=adapter_config_path,
            max_steps_for_stuck=adapter_max_stuck_steps
        )
        self.wrapper_render_mode = wrapper_render_mode # "human" or "rgb_array"
        
        # Action processing parameters (can be tuned)
        # These were found to be important from previous iterations
        self.num_frames_to_hold_action = 1 # User set this to 1 (previously 5)
        self.num_frames_for_no_op_pause = 200 # User set this to 200

        # Screenshot skipping during no-op (Phase 2)
        # If true, only the first frame of Phase 2 gets a screenshot.
        self.skip_later_noop_screenshots = True # User set this via frame_num_phase2 >= 1 (equivalent to this bool)
        
        print(f"[AceAttorneyEnv __init__] Initialized with state: {self.initial_retro_state_name}, obs_mode for adapter: {adapter_observation_mode}")
        print(f"[AceAttorneyEnv __init__] Action hold: {self.num_frames_to_hold_action} frames, No-op pause: {self.num_frames_for_no_op_pause} frames.")

    def _load_env_specific_config(self, config_path: str):
        # This method might become very simple or be removed if all necessary
        # config (like button names for retro env) is handled by adapter or hardcoded if stable.
        # For now, let it pass, as adapter loads its own action_mapping directly.
        if os.path.exists(config_path):
            # The adapter loads the action_mapping. This env doesn't need to parse it further for itself.
            print(f"[AceAttorneyEnv] Env-specific config found at {config_path}. Adapter will handle its contents.")
        else:
            print(f"[AceAttorneyEnv] WARNING: Env specific config {config_path} not found.")

    def _load_game_script_data(self, script_path: str):
        """Loads game script data (dialogue, skips, scene details) from a JSON file."""
        if os.path.exists(script_path):
            try:
                with open(script_path, 'r') as f:
                    self.game_script_data = json.load(f)
                print(f"[AceAttorneyEnv] Successfully loaded game script data from: {script_path}")
                
                # ADDED: Load dialogue keyword map for the current state
                if self.game_script_data and self.current_retro_state_name in self.game_script_data:
                    current_state_data = self.game_script_data[self.current_retro_state_name]
                    if isinstance(current_state_data, dict): # Ensure it's a dictionary
                        self.dialogue_keyword_map = current_state_data.get("dialogue_keyword_map", {})
                        if self.dialogue_keyword_map:
                            print(f"[AceAttorneyEnv] Loaded dialogue_keyword_map for state '{self.current_retro_state_name}'.")
                        else:
                            print(f"[AceAttorneyEnv] No 'dialogue_keyword_map' found for state '{self.current_retro_state_name}' in {script_path}.")
                    else:
                        print(f"[AceAttorneyEnv] Data for state '{self.current_retro_state_name}' in {script_path} is not a dictionary.")
                elif not self.game_script_data:
                     print(f"[AceAttorneyEnv] Game script data loaded from {script_path} is empty.")
                else:
                    print(f"[AceAttorneyEnv] Current state '{self.current_retro_state_name}' not found in game script data keys: {list(self.game_script_data.keys())}")

            except json.JSONDecodeError as e:
                print(f"[AceAttorneyEnv] Error decoding JSON from game script {script_path}: {e}")
                self.game_script_data = {}
            except Exception as e:
                print(f"[AceAttorneyEnv] Error loading game script {script_path}: {e}")
                self.game_script_data = {}
        else:
            print(f"[AceAttorneyEnv] Warning: Game script file {script_path} not found.")
            self.game_script_data = {}

    def _load_skip_conversation_data(self, skip_path: str):
        if os.path.exists(skip_path):
            try:
                with open(skip_path, 'r', encoding='utf-8') as f:
                    self.skip_conversation_data = json.load(f)
                print(f"[AceAttorneyEnv] Loaded skip conversation data: {skip_path}")
            except Exception as e:
                print(f"[AceAttorneyEnv] ERROR loading skip conversation data {skip_path}: {e}")
        else:
            print(f"[AceAttorneyEnv] WARNING: Skip conversation data file not found: {skip_path}")

    def _initialize_level_specific_data(self):
        self.current_level_background = []
        self.current_level_initial_evidence = []
        self.current_level_name_map = {}
        self.current_level_dialog_map = {}
        self.current_level_evidence_map = {}
        self.current_level_all_scripted_evidence = []
        self.current_level_dialogue_log = []
        self.current_level_skip_map = {}
        self.current_level_end_statements = []

        if not self.current_retro_state_name or not self.game_script_data:
            print("[AceAttorneyEnv] Cannot initialize level data: retro_state_name or game_script_data missing.")
            return

        level_data = self.game_script_data.get(self.current_retro_state_name)
        if not level_data:
            print(f"[AceAttorneyEnv] WARNING: No data in mapping.json for level: {self.current_retro_state_name}")
            return

        self.current_level_background = level_data.get("background_transcript", [])
        self.current_level_initial_evidence = level_data.get("evidences", [])
        self.current_level_name_map = level_data.get("name_mappings", {})
        self.current_level_dialog_map = level_data.get("dialog", {})
        self.current_level_evidence_map = level_data.get("evidence_mappings", {})
        self.current_level_all_scripted_evidence = level_data.get("evidences", [])

        level_skip_data = self.skip_conversation_data.get(self.current_retro_state_name)
        if level_skip_data:
            self.current_level_skip_map = {k: v for k, v in level_skip_data.items() if k != "end_statement"}
            self.current_level_end_statements = level_skip_data.get("end_statement", [])
            print(f"[AceAttorneyEnv] Initialized skip/end data for level: {self.current_retro_state_name}")
        else:
            print(f"[AceAttorneyEnv] WARNING: No skip/end data for level: {self.current_retro_state_name}")

        print(f"[AceAttorneyEnv] Initialized data for level: {self.current_retro_state_name}")

    def _update_internal_game_state(self, core_info: Dict[str, Any]):
        """Updates internal game state like lives from core_info (RAM)."""
        if LIVES_RAM_VARIABLE_NAME and LIVES_RAM_VARIABLE_NAME in core_info:
            new_lives_value = int(core_info[LIVES_RAM_VARIABLE_NAME])
            if new_lives_value != self.current_lives:
                print(f"[AceAttorneyEnv DEBUG] Lives changed from {self.current_lives} to {new_lives_value}. RAM Variable: '{LIVES_RAM_VARIABLE_NAME}', Value in RAM: {core_info[LIVES_RAM_VARIABLE_NAME]}")
                self.current_lives = new_lives_value
            # If new_lives_value is the same as self.current_lives, no need to print or update.
        elif LIVES_RAM_VARIABLE_NAME:
            # This case means LIVES_RAM_VARIABLE_NAME is defined but not found in core_info.
            # Could print a one-time warning or handle as needed, for now, it's silent unless it changes.
            pass

    def _extract_dialogue_from_info(self, core_info: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        # Environment no longer extracts dialogue from RAM.
        # This is now the agent's responsibility (e.g., via PerceptionModule).
        # self.current_level_dialogue_log will not be populated by the env.
        return None, None # Return None for speaker and dialogue

    def _get_agent_info(self) -> Dict[str, Any]:
        score = self.current_core_info.get("score", 0)
        agent_info = {
            "score": float(score),
            "current_lives": self.current_lives,
            "current_retro_state": self.current_retro_state_name,
            "level_background_info": self.current_level_background[:3],
            "current_level_evidence": self.current_level_initial_evidence[:5],
            "raw_retro_info": {k: v for k, v in self.current_core_info.items() if isinstance(v, (int, float, str, bool))}
        }
        return agent_info

    def _trigger_skip_actions(self, num_skip_actions: int):
        # This logic relied on dialogue triggers from skip_conversation_data which the env no longer processes.
        if num_skip_actions <= 0: return
        return

    def _save_frame_to_path(self, frame_to_save: Optional[np.ndarray]) -> Optional[str]:
        """Saves the given frame to a uniquely named PNG file and returns its path."""
        if frame_to_save is None:
            # print(f"[AceAttorneyEnv _save_frame_to_path] frame_to_save is None. Cannot save image for E{self.adapter.current_episode_id} S{self.adapter.current_step_num}.")
            return None
        
        # Ensure frame_to_save is a numpy array
        if not isinstance(frame_to_save, np.ndarray):
            print(f"[AceAttorneyEnv _save_frame_to_path] Warning: frame_to_save is not a numpy array (type: {type(frame_to_save)}). Cannot save image for E{self.adapter.current_episode_id} S{self.adapter.current_step_num}.")
            return None

        img_path = self.adapter._create_agent_observation_path(
            self.adapter.current_episode_id, 
            self.adapter.current_step_num
        )
        try:
            save_dir = os.path.dirname(img_path)
            if save_dir: 
                os.makedirs(save_dir, exist_ok=True)

            pil_image = Image.fromarray(frame_to_save)
            pil_image.save(img_path)
            return img_path
        except Exception as e:
            print(f"[AceAttorneyEnv _save_frame_to_path] Error saving frame to {img_path}: {e}")
            return None

    def _build_agent_observation_components(self, agent_facing_info: Dict, skip_screenshot: bool = False) -> Tuple[Optional[str], Optional[str]]:
        """
        Builds the image and text components for the agent's observation.
        Image is saved if not skipped. Text component is now always None.
        """
        img_path_component: Optional[str] = None
        text_obs_component: Optional[str] = None # MODIFIED: Always None

        if self.adapter.observation_mode in ["vision", "both"]:
            if skip_screenshot:
                # print(f"[AceAttorneyEnv _build_agent_obs] E{self.adapter.current_episode_id} S{self.adapter.current_step_num}: Screenshot SKIPPED for this frame.")
                img_path_component = None # Explicitly None if skipped
            else:
                # Ensure self.current_raw_frame is up-to-date before saving
                # self.current_raw_frame should have been updated in the step method right after super().step()
                if self.current_raw_frame is not None:
                    img_path_component = self._save_frame_to_path(self.current_raw_frame)
                    # if img_path_component:
                    #     print(f"[AceAttorneyEnv _build_agent_obs] E{self.adapter.current_episode_id} S{self.adapter.current_step_num}: Screenshot SAVED to {img_path_component}")
                    # else:
                    #     print(f"[AceAttorneyEnv _build_agent_obs] E{self.adapter.current_episode_id} S{self.adapter.current_step_num}: Screenshot SAVE FAILED.")
                else:
                    # This case should ideally not be hit if current_raw_frame is managed correctly
                    # print(f"[AceAttorneyEnv _build_agent_obs] E{self.adapter.current_episode_id} S{self.adapter.current_step_num}: self.current_raw_frame is None. Cannot save screenshot.")
                    img_path_component = None
        
        # MODIFIED: Textual observation is explicitly not generated by the environment.
        # The agent (e.g. PerceptionModule or BaseModule with vision) is responsible for interpreting the scene.
        # if self.adapter.observation_mode in ["text", "both"]:
        # text_obs_component = f"Lives: {self.current_lives}. Dialogue and scene details to be extracted by agent from vision."
        # text_obs_component = None # Ensure it's None

        return img_path_component, text_obs_component

    def reset(self, *, seed: Optional[int]=None, options: Optional[Dict[str,Any]]=None, episode_id:int=1) -> Tuple[Observation, Dict[str,Any]]:
        if self.current_retro_state_name != self.initial_retro_state_name:
            print(f"[AceAttorneyEnv RESET] Current state '{self.current_retro_state_name}' differs from initial '{self.initial_retro_state_name}'. Loading initial state.")
            self.load_state(self.initial_retro_state_name, inttype=self.retro_inttype)
            self.current_retro_state_name = self.initial_retro_state_name 
        
        # super().reset() returns obs (RAM data if obs_type=RAM) and info
        ram_observation, self.current_core_info = super().reset(seed=seed, options=options)
        self.current_raw_frame = self.em.get_screen() # Explicitly get screen pixels via self.em

        self.current_retro_state_name = self.initial_retro_state_name # Re-affirm
        self.current_lives = self.initial_lives
        print(f"[AceAttorneyEnv DEBUG] RESET: Initial state set to '{self.current_retro_state_name}', Lives reset to {self.current_lives}")
        
        self._initialize_level_specific_data()
        self._update_internal_game_state(self.current_core_info)
        self._extract_dialogue_from_info(self.current_core_info)
        
        agent_facing_info = self._get_agent_info()
        self.adapter.reset_episode(episode_id)

        img_path, txt_rep = self._build_agent_observation_components(agent_facing_info) # skip_screenshot is False by default
        agent_obs = self.adapter.create_agent_observation(img_path=img_path, text_representation=txt_rep)
        
        initial_step_perf_score = self.adapter.calculate_perf_score(0.0, agent_facing_info)
        self.adapter.log_step_data(
            agent_action_str="<RESET>",
            thought_process="Episode reset.",
            reward=0.0,
            info=agent_facing_info,
            terminated=False,
            truncated=False,
            time_taken_s=0.0,
            perf_score=initial_step_perf_score,
            agent_observation=agent_obs
        )

        if self.wrapper_render_mode == "human":
            self.render()
        return agent_obs, agent_facing_info

    def step(self, agent_action_str:Optional[str], thought_process:str="",time_taken_s:float=0.0) -> Tuple[Observation,SupportsFloat,bool,bool,Dict[str,Any],float]:
        num_frames_to_hold_action = 1 # User's file had this as 1
        num_frames_for_no_op_pause = 200

        base_env_action_from_agent = self.adapter.map_agent_action_to_env_action(agent_action_str)

        if base_env_action_from_agent is None:
            agent_intended_action_for_phase1 = np.zeros(self.num_buttons, dtype=bool)
            effective_agent_action_str_for_log = agent_action_str if agent_action_str else "<NONE>"
            # print(f"[AceAttorneyEnv DEBUG] Phase 1: Agent action '{effective_agent_action_str_for_log}' mapped to None or invalid, using no-op for {num_frames_to_hold_action} frames.")
        elif not isinstance(base_env_action_from_agent, np.ndarray):
            # print(f"[AceAttorneyEnv ERROR] Phase 1: Adapter returned non-array action {base_env_action_from_agent} for agent_action_str '{agent_action_str}'. Using no-op for {num_frames_to_hold_action} frames.")
            agent_intended_action_for_phase1 = np.zeros(self.num_buttons, dtype=bool)
            effective_agent_action_str_for_log = agent_action_str
        else:
            agent_intended_action_for_phase1 = base_env_action_from_agent
            effective_agent_action_str_for_log = agent_action_str
        
        overall_accumulated_reward = 0.0
        overall_accumulated_perf_score = 0.0
        
        current_observation_to_return = None
        current_agent_facing_info_to_return = {}
        current_terminated_overall = False
        current_truncated_overall = False
        original_time_taken_s_for_agent_action = time_taken_s

        # --- Phase 1: Execute Agent's Chosen Action ---
        for frame_num_phase1 in range(num_frames_to_hold_action):
            self.adapter.increment_step()
            current_frame_time_taken = original_time_taken_s_for_agent_action if frame_num_phase1 == 0 else 0.0
            
            # super().step() returns obs (RAM data if obs_type=RAM), reward, terminated, truncated, info
            ram_obs_frame_p1, p1_step_reward, p1_terminated_frame, p1_truncated_frame, self.current_core_info = super().step(agent_intended_action_for_phase1)
            self.current_raw_frame = self.em.get_screen() # Explicitly get screen pixels for phase 1 via self.em

            overall_accumulated_reward += float(p1_step_reward)
            self._update_internal_game_state(self.current_core_info)
            p1_speaker, p1_dialogue = self._extract_dialogue_from_info(self.current_core_info)

            if self.current_lives <= 0: p1_terminated_frame = True
            if not p1_terminated_frame and p1_speaker is not None and p1_dialogue is not None: pass

            p1_agent_facing_info = self._get_agent_info()
            p1_step_perf_score = self.adapter.calculate_perf_score(float(p1_step_reward), p1_agent_facing_info)
            overall_accumulated_perf_score += p1_step_perf_score
            
            p1_img_path, p1_txt_rep = self._build_agent_observation_components(p1_agent_facing_info) # skip_screenshot is False
            p1_current_agent_obs = self.adapter.create_agent_observation(img_path=p1_img_path, text_representation=p1_txt_rep)
            
            p1_term_adapter, p1_trunc_adapter = self.adapter.verify_termination(p1_current_agent_obs, p1_terminated_frame, p1_truncated_frame)
            current_terminated_overall = p1_terminated_frame or p1_term_adapter
            current_truncated_overall = p1_truncated_frame or p1_trunc_adapter

            self.adapter.log_step_data(effective_agent_action_str_for_log, thought_process, float(p1_step_reward), p1_agent_facing_info.copy(), current_terminated_overall, current_truncated_overall, current_frame_time_taken, p1_step_perf_score, p1_current_agent_obs)
            current_observation_to_return = p1_current_agent_obs
            current_agent_facing_info_to_return = p1_agent_facing_info
            
            if self.wrapper_render_mode == "human": self.render()
            if current_terminated_overall or current_truncated_overall:
                # print(f"[AceAttorneyEnv DEBUG] Phase 1 (Frame {frame_num_phase1+1}): Terminating/Truncating. Skipping Phase 2.")
                return current_observation_to_return, overall_accumulated_reward, current_terminated_overall, current_truncated_overall, current_agent_facing_info_to_return, overall_accumulated_perf_score

        # --- Phase 2: Execute Automatic No-Op Pause ---
        # This phase runs only if Phase 1 did not terminate the episode.
        if num_frames_for_no_op_pause > 0:
            no_op_action = np.zeros(self.num_buttons, dtype=bool)
            accumulated_reward_phase2 = 0.0
            
            phase2_internal_terminated = False
            phase2_internal_truncated = False

            for frame_num_phase2 in range(num_frames_for_no_op_pause):
                self.adapter.increment_step() # Increment step for adapter's internal count

                ram_obs_frame_p2, p2_step_reward_frame, p2_terminated_frame_internal, p2_truncated_frame_internal, self.current_core_info = super().step(no_op_action)
                self.current_raw_frame = self.em.get_screen() # Keep updating current_raw_frame

                accumulated_reward_phase2 += float(p2_step_reward_frame)
                self._update_internal_game_state(self.current_core_info)
                # self._extract_dialogue_from_info(self.current_core_info) # Dialogue not used here currently

                if self.current_lives <= 0: p2_terminated_frame_internal = True
                
                phase2_internal_terminated = p2_terminated_frame_internal
                phase2_internal_truncated = p2_truncated_frame_internal

                if phase2_internal_terminated or phase2_internal_truncated:
                    # print(f"[AceAttorneyEnv DEBUG] Phase 2 (Internal Frame {frame_num_phase2+1}): Terminating/Truncating during no-op pause.")
                    break 
            
            # After the no-op loop (or early break), create one observation and log entry for the entire phase.
            overall_accumulated_reward += accumulated_reward_phase2

            p2_agent_facing_info = self._get_agent_info()
            p2_block_perf_score = self.adapter.calculate_perf_score(accumulated_reward_phase2, p2_agent_facing_info)
            overall_accumulated_perf_score += p2_block_perf_score
            
            # For the single observation at the end of phase 2, we will save a screenshot.
            # The previous `skip_screenshot_for_this_noop_frame` is not needed as we only build components once.
            p2_img_path, p2_txt_rep = self._build_agent_observation_components(p2_agent_facing_info, skip_screenshot=False)
            p2_final_agent_obs = self.adapter.create_agent_observation(img_path=p2_img_path, text_representation=p2_txt_rep)

            # Verify termination status *after* phase 2, considering stuck states on the final observation
            p2_term_adapter, p2_trunc_adapter = self.adapter.verify_termination(p2_final_agent_obs, phase2_internal_terminated, phase2_internal_truncated)
            
            current_terminated_overall = current_terminated_overall or phase2_internal_terminated or p2_term_adapter
            current_truncated_overall = current_truncated_overall or phase2_internal_truncated or p2_trunc_adapter
            
            self.adapter.log_step_data(
                agent_action_str="<AUTO_NO_OP_BLOCK>", 
                thought_process=f"Automatic no-op pause for {frame_num_phase2 + 1}/{num_frames_for_no_op_pause} frames.", # Log actual frames executed
                reward=accumulated_reward_phase2, 
                info=p2_agent_facing_info.copy(), 
                terminated=current_terminated_overall, 
                truncated=current_truncated_overall, 
                time_taken_s=0.0, # Time for this block action
                perf_score=p2_block_perf_score, 
                agent_observation=p2_final_agent_obs
            )
            current_observation_to_return = p2_final_agent_obs
            current_agent_facing_info_to_return = p2_agent_facing_info
        
        # If num_frames_for_no_op_pause was 0, the observation and info from Phase 1 are still the latest.
        # The overall_accumulated_reward and overall_accumulated_perf_score are also correct.
        # current_terminated_overall and current_truncated_overall would be from Phase 1.

        if self.wrapper_render_mode == "human": self.render()
        return current_observation_to_return, overall_accumulated_reward, current_terminated_overall, current_truncated_overall, current_agent_facing_info_to_return, overall_accumulated_perf_score

    def render(self) -> Optional[RenderFrame]:
        if self.wrapper_render_mode == 'human':
            return super().render()
        elif self.wrapper_render_mode == 'rgb_array' and self.current_raw_frame is not None:
            return self.current_raw_frame.copy()
        return None

    def store_llm_extracted_dialogue(self, dialogue_data: Dict[str, str]):
        """Stores dialogue extracted by the LLM into a JSON file in the agent_cache_dir."""
        print(f"[AceAttorneyEnv DEBUG store_llm_extracted_dialogue] Method called. Received dialogue_data: {dialogue_data}")

        if not self.current_retro_state_name:
            print("[AceAttorneyEnv store_llm_extracted_dialogue] ERROR: current_retro_state_name is not set. Cannot determine episode for storing dialogue.")
            return

        if not dialogue_data or not isinstance(dialogue_data, dict) or "speaker" not in dialogue_data or "text" not in dialogue_data:
            print(f"[AceAttorneyEnv store_llm_extracted_dialogue] ERROR: Invalid dialogue_data format: {dialogue_data}. Required keys: 'speaker', 'text'.")
            return

        # ADDED: Update last_llm_dialogue_info
        self.last_llm_dialogue_info = {
            "state_name": self.current_retro_state_name,
            "speaker": dialogue_data["speaker"],
            "text": dialogue_data["text"],
            "timestamp": time.time()
        }
        # print(f"[AceAttorneyEnv store_llm_extracted_dialogue] Updated self.last_llm_dialogue_info: {self.last_llm_dialogue_info}")


        # Create a unique hash for the dialogue content to use as part of the filename or key
        dialogue_str_for_hash = f"{dialogue_data['speaker']}:{dialogue_data['text']}"
        dialogue_hash = hashlib.md5(dialogue_str_for_hash.encode('utf-8')).hexdigest()[:8]

        # Define directory structure: cache_dir/dialogue_store/EpisodeName/LevelName/
        # Using current_retro_state_name as a proxy for LevelName or SceneGroup
        # This path is relative to the adapter's agent_cache_dir
        dialogue_storage_base_dir = os.path.join(self.adapter.agent_cache_dir, "llm_dialogue_store")
        # episode_name_for_path = f"episode_{self.adapter.current_episode_id:03d}" # Or a more meaningful episode name if available
        print(f"[AceAttorneyEnv DEBUG store_llm_extracted_dialogue] dialogue_storage_base_dir: {dialogue_storage_base_dir}")
        
        # For Ace Attorney, self.initial_retro_state_name often is the "Episode Name" like "The_First_Turnabout"
        # And self.current_retro_state_name can be a specific scene/save state within that episode.
        # Let's use self.initial_retro_state_name for the main episode folder, and current_retro_state_name for sub-context.
        
        # If self.current_retro_state_name changes often (e.g. after every save state), this might create many folders.
        # Consider if current_retro_state_name is stable enough for a "scene group" concept.
        # For simplicity, let's assume current_retro_state_name is a good grouping for now.

        # Simplified path: cache_dir/llm_dialogue_store/STATENAME/HASH.json
        state_specific_dialogue_dir = os.path.join(dialogue_storage_base_dir, self.current_retro_state_name)
        os.makedirs(state_specific_dialogue_dir, exist_ok=True)
        print(f"[AceAttorneyEnv DEBUG store_llm_extracted_dialogue] Ensured state_specific_dialogue_dir exists: {state_specific_dialogue_dir}")
        
        # Filename includes a hash of the dialogue to distinguish different dialogues within the same state/scene.
        # And step number to ensure uniqueness if dialogue is identical but occurs at different times.
        dialogue_file_name = f"dialogue_S{self.adapter.current_step_num:04d}_{dialogue_hash}.json"
        dialogue_file_path = os.path.join(state_specific_dialogue_dir, dialogue_file_name)
        print(f"[AceAttorneyEnv DEBUG store_llm_extracted_dialogue] Final dialogue_file_path: {dialogue_file_path}")

        try:
            with open(dialogue_file_path, 'w') as f:
                json.dump({
                    "episode_id": self.adapter.current_episode_id,
                    "step_num": self.adapter.current_step_num,
                    "retro_state_name": self.current_retro_state_name,
                    "dialogue_hash": dialogue_hash,
                    "speaker": dialogue_data["speaker"],
                    "text": dialogue_data["text"],
                    "timestamp_saved": time.time()
                }, f, indent=2)
            # print(f"[AceAttorneyEnv store_llm_extracted_dialogue] LLM-extracted dialogue saved to: {dialogue_file_path}")
            print(f"[AceAttorneyEnv DEBUG store_llm_extracted_dialogue] Successfully saved dialogue to: {dialogue_file_path}")
        except Exception as e:
            print(f"[AceAttorneyEnv store_llm_extracted_dialogue] CRITICAL ERROR: Failed to save LLM dialogue to {dialogue_file_path}. Details: {e}")

    def get_mapped_dialogue_event_for_prompt(self) -> Optional[str]:
        """
        Retrieves the last stored LLM dialogue, maps its text to a keyword using
        self.dialogue_keyword_map, and returns the keyword.
        """
        if not self.last_llm_dialogue_info or "text" not in self.last_llm_dialogue_info:
            # print("[AceAttorneyEnv get_mapped_dialogue_event] No last LLM dialogue info available.")
            return None

        dialogue_text = self.last_llm_dialogue_info["text"]
        
        if not self.dialogue_keyword_map:
            # print("[AceAttorneyEnv get_mapped_dialogue_event] Dialogue keyword map is empty.")
            return None # Or a default like "DIALOGUE_OCCURRED_NO_MAP"

        map_patterns = self.dialogue_keyword_map.get("patterns", [])
        default_keyword = self.dialogue_keyword_map.get("default_keyword", "UNMAPPED_DIALOGUE")

        for item in map_patterns:
            regex = item.get("regex")
            keyword = item.get("keyword")
            if regex and keyword:
                try:
                    if re.search(regex, dialogue_text, re.IGNORECASE):
                        # print(f"[AceAttorneyEnv get_mapped_dialogue_event] Matched regex '{regex}' to keyword '{keyword}' for text: '{dialogue_text[:50]}...'")
                        return keyword
                except re.error as e:
                    print(f"[AceAttorneyEnv get_mapped_dialogue_event] Regex error for pattern '{regex}': {e}")
        
        # print(f"[AceAttorneyEnv get_mapped_dialogue_event] No pattern matched for text: '{dialogue_text[:50]}...'. Returning default keyword '{default_keyword}'.")
        return default_keyword

    def close(self):
        """Closes the environment and the adapter's log file."""
        super().close()
        self.adapter.close_log_file()
        print("[AceAttorneyEnv] Environment closed.")
