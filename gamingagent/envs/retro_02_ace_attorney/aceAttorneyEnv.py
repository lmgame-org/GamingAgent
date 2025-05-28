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
        state: Optional[str] = "level1_1_5", # This will be our primary level/scene identifier
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
        # print(f"[AceAttorneyEnv __init__] RetroEnv initialized. Action space: {self.action_space}, Buttons: {self.buttons}")

        # --- Game Specific Variables & Configs ---
        self.initial_retro_state_name: str = state if state is not None else "level1_1_5" # Default if None
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
        self.raw_llm_output_from_previous_step: Optional[str] = None # ADDED: To store raw LLM output from previous step
        self.dialogue_history_for_agent: List[str] = [] # ADDED: To accumulate dialogue for agent observation

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
        
        # print(f"[AceAttorneyEnv __init__] Initialized with state: {self.initial_retro_state_name}, obs_mode for adapter: {adapter_observation_mode}")
        # print(f"[AceAttorneyEnv __init__] Action hold: {self.num_frames_to_hold_action} frames, No-op pause: {self.num_frames_for_no_op_pause} frames.")

    def _load_env_specific_config(self, config_path: str):
        # This method might become very simple or be removed if all necessary
        # config (like button names for retro env) is handled by adapter or hardcoded if stable.
        # For now, let it pass, as adapter loads its own action_mapping directly.
        if os.path.exists(config_path):
            # The adapter loads the action_mapping. This env doesn't need to parse it further for itself.
            # print(f"[AceAttorneyEnv] Env-specific config found at {config_path}. Adapter will handle its contents.")
            pass
        else:
            # print(f"[AceAttorneyEnv] WARNING: Env specific config {config_path} not found.")
            pass

    def _load_game_script_data(self, script_path: str):
        """Loads game script data (dialogue, skips, scene details) from a JSON file."""
        if os.path.exists(script_path):
            try:
                with open(script_path, 'r') as f:
                    self.game_script_data = json.load(f)
                # print(f"[AceAttorneyEnv] Successfully loaded game script data from: {script_path}")
                
            except json.JSONDecodeError as e:
                print(f"[AceAttorneyEnv] Error decoding JSON from game script {script_path}: {e}")
        else:
            # print(f"[AceAttorneyEnv] Warning: Game script file {script_path} not found.")
            self.game_script_data = {}

    def _load_skip_conversation_data(self, skip_path: str):
        if os.path.exists(skip_path):
            try:
                with open(skip_path, 'r', encoding='utf-8') as f:
                    self.skip_conversation_data = json.load(f)
                # print(f"[AceAttorneyEnv] Loaded skip conversation data: {skip_path}")
            except Exception as e:
                print(f"[AceAttorneyEnv] ERROR loading skip conversation data {skip_path}: {e}")
        else:
            # print(f"[AceAttorneyEnv] WARNING: Skip conversation data file not found: {skip_path}")
            pass

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
            # print("[AceAttorneyEnv] Cannot initialize level data: retro_state_name or game_script_data missing.")
            return

        level_data = self.game_script_data.get(self.current_retro_state_name)
        if not level_data:
            # print(f"[AceAttorneyEnv] WARNING: No data in mapping.json for level: {self.current_retro_state_name}")
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
        else:
            # print(f"[AceAttorneyEnv] WARNING: No skip/end data for level: {self.current_retro_state_name}")
            pass

        # print(f"[AceAttorneyEnv] Initialized data for level: {self.current_retro_state_name}")

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
            # print(f"[AceAttorneyEnv _save_frame_to_path] Warning: frame_to_save is not a numpy array (type: {type(frame_to_save)}). Cannot save image for E{self.adapter.current_episode_id} S{self.adapter.current_step_num}.")
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
            # print(f"[AceAttorneyEnv _save_frame_to_path] Error saving frame to {img_path}: {e}")
            return None

    def _build_agent_observation_components(
        self,
        agent_facing_info: Dict,
        skip_screenshot: bool = False
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Builds the image **and text** components for the agent's observation.

        * Image is saved unless `skip_screenshot` is True or obs‑mode lacks vision.
        * Text component is now constructed from:
            - Comprehensive memory (background, evidence)
            - The latest LLM-parsed dialogue line.
        """
        # Vision
        img_path_component: Optional[str] = None
        if self.adapter.observation_mode in ("vision", "both"):
            if not skip_screenshot and self.current_raw_frame is not None:
                img_path_component = self._save_frame_to_path(self.current_raw_frame)

        # Text part
        text_obs_component: Optional[str] = None
        if self.adapter.observation_mode in ("text", "both"):
            parts = []
            
            # 1. Comprehensive Memory (Background & Evidence)
            # get_comprehensive_memory_string() returns a single string with both.
            comprehensive_memory = self.get_comprehensive_memory_string()
            if comprehensive_memory:
                parts.append(comprehensive_memory) # This already includes "Background Transcript:" and "Evidence List:" headers

            # 2. Process Dialogue from previous LLM output
            if self.raw_llm_output_from_previous_step:
                dialogue_match = re.search(r"^[Dd][Ii][Aa][Ll][Oo][Gg]:\s*([^:]+):\s*(.+)$", self.raw_llm_output_from_previous_step, re.MULTILINE)
                if dialogue_match:
                    speaker = dialogue_match.group(1).strip()
                    text = dialogue_match.group(2).strip()
                    
                    current_parsed_dialogue_line = f"{speaker}: {text}"
                    
                    # Check for duplicates before appending
                    if not self.dialogue_history_for_agent or self.dialogue_history_for_agent[-1] != current_parsed_dialogue_line:
                        self.dialogue_history_for_agent.append(current_parsed_dialogue_line)
                    
                    # Store the parsed dialogue to update self.last_llm_dialogue_info for other internal uses
                    parsed_dialogue_data_for_storage = {"speaker": speaker, "text": text}
                    if hasattr(self, "store_llm_extracted_dialogue"):
                        self.store_llm_extracted_dialogue(parsed_dialogue_data_for_storage)
            
            # 3. Construct Dialogue History for observation with mapping logic
            if self.dialogue_history_for_agent:
                displayed_dialogue_lines = []
                for history_line in self.dialogue_history_for_agent:
                    # Attempt to find a full-line match in the current level's dialog map
                    if self.current_level_dialog_map and history_line in self.current_level_dialog_map:
                        displayed_dialogue_lines.append(self.current_level_dialog_map[history_line])
                    else:
                        # If no full match, try to map just the speaker
                        # Use a different variable name here to avoid overwriting the outer 'parts' list
                        speaker_text_pair = history_line.split(": ", 1)
                        if len(speaker_text_pair) == 2:
                            speaker_orig, text_orig = speaker_text_pair[0], speaker_text_pair[1]
                            mapped_speaker = speaker_orig # Default to original if not in name_map
                            if self.current_level_name_map:
                                mapped_speaker = self.current_level_name_map.get(speaker_orig.lower(), speaker_orig)
                            displayed_dialogue_lines.append(f"{mapped_speaker}: {text_orig}")
                        else:
                            displayed_dialogue_lines.append(history_line) # Should not happen if parsing was correct
                
                dialogue_history_str = "Dialogue History (older -> newer):\n" + "\n".join(displayed_dialogue_lines)
                parts.append(dialogue_history_str)
            else:
                parts.append("Dialogue History: None available yet.")
            
            text_obs_component = "\n\n".join(parts).strip()
            if not text_obs_component: # Ensure it's not an empty string if all parts are None/empty
                text_obs_component = "No textual information available."


        return img_path_component, text_obs_component

    def reset(self, *, seed: Optional[int]=None, options: Optional[Dict[str,Any]]=None, episode_id:int=1) -> Tuple[Observation, Dict[str,Any]]:
        # Always reset to the designated initial_retro_state_name for this environment instance.
        # Level progression is handled within step() by directly calling load_state().
        # print(f"[AceAttorneyEnv RESET] Initiating reset. Target initial state: '{self.initial_retro_state_name}'.")
        
        # Load the initial state defined for this environment instance.
        # This ensures that a general reset always brings us back to the very start state of this env config.
        if self.current_retro_state_name != self.initial_retro_state_name or not self.data:
            # self.data check is to ensure core retro env is properly initialized if it's the first reset.
            # print(f"[AceAttorneyEnv RESET] Current state ('{self.current_retro_state_name}') differs from initial or core data missing. Loading initial state: '{self.initial_retro_state_name}'.")
            try:
                self.load_state(self.initial_retro_state_name, self.retro_inttype)
            except Exception as e:
                print(f"[AceAttorneyEnv RESET] CRITICAL ERROR loading initial state '{self.initial_retro_state_name}': {e}. Attempting to proceed with super().reset() but state might be inconsistent.")
                # If load_state fails, super().reset() might reset to a default or last valid state, which could be problematic.
        
        self.current_retro_state_name = self.initial_retro_state_name # Ensure this is set before super().reset()
        
        # super().reset() handles the core emulator reset, provides initial RAM observation and info.
        ram_observation, self.current_core_info = super().reset(seed=seed, options=options)
        self.current_raw_frame = self.em.get_screen() # Get screen pixels after core reset

        # Reset internal game logic state variables to their initial values for this env instance.
        self.current_lives = self.initial_lives 
        # print(f"[AceAttorneyEnv RESET] Lives reset to initial value: {self.current_lives}")
        self.dialogue_history_for_agent = [] # ADDED: Clear dialogue history
        
        self._initialize_level_specific_data() # Re-initialize dialogue logs, skip maps, etc., for the initial_retro_state_name.
        self._update_internal_game_state(self.current_core_info) # Update lives from RAM if different (should match initial_lives now)
        # self._extract_dialogue_from_info(self.current_core_info) # Dialogue not extracted by env
        
        agent_facing_info = self._get_agent_info()
        self.adapter.reset_episode(episode_id) # Reset adapter's episode tracking.

        # Build the first observation for the agent.
        img_path, txt_rep = self._build_agent_observation_components(agent_facing_info, skip_screenshot=False)
        agent_obs = self.adapter.create_agent_observation(img_path=img_path, text_representation=txt_rep)
        
        initial_step_perf_score = self.adapter.calculate_perf_score(0.0, agent_facing_info)
        self.adapter.log_step_data(
            agent_action_str="<RESET>",
            thought_process="Episode reset to initial state.",
            reward=0.0,
            info=agent_facing_info,
            terminated=False, # A reset implies not terminated at this point
            truncated=False,
            time_taken_s=0.0,
            perf_score=initial_step_perf_score,
            agent_observation=agent_obs
        )

        if self.wrapper_render_mode == "human":
            self.render()
        
        # print(f"[AceAttorneyEnv RESET] Reset complete. Current state: '{self.current_retro_state_name}', Lives: {self.current_lives}. Returning initial observation.")
        return agent_obs, agent_facing_info

    def _check_and_trigger_skip_sequence(self) -> Optional[Tuple[Observation, SupportsFloat, bool, bool, Dict[str, Any], float]]:
        """
        Checks if the last dialogue line in history triggers a skip sequence.
        If so, executes 'A' presses and returns the state after skipping.
        Returns None if no skip sequence is triggered.
        """
        if not self.dialogue_history_for_agent: # Check if history is empty
            return None

        trigger_key = self.dialogue_history_for_agent[-1] # Use the last line from history

        # Ensure current_level_skip_map is populated
        if not hasattr(self.adapter, 'map_agent_action_to_env_action'):
            print("[AceAttorneyEnv _check_and_trigger_skip_sequence] ERROR: adapter not fully initialized or missing map_agent_action_to_env_action.")
            return None
        
        # Assuming 'a' is the action string for pressing the A button.
        # We need the actual button array for 'A'.
        action_A_array = self.adapter.map_agent_action_to_env_action("a")
        if action_A_array is None:
            print("[AceAttorneyEnv _check_and_trigger_skip_sequence] ERROR: Could not map 'a' to an action array. Check game_env_config.json.")
            # Fallback: try to create a plausible 'A' button array if num_buttons is known
            # This assumes 'A' is the 9th button (index 8) for a 12-button layout.
            if self.num_buttons == 12:
                action_A_array = np.array([0,0,0,0,0,0,0,0,1,0,0,0], dtype=bool)
                # print("[AceAttorneyEnv _check_and_trigger_skip_sequence] WARNING: Using hardcoded 'A' button for skipping due to mapping failure.")
            else:
                print("[AceAttorneyEnv _check_and_trigger_skip_sequence] CRITICAL: Cannot determine 'A' button action for skipping.")
                return None


        if self.current_level_skip_map and trigger_key in self.current_level_skip_map:
            lines_to_skip = self.current_level_skip_map[trigger_key]
            num_skips = len(lines_to_skip)

            if num_skips == 0:
                return None # No lines to skip for this trigger

            print(f"[AceAttorneyEnv DEBUG _check_and_trigger_skip_sequence] Matched skip trigger: '{trigger_key}'. Skipping {num_skips} lines.")

            accumulated_skip_reward: float = 0.0
            skip_terminated: bool = False
            skip_truncated: bool = False
            final_skip_obs: Optional[Observation] = None
            final_skip_info: Dict[str, Any] = {}
            
            num_frames_for_no_op_pause_in_skip = 200 # Define the pause length
            no_op_action_for_skip_pause = np.zeros(self.num_buttons, dtype=bool)

            for i in range(num_skips):
                # --- 1. Press 'A' (1 frame) ---
                # print(f"[AceAttorneyEnv DEBUG _check_and_trigger_skip_sequence] Skip Turn {i+1}/{num_skips}: Pressing 'A'.")
                ram_obs_A, reward_A, term_A, trunc_A, self.current_core_info = super().step(action_A_array)
                self.current_raw_frame = self.em.get_screen()
                accumulated_skip_reward += float(reward_A)
                self._update_internal_game_state(self.current_core_info)
                if self.current_lives <= 0: term_A = True
                
                skip_terminated = term_A
                skip_truncated = trunc_A
                if self.wrapper_render_mode == "human": self.render()

                if skip_terminated or skip_truncated:
                    print(f"[AceAttorneyEnv DEBUG _check_and_trigger_skip_sequence] Terminated/Truncated after 'A' press in skip turn {i+1}.")
                    break # Exit the main skip loop

                # --- 2. Execute No-Op Pause (e.g., 200 frames) ---
                # print(f"[AceAttorneyEnv DEBUG _check_and_trigger_skip_sequence] Skip Turn {i+1}/{num_skips}: Starting {num_frames_for_no_op_pause_in_skip}-frame no-op pause.")
                for _ in range(num_frames_for_no_op_pause_in_skip):
                    ram_obs_noop, reward_noop, term_noop, trunc_noop, self.current_core_info = super().step(no_op_action_for_skip_pause)
                    self.current_raw_frame = self.em.get_screen()
                    accumulated_skip_reward += float(reward_noop)
                    self._update_internal_game_state(self.current_core_info)
                    if self.current_lives <= 0: term_noop = True
                    
                    skip_terminated = term_noop
                    skip_truncated = trunc_noop
                    if self.wrapper_render_mode == "human": self.render()

                    if skip_terminated or skip_truncated:
                        print(f"[AceAttorneyEnv DEBUG _check_and_trigger_skip_sequence] Terminated/Truncated during no-op pause in skip turn {i+1}.")
                        break # Exit no-op pause loop
                
                if skip_terminated or skip_truncated: # Check again to break main skip loop if no-op pause terminated
                    break
            
            # After skip loop, build the final observation and info
            final_skip_info = self._get_agent_info()
            skip_block_perf_score = self.adapter.calculate_perf_score(accumulated_skip_reward, final_skip_info)

            img_path_skip, txt_rep_skip = self._build_agent_observation_components(final_skip_info, skip_screenshot=False)
            final_skip_obs = self.adapter.create_agent_observation(img_path=img_path_skip, text_representation=txt_rep_skip)
            
            # Log this entire skip block as a single meta-action
            self.adapter.log_step_data(
                agent_action_str=f"<AUTO_SKIP_BLOCK_{num_skips}_FRAMES>",
                thought_process=f"Auto-skipped {num_skips} dialogue lines triggered by: {trigger_key}",
                reward=accumulated_skip_reward,
                info=final_skip_info.copy(),
                terminated=skip_terminated,
                truncated=skip_truncated,
                time_taken_s=0.0, # Mechanical skip, effectively instantaneous from agent's perspective
                perf_score=skip_block_perf_score,
                agent_observation=final_skip_obs
            )
            return final_skip_obs, accumulated_skip_reward, skip_terminated, skip_truncated, final_skip_info, skip_block_perf_score
        
        return None # No skip triggered

    def _check_for_end_statement_match(self) -> bool:
        """Checks if the last dialogue line in history matches any defined end statement for the current level."""
        if not self.dialogue_history_for_agent: # Check if history is empty
            return False
        if not self.current_level_end_statements: # No end statements defined for this level
            return False

        current_dialogue_line = self.dialogue_history_for_agent[-1] # Use the last line from history
        
        for end_statement in self.current_level_end_statements:
            if current_dialogue_line == end_statement: # Exact match
                print(f"[AceAttorneyEnv DEBUG _check_for_end_statement_match] Matched end statement: '{end_statement}'")
                return True
        return False

    def step(self, agent_action_str:Optional[str], thought_process:str="",time_taken_s:float=0.0, raw_llm_output_for_next_obs: Optional[str] = None) -> Tuple[Observation,SupportsFloat,bool,bool,Dict[str,Any],float]:
        # Store the raw LLM output from this step, to be used in the *next* step's observation construction
        self.raw_llm_output_from_previous_step = raw_llm_output_for_next_obs
        # --- Initial End Statement Check (based on PREVIOUS turn's dialogue) ---
        if self._check_for_end_statement_match():
            # print(f"[AceAttorneyEnv STEP] Terminating level '{self.current_retro_state_name}' due to matched end statement from previous observation.")
            
            # --- LEVEL PROGRESSION LOGIC ---
            next_level_state_name: Optional[str] = None
            if self.current_retro_state_name == "level1_1_5":
                next_level_state_name = "level2_1_9" 
            if self.current_retro_state_name == "level2_1_9":
                next_level_state_name = "level3_1_8" 
            if next_level_state_name:
                try:
                    # Load the new state into the retro emulator
                    self.load_state(next_level_state_name, self.retro_inttype) # This effectively resets the core game to the new state
                    self.current_retro_state_name = next_level_state_name
                    self.current_lives = self.initial_lives # Reset lives for the new level
                    self._initialize_level_specific_data() # Reload scripts, skip maps for the new level
                    
                    # The core game is reset to the new level. Get its initial observation.
                    ram_observation_new_level, self.current_core_info = super().reset(seed=None, options=None) # Perform a soft reset of the core for new level's initial info
                    self.current_raw_frame = self.em.get_screen()
                    self._update_internal_game_state(self.current_core_info)

                    # print(f"[AceAttorneyEnv STEP] Successfully loaded new level: {self.current_retro_state_name}. Lives set to {self.current_lives}.")
                    
                    # Construct observation for the agent for this new level's start
                    current_info = self._get_agent_info() # Info for the new level
                    img_path_new_level, txt_rep_new_level = self._build_agent_observation_components(current_info, skip_screenshot=False)
                    obs_for_new_level_start = self.adapter.create_agent_observation(img_path=img_path_new_level, text_representation=txt_rep_new_level)
                    perf_for_new_level_start = self.adapter.calculate_perf_score(0.0, current_info)

                    # Log that the level ended and new one is starting
                    self.adapter.log_step_data(
                        agent_action_str="<LEVEL_COMPLETE_PROCEED_TO_NEXT>",
                        thought_process=f"Level {self.initial_retro_state_name} ended. Starting {self.current_retro_state_name}.", # initial_retro_state_name here is a bit off, but conveys transition
                        reward=0.0, # No specific reward for level transition itself in this step
                        info=current_info.copy(),
                        terminated=True, # The step in the *previous* level is terminated
                        truncated=False,
                        time_taken_s=0.0,
                        perf_score=perf_for_new_level_start,
                        agent_observation=obs_for_new_level_start # Provide the first obs of the new level
                    )
                    # The runner will likely see terminated=True and might reset or just continue if designed for sequential levels.
                    # We return the first observation of the new level.
                    return obs_for_new_level_start, 0.0, True, False, current_info, perf_for_new_level_start

                except Exception as e:
                    print(f"[AceAttorneyEnv STEP] CRITICAL ERROR loading next level '{next_level_state_name}': {e}. Treating as game over.")
                    # Fall through to game over logic if next level load fails
                    current_info = self._get_agent_info()
                    img_path_err, txt_rep_err = self._build_agent_observation_components(current_info, skip_screenshot=False)
                    obs_err = self.adapter.create_agent_observation(img_path=img_path_err, text_representation=txt_rep_err)
                    perf_err = self.adapter.calculate_perf_score(0.0, current_info)
                    self.adapter.log_step_data("<LEVEL_LOAD_ERROR_GAME_OVER>", f"Failed to load {next_level_state_name}", 0.0, current_info.copy(), True, False, 0.0, perf_err, obs_err)
                    return obs_err, 0.0, True, False, current_info, perf_err
            else:
                # print(f"[AceAttorneyEnv STEP] End statement matched for '{self.current_retro_state_name}', but no next level defined. Treating as game completion/final level.")
                # Standard termination for the last defined level
                last_obs_for_agent = self.adapter.get_last_observation_for_agent()
                if last_obs_for_agent is None:
                    img_path_term, txt_rep_term = self._build_agent_observation_components(self._get_agent_info(), skip_screenshot=False)
                    last_obs_for_agent = self.adapter.create_agent_observation(img_path=img_path_term, text_representation=txt_rep_term)
                current_info = self._get_agent_info()
                current_perf = self.adapter.calculate_perf_score(0.0, current_info)
                self.adapter.log_step_data("<FINAL_LEVEL_COMPLETE>", "Final level completed via end statement.", 0.0, current_info.copy(), True, False, 0.0, current_perf, last_obs_for_agent)
                return last_obs_for_agent, 0.0, True, False, current_info, current_perf

        # --- LIVES CHECK (can lead to Game Over) ---
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
                # Check for game over due to lives AFTER processing the action that might have caused it
                if self.current_lives <= 0 and not current_terminated_overall: # ensure we don't double-log termination if already terminated for other reasons
                    # print(f"[AceAttorneyEnv STEP] GAME OVER. Lives reached 0 after Phase 1 action.")
                    current_terminated_overall = True # Explicitly set game over
                    # Log this specific game over event. current_observation_to_return and others are from this step.
                    self.adapter.log_step_data(
                        agent_action_str=effective_agent_action_str_for_log, # Action that led to game over
                        thought_process=thought_process + " (Resulted in Game Over - Lives 0)",
                        reward=overall_accumulated_reward, # Reward up to this point
                        info=current_agent_facing_info_to_return.copy(),
                        terminated=True, # Game Over
                        truncated=current_truncated_overall, # Keep existing truncation status
                        time_taken_s=current_frame_time_taken,
                        perf_score=overall_accumulated_perf_score, # Perf score up to this point
                        agent_observation=current_observation_to_return
                    )
                # print(f"[AceAttorneyEnv DEBUG] Phase 1 (Frame {frame_num_phase1+1}): Terminating/Truncating. Skipping Phase 2.")
                return current_observation_to_return, overall_accumulated_reward, current_terminated_overall, current_truncated_overall, current_agent_facing_info_to_return, overall_accumulated_perf_score

        # --- Check and Execute Skip Sequence (NEW) ---
        # This occurs after agent's action (Phase 1) and before automatic no-op (Phase 2)
        # It uses self.last_llm_dialogue_info which was set by BaseAgent based on the LLM's parsing
        # of the visual state *before* the current agent_action_str was executed.
        
        skip_results = self._check_and_trigger_skip_sequence()
        
        if skip_results:
            # A skip sequence was executed. Update overall state.
            # print("[AceAttorneyEnv DEBUG Step] Skip sequence executed.")
            (current_observation_to_return, skip_reward, skip_terminated, 
             skip_truncated, current_agent_facing_info_to_return, skip_perf_score) = skip_results
            
            overall_accumulated_reward += skip_reward # Add reward from skip sequence
            overall_accumulated_perf_score += skip_perf_score
            current_terminated_overall = current_terminated_overall or skip_terminated
            current_truncated_overall = current_truncated_overall or skip_truncated
            
            # Check for game over due to lives AFTER skip sequence
            if self.current_lives <= 0 and not current_terminated_overall:
                # print(f"[AceAttorneyEnv STEP] GAME OVER. Lives reached 0 after skip sequence.")
                current_terminated_overall = True
                self.adapter.log_step_data(
                    agent_action_str=f"<AUTO_SKIP_BLOCK_GAME_OVER>", # Indicate skip led to game over
                    thought_process=f"Auto-skipped, resulted in Game Over - Lives 0",
                    reward=overall_accumulated_reward,
                    info=current_agent_facing_info_to_return.copy(),
                    terminated=True,
                    truncated=current_truncated_overall,
                    time_taken_s=0.0,
                    perf_score=overall_accumulated_perf_score,
                    agent_observation=current_observation_to_return
                )

            if current_terminated_overall or current_truncated_overall:
                # print("[AceAttorneyEnv DEBUG Step] Terminated/Truncated after skip sequence. Skipping Phase 2 (no-op).")
                return current_observation_to_return, overall_accumulated_reward, current_terminated_overall, current_truncated_overall, current_agent_facing_info_to_return, overall_accumulated_perf_score

        # --- Phase 2: Execute Automatic No-Op Pause ---
        # This phase runs only if Phase 1 (and skip sequence, if any) did not terminate/truncate the episode.
        if not (current_terminated_overall or current_truncated_overall) and num_frames_for_no_op_pause > 0:
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
            
            # Check for game over due to lives AFTER Phase 2 (No-Op)
            if self.current_lives <= 0 and not current_terminated_overall:
                # print(f"[AceAttorneyEnv STEP] GAME OVER. Lives reached 0 after Phase 2 (No-Op).")
                current_terminated_overall = True
                # Update the log entry for the no-op block if it's the one causing game over.
                # This is a bit tricky as the log for no-op is made with its own values.
                # For simplicity, we'll let the existing no-op log stand, and this game over is an additional state.
                # The 'terminated' flag in the final return will reflect this.
                # If precise logging of which block (action, skip, no-op) caused game over is needed,
                # the log_step_data calls would need to be more conditional or updated.

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
        
        # Final check for game over if not caught by specific phase checks (e.g., if lives dropped due to an earlier phase but termination wasn't processed there)
        if self.current_lives <= 0 and not current_terminated_overall:
            # print(f"[AceAttorneyEnv STEP] GAME OVER. Lives at {self.current_lives} at end of step processing.")
            current_terminated_overall = True
            # Potentially log a generic game over event here if not already logged by a specific phase.
            # However, the phase-specific checks should ideally catch it.

        return current_observation_to_return, overall_accumulated_reward, current_terminated_overall, current_truncated_overall, current_agent_facing_info_to_return, overall_accumulated_perf_score

    def render(self) -> Optional[RenderFrame]:
        if self.wrapper_render_mode == 'human':
            return super().render()
        elif self.wrapper_render_mode == 'rgb_array' and self.current_raw_frame is not None:
            return self.current_raw_frame.copy()
        return None

    def store_llm_extracted_dialogue(self, dialogue_data: Dict[str, str]):
        """Stores dialogue extracted by the LLM into a JSONL file in the agent_cache_dir."""
        if not self.current_retro_state_name:
            print("[AceAttorneyEnv store_llm_extracted_dialogue] ERROR: current_retro_state_name is not set. Cannot reliably tag dialogue origin.")
        if not dialogue_data or not isinstance(dialogue_data, dict) or "speaker" not in dialogue_data or "text" not in dialogue_data:
            print(f"[AceAttorneyEnv store_llm_extracted_dialogue] ERROR: Invalid dialogue_data format: {dialogue_data}. Required keys: 'speaker', 'text'.")
            return

        # Update last_llm_dialogue_info for immediate use (e.g., by get_mapped_dialogue_event_for_prompt)
        self.last_llm_dialogue_info = {
            "state_name": self.current_retro_state_name, # Store current state for context
            "speaker": dialogue_data["speaker"],
            "text": dialogue_data["text"],
            "timestamp": time.time()
        }
        # print(f"[AceAttorneyEnv store_llm_extracted_dialogue] Updated self.last_llm_dialogue_info: {self.last_llm_dialogue_info}")

        # Define the single dialogues.jsonl file path in the root of agent_cache_dir
        dialogue_file_path = os.path.join(self.adapter.agent_cache_dir, "dialogues.jsonl")
        # print(f"[AceAttorneyEnv DEBUG store_llm_extracted_dialogue] Dialogue log file path: {dialogue_file_path}")

        entry_to_save = {
            "state_name": self.current_retro_state_name, # Current game state/level
            "speaker": dialogue_data["speaker"],
            "text": dialogue_data["text"],
        }

        try:
            # Open in append mode ('a') to add new dialogue entries
            with open(dialogue_file_path, 'a') as f:
                json.dump(entry_to_save, f)
                f.write('\n') # Add a newline to separate JSON objects (JSONL format)
            #  print(f"[AceAttorneyEnv DEBUG store_llm_extracted_dialogue] Successfully appended dialogue to: {dialogue_file_path}")
        except Exception as e:
            print(f"[AceAttorneyEnv store_llm_extracted_dialogue] CRITICAL ERROR: Failed to save LLM dialogue to {dialogue_file_path}. Details: {e}")

    def get_mapped_dialogue_event_for_prompt(self) -> Optional[str]:
        """
        Retrieves the last stored LLM dialogue, maps its speaker name using
        the current state's 'name_mappings' in mapping.json, and returns a formatted string
        with the original text for the LLM prompt.
        """
        # print(f"[AceAttorneyEnv DEBUG get_mapped_dialogue_event] Method called (Simplified: Name Mapping Only).")
        if not self.last_llm_dialogue_info or \
           "speaker" not in self.last_llm_dialogue_info or \
           "text" not in self.last_llm_dialogue_info or \
           "state_name" not in self.last_llm_dialogue_info:
            # print("[AceAttorneyEnv get_mapped] Essential info missing from self.last_llm_dialogue_info. Cannot create prompt context.")
            return None

        original_speaker = self.last_llm_dialogue_info["speaker"].strip()
        original_text = self.last_llm_dialogue_info["text"].strip()
        current_state_name = self.last_llm_dialogue_info["state_name"]

        # print(f"[AceAttorneyEnv get_mapped] Processing: State='{current_state_name}', Speaker='{original_speaker}', Text='{original_text[:60]}...'")

        final_speaker_to_use = original_speaker # Default to original speaker

        if not self.game_script_data or current_state_name not in self.game_script_data:
            # print(f"[AceAttorneyEnv get_mapped] No game_script_data or state '{current_state_name}' not found in script. Using original speaker and text.")
            pass
        else:
            state_data = self.game_script_data[current_state_name]
            state_name_map = state_data.get("name_mappings", {})
            
            if not state_name_map:
                # print(f"[AceAttorneyEnv get_mapped] 'name_mappings' not found or empty for state '{current_state_name}'. Using original speaker.")
                pass
            else:
                # Apply name mapping to the speaker
                # Speaker names in name_mappings are lowercase keys
                canonical_speaker = state_name_map.get(original_speaker.lower(), original_speaker)
                if canonical_speaker != original_speaker:
                    # print(f"[AceAttorneyEnv get_mapped] Speaker '{original_speaker}' mapped to '{canonical_speaker}' via 'name_mappings'.")
                    final_speaker_to_use = canonical_speaker
                # else:
                    # print(f"[AceAttorneyEnv get_mapped] Speaker '{original_speaker}' not found in 'name_mappings' for state '{current_state_name}' or already canonical.")
            
        # Construct the final context string using original_text
        # Return only speaker: text, the BaseAgent will add the prefix.
        prompt_context = f"{final_speaker_to_use}: {original_text}"
        # print(f"[AceAttorneyEnv get_mapped] Generated mapped dialogue: {prompt_context[:100]}...")
        return prompt_context

    def get_comprehensive_memory_string(self) -> str:
        """
        Compiles a comprehensive memory string including mapped background transcript
        and mapped evidence list for the current game state.
        Now structured with clear headers for each section.
        """
        # print(f"[AceAttorneyEnv DEBUG get_comprehensive_memory_string] Method called for state: {self.current_retro_state_name}")
        if not self.current_retro_state_name or not self.game_script_data or \
           self.current_retro_state_name not in self.game_script_data:
            # print(f"[AceAttorneyEnv get_comprehensive] Essential data missing for state '{self.current_retro_state_name}'. Returning empty context.")
            return "Background Transcript: Not available for current state.\\n\\nEvidence List: Not available for current state."

        state_data = self.game_script_data[self.current_retro_state_name]
        name_map = state_data.get("name_mappings", {})
        evidence_map = state_data.get("evidence_mappings", {})
        
        background_transcript_orig = state_data.get("background_transcript", [])
        evidences_orig = state_data.get("evidences", [])

        # Process Background Transcript
        processed_background_lines = []
        if background_transcript_orig:
            for line in background_transcript_orig:
                parts = line.split(": ", 1)
                mapped_line = line
                if len(parts) == 2:
                    speaker, text = parts[0], parts[1]
                    mapped_speaker = name_map.get(speaker.lower(), speaker)
                    mapped_line = f"{mapped_speaker}: {text}"
                processed_background_lines.append(mapped_line)
        
        background_section = "Background Transcript:\\n" + ("\\n".join(processed_background_lines) if processed_background_lines else "None available.")
        
        # Process Evidences
        processed_evidences_lines = []
        if evidences_orig:
            for ev_line in evidences_orig:
                parts = ev_line.split(": ", 1)
                mapped_ev_line = ev_line
                if len(parts) == 2:
                    ev_name, ev_desc = parts[0], parts[1]
                    short_ev_name = evidence_map.get(ev_name.upper(), None)
                    if short_ev_name:
                        mapped_ev_line = f"{short_ev_name}: {ev_desc}"
                    else:
                        mapped_ev_line = f"{ev_name}: {ev_desc}"
                processed_evidences_lines.append(mapped_ev_line)
        
        evidence_section = "Evidence List:\\n" + ("\\n".join(processed_evidences_lines) if processed_evidences_lines else "None available.")

        final_context_string = f"{background_section}\\n\\n{evidence_section}"
        # print(f"[AceAttorneyEnv get_comprehensive] Generated comprehensive context (first 200 chars): {final_context_string[:200]}...")
        return final_context_string

    def prepare_prompt(self, base_prompt: str) -> Tuple[str, Optional[str]]:
        """
        Return (prompt_body, additional_prefix) ready for the LLM.
        Called by BaseAgent in non‑harness mode.
        """
        prompt_body = base_prompt

        # 1.  {memory_context} placeholder
        if "{memory_context}" in prompt_body:
            prompt_body = prompt_body.replace(
                "{memory_context}",
                self.get_comprehensive_memory_string()
            )

        # 2.  Dialogue prefix
        additional_prefix = None
        dialogue_line = self.get_mapped_dialogue_event_for_prompt()
        if dialogue_line:
            additional_prefix = f"Previous Dialogue Context: {dialogue_line}\\n\\n"

        return prompt_body, additional_prefix

    def close(self):
        """Closes the environment and the adapter's log file."""
        super().close()
        self.adapter.close_log_file()
        # print("[AceAttorneyEnv] Environment closed.")
