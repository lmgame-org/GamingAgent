import gymnasium as gym
import numpy as np

import copy
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, SupportsFloat, List
from gymnasium.core import RenderFrame

import cv2 # For rendering 'human' mode

import os
import re

from gamingagent.envs.gym_env_adapter import GymEnvAdapter
from gamingagent.modules.core_module import Observation
from gamingagent.envs.env_utils import create_board_image_tetris

# --- Core Tetris Components with Color ---
@dataclass
class Pixel:
    value: int
    color_rgb: List[int]

class Tetromino:
    def __init__(self, id_val: int, color_rgb: List[int], matrix: np.ndarray):
        self.id = id_val # This will be the unique ID on the board after offset
        self.color_rgb = color_rgb
        self.matrix = matrix
# --- End of Core Tetris Components ---

class TetrisEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "ansi"], "render_fps": 30}

    # BASE_PIXELS_DATA now includes color, as per tetris.py structure
    # value, [R,G,B]
    BASE_PIXELS_DATA = [
        (0, [0, 0, 0]),      # 0: Empty (Black)
        (1, [128, 128, 128]) # 1: Bedrock (Grey)
    ]
    
    # Directly define TETROMINOES as a list of Tetromino objects, like in tetris.py
    TETROMINOES = [
        Tetromino(0, [0, 240, 240], np.array([[0,0,0,0],[1,1,1,1],[0,0,0,0],[0,0,0,0]], dtype=np.uint8)), # I
        Tetromino(1, [240, 240, 0], np.array([[1,1],[1,1]], dtype=np.uint8)),                             # O
        Tetromino(2, [160, 0, 240], np.array([[0,1,0],[1,1,1],[0,0,0]], dtype=np.uint8)),                 # T
        Tetromino(3, [0, 240, 0],   np.array([[0,1,1],[1,1,0],[0,0,0]], dtype=np.uint8)),                 # S
        Tetromino(4, [240, 0, 0],   np.array([[1,1,0],[0,1,1],[0,0,0]], dtype=np.uint8)),                 # Z
        Tetromino(5, [0, 0, 240],   np.array([[1,0,0],[1,1,1],[0,0,0]], dtype=np.uint8)),                 # J
        Tetromino(6, [240, 160, 0], np.array([[0,0,1],[1,1,1],[0,0,0]], dtype=np.uint8))                  # L
    ]

    # Action Constants (replaces ActionsMapping)
    ACTION_NO_OP = 0
    ACTION_MOVE_LEFT = 1
    ACTION_MOVE_RIGHT = 2
    ACTION_ROTATE_COUNTERCLOCKWISE = 3
    ACTION_ROTATE_CLOCKWISE = 4
    ACTION_SOFT_DROP = 5
    ACTION_HARD_DROP = 6

    # Updated Reward Constants
    REWARD_PIECE_PLACED = 1.0
    REWARD_PER_LINE_CLEARED = 10.0

    # Symbols for text representation
    EMPTY_SYMBOL = '.'
    BEDROCK_SYMBOL = '#'
    TETROMINO_SYMBOLS = ['I', 'O', 'T', 'S', 'Z', 'J', 'L']

    def __init__(
        self,
        render_mode: Optional[str] = None,
        board_width: int = 10,
        board_height: int = 20,
        gravity: bool = True,
        render_upscale: int = 25,
        queue_size: int = 4,
        game_name_for_adapter: str = "tetris",
        observation_mode_for_adapter: str = "vision",
        agent_cache_dir_for_adapter: str = "cache/tetris/default_run",
        game_specific_config_path_for_adapter: str = "gamingagent/envs/custom_04_tetris/game_env_config.json",
        max_stuck_steps_for_adapter: Optional[int] = 30,
        seed: Optional[int] = None # Added seed parameter
    ):
        super().__init__()
        self.width = board_width
        self.height = board_height
        self.gravity_enabled = gravity
        self.render_scaling_factor = render_upscale
        self.window_name: Optional[str] = None
        self.queue_size = queue_size

        self.base_pixels = [Pixel(val, color) for val, color in self.BASE_PIXELS_DATA]
        
        # Use self.TETROMINOES (the list of Tetromino objects)
        # Deepcopy to ensure that any modifications to Tetromino instances (e.g. matrix during offset)
        # don't affect the class-level definition for subsequent TetrisEnv instantiations.
        raw_tetrominoes = [copy.deepcopy(t) for t in self.TETROMINOES]
        self.tetrominoes = self._offset_tetromino_ids_and_values(raw_tetrominoes, len(self.base_pixels))

        # Create a direct mapping from piece ID on board to its color (dictionary - for info panel, etc.)
        self.pixel_id_to_color_map: Dict[int, List[int]] = {}
        max_id_for_lookup_array = 0
        for bp in self.base_pixels:
            self.pixel_id_to_color_map[bp.value] = bp.color_rgb
            if bp.value > max_id_for_lookup_array: max_id_for_lookup_array = bp.value
        for t_obj in self.tetrominoes:
            self.pixel_id_to_color_map[t_obj.id] = t_obj.color_rgb
            if t_obj.id > max_id_for_lookup_array: max_id_for_lookup_array = t_obj.id
        
        # Create a NumPy array for fast color lookup for the main board rendering
        self.colors_lookup_array = np.zeros((max_id_for_lookup_array + 1, 3), dtype=np.uint8)
        for piece_id, color_rgb in self.pixel_id_to_color_map.items():
            if piece_id <= max_id_for_lookup_array:
                 self.colors_lookup_array[piece_id] = color_rgb
            else:
                print(f"[WARN] TetrisEnv __init__: Piece ID {piece_id} out of bounds for colors_lookup_array (max_idx {max_id_for_lookup_array}).")

        self.padding = 0
        if self.tetrominoes: self.padding = max(max(t.matrix.shape) for t in self.tetrominoes)
        else: self.padding = 4 

        self.width_padded = self.width + 2 * self.padding
        self.height_padded = self.height + self.padding
        
        self.rng = np.random.default_rng(seed)
        self.tetromino_bag: List[int] = [] 
        self._fill_tetromino_bag()
        self.piece_queue: List[int] = [] 
        self._fill_piece_queue()
        
        self.board = self._create_board()
        self.active_tetromino: Optional[Tetromino] = None
        self.active_tetromino_original_idx: Optional[int] = None
        self.x: int = 0; self.y: int = 0
        self.game_over = False
        self.current_score = 0.0 
        self.total_perf_score_episode = 0.0
        self.lines_cleared_total = 0
        self.level = 1
        
        self.action_space = gym.spaces.Discrete(7)
        max_pixel_id_on_board = max_id_for_lookup_array
        self.observation_space = gym.spaces.Dict({
            "board": gym.spaces.Box(low=0, high=max_pixel_id_on_board, shape=(self.height_padded, self.width_padded), dtype=np.uint8),
            "active_tetromino_mask": gym.spaces.Box(low=0, high=1, shape=(self.height_padded, self.width_padded), dtype=np.uint8),
            "queue_piece_ids": gym.spaces.Box(low=0, high=max_pixel_id_on_board, shape=(self.queue_size,), dtype=np.uint8)
        })
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Headless mode fallback
        if self.render_mode == "human" and not os.environ.get("DISPLAY"):
            print("[TetrisEnv] DISPLAY not found – switching render_mode from 'human' to 'rgb_array'")
            self.render_mode = "rgb_array"

        self.adapter = GymEnvAdapter(
            game_name=game_name_for_adapter, observation_mode=observation_mode_for_adapter,
            agent_cache_dir=agent_cache_dir_for_adapter, game_specific_config_path=game_specific_config_path_for_adapter,
            max_steps_for_stuck=max_stuck_steps_for_adapter
        )
        self.current_raw_obs_dict: Dict[str, Any] = {}
        self.current_info_dict: Dict[str, Any] = {}
        self.game_name = game_name_for_adapter

    # --- Methods for Inlined Randomizer/Queue/Holder ---
    def _fill_tetromino_bag(self):
        # Bag contains original indices (0 to len(self.TETROMINOES)-1)
        self.tetromino_bag = list(range(len(self.TETROMINOES))) # Use self.TETROMINOES here
        self.rng.shuffle(self.tetromino_bag)

    def _get_next_tetromino_idx_from_bag(self) -> int:
        if not self.tetromino_bag:
            self._fill_tetromino_bag()
        return self.tetromino_bag.pop(0)

    def _fill_piece_queue(self):
        while len(self.piece_queue) < self.queue_size:
            self.piece_queue.append(self._get_next_tetromino_idx_from_bag())
    
    def _pop_from_piece_queue(self) -> int:
        idx = self.piece_queue.pop(0)
        self._fill_piece_queue()
        return idx
    # --- End Inlined Logic Methods ---

    def _offset_tetromino_ids_and_values(self, tetromino_list: List[Tetromino], offset: int) -> List[Tetromino]:
        processed = []
        for t_orig in tetromino_list:
            new_id = t_orig.id + offset 
            new_matrix = t_orig.matrix.copy()
            new_matrix[new_matrix > 0] = new_id 
            processed.append(Tetromino(new_id, t_orig.color_rgb, new_matrix))
        return processed

    def _create_board(self) -> np.ndarray:
        board = np.full((self.height, self.width), self.base_pixels[0].value, dtype=np.uint8)
        board = np.pad(board, ((0, self.padding), (self.padding, self.padding)), mode="constant", constant_values=self.base_pixels[1].value)
        return board

    def _spawn_tetromino(self) -> bool:
        tetromino_original_idx = self._pop_from_piece_queue() 
        self.active_tetromino_original_idx = tetromino_original_idx
        self.active_tetromino = copy.deepcopy(self.tetrominoes[tetromino_original_idx])
        
        self.x = self.width_padded // 2 - self.active_tetromino.matrix.shape[1] // 2
        self.y = 0
        if self._collision(self.active_tetromino, self.x, self.y):
            self.game_over = True
            return False
        return True

    def _collision(self, tetromino: Optional[Tetromino], x: int, y: int) -> bool:
        if tetromino is None: return True
        t_h, t_w = tetromino.matrix.shape
        for r_idx in range(t_h):
            for c_idx in range(t_w):
                if tetromino.matrix[r_idx, c_idx] != 0:
                    board_r, board_c = y + r_idx, x + c_idx
                    if not (0 <= board_r < self.height_padded and 0 <= board_c < self.width_padded) or \
                       self.board[board_r, board_c] != self.base_pixels[0].value:
                        return True
        return False

    def _rotate_tetromino(self, tetromino: Optional[Tetromino], clockwise: bool = True) -> Optional[Tetromino]:
        if tetromino is None: return None
        rotated_matrix = np.rot90(tetromino.matrix, k=-1 if clockwise else 1)
        return Tetromino(tetromino.id, tetromino.color_rgb, rotated_matrix)

    def _place_active_tetromino(self):
        if self.active_tetromino is None: return
        t_h, t_w = self.active_tetromino.matrix.shape
        for r_idx in range(t_h):
            for c_idx in range(t_w):
                if self.active_tetromino.matrix[r_idx, c_idx] != 0:
                    self.board[self.y + r_idx, self.x + c_idx] = self.active_tetromino.matrix[r_idx, c_idx]
        self.active_tetromino = None

    def _clear_filled_rows(self) -> int:
        lines_cleared = 0; r = self.height - 1 
        while r >= 0:
            game_row_on_board = self.board[r, self.padding : self.padding + self.width]
            if np.all(game_row_on_board != self.base_pixels[0].value):
                lines_cleared += 1
                self.board[1 : r + 1, self.padding : self.padding + self.width] = \
                    self.board[0 : r, self.padding : self.padding + self.width]
                self.board[0, self.padding : self.padding + self.width] = self.base_pixels[0].value
            else: r -= 1
        return lines_cleared

    def _calculate_score(self, lines: int) -> float:
        # Simplified scoring: +10 per line cleared
        return float(lines * self.REWARD_PER_LINE_CLEARED)

    def _commit_active_tetromino(self) -> Tuple[float, int]:
        """
        Places the active piece, clears full rows, updates game stats
        and returns:

            (tetris_reward , perf_increment)

        * tetris_reward   – +1 for piece +10 per line
        * perf_increment  – +1 for the commit itself, plus +1 for every
                            line cleared (so a T-Spin Triple would add 4).
        """
        if self.active_tetromino is None:
            return 0.0, 0

        # place piece on board
        self._place_active_tetromino()

        # line clears
        lines = self._clear_filled_rows()

        # reward bookkeeping
        commit_reward = (
            self.REWARD_PIECE_PLACED          # +1 for placing a piece
            + self._calculate_score(lines)    # +10 per cleared line
        )
        self.current_score += self._calculate_score(lines)
        self.lines_cleared_total += lines
        if self.lines_cleared_total // 10 >= self.level:
            self.level = (self.lines_cleared_total // 10) + 1

        # perf‑score increment (+1 commit, +1 per line)
        perf_increment = 1 + lines

        # next piece / game‑over check
        if not self._spawn_tetromino():
            self.game_over = True

        return commit_reward, perf_increment

    def _get_raw_board_obs_for_render(self) -> np.ndarray:
        projection = self.board.copy()
        if self.active_tetromino is not None:
            t_h, t_w = self.active_tetromino.matrix.shape
            for r_idx in range(t_h):
                for c_idx in range(t_w):
                    if self.active_tetromino.matrix[r_idx, c_idx] != 0:
                        proj_r, proj_c = self.y + r_idx, self.x + c_idx
                        if 0 <= proj_r < self.height_padded and 0 <= proj_c < self.width_padded:
                             projection[proj_r, proj_c] = self.active_tetromino.matrix[r_idx, c_idx]
        return projection

    def _get_obs(self) -> Dict[str, Any]:
        board_with_active = self._get_raw_board_obs_for_render()
        active_mask = np.zeros_like(self.board, dtype=np.uint8)
        if self.active_tetromino:
            t_h, t_w = self.active_tetromino.matrix.shape
            for r_idx in range(t_h):
                for c_idx in range(t_w):
                    if self.active_tetromino.matrix[r_idx,c_idx]!=0 and 0<=self.y+r_idx<self.height_padded and 0<=self.x+c_idx<self.width_padded:
                        active_mask[self.y+r_idx, self.x+c_idx] = 1
        
        q_indices = list(self.piece_queue) # Take a copy for debug/consistency
        q_ids = [self.tetrominoes[idx].id for idx in q_indices] 
        while len(q_ids) < self.observation_space["queue_piece_ids"].shape[0]: 
            q_ids.append(self.base_pixels[0].value)
        
        self.current_raw_obs_dict = {
            "board": board_with_active.astype(np.uint8),
            "active_tetromino_mask": active_mask.astype(np.uint8),
            "queue_piece_ids": np.array(q_ids[:self.observation_space["queue_piece_ids"].shape[0]], dtype=np.uint8)
        }
        return self.current_raw_obs_dict

    def _get_info(self) -> Dict[str, Any]:
        next_actual_ids = [self.tetrominoes[orig_idx].id for orig_idx in self.piece_queue]
        
        return {
            "score": self.current_score, 
            "total_perf_score_episode": self.total_perf_score_episode, 
            "lines": self.lines_cleared_total, 
            "level": self.level,
            "next_piece_ids": next_actual_ids, 
        }

    def _get_symbol_for_id(self, piece_id: int) -> str:
        if piece_id == self.base_pixels[0].value:
            return self.EMPTY_SYMBOL
        elif piece_id == self.base_pixels[1].value:
            return self.BEDROCK_SYMBOL
        else:
            # Tetromino IDs are offset by len(self.base_pixels)
            original_idx = piece_id - len(self.base_pixels)
            if 0 <= original_idx < len(self.TETROMINO_SYMBOLS):
                return self.TETROMINO_SYMBOLS[original_idx]
            return '?' # Unknown piece

    def _get_board_text_representation(self, board_array: np.ndarray) -> str:
        # Crop to the actual game area (height x width), excluding padding used for mechanics
        game_area = board_array[0:self.height, self.padding : self.padding + self.width]
        
        symbolic_rows = []
        for row in game_area:
            symbolic_rows.append("".join([self._get_symbol_for_id(cell_id) for cell_id in row]))
        return "\n".join(symbolic_rows)

    def _get_next_pieces_symbols_list(self, next_piece_ids: List[int]) -> List[str]:
        return [self._get_symbol_for_id(pid) for pid in next_piece_ids]

    def _get_all_rotations_text_representations(self) -> str:
        if self.active_tetromino is None or self.active_tetromino_original_idx is None:
            return "Rotations: (No active piece)\n"

        original_tetromino_def = self.TETROMINOES[self.active_tetromino_original_idx]
        # We need to use the offset ID for the piece on the board, but the original matrix for rotation logic
        current_piece_board_id = self.active_tetromino.id 

        # Determine max unique rotations (I,S,Z have 2; O has 1; T,L,J have 4)
        # This logic mirrors tetris_modules.py's TetrisPerceptionModule
        # Assuming original_tetromino_def.id is 0 for I, 1 for O, etc.
        og_id = original_tetromino_def.id
        max_rotations = 1
        if og_id == 0: # I
            max_rotations = 2
        elif og_id == 1: # O
            max_rotations = 1
        elif og_id == 2: # T
            max_rotations = 4
        elif og_id == 3: # S
            max_rotations = 2
        elif og_id == 4: # Z
            max_rotations = 2
        elif og_id == 5: # J
            max_rotations = 4
        elif og_id == 6: # L
            max_rotations = 4

        rotations_text = "Potential Clockwise Rotations (from canonical base, at current x,y - if valid):\n"
        temp_piece_matrix = original_tetromino_def.matrix.copy()

        for r_idx in range(max_rotations):
            if r_idx == 0:
                rotations_text += f"Rotation 0 (Canonical Base):\n"
            else:
                rotations_text += f"Rotation {r_idx} (Base + {r_idx} clockwise):\n"
            
            # Create a temporary Tetromino object for collision checking with the correct ID
            # The matrix needs to be non-offset for rotation, then values changed to board ID
            rotated_matrix_with_board_id = temp_piece_matrix.copy()
            rotated_matrix_with_board_id[rotated_matrix_with_board_id > 0] = current_piece_board_id
            temp_tetromino_for_collision = Tetromino(id_val=current_piece_board_id, 
                                                   color_rgb=original_tetromino_def.color_rgb, # Color doesn't matter for text
                                                   matrix=rotated_matrix_with_board_id)

            if not self._collision(temp_tetromino_for_collision, self.x, self.y):
                # Create a temporary board and place the piece
                temp_board = self.board.copy()
                t_h, t_w = temp_tetromino_for_collision.matrix.shape
                for row_idx in range(t_h):
                    for col_idx in range(t_w):
                        if temp_tetromino_for_collision.matrix[row_idx, col_idx] != 0:
                            board_r, board_c = self.y + row_idx, self.x + col_idx
                            if 0 <= board_r < self.height_padded and 0 <= board_c < self.width_padded:
                                temp_board[board_r, board_c] = temp_tetromino_for_collision.matrix[row_idx, col_idx]
                rotations_text += self._get_board_text_representation(temp_board) + "\n"
            else:
                rotations_text += "(Collision at current x,y)\n"
            
            if r_idx < max_rotations -1: # Don't rotate beyond the last needed view
                 # Rotate the original definition's matrix (which has 1s)
                temp_piece_matrix = np.rot90(temp_piece_matrix, k=-1) # k=-1 for clockwise
        
        return rotations_text

    def reset(self, *, seed: Optional[int]=None, options: Optional[Dict[str,Any]]=None, max_memory: Optional[int] = 10, episode_id:int=1) -> Tuple[Observation, Dict[str,Any]]:
        super().reset(seed=seed)
        if seed is not None: self.rng = np.random.default_rng(seed)
        
        self._fill_tetromino_bag() 
        self.piece_queue = []
        self._fill_piece_queue()
        
        self.board = self._create_board()
        self.game_over=False;
        self.current_score=0.0; 
        self.total_perf_score_episode = 0.0 # NEW: Reset total perf score for the episode
        self.lines_cleared_total=0; self.level=1; self.x=0; self.y=0
        self.active_tetromino_original_idx = None
        if not self._spawn_tetromino(): self.game_over = True
        
        self.adapter.reset_episode(episode_id)
        obs_dict_for_adapter = self._get_obs() 
        self.current_info_dict = self._get_info() 
        
        initial_step_perf_score = self.adapter.calculate_perf_score(0.0, self.current_info_dict) # Perf score for the reset state (usually 0)
        
        img_path, txt_rep = None, None
        if self.adapter.observation_mode in ["vision", "both"]:
            img_path = self.adapter._create_agent_observation_path(self.adapter.current_episode_id, self.adapter.current_step_num)
            create_board_image_tetris(
                board=obs_dict_for_adapter["board"], 
                save_path=img_path,
                pixel_color_mapping=self.pixel_id_to_color_map, 
                all_tetromino_objects=self.tetrominoes,
                score=int(self.current_score), 
                lines=self.lines_cleared_total,
                level=self.level, 
                next_pieces_ids=self.current_info_dict["next_piece_ids"], 
                perf_score=initial_step_perf_score
            )
        if self.adapter.observation_mode in ["text", "both"]:
            board_text_rep = self._get_board_text_representation(obs_dict_for_adapter["board"])
            next_symbols = self._get_next_pieces_symbols_list(self.current_info_dict['next_piece_ids'])
            stats_str = f"PerfS:{initial_step_perf_score:.0f} S:{self.current_score:.0f} L:{self.lines_cleared_total} Lv:{self.level}"
            rotations_viz = self._get_all_rotations_text_representations()
            txt_rep = (
                f"Board:\n{board_text_rep}\n"
                f"( '.' = empty, {''.join(self.TETROMINO_SYMBOLS)} = tetrominoes. Active piece is rendered on board.)\n"
                f"Next Pieces: {','.join(next_symbols)}\n"
                f"Game Stats: {stats_str}\n"
                f"\n{rotations_viz}"
                f"^ These show how your piece would look after rotating right to that orientation. "
                f"Use 'rotate_right' or 'rotate_left' actions to achieve a rotation.\n"
            )
        
        agent_obs = self.adapter.create_agent_observation(img_path, txt_rep, max_memory=max_memory)
        if self.render_mode == "human": self.render()
        return agent_obs, self.current_info_dict

    def step(
        self,
        agent_action_str: Optional[str],
        thought_process: str = "",
        time_taken_s: float = 0.0,
    ) -> Tuple[Observation, SupportsFloat, bool, bool, Dict[str, Any], float]:
        """
        Executes a *sequence* of actions in one call, where every action may
        optionally specify how many physics frames it lasts.

        Examples accepted
        -----------------
        "(move_left,2)(rotate_clockwise,1)(hard_drop,1)"
        "move_left,2; rotate_clockwise,1 ; hard_drop"
        "soft_drop"
        """

        # ---------------------------------------------------------------
        # 1)  Parse the incoming string → [(action_name, repeat_frames)]
        # ---------------------------------------------------------------
        actions_sequence: List[Tuple[str, int]] = []

        if agent_action_str:
            # Matches:  (optional_quote? action_name optional_quote?,  optional_int)?
            token_pattern = r"(\w+(?:_\w+)*)(?:\s*,\s*(\d+))?"
            for name, cnt in re.findall(token_pattern, agent_action_str):
                repeat = int(cnt) if cnt else 1
                repeat = max(1, repeat)
                actions_sequence.append((name.lower(), repeat))

        # Fallback → a single NO‑OP (so we still advance the adapter timers)
        if not actions_sequence:
            actions_sequence.append(("noop", 1))

        # ---------------------------------------------------------------
        # 2)  Execute every (action, frames) pair
        # ---------------------------------------------------------------
        total_reward: float = 0.0
        total_perf:   float = 0.0
        terminated = truncated = False
        last_obs: Optional[Observation] = None
        # to propagate user‑supplied time_taken_s
        first_outer = True 
        executed_any_action = False  # Track if we executed any actions

        for action_name, frames in actions_sequence:
            env_action_idx = self.adapter.map_agent_action_to_env_action(action_name)
            if env_action_idx is None:
                env_action_idx = self.ACTION_NO_OP
                continue

            executed_any_action = True

            for frame_idx in range(frames):
                self.adapter.increment_step()
                action_idx = env_action_idx

                current_step_reward   = 0.0
                perf_increment_step   = 0        # NEW
                terminated_flag       = self.game_over
                truncated_flag        = False

                if not terminated_flag:
                    if action_idx == self.ACTION_MOVE_LEFT:
                        if self.active_tetromino and not self._collision(
                            self.active_tetromino, self.x - 1, self.y
                        ):
                            self.x -= 1
                    elif action_idx == self.ACTION_MOVE_RIGHT:
                        if self.active_tetromino and not self._collision(
                            self.active_tetromino, self.x + 1, self.y
                        ):
                            self.x += 1
                    elif action_idx == self.ACTION_ROTATE_CLOCKWISE:
                        if self.active_tetromino:
                            rotated = self._rotate_tetromino(
                                self.active_tetromino, True
                            )
                            if not self._collision(rotated, self.x, self.y):
                                self.active_tetromino = rotated
                    elif action_idx == self.ACTION_ROTATE_COUNTERCLOCKWISE:
                        if self.active_tetromino:
                            rotated = self._rotate_tetromino(
                                self.active_tetromino, False
                            )
                            if not self._collision(rotated, self.x, self.y):
                                self.active_tetromino = rotated
                    elif action_idx == self.ACTION_SOFT_DROP:
                        if self.active_tetromino and not self._collision(
                            self.active_tetromino, self.x, self.y + 1
                        ):
                            self.y += 1
                        elif self.active_tetromino:
                            reward, perf = self._commit_active_tetromino()
                            current_step_reward += reward
                            perf_increment_step += perf
                    elif action_idx == self.ACTION_HARD_DROP:
                        if self.active_tetromino:
                            while not self._collision(self.active_tetromino, self.x, self.y + 1):
                                self.y += 1
                            # Only commit once after dropping
                            if self.active_tetromino is not None:
                                reward, perf = self._commit_active_tetromino()
                                current_step_reward += reward
                                perf_increment_step += perf

                    # gravity
                    if (
                        self.gravity_enabled
                        and self.active_tetromino
                        and action_idx != self.ACTION_HARD_DROP
                        and not (
                            action_idx == self.ACTION_SOFT_DROP
                            and self.active_tetromino is None
                        )
                    ):
                        if not self._collision(
                            self.active_tetromino, self.x, self.y + 1
                        ):
                            self.y += 1
                        else:
                            if self.active_tetromino is not None:
                                reward, perf = self._commit_active_tetromino()
                                current_step_reward += reward
                                perf_increment_step += perf

                terminated_flag = self.game_over
                obs_dict = self._get_obs()
                self.current_info_dict = self._get_info()

                current_step_perf = self.adapter.calculate_perf_score(
                    current_step_reward, self.current_info_dict
                )
                self.total_perf_score_episode += current_step_perf
                self.current_info_dict = self._get_info()

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
                        perf_score=current_step_perf,
                    )
                if self.adapter.observation_mode in ["text", "both"]:
                    board_text_rep = self._get_board_text_representation(obs_dict["board"])
                    next_symbols = self._get_next_pieces_symbols_list(self.current_info_dict['next_piece_ids'])
                    stats_str = f"PerfS:{current_step_perf:.0f} S:{self.current_score:.0f} L:{self.lines_cleared_total} Lv:{self.level}"
                    rotations_viz = self._get_all_rotations_text_representations()
                    txt_rep = (
                        f"Board:\n{board_text_rep}\n"
                        f"( '.' = empty, {''.join(self.TETROMINO_SYMBOLS)} = tetrominoes. Active piece is rendered on board.)\n"
                        f"Next Pieces: {','.join(next_symbols)}\n"
                        f"Game Stats: {stats_str}\n"
                        f"\n{rotations_viz}"
                        f"^ These show how your piece would look after rotating right to that orientation. "
                        f"Use 'rotate_right' or 'rotate_left' actions to achieve a rotation.\n"
                    )
                agent_obs = self.adapter.create_agent_observation(img_path, txt_rep)

                final_term, final_trunc = self.adapter.verify_termination(
                    agent_obs, terminated_flag, truncated_flag
                )
                self.adapter.log_step_data(
                    agent_action_str=action_name,
                    thought_process=thought_process,
                    reward=current_step_reward,
                    info=self.current_info_dict,
                    terminated=final_term,
                    truncated=final_trunc,
                    time_taken_s=time_taken_s if first_outer and frame_idx == 0 else 0.0,
                    perf_score=current_step_perf,
                    agent_observation=agent_obs,
                )
                if self.render_mode == "human":
                    self.render()

                # accumulate & maybe early‑exit
                total_reward += current_step_reward
                total_perf += current_step_perf # our customized performance metrics
                last_obs = agent_obs
                terminated, truncated = final_term, final_trunc
                if terminated or truncated:
                    break

            first_outer = False
            if terminated or truncated:
                break

        # If no valid actions were executed, run one NO_OP
        # Here current_step_perf is 0.0, it should be perf_increment_step but here we use current_step_perf instead
        if not executed_any_action:
            env_action_idx = self.ACTION_NO_OP
            self.adapter.increment_step()
            # Execute the NO_OP logic (which is essentially do nothing, just update observations)
            obs_dict = self._get_obs()
            self.current_info_dict = self._get_info()
            
            current_step_perf = self.adapter.calculate_perf_score(0.0, self.current_info_dict)
            self.total_perf_score_episode += current_step_perf
            self.current_info_dict = self._get_info()

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
                    perf_score=current_step_perf,
                )
            if self.adapter.observation_mode in ["text", "both"]:
                board_text_rep = self._get_board_text_representation(obs_dict["board"])
                next_symbols = self._get_next_pieces_symbols_list(self.current_info_dict['next_piece_ids'])
                stats_str = f"PerfS:{current_step_perf:.0f} S:{self.current_score:.0f} L:{self.lines_cleared_total} Lv:{self.level}"
                rotations_viz = self._get_all_rotations_text_representations()
                txt_rep = (
                    f"Board:\n{board_text_rep}\n"
                    f"( '.' = empty, {''.join(self.TETROMINO_SYMBOLS)} = tetrominoes. Active piece is rendered on board.)\n"
                    f"Next Pieces: {','.join(next_symbols)}\n"
                    f"Game Stats: {stats_str}\n"
                    f"\n{rotations_viz}"
                    f"^ These show how your piece would look after rotating right to that orientation. "
                    f"Use 'rotate_right' or 'rotate_left' actions to achieve a rotation.\n"
                )
            last_obs = self.adapter.create_agent_observation(img_path, txt_rep)

            final_term, final_trunc = self.adapter.verify_termination(
                last_obs, self.game_over, False
            )
            self.adapter.log_step_data(
                agent_action_str="no_op",
                thought_process=thought_process,
                reward=0.0,
                info=self.current_info_dict,
                terminated=final_term,
                truncated=final_trunc,
                time_taken_s=time_taken_s,
                perf_score=current_step_perf,
                agent_observation=last_obs,
            )
            terminated, truncated = final_term, final_trunc

        return (
            last_obs,
            total_reward,
            terminated,
            truncated,
            self.current_info_dict,
            total_perf,
        )

    def render(self) -> Optional[RenderFrame]:
        if self.render_mode == "ansi":
            board_render_ansi = self._get_raw_board_obs_for_render()
            cropped_ansi = board_render_ansi[0:self.height, self.padding:self.padding+self.width]
            char_field_ansi = np.full(cropped_ansi.shape, ".", dtype=str)
            for val_id in np.unique(cropped_ansi):
                 if val_id == self.base_pixels[0].value: char_field_ansi[cropped_ansi == val_id] = ' '
                 elif val_id == self.base_pixels[1].value: char_field_ansi[cropped_ansi == val_id] = '#'
                 elif val_id != 0: char_field_ansi[cropped_ansi == val_id] = str(val_id % 10) 
            ansi_str_out = "\n".join("|" + "".join(r) + "|" for r in char_field_ansi)
            border_line_out = "+" + "-" * self.width + "+"
            info_dict_ansi = self.current_info_dict # Use up-to-date info
            # Display total accumulated perf score for ANSI as well
            score_to_display_ansi = info_dict_ansi.get("total_perf_score_episode", 0.0) 
            lines_ansi = info_dict_ansi.get("lines", 0)
            level_ansi = info_dict_ansi.get("level", 1)
            next_ids_ansi = info_dict_ansi.get("next_piece_ids", [])
            next_disp_ansi = [pid % 10 if pid !=0 and pid != self.base_pixels[0].value else '-' for pid in next_ids_ansi[:3]]
            info_str_out = f"PerfS:{score_to_display_ansi:.0f} L:{lines_ansi} Lv:{level_ansi} N:{next_disp_ansi}"
            return f"{info_str_out}\n{border_line_out}\n{ansi_str_out}\n{border_line_out}"
        
        board_ids_to_render = self.current_raw_obs_dict.get("board", self._get_raw_board_obs_for_render())
        
        # Use the NumPy lookup array for efficient color mapping of the main board
        # Ensure board_ids_to_render contains IDs within the bounds of colors_lookup_array.
        # If an ID is too large, it will cause an IndexError.
        # We can clip IDs or ensure they are always valid.
        # For safety, clip to max valid index if necessary, though ideally not needed.
        clipped_board_ids = np.clip(board_ids_to_render, 0, self.colors_lookup_array.shape[0] - 1)
        rgb_image_render_padded = self.colors_lookup_array[clipped_board_ids] # This is (H_pad, W_pad, 3) and RGB
        
        # Crop to game area. rgb_image_render_padded is already RGB.
        display_img_render = rgb_image_render_padded[0:self.height, self.padding:self.padding+self.width].copy() # Use .copy() to ensure it's writeable for cv2 overlays

        if self.render_scaling_factor > 0:
            target_width = self.width * self.render_scaling_factor
            target_height = self.height * self.render_scaling_factor
            if target_width > 0 and target_height > 0:
                display_img_render = cv2.resize(display_img_render, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
        
        if self.render_mode == "human" or self.render_mode == "rgb_array":
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5 * max(1, self.render_scaling_factor / 25)
            text_color_cv = (220, 220, 220) 
            line_type=1
            text_y_offset = int(20 * max(1, self.render_scaling_factor / 25))
            current_text_y_cv = text_y_offset
            
            mini_shape_cell_size = 4 # Define it here, as it was for held piece display

            info_dict_cv = self.current_info_dict
            score_to_display_cv = info_dict_cv.get("total_perf_score_episode", 0.0)
            lines_cv = info_dict_cv.get("lines", 0)
            level_cv = info_dict_cv.get("level", 1)
            cv2.putText(display_img_render,f"PerfS:{score_to_display_cv:.0f} L:{lines_cv} Lv:{level_cv}",(10,current_text_y_cv),font,font_scale,text_color_cv,line_type)
            current_text_y_cv += text_y_offset

            # --- Render Next Pieces Shapes ---
            cv2.putText(display_img_render, "Next:", (10, current_text_y_cv), font, font_scale, text_color_cv, line_type)
            next_ids_cv = info_dict_cv.get("next_piece_ids", [])
            shape_area_start_y_next = current_text_y_cv + 5

            for i, piece_id_cv in enumerate(next_ids_cv[:3]): # Show up to 3 next pieces
                display_y_offset_for_this_piece = shape_area_start_y_next + i * (mini_shape_cell_size * 4 + 10) # Vertical spacing for each next piece
                if piece_id_cv != self.base_pixels[0].value:
                    next_tetromino_obj = next((t for t in self.tetrominoes if t.id == piece_id_cv), None)
                    if next_tetromino_obj:
                        color_rgb_list = next_tetromino_obj.color_rgb # This is [R, G, B]
                        # Instead of converting to BGR here, pass RGB tuple directly
                        # bgr_color = (color_rgb_list[2], color_rgb_list[1], color_rgb_list[0]) 
                        rgb_color_tuple = tuple(color_rgb_list) # Use RGB: (R, G, B)

                        matrix = next_tetromino_obj.matrix
                        shape_matrix = (matrix > 0).astype(np.uint8)
                        for r_idx, row in enumerate(shape_matrix):
                            for c_idx, cell in enumerate(row):
                                if cell == 1:
                                    x0 = 10 + c_idx * mini_shape_cell_size
                                    y0 = display_y_offset_for_this_piece + r_idx * mini_shape_cell_size
                                    # Draw with the RGB tuple
                                    cv2.rectangle(display_img_render, (x0, y0), (x0 + mini_shape_cell_size, y0 + mini_shape_cell_size), rgb_color_tuple, -1)
                    else:
                         cv2.putText(display_img_render, "?", (10 + mini_shape_cell_size, display_y_offset_for_this_piece + mini_shape_cell_size), font, font_scale*0.8, text_color_cv, line_type)
                else:
                    cv2.putText(display_img_render, "-", (10 + mini_shape_cell_size, display_y_offset_for_this_piece + mini_shape_cell_size), font, font_scale*0.8, text_color_cv, line_type)
            # current_text_y_cv += text_y_offset # This might not be needed or adjusted based on next piece area

        if self.render_mode == "rgb_array": 
            return display_img_render 
        elif self.render_mode == "human":
            if self.window_name is None: self.window_name="TetrisEnv Human"; cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            bgr_display_for_human = cv2.cvtColor(display_img_render, cv2.COLOR_RGB2BGR)
            cv2.imshow(self.window_name, bgr_display_for_human) 
            cv2.waitKey(1)
        return None

    def close(self):
        if self.window_name: cv2.destroyWindow(self.window_name); self.window_name=None
        self.adapter.close_log_file()
        print("[TetrisEnv] Self-contained env closed.") 