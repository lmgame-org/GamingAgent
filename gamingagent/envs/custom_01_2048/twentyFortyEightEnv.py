# Credits to https://github.com/Quentin18/gymnasium-2048/tree/main for the original 2048 game implementation.
# We thank the author for their work, which serves as an excellent testbed for our agent.

from typing import Any, Dict, Tuple, Optional, List

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.core import ActType, ObsType, RenderFrame, SupportsFloat

# Import the adapter and Observation dataclass
from gamingagent.envs.gym_env_adapter import GymEnvAdapter
from gamingagent.modules.core_module import Observation
from gamingagent.envs.env_utils import create_board_image_2048 # Ensure this is imported

WINDOW_WIDTH = 400
WINDOW_HEIGHT = 400
WINDOW_SCORE_HEIGHT = 60
WINDOW_BG_COLOR = (250, 248, 238)

BOARD_PADDING = 20
BOARD_BG_COLOR = (186, 172, 160)
TILE_PADDING = 5
TILE_COLOR_MAP = {
    0: (204, 193, 178),
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46),
}
TILE_COLOR_DEFAULT = (60, 58, 50)
BORDER_RADIUS = 4

FONT_NAME = "Comic Sans MS"
FONT_DARK_COLOR = (119, 110, 101)
FONT_LIGHT_COLOR = (249, 246, 242)
FONT_SCORE_COLOR = (0, 0, 0)
FONT_SIZE = 40


class TwentyFortyEightEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        render_mode: str | None = None,
        size: int = 4, # Default from game_env_config.json
        max_pow: int = 16, # Default from game_env_config.json
        # Adapter related parameters (will be passed by the runner)
        game_name_for_adapter: str = "twenty_forty_eight",
        observation_mode_for_adapter: str = "vision",
        agent_cache_dir_for_adapter: str = "cache/twenty_forty_eight/default_run",
        game_specific_config_path_for_adapter: str = "gamingagent/envs/gym_01_2048/game_env_config.json",
        max_stuck_steps_for_adapter: Optional[int] = 10,
    ) -> None:
        assert size >= 2, "size must be greater of equal than 2"

        # Standard Gym Observation Space (raw board powers)
        # The adapter will convert this to agent's Observation object
        self.observation_space = spaces.Box(
            low=0,
            high=max_pow -1, # Powers, so if max_pow is 16 (for 2^15), values go 0-15
            shape=(size, size),
            dtype=np.uint8,
        )

        self.action_space = spaces.Discrete(4)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None
        self.font = None
        
        self.board_size = size # Store board size

        # Initialize the adapter
        self.adapter = GymEnvAdapter(
            game_name=game_name_for_adapter,
            observation_mode=observation_mode_for_adapter,
            agent_cache_dir=agent_cache_dir_for_adapter,
            game_specific_config_path=game_specific_config_path_for_adapter,
            max_steps_for_stuck=max_stuck_steps_for_adapter
        )
        self.current_raw_board: Optional[np.ndarray] = None
        self.current_info_dict: Dict[str, Any] = {}

    # _get_obs from original TwentyFortyEightEnv returns one-hot encoded. 
    # We want to return the raw board powers for the adapter.
    def _get_raw_board_obs(self) -> np.ndarray:
        return self.board.copy() # Return copy of the board (powers)

    def _get_info(self) -> dict[str, Any]:
        return {
            "board": self.board.copy(), # board with powers
            "step_score": self.step_score,
            "total_score": self.total_score,
            "max_tile_power": np.max(self.board),
            "is_legal_move": self.is_legal_move,
            "illegal_move_count": self.illegal_move_count,
        }

    def _spawn_tile(self) -> None:
        rows, cols = np.where(self.board == 0)
        if not len(rows):
            return # No space to spawn
        index = self.np_random.choice(len(rows))
        # value is power of 2 (1 for 2^1=2, 2 for 2^2=4)
        value = 1 if self.np_random.random() > 0.1 else 2 
        self.board[rows[index], cols[index]] = value

    def reset(
        self,
        *, 
        seed: int | None = None,
        options: dict[str, Any] | None = None, # Standard gym signature
        # Custom args for runner compatibility
        max_memory: Optional[int] = 10,
        episode_id: int = 1 
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)

        self.board = np.zeros(
            (self.board_size, self.board_size),
            dtype=np.uint8,
        )
        self.step_score = 0
        self.total_score = 0
        self.is_legal_move = True
        self.illegal_move_count = 0

        self._spawn_tile()
        self._spawn_tile()

        self.adapter.reset_episode(episode_id)
        self.current_raw_board = self._get_raw_board_obs()
        self.current_info_dict = self._get_info()

        # Prepare observation components for the adapter
        img_path_for_adapter = None
        text_representation_for_adapter = None
        initial_perf_score = self.adapter.calculate_perf_score(0, self.current_info_dict) # Score for initial image

        if self.adapter.observation_mode in ["vision", "both"]:
            img_path_for_adapter = self.adapter._create_agent_observation_path(
                self.adapter.current_episode_id, self.adapter.current_step_num
            )
            create_board_image_2048(self.current_raw_board, img_path_for_adapter, perf_score=initial_perf_score)
        
        if self.adapter.observation_mode in ["text", "both"]:
            # if isinstance(self.current_raw_board, list):
            #     text_representation_for_adapter = str(self.current_raw_board)
            # elif hasattr(self.current_raw_board, 'tolist'): # For numpy arrays
            #     text_representation_for_adapter = str(self.current_raw_board.tolist())
            # else:
            #     text_representation_for_adapter = str(self.current_raw_board)
            board = {}
            board['board'] = [[2 ** entry if entry != 0 else 0 for entry in row] for row in self.current_raw_board]
            board['highest_tile'] = np.max(board['board'])
            board['analysis'] = f"Board has {16 - np.count_nonzero(self.current_raw_board)} empty spaces"
            text_representation_for_adapter = str(board)

        agent_observation = self.adapter.create_agent_observation(
            img_path=img_path_for_adapter,
            text_representation=text_representation_for_adapter,
            max_memory=max_memory
        )

        if self.render_mode == "human":
            self._render_frame()

        return agent_observation, self.current_info_dict

    @staticmethod
    def _transpose(board: np.ndarray) -> np.ndarray:
        return np.transpose(board)

    @staticmethod
    def _reverse(board: np.ndarray) -> np.ndarray:
        return np.flipud(board)

    @staticmethod
    def _cover_up(board: np.ndarray) -> np.ndarray:
        cover_board = np.zeros_like(board)
        for col in range(board.shape[1]):
            up = 0
            for row in range(board.shape[0]):
                if board[row, col] != 0:
                    cover_board[up, col] = board[row, col]
                    up += 1
        return cover_board

    @staticmethod
    def _merge(board: np.ndarray) -> tuple[np.ndarray, int]:
        score = 0
        for row in range(1, board.shape[0]):
            for col in range(board.shape[1]):
                if board[row, col] != 0 and board[row, col] == board[row - 1, col]:
                    score += 2 ** (board[row, col] + 1)
                    board[row - 1, col] = board[row - 1, col] + 1
                    board[row, col] = 0
        return board, score

    @classmethod
    def _up(cls, board: np.ndarray) -> tuple[np.ndarray, int]:
        next_board = cls._cover_up(board)
        next_board, score = cls._merge(next_board)
        next_board = cls._cover_up(next_board)
        return next_board, score

    @classmethod
    def _right(cls, board: np.ndarray) -> tuple[np.ndarray, int]:
        next_board = cls._reverse(cls._transpose(board))
        next_board = cls._cover_up(next_board)
        next_board, score = cls._merge(next_board)
        next_board = cls._cover_up(next_board)
        next_board = cls._transpose(cls._reverse(next_board))
        return next_board, score

    @classmethod
    def _down(cls, board: np.ndarray) -> tuple[np.ndarray, int]:
        next_board = cls._reverse(board)
        next_board = cls._cover_up(next_board)
        next_board, score = cls._merge(next_board)
        next_board = cls._cover_up(next_board)
        next_board = cls._reverse(next_board)
        return next_board, score

    @classmethod
    def _left(cls, board: np.ndarray) -> tuple[np.ndarray, int]:
        next_board = cls._transpose(board)
        next_board = cls._cover_up(next_board)
        next_board, score = cls._merge(next_board)
        next_board = cls._cover_up(next_board)
        next_board = cls._transpose(next_board)
        return next_board, score

    @classmethod
    def apply_action(
        cls,
        board: np.ndarray,
        action: ActType, # This is the raw integer action for the gym env
    ) -> tuple[np.ndarray, int, bool]:
        action_func = (cls._up, cls._right, cls._down, cls._left)
        next_board, score = action_func[action](board.copy()) # Operate on a copy
        is_legal = not np.array_equal(board, next_board)
        return next_board, score, is_legal

    @staticmethod
    def is_terminated(board: np.ndarray) -> bool:
        if (board == 0).any(): return False
        for r in range(board.shape[0]):
            for c in range(board.shape[1]):
                if c + 1 < board.shape[1] and board[r, c] == board[r, c+1]: return False
                if r + 1 < board.shape[0] and board[r, c] == board[r+1, c]: return False
        return True

    def step(
        self,
        agent_action_str: Optional[str], 
        thought_process: str = "", 
        time_taken_s: float = 0.0
    ) -> tuple[Observation, SupportsFloat, bool, bool, dict[str, Any], float]: # Added perf_score to return
        
        self.adapter.increment_step()
        env_action_idx = self.adapter.map_agent_action_to_env_action(agent_action_str)

        reward = 0.0
        terminated = False
        truncated = False # Gymnasium standard, not typically used by 2048 logic itself
        self.is_legal_move = False # Assume illegal/skip unless a valid move is made
        self.step_score = 0

        if env_action_idx is not None and self.action_space.contains(env_action_idx):
            next_board_state, current_step_score, self.is_legal_move = self.apply_action(
                board=self.board, 
                action=env_action_idx
            )
            self.step_score = current_step_score
            self.total_score += self.step_score
            reward = float(self.step_score) # Reward for this step

            if self.is_legal_move:
                self.board = next_board_state
                self._spawn_tile()
            else:
                self.illegal_move_count += 1
            
            terminated = self.is_terminated(board=self.board)
        else:
            # Skipped or invalid action from agent
            print(f"[TwentyFortyEightEnv] Action '{agent_action_str}' is skip/invalid. Gym env not stepped.")
            # Keep current board, reward is 0, terminated state from previous step or check current
            terminated = self.is_terminated(board=self.board) 

        self.current_raw_board = self._get_raw_board_obs()
        self.current_info_dict = self._get_info()
        
        current_perf_score = self.adapter.calculate_perf_score(reward, self.current_info_dict)

        # Prepare observation components for the adapter
        img_path_for_adapter = None
        text_representation_for_adapter = None

        if self.adapter.observation_mode in ["vision", "both"]:
            img_path_for_adapter = self.adapter._create_agent_observation_path(
                self.adapter.current_episode_id, self.adapter.current_step_num
            )
            create_board_image_2048(self.current_raw_board, img_path_for_adapter, perf_score=current_perf_score)

        if self.adapter.observation_mode in ["text", "both"]:
            # if isinstance(self.current_raw_board, list):
            #     text_representation_for_adapter = str(self.current_raw_board)
            # elif hasattr(self.current_raw_board, 'tolist'): # For numpy arrays
            #     text_representation_for_adapter = str(self.current_raw_board.tolist())
            # else:
            #     text_representation_for_adapter = str(self.current_raw_board)
            board = {}
            board['board'] = [[2 ** entry if entry != 0 else 0 for entry in row] for row in self.current_raw_board]
            board['highest_tile'] = np.max(board['board'])
            board['analysis'] = f"Board has {16 - np.count_nonzero(self.current_raw_board)} empty spaces"
            text_representation_for_adapter = str(board)
        
        agent_observation = self.adapter.create_agent_observation(
            img_path=img_path_for_adapter,
            text_representation=text_representation_for_adapter
        )
        
        # Use adapter for final termination check (e.g., stuck detection)
        final_terminated, final_truncated = self.adapter.verify_termination(agent_observation, terminated, truncated)

        self.adapter.log_step_data(
            agent_action_str=agent_action_str,
            thought_process=thought_process,
            reward=reward,
            info=self.current_info_dict,
            terminated=final_terminated,
            truncated=final_truncated,
            time_taken_s=time_taken_s,
            perf_score=current_perf_score,
            agent_observation=agent_observation
        )

        if self.render_mode == "human":
            self._render_frame()

        return agent_observation, reward, final_terminated, final_truncated, self.current_info_dict, current_perf_score

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        if self.render_mode == "rgb_array":
            return self._render_frame()
        # Human rendering is handled in reset and step if mode is "human"
        return None

    def _get_value(self, row: int, col: int) -> int:
        return 2 ** self.board[row, col] if self.board[row, col] > 0 else 0

    @staticmethod
    def _get_background_color(value: int) -> tuple[int, int, int]:
        return TILE_COLOR_MAP.get(value, TILE_COLOR_DEFAULT)

    @staticmethod
    def _get_text_color(value: int) -> tuple[int, int, int]:
        return FONT_DARK_COLOR if value < 8 else FONT_LIGHT_COLOR

    def _draw_board(self, canvas: pygame.Surface) -> None:
        board_left = BOARD_PADDING
        board_top = BOARD_PADDING # Corrected from board_right
        board_width = WINDOW_WIDTH - 2 * BOARD_PADDING
        board_height = WINDOW_HEIGHT - 2 * BOARD_PADDING
        tile_width = (board_width - TILE_PADDING * (self.board_size +1) ) // self.board_size # Adjusted for padding between tiles
        tile_height = (board_height - TILE_PADDING * (self.board_size +1)) // self.board_size # Adjusted for padding between tiles
        
        pygame.draw.rect(
            surface=canvas,
            color=BOARD_BG_COLOR,
            rect=(board_left, board_top, board_width, board_height),
            border_radius=BORDER_RADIUS,
        )
        for row in range(self.board.shape[0]):
            for col in range(self.board.shape[1]):
                value = self._get_value(row=row, col=col)
                # Corrected rect calculation for tiles to include padding
                rect_x = board_left + TILE_PADDING * (col + 1) + col * tile_width
                rect_y = board_top + TILE_PADDING * (row + 1) + row * tile_height
                
                tile_rect = pygame.Rect(
                    rect_x,
                    rect_y,
                    tile_width,
                    tile_height,
                )
                pygame.draw.rect(
                    surface=canvas,
                    color=self._get_background_color(value=value),
                    rect=tile_rect,
                    border_radius=BORDER_RADIUS,
                )
                if value == 0:
                    continue
                
                # Adjust font size dynamically based on tile value length for better fit
                str_value = str(value)
                current_font_size = FONT_SIZE
                if len(str_value) > 2: current_font_size = int(FONT_SIZE * 0.7)
                if len(str_value) > 3: current_font_size = int(FONT_SIZE * 0.5)
                dynamic_font = pygame.font.SysFont(FONT_NAME, current_font_size)

                text_surface = dynamic_font.render(
                    str_value,
                    True,
                    self._get_text_color(value=value),
                )
                text_rect = text_surface.get_rect(center=tile_rect.center)
                canvas.blit(source=text_surface, dest=text_rect)

    def _draw_score(self, canvas: pygame.Surface) -> None:
        board_width = WINDOW_WIDTH - 2 * BOARD_PADDING
        # Ensure font is initialized for score drawing as well
        if self.font is None:
            pygame.font.init()
            self.font = pygame.font.SysFont(FONT_NAME, FONT_SIZE) 
            
        score_surface = self.font.render(
            f"Score: {self.total_score}",
            True,
            FONT_SCORE_COLOR,
        )
        score_height = self.font.get_height()
        score_rect = pygame.Rect(
            BOARD_PADDING,
            WINDOW_HEIGHT + (WINDOW_SCORE_HEIGHT - score_height) // 2,
            board_width,
            score_height,
        )
        canvas.blit(source=score_surface, dest=score_rect)

    def _render_frame(self) -> RenderFrame | list[RenderFrame]:
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (WINDOW_WIDTH, WINDOW_HEIGHT + WINDOW_SCORE_HEIGHT)
            )
            pygame.display.set_caption("2048")

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        if self.font is None: # Ensure font is initialized before drawing
            pygame.font.init()
            self.font = pygame.font.SysFont(FONT_NAME, FONT_SIZE)

        canvas = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT + WINDOW_SCORE_HEIGHT))
        canvas.fill(WINDOW_BG_COLOR)

        self._draw_board(canvas=canvas)
        self._draw_score(canvas=canvas)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
            return None # For human mode, render handles display, returns None
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self) -> None:
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
            self.font = None
        self.adapter.close_log_file()