import gymnasium as gym
import numpy as np

from gymnasium.spaces import Discrete, Box
from typing import Optional, List, Tuple, Union
from collections import OrderedDict

from tile_match_gym.board import Board
from tile_match_gym.board import is_move_effective

from tile_match_gym.renderer import Renderer


class TileMatchEnv(gym.Env):
    metadata = {"render_modes": ["string", "human", "rgb_array"], "render_fps": 2}

    def __init__(
            self,
            num_rows: int,
            num_cols: int,
            num_colours: int,
            num_moves: int,
            colourless_specials: List[str],
            colour_specials: List[str],
            seed: Optional[int] = 1,
            render_mode: str = "string"
    ) -> None:
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_colours = num_colours

        self.colourless_specials = colourless_specials
        self.colour_specials = colour_specials
        self.num_moves = num_moves

        self.renderer = None
        if render_mode == "string":
            self.colour_map = self.np_random.choice(range(105, 230), size=self.num_colours + 1, replace=False)
        elif render_mode in ["human", "rgb_array"]:
            self.renderer = Renderer(num_rows, num_cols, num_colours, num_moves, render_fps=self.metadata["render_fps"], render_mode=render_mode)


        self.render_mode = render_mode
        self.num_colour_specials = len(self.colour_specials)
        self.num_colourless_specials = len(self.colourless_specials)

        self.seed = seed

        np_random = np.random.default_rng(seed=seed)
        self.board = Board(num_rows, num_cols, num_colours, colourless_specials, colour_specials, np_random)
        self.np_random = self.board.np_random
        obs_low = np.array([np.zeros((self.num_rows, self.num_cols), dtype=np.int32),
                            np.full((self.num_rows, self.num_cols), - self.num_colourless_specials, dtype=np.int32)])
        obs_high = np.array([np.full((self.num_rows, self.num_cols), self.num_colours, dtype=np.int32),
                             np.full((self.num_rows, self.num_cols), self.num_colour_specials + 2,
                                     dtype=np.int32)])  # + 1 for empty

        self.num_actions = int((self.num_rows * self.num_cols * 2) - self.num_rows - self.num_cols)
        self._action_to_coords = self.board.action_to_coords
        self._board_observation_space = Box(
            low=obs_low,
            high=obs_high,
            shape=(2, self.num_rows, self.num_cols),
            dtype=np.int32,
            seed=self.seed)

        self._moves_left_observation_space = Discrete(self.num_moves + 1, seed=self.seed)

        self.observation_space = gym.spaces.Dict({
            "board": self._board_observation_space,
            "num_moves_left": self._moves_left_observation_space
        })

        self.last_board = None
        self.timer = None

        self.action_space = Discrete(self.num_actions, seed=self.seed)

    def set_seed(self, seed: int) -> None:
        self.action_space.seed = seed
        self.observation_space.seed = seed
        self.board.np_random = np.random.default_rng(seed=seed)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[dict, dict]:
        if seed is not None:
            self.set_seed(seed)
        self.board.generate_board()
        self.timer = 0
        obs = self._get_obs()
        info = {'effective_actions': self._get_effective_actions()}
        return obs, info

    def step(self, action: int) -> Tuple[dict, int, bool, bool, dict]:
        if self.timer is None or self.timer >= self.num_moves:
            raise Exception("You must call reset before calling step")

        coord1, coord2 = self._action_to_coords[action]
        num_eliminations, is_combination_match, num_new_specials, num_specials_activated, shuffled = self.board.move(coord1, coord2)

        self.timer += 1
        done = self.timer == self.num_moves
        effective_actions = self._get_effective_actions()
        info = {
            "is_combination_match": is_combination_match,
            "num_new_specials": num_new_specials,
            "num_specials_activated": num_specials_activated,
            "shuffled": shuffled,
            "effective_actions": effective_actions
        }
        next_obs = self._get_obs()

        return next_obs, num_eliminations, done, False, info

    def _get_obs(self) -> dict[str, Union[np.ndarray, int]]:
        return OrderedDict([("board", self.board.board), ("num_moves_left", self.num_moves - self.timer)])


    def _get_effective_actions(self) -> List[int]:
        if self.timer == self.num_moves:
            return []

        action_check = lambda a: is_move_effective(self.board.board, *self._action_to_coords[a])
        effective_actions = list(filter(action_check, range(self.num_actions)))
        return effective_actions

    def render(self) -> Union[None, np.ndarray]:
        if self.render_mode == "string":
            color = lambda id, c: "\033[48;5;16m" + f"\033[38;5;{self.colour_map[id]}m{c}\033[0m"
            height = self.board.board.shape[1]
            width = self.board.board.shape[2]

            print(" " + "-" * (width * 2 + 1))
            for row_num in range(height):
                print("| ", end="\033[48;5;16m")
                for col in range(width):
                    tile_colour = self.board.board[0, row_num, col]
                    tile_type = self.board.board[1, row_num, col]

                    print(color(tile_colour, tile_type), end="\033[48;5;16m ")
                    print("\033[0m", end="")

                print("|", end="\n")
            print(" " + "-" * (width * 2 + 1))

        else:
            return self.renderer.render(self.board.board, self.num_moves - self.timer)

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
