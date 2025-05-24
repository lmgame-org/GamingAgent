"""Tetris environment for Gymnasium."""
import copy
from dataclasses import dataclass, fields
from typing import Any, List

import cv2
import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, RenderFrame
from gymnasium.spaces import Box, Discrete

from tetris_gymnasium.components.tetromino import Pixel, Tetromino
from tetris_gymnasium.components.tetromino_holder import TetrominoHolder
from tetris_gymnasium.components.tetromino_queue import TetrominoQueue
from tetris_gymnasium.components.tetromino_randomizer import BagRandomizer, Randomizer
from tetris_gymnasium.mappings.actions import ActionsMapping
from tetris_gymnasium.mappings.rewards import RewardsMapping


@dataclass
class TetrisState:
    """State of the Tetris environment."""

    board: np.ndarray
    active_tetromino: Tetromino
    x: int
    y: int
    queue: TetrominoQueue
    holder: TetrominoHolder
    randomizer: Randomizer
    has_swapped: bool
    game_over: bool
    score: int


class Tetris(gym.Env):
    """Tetris environment for Gymnasium."""

    metadata = {
        "render_modes": ["human", "rgb_array", "ansi"],
        "render_fps": 1,
        "render_human": True,
    }

    BASE_PIXELS = [Pixel(0, [0, 0, 0]), Pixel(1, [128, 128, 128])]  # Empty  # Bedrock

    TETROMINOES = [
        Tetromino(
            0,
            [0, 240, 240],
            np.array(
                [[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint8
            ),
        ),  # I
        Tetromino(1, [240, 240, 0], np.array([[1, 1], [1, 1]], dtype=np.uint8)),  # O
        Tetromino(
            2,
            [160, 0, 240],
            np.array([[0, 1, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8),
        ),  # T
        Tetromino(
            3, [0, 240, 0], np.array([[0, 1, 1], [1, 1, 0], [0, 0, 0]], dtype=np.uint8)
        ),  # S
        Tetromino(
            4, [240, 0, 0], np.array([[1, 1, 0], [0, 1, 1], [0, 0, 0]], dtype=np.uint8)
        ),  # Z
        Tetromino(
            5, [0, 0, 240], np.array([[1, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
        ),  # J
        Tetromino(
            6,
            [240, 160, 0],
            np.array([[0, 0, 1], [1, 1, 1], [0, 0, 0]], dtype=np.uint8),
        ),  # L
    ]

    def __init__(
        self,
        render_mode=None,
        width=10,
        height=20,
        gravity=True,
        actions_mapping=ActionsMapping(),
        rewards_mapping=RewardsMapping(),
        queue: TetrominoQueue = None,
        holder: TetrominoHolder = None,
        randomizer: Randomizer = None,
        base_pixels=None,
        tetrominoes=None,
        render_upscale: int = 10,
    ):
        """Creates a new Tetris environment.

        Args:
            render_mode: The mode to use for rendering. If None, no rendering will be done.
            width: The width of the board.
            height: The height of the board.
            gravity: Whether gravity is enabled in the game..
            actions_mapping: The mapping for the actions that the agent can take.
            rewards_mapping: The mapping for the rewards that the agent can receive.
            queue: The :class:`TetrominoQueue` to use for holding tetrominoes temporarily.
            holder: The :class:`TetrominoHolder` to use for storing tetrominoes.
            randomizer: The :class:`Randomizer` to use for selecting tetrominoes
            base_pixels: A list of base (non-Tetromino) :class:`Pixel` to use for the environment (e.g. empty, bedrock).
            tetrominoes: A list of :class:`Tetromino` to use in the environment.
            render_upscale: The factor to upscale the rendered board by.
        """
        # Dimensions
        self.game_over = False
        self.height: int = height
        self.width: int = width

        # Base Pixels
        if base_pixels is None:
            self.base_pixels = copy.deepcopy(self.BASE_PIXELS)

        # Tetrominoes
        if tetrominoes is None:
            tetrominoes = copy.deepcopy(self.TETROMINOES)
        self.tetrominoes = tetrominoes
        self.tetrominoes: List[Tetromino] = self.offset_tetromino_id(
            self.tetrominoes, len(self.base_pixels)
        )
        self.active_tetromino: Tetromino = None

        # Pixels
        self.pixels: List[Pixel] = self.parse_pixels(self.tetrominoes)

        # Padding
        self.padding: int = max(max(t.matrix.shape) for t in self.tetrominoes)
        self.width_padded: int = self.width + 2 * self.padding
        self.height_padded: int = self.height + self.padding

        # Board
        self.board = self.create_board()

        # Game engine
        # Reason for this kind of initialization: https://stackoverflow.com/q/41686829
        if randomizer is None:
            self.randomizer = BagRandomizer(len(self.tetrominoes))
        if queue is None:
            self.queue = TetrominoQueue(self.randomizer)
        if holder is None:
            self.holder = TetrominoHolder()
        self.has_swapped = False
        self.gravity_enabled = gravity

        # Position
        self.x: int = 0
        self.y: int = 0

        # Gymnasium
        self.observation_space = gym.spaces.Dict(
            {
                "board": Box(
                    low=0,
                    high=len(self.pixels),
                    shape=(self.height_padded, self.width_padded),
                    dtype=np.uint8,
                ),
                "active_tetromino_mask": Box(
                    low=0,
                    high=1,
                    shape=(self.height_padded, self.width_padded),
                    dtype=np.uint8,
                ),
                "holder": Box(
                    low=0,
                    high=len(self.pixels),
                    shape=(
                        self.padding,
                        self.padding * self.holder.size,
                    ),
                    dtype=np.uint8,
                ),
                "queue": gym.spaces.Box(
                    low=0,
                    high=len(self.pixels),
                    shape=(
                        self.padding,
                        self.padding * self.queue.size,
                    ),
                    dtype=np.uint8,
                ),
            }
        )

        # Mappings for rewards  & actions (readability in code)
        self.actions = actions_mapping
        self.rewards = rewards_mapping

        self.action_space = Discrete(len(fields(self.actions)))
        self.reward_range = (
            min(vars(self.rewards).values()),
            max(vars(self.rewards).values()),
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.render_scaling_factor = render_upscale
        self.window_name = None

    def step(self, action: ActType) -> "tuple[dict, float, bool, bool, dict]":
        """Perform one step of the environment's dynamics.

        Args:
            action: The action to be executed.

        Returns:
            observation: The observation of the current board as np array.
            reward: Amount of reward returned after previous action.
            done: Whether the episode has ended, in which case further step() calls will return undefined results.
            info: Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"

        truncated = False  # Tetris without levels will never truncate
        reward = 0
        lines_cleared = 0

        if action == self.actions.move_left:
            if not self.collision(self.active_tetromino, self.x - 1, self.y):
                self.x -= 1
        elif action == self.actions.move_right:
            if not self.collision(self.active_tetromino, self.x + 1, self.y):
                self.x += 1
        elif action == self.actions.move_down:
            if not self.collision(self.active_tetromino, self.x, self.y + 1):
                self.y += 1
        elif action == self.actions.rotate_clockwise:
            if not self.collision(
                self.rotate(self.active_tetromino, True), self.x, self.y
            ):
                self.active_tetromino = self.rotate(self.active_tetromino, True)
        elif action == self.actions.rotate_counterclockwise:
            if not self.collision(
                self.rotate(self.active_tetromino, False), self.x, self.y
            ):
                self.active_tetromino = self.rotate(self.active_tetromino, False)
        elif action == self.actions.swap:
            if not self.has_swapped:
                # Swap the active tetromino with the one in the holder (saves orientation)
                self.active_tetromino = self.holder.swap(self.active_tetromino)
                self.has_swapped = True
                if self.active_tetromino is None:
                    # If the holder is empty, spawn the next tetromino
                    # No need for collision check, as the holder is only empty at the start
                    self.spawn_tetromino()
                else:
                    self.reset_tetromino_position()
        elif action == self.actions.hard_drop:
            reward, self.game_over, lines_cleared = self.commit_active_tetromino()
        elif action == self.actions.no_op:
            pass

        # Gravity
        if self.gravity_enabled and action != self.actions.hard_drop:
            if not self.collision(self.active_tetromino, self.x, self.y + 1):
                self.y += 1
            else:
                # If there's no more room to move, lock in the tetromino
                reward, self.game_over, lines_cleared = self.commit_active_tetromino()

        return (
            self._get_obs(),
            reward,
            self.game_over,
            truncated,
            {"lines_cleared": lines_cleared},
        )

    def reset(
        self, *, seed: "int | None" = None, options: "dict[str, Any] | None" = None
    ) -> "tuple[dict[str, Any], dict[str, Any]]":
        """Resets the state of the environment.

        As with all Gymnasium environments, the reset method is called once at the beginning of an episode.

        Args:
            seed: The random seed to use for the reset.
            options: A dictionary of options to use for the reset.

        Returns: The initial observation of the space.
        """
        super().reset(seed=seed, options=options)

        # Initialize fresh board
        self.board = self.create_board()
        self.game_over = False

        # Reset the randomizer
        self.queue.reset(seed=seed)

        # Get the next tetromino and spawn it
        self.active_tetromino = self.tetrominoes[self.queue.get_next_tetromino()]
        self.reset_tetromino_position()

        # Holder
        self.holder.reset()
        self.has_swapped = False

        # Render
        self.window_name = None

        return self._get_obs(), self._get_info()

    def get_rgb(self, observation):
        """Observation wrapper that displays all observations (board, holder, queue) as one single RGB Image.

        The observation contains the board on the left, the queue on the top right and the holder on the bottom right.
        """
        # Board
        board_obs = observation["board"]
        # Holder
        holder_obs = observation["holder"]
        # Queue
        queue_obs = observation["queue"]

        max_size = holder_obs.shape[0]
        max_len = max(holder_obs.shape[1], queue_obs.shape[1])

        # make holder and queue same length by adding optional padding
        holder_obs = np.hstack(
            (holder_obs, np.ones((max_size, max_len - holder_obs.shape[1])))
        )
        queue_obs = np.hstack(
            (queue_obs, np.ones((max_size, max_len - queue_obs.shape[1])))
        )

        # add vertical padding between the board and the holder/queue
        v_padding = np.ones((board_obs.shape[0] - 2 * max_size, max_len))
        cnn_extra = np.vstack((queue_obs, v_padding, holder_obs))

        stack = np.hstack((board_obs, cnn_extra)).astype(np.integer)

        # Convert to RGB
        rgb = np.zeros((stack.shape[0], stack.shape[1], 3))
        colors = np.array(list(p.color_rgb for p in self.pixels), dtype=np.uint8)
        rgb[...] = colors[stack]

        return rgb.astype(np.uint8)

    def render(self) -> "RenderFrame | list[RenderFrame] | None":
        """Renders the environment in various formats.

        This render function is different from the default as it uses the values from :func:`observation`  to render
        the environment.
        """
        if self.render_mode == "ansi":
            # Render active tetromino (because it's not on self.board)
            projection = self.project_tetromino()

            # Crop padding away as we don't want to render it
            projection = self.crop_padding(projection)

            # Convert to string
            char_field = np.where(projection == 0, ".", projection.astype(str))
            field_str = "\n".join("".join(row) for row in char_field)
            return field_str

        matrix = self.get_rgb(self._get_obs())

        if self.render_mode == "human" or self.render_mode == "rgb_array":
            # Upscale the matrix for better visualization
            kernel = np.ones(
                (self.render_scaling_factor, self.render_scaling_factor, 1),
                dtype=np.uint8,
            )
            matrix = np.kron(matrix, kernel)

            if self.render_mode == "rgb_array":
                return matrix

            if self.render_mode == "human":
                if self.window_name is None:
                    self.window_name = "Tetris Gymnasium"
                    cv2.namedWindow(self.window_name, cv2.WINDOW_GUI_NORMAL)

                    h, w = (
                        matrix.shape[0],
                        matrix.shape[1],
                    )
                    cv2.resizeWindow(self.window_name, w, h)
                cv2.imshow(
                    self.window_name,
                    cv2.cvtColor(matrix, cv2.COLOR_RGB2BGR),
                )

        return None

    def spawn_tetromino(self) -> bool:
        """Spawns a new tetromino at the top of the board and checks for collision.

        Returns
            True if the tetromino can be successfully spawned, False otherwise.
        """
        self.active_tetromino = self.tetrominoes[self.queue.get_next_tetromino()]
        self.reset_tetromino_position()
        return not self.collision(self.active_tetromino, self.x, self.y)

    def place_active_tetromino(self):
        """Locks the active tetromino in place on the board."""
        self.board = self.project_tetromino()
        self.active_tetromino = None

    def collision(self, tetromino: Tetromino, x: int, y: int) -> bool:
        """Check if the tetromino collides with the board at the given position.

        A collision is detected if the tetromino overlaps with any non-zero cell on the board.
        These non-zero cells represent the padding / bedrock (value 1) or other tetrominoes (values >=2).

        Args:
            tetromino: The tetromino to check for collision.
            x: The x position of the tetromino to check collision for.
            y: The y position of the tetromino to check collision for.

        Returns:
            True if the tetromino collides with the board at the given position, False otherwise.
        """
        # Extract the part of the board that the tetromino would occupy.
        slices = self.get_tetromino_slices(tetromino, x, y)
        board_subsection = self.board[slices]

        # Check collision using numpy element-wise operations.
        return np.any(board_subsection[tetromino.matrix > 0] > 0)

    def rotate(self, tetromino: Tetromino, clockwise=True) -> Tetromino:
        """Rotate a tetromino by 90 degrees.

        Args:
            tetromino: The tetromino to rotate.
            clockwise: Whether to rotate the tetromino clockwise or counterclockwise.

        Returns:
            The rotated tetromino.
        """
        return Tetromino(
            tetromino.id,
            tetromino.color_rgb,
            np.rot90(tetromino.matrix, k=(1 if clockwise else -1)),
        )

    def drop_active_tetromino(self):
        """Drop the active tetromino to the lowest possible position on the board."""
        while not self.collision(self.active_tetromino, self.x, self.y + 1):
            self.y += 1

    def commit_active_tetromino(self):
        """Commit the active tetromino to the board.

        After locking in the tetromino, the game checks if any rows are filled and clears them.
        Finally, it spawns the next tetromino.

        Returns
            The reward for the current step and whether the game is over.
        """
        # 1. Drop the tetromino and lock it in place
        lines_cleared = 0
        if self.collision(self.active_tetromino, self.x, self.y):
            reward = self.rewards.game_over
            self.game_over = True
        else:
            self.drop_active_tetromino()
            self.place_active_tetromino()
            self.board, lines_cleared = self.clear_filled_rows(self.board)
            reward = self.score(lines_cleared)

            # 2. Spawn the next tetromino and check if the game continues
            self.game_over = not self.spawn_tetromino()
            reward += self.rewards.alife
            if self.game_over:
                reward = self.rewards.game_over

            # 3. Reset the swap flag (agent can swap once per tetromino)
            self.has_swapped = False

        return reward, self.game_over, lines_cleared

    def clear_filled_rows(self, board) -> "tuple(np.ndarray, int)":
        """Clear any filled rows on the board.

        The clearing is performed using numpy by indexing only the rows that are not filled and
        concatenating them with a new top part of the board that contains zeros.

        With this implementation, the clearing operation is efficient and does not require loops.

        Returns:
            The number of rows that were cleared.
        """
        # A row is filled if it doesn't contain any free space (0) and doesn't contain any bedrock / padding (1).
        filled_rows = (~(board == 0).any(axis=1)) & (~(board == 1).all(axis=1))
        n_filled = np.sum(filled_rows)

        if n_filled > 0:
            # Identify the rows that are not filled.
            unfilled_rows = board[~filled_rows]

            # Create a new top part of the board with free space (0) to compensate for the cleared rows.
            free_space = np.zeros((n_filled, self.width), dtype=np.uint8)
            free_space = np.pad(
                free_space,
                ((0, 0), (self.padding, self.padding)),
                mode="constant",
                constant_values=1,
            )

            # Concatenate the new top with the unfilled rows to form the updated board.
            board[:] = np.concatenate((free_space, unfilled_rows), axis=0)

        return board, n_filled

    def crop_padding(self, matrix: np.ndarray) -> np.ndarray:
        """Crop the padding from the given matrix.

        The Tetris board has padding on all sides except the top to simplify collision detection.
        This method crops the padding from the given matrix to return the actual board, which is useful for rendering.

        Returns
            The matrix with the padding cropped.
        """
        return matrix[0 : -self.padding, self.padding : -self.padding]

    def get_tetromino_slices(
        self, tetromino: Tetromino, x: int, y: int
    ) -> "tuple(slice, slice)":
        """Get the slices of the active tetromino on the board.

        Returns:
            The slices of the active tetromino on the board.
        """
        tetromino_height, tetromino_width = tetromino.matrix.shape
        return tuple((slice(y, y + tetromino_height), slice(x, x + tetromino_width)))

    def reset_tetromino_position(self) -> None:
        """Reset the x and y position of the active tetromino to the center of the board."""
        self.x, self.y = (
            self.width_padded // 2 - self.active_tetromino.matrix.shape[0] // 2,
            0,
        )

    def project_tetromino(
        self, tetromino: Tetromino = None, x: int = None, y: int = None
    ) -> np.ndarray:
        """Project the active tetromino on the board.

        By default, the active (moving) tetromino is not part of the board. This function projects the active tetromino
        on the board to render it.
        """
        if tetromino is None:
            tetromino = self.active_tetromino
        if x is None:
            x = self.x
        if y is None:
            y = self.y

        projection = self.board.copy()
        if self.collision(tetromino, x, y):
            return projection

        slices = self.get_tetromino_slices(tetromino, x, y)
        projection[slices] += tetromino.matrix
        return projection

    def _get_obs(self) -> "dict[str, Any]":
        """Return the current board as an observation."""
        # Include the active tetromino on the board for the observation.
        board_obs = self.project_tetromino()

        # Create a mask where the active tetromino is
        active_tetromino_slices = self.get_tetromino_slices(
            self.active_tetromino, self.x, self.y
        )
        active_tetromino_mask = np.zeros_like(board_obs)
        active_tetromino_mask[active_tetromino_slices] = 1

        # Holder
        max_size = self.padding
        holder_tetrominoes = self.holder.get_tetrominoes()
        if len(holder_tetrominoes) > 0:
            # Pad all tetrominoes to be the same size
            for index, t in enumerate(holder_tetrominoes):
                holder_tetrominoes[index] = np.pad(
                    t.matrix,
                    (
                        (0, max_size - t.matrix.shape[0]),
                        (0, max_size - t.matrix.shape[1]),
                    ),
                )
            # Concatenate all tetrominoes horizontally
            holder_obs = np.hstack(holder_tetrominoes)
        else:
            holder_obs = np.ones((max_size, max_size * self.holder.size))

        # Queue
        queue_tetrominoes = self.queue.get_queue()
        for index, t_id in enumerate(queue_tetrominoes):
            # Pad all tetrominoes to be the same size
            t = copy.deepcopy(self.tetrominoes[t_id])
            t.matrix = np.pad(
                t.matrix,
                ((0, max_size - t.matrix.shape[0]), (0, max_size - t.matrix.shape[1])),
            )
            # Safe padded result back to the array
            queue_tetrominoes[index] = t.matrix
        # Concatenate all tetrominoes horizontally
        queue_obs = np.hstack(queue_tetrominoes)

        return {
            "board": board_obs.astype(np.uint8),
            "active_tetromino_mask": active_tetromino_mask.astype(np.uint8),
            "holder": holder_obs.astype(np.uint8),
            "queue": queue_obs.astype(np.uint8),
        }

    def _get_info(self) -> dict:
        """Return the current game state as info."""
        return {"lines_cleared": 0}

    def score(self, rows_cleared) -> int:
        """Calculate the score based on the number of lines cleared.

        Args:
            rows_cleared: The number of lines cleared in the last step.

        Returns
            The score for the given number of lines cleared.
        """
        return (rows_cleared**2) * self.width

    def create_board(self) -> np.ndarray:
        """Create a new board with the given dimensions."""
        board = np.zeros((self.height, self.width), dtype=np.uint8)
        board = np.pad(
            board,
            ((0, self.padding), (self.padding, self.padding)),
            mode="constant",
            constant_values=1,
        )
        return board

    def parse_pixels(self, tetrominoes: "List[Tetromino]") -> "List[Pixel]":
        """Creates a list of pixels from the base pixels and the tetrominoes.

        Pixels are used to represent the board and the tetrominoes in the environment.

        Args:
            tetrominoes: The tetrominoes to add to the base pixels.

        Returns:
            The list of pixels for the environment.
        """
        return self.base_pixels + [
            Pixel(t.id + len(self.base_pixels), t.color_rgb) for t in tetrominoes
        ]

    def offset_tetromino_id(
        self, tetrominoes: "List[Tetromino]", offset: int
    ) -> "List[Tetromino]":
        """In order to make the tetominos distinguishable, each tetromino should have a unique value.

        The tetrominoes already possess a unique ID, but the matrix should also be updated to reflect this.
        Additionally, the tetrominoes should be offset by a certain value to avoid conflicts with the board.
        The board already contains a number of pixels which are not part of the tetrominoes (empty cells, bedrock).
        So, the tetrominoes should be offset by the number of pixels in the board that are  not tetrominoes.

        Args:
            tetrominoes: The tetrominoes to preprocess.
            offset: The offset to apply to the tetrominoes. This is usually the number of non-tetromino pixels in the board.

        Returns:
            The preprocessed tetrominoes (= id and matrix values offset by number of non-tetromino pixels).
        """
        for i in range(len(tetrominoes)):
            tetrominoes[i].id += offset
            tetrominoes[i].matrix = tetrominoes[i].matrix * (i + offset)

        return tetrominoes

    def set_state(self, state: TetrisState) -> None:
        """Restore the state of the environment. Should be used instead of deepcopy for performance."""
        self.board = state.board
        self.active_tetromino = state.active_tetromino
        self.x = state.x
        self.y = state.y
        self.queue = state.queue
        self.holder = state.holder
        self.randomizer = state.randomizer
        self.has_swapped = state.has_swapped
        self.game_over = state.game_over
        self.score = state.score

    def get_state(self) -> TetrisState:
        """Clone the current state of the environment. Should be used instead of deepcopy for performance."""
        randomizer = copy.copy(self.randomizer)
        return TetrisState(
            board=self.board.copy(),
            active_tetromino=copy.copy(self.active_tetromino),
            x=self.x,
            y=self.y,
            queue=self.queue.copy(randomizer),
            holder=copy.copy(self.holder),
            randomizer=randomizer,
            has_swapped=self.has_swapped,
            game_over=self.game_over,
            score=self.score,
        )
