from gymnasium import ObservationWrapper, RewardWrapper
from collections import OrderedDict
from gymnasium.spaces import Box

import gymnasium as gym
import numpy as np

# Have to use these because the special types have hardcoded ids in the environment.
COLOURLESS_SPECIALS = {"cookie": -1}
COLOUR_SPECIALS = {"vertical_laser": 2, "horizontal_laser": 3, "bomb": 4}

# First num_colours slices are for colour. Absence in these slices means colourless.
# Then the next 1 is for ordinary type. 
# Then the next num_colourless_special slices are for colourless specials. 
# Finally the last num_colour_specials slices are for colour specials.
class OneHotWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.num_colours = self.unwrapped.num_colours
        self.num_colour_specials = self.unwrapped.num_colour_specials
        self.num_colourless_specials = self.unwrapped.num_colourless_specials
        self.num_rows = self.unwrapped.num_rows
        self.num_cols = self.unwrapped.num_cols
        self.board_obs_space = Box(low=0, high=1, dtype=np.int32, shape = (self.num_colours + self.num_colour_specials + self.num_colourless_specials, self.num_rows, self.num_cols))

        self.observation_space = gym.spaces.Dict({
            "board": self.board_obs_space,
            "num_moves_left": self.unwrapped._moves_left_observation_space
        })

        self.colour_specials = self.unwrapped.colour_specials
        self.colourless_specials =   self.unwrapped.colourless_specials

        self.global_num_colourless_specials = len(COLOURLESS_SPECIALS)
        self.global_num_colour_specials = len(COLOUR_SPECIALS)

        self._global_specials = {**COLOURLESS_SPECIALS, **COLOUR_SPECIALS}
        
        self.type_slices = [] # Don't track ordinary type
        for special, idx in self._global_specials.items():
            if special in self.colour_specials or special in self.colourless_specials:
                self.type_slices.append(idx)

        self.type_slices = np.array(sorted(self.type_slices)) + self.global_num_colourless_specials
        self.num_type_slices = len(self.type_slices)

    def observation(self, obs) -> dict:
        board = obs["board"]
        ohe_board = self._one_hot_encode_board(board)
        return OrderedDict([("board", ohe_board), ("num_moves_left", obs["num_moves_left"])])
    
    
    def _one_hot_encode_board(self, board: np.ndarray) -> np.ndarray:
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
        
class ProportionRewardWrapper(RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.flat_size = self.unwrapped.num_rows * self.unwrapped.num_cols
    
    def reward(self, reward: float):
        return reward / self.flat_size
    


if __name__=="__main__":
    import gymnasium as gym
    import tile_match_gym
    env = gym.make("TileMatch-v0", num_rows=5, num_cols=4, num_colours=2, num_moves = 10, colour_specials=["vertical_laser", "horizontal_laser", "bomb"], colourless_specials=["cookie"], seed=2)
    env = OneHotWrapper(ProportionRewardWrapper(env))

    obs, _ = env.reset()
