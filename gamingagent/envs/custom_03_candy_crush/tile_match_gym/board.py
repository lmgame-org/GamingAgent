import numpy as np
import numba
from numba import njit, types
from typing import Optional, List, Tuple


"""
tile_colours = {
    0: Colourless
    1: colour1,
    2: ...
}

empty_tile: [0, 0] if tile_

"""

TILE_TYPES = {
    "empty": 0,
    "normal": 1,
    "vertical_laser": 2,
    "horizontal_laser": 3,
    "bomb": 4,
    "cookie": -1,
}


numba_spec = [
    ('num_rows', types.int32),
    ('num_cols', types.int32),
    ('num_colours', types.int32),
    ('flat_size', types.int32),
    ('colourless_specials', types.ListType(types.string)),
    ('colour_specials', types.ListType(types.string)),
    ('specials', types.ListType(types.string)),
    ('np_random', numba.typeof(np.random.default_rng(0))),
    ('board', types.Optional(types.Array(types.int32, ndim=3, layout='C'))),
    ('indices', types.Array(types.int32, ndim=3, layout='C')),
]

class Board:
    def __init__(
        self,
        num_rows: int,
        num_cols: int,
        num_colours: int,
        colourless_specials: List[str] = ["cookie"],
        colour_specials: List[str] = ["vertical_laser", "horizontal_laser", "bomb"],
        np_random: np.random.Generator = np.random.default_rng(0),
        board: Optional[np.ndarray] = None,
    ):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_colours = num_colours

        self.flat_size = int(self.num_cols * self.num_rows)

        self.colourless_specials = colourless_specials
        self.colour_specials = colour_specials
        
        self.specials = set(self.colourless_specials + self.colour_specials)
        
        self.np_random = np_random
        # handle the case where we are given a board
        if board is not None:
            if isinstance(board, list):
                board = np.array(board, dtype=np.int32)
            if len(board.shape) < 3:
                self.board = np.array([board,np.ones_like(board)])
            else:
                self.board = board

            self.num_rows = self.board.shape[1]
            self.num_cols = self.board.shape[2]


        self.num_actions = int((self.num_rows * self.num_cols * 2) - self.num_rows - self.num_cols)
        self.action_to_coords = []

        for i in range(self.num_actions):
            # down
            if i < self.num_cols * (self.num_rows - 1):
                row = i // self.num_cols
                col = i % self.num_cols
                self.action_to_coords.append(((row, col), (row + 1, col)))
            # right
            else:
                i_ = i - self.num_cols * (self.num_rows - 1)
                row = i_ // (self.num_cols - 1)
                col = i_ % (self.num_cols - 1)
                self.action_to_coords.append(((row, col), (row, col + 1)))

        self.action_to_coords = tuple(self.action_to_coords)

    def generate_board(self):
        self.board = np.ones((2, self.num_rows, self.num_cols), dtype=np.int32)
        self.board[0] = self.np_random.integers(1, self.num_colours+1, self.flat_size).reshape(self.num_rows, self.num_cols)

        line_matches = self.get_colour_lines()
        num_line_matches = len(line_matches)

        while not self.possible_move() or num_line_matches > 0:
            if num_line_matches > 0:
                self.remove_colour_lines(line_matches)
            else:
                self.shuffle()

            line_matches = self.get_colour_lines()
            num_line_matches = len(line_matches)
        
        # assert self.possible_move()
        # assert self.get_colour_lines() == []

    def shuffle(self):
        shuffled_idcs = np.arange(self.num_rows * self.num_cols)
        self.np_random.shuffle(shuffled_idcs)
        shuffled_idcs = shuffled_idcs.reshape(self.num_rows, self.num_cols)
        self.board = self.board[:, shuffled_idcs // self.num_cols, shuffled_idcs % self.num_cols]

    def remove_colour_lines(self, line_matches: List[List[Tuple[int, int]]]) -> None:
        """Given a board and list of lines where each line is a list of coordinates where the colour of the tiles at each coordinate in one line is the same, changes the board such that no colour matches exist.
            This is only used for generating the board or when a shuffle is needed. This function does not touch the type of tiles in the board, only the colours.
        Args:
            line_matches (List[List[Tuple[int, int]]]): List of lines where each line is colour match.
        """
        while len(line_matches) > 0:
            l = line_matches[0]
            row = min(self.num_rows - 1, l[0][0] + 1)
            self.board[0, :row+1, :] = self.np_random.integers(1, self.num_colours+1, int((row+1) * self.num_cols)).reshape(-1, self.num_cols)
            line_matches = self.get_colour_lines()
            # line_matches = get_colour_lines(self.board)

    def detect_colour_matches(self) -> Tuple[List[List[Tuple[int, int]]], List[str], List[int]]:
        """
        Returns the types and locations of tiles involved in the bottom-most colour matches.
        """
        # For an empty board we can skip this.
        # if np.all(self.board[0] == 0):
        #     return [], [], []
        
        lines = self.get_colour_lines()

        if len(lines) == 0:
            return [], [], []
        else:
            tile_coords, tile_names, tile_colours = self.process_colour_lines(lines)
            return tile_coords, tile_names, tile_colours

    def get_colour_lines(self) -> List[List[Tuple[int, int]]]:
        """
        Starts from the top and checks for 3 or more in a row vertically or horizontally.
        Returns contiguous lines of 3 or more tiles.
        """
        lines = []
        vertical_line_coords = set()
        horizontal_line_coords = set()
        found_line = False
        for row in range(self.num_rows - 1, -1, -1):
            if found_line:
                break  # Only get lowest lines.
            for col in range(self.num_cols):
                # Vertical lines
                if 1 < row and (row, col) not in vertical_line_coords:
                    if self.board[1, row, col] > 0 : # Not Colourless special
                        if self.board[0, row, col] == self.board[0, row-1, col]: # Don't have to check the other one isn't a colourless special since colourless specials should be 0 in first axis.
                            line_start = row - 1
                            line_end = row
                            while line_start > 0:
                                if self.board[0, row, col] == self.board[0, line_start - 1, col]:
                                    line_start -= 1
                                else:
                                    break
                            if line_end - line_start >= 2:
                                found_line = True
                                line = [(i, col) for i in range(line_start, line_end + 1)]      
                                vertical_line_coords.update(line)
                                lines.append(line)
                # Horizontal lines
                if col < self.num_cols - 2 and (row, col) not in horizontal_line_coords:
                    if self.board[1, row, col] > 0 : # Not Colourless special
                        if self.board[0, row, col] == self.board[0, row, col + 1]: # Don't have to check the other one isn't a colourless special since colourless specials should be 0 in first axis.
                            line_start = col
                            line_end = col + 1
                            while line_end < self.num_cols - 1:
                                if self.board[0, row, col] == self.board[0, row, line_end + 1]:
                                    line_end += 1
                                else:
                                    break
                            if line_end - line_start >= 2:
                                found_line = True
                                line = [(row, i) for i in range(line_start, line_end + 1)]
                                horizontal_line_coords.update(line)
                                lines.append(line)
        
        # go through all the coordinates as a list
        # find neighbours that are not in the coordinates list but have the same colour and (and are not colourless)
        # follow the neighbours until the end of the line is reached if the line is long enough, add it to the list of lines
        valid_coord = lambda coord: 0 <= coord[0] < self.num_rows and 0 <= coord[1] < self.num_cols
        match_color = lambda coord1, coord2: self.board[0, coord1[0], coord1[1]] == self.board[0, coord2[0], coord2[1]] and self.board[1, coord1[0], coord1[1]] > 0 and self.board[1, coord2[0], coord2[1]] > 0
        coords = [(i, j) for l in lines for i, j in l]
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        for c in coords:
            for d in directions:
                line = [c]
                for direction in [d, (-d[0], -d[1])]:
                    n = (c[0] + direction[0], c[1] + direction[1])
                    while n not in coords and valid_coord(n) and match_color(c, n):
                        line.append(n)
                        n = (n[0] + direction[0], n[1] + direction[1])
                if len(line) >= 3:
                    sorted_line = sorted(line, key=lambda x: (x[0], x[1]))
                    if sorted_line not in lines:
                        lines.append(sorted_line)
        return lines

    def gravity(self) -> None:
        """
        Given a board with zeroes, push the zeroes to the top of the board.
        If an activation queue of coordinates is passed in, then the coordinates in the queue are updated as gravity pushes the coordinates down.
        """
        colour_zero_mask_T = self.board[0].T == 0
        type_zero_mask_T = self.board[1].T == 0
        zero_mask_T = colour_zero_mask_T & type_zero_mask_T
        non_zero_mask_T = ~zero_mask_T
        
        for j in range(self.num_cols):
            self.board[0][:, j] = np.concatenate([self.board[0][:, j][zero_mask_T[j]], self.board[0][:, j][non_zero_mask_T[j]]])
            self.board[1][:, j] = np.concatenate([self.board[1][:, j][zero_mask_T[j]], self.board[1][:, j][non_zero_mask_T[j]]])
            
    def refill(self) -> None:
        """Replace all empty tiles."""
        zero_mask_colour = self.board[0] == 0
        zero_mask_type = self.board[1] == 0
        zero_mask = zero_mask_colour & zero_mask_type

        num_zeros = zero_mask.sum()
        if num_zeros > 0:
            rand_vals = self.np_random.integers(1, self.num_colours + 1, size=num_zeros)
            self.board[0, zero_mask] = rand_vals
            self.board[1, zero_mask] = 1

    def is_move_legal(self, coord1: Tuple[int, int], coord2: Tuple[int, int]) -> bool:
        """This function checks if the move is actually possible regardless of whether the move results in a match. 

        Args:
            coord1 (Tuple[int, int]): The first coordinate on grid corresponding to the action taken. This will always be above or to the left of the second coordinate below.
            coord2 (Tuple[int, int]): Second coordinate on grid corresponding to the action taken.
        Returns:
            bool: Whether the move is legal.
        """
        ## Check both coords are on the board. ##
        if not (0 <= coord1[0] < self.num_rows and 0 <= coord1[1] < self.num_cols):
            return False
        if not (0 <= coord2[0] < self.num_rows and 0 <= coord2[1] < self.num_cols):
            return False
        
        # Check coordinates are not identical.
        if coord1 == coord2:
            return False
        
        # Check coords are next to each other.
        if not (coord1[0] == coord2[0] or coord1[1] == coord2[1]) or np.abs(coord1[0] - coord2[0]) > 1 or  np.abs(coord1[1] - coord2[1]) > 1:
            return False

        return True


    def process_colour_lines(self, lines: List[List[Tuple[int, int]]]) -> Tuple[List[List[Tuple[int, int]]], List[str], List[int]]:
        """
        Given list of contiguous lines (each a list of coords), this function detects the match type from the bottom up, merging any lines that share a coordinate.
        It greedily extracts the maximum match from the bottom up. So first look at what the most powerful thing you can extract from the bottom up.
        Note: concurrent groups can be matched at the same time.

        Returns:
            Tuple[List[List[Tuple[int, int]]], List[str], List[int]]: List of coordinates, list of match types, list of match colours.
        """
        tile_names = []
        tile_coords = []
        tile_colours = []

        lines = sorted([sorted(i, key=lambda x: (x[0], x[1])) for i in lines], key=lambda y: (y[0][0]), reverse=False)

        while len(lines) > 0:
            line = lines.pop(0)
            # check for cookie
            if len(line) >= 5 and "cookie" in self.specials:
                tile_names.append("cookie")
                tile_coords.append(line[:5])
                tile_colours.append(0)
                if len(line[5:]) > 2:
                    lines.append(line[5:])  # TODO - should just not pop the line rather than removing and adding again.
            # check for laser
            elif len(line) == 4:
                tile_colours.append(self.board[0, line[0][0], line[0][1]])
                tile_coords.append(line)
                if line[0][0] == line[1][0] and "horizontal_laser" in self.specials:
                    tile_names.append("horizontal_laser")
                elif "vertical_laser" in self.specials:
                    tile_names.append("vertical_laser")
                else:
                    tile_names.append("normal")
            # check for bomb (coord should appear in another line)
            elif "bomb" in self.specials and any([coord in l for coord in line for l in lines]):  # TODO - REMOVE THIS AS SLOW AND IS DONE TWICE
                for l in lines:
                    shared = [c for c in line if c in l]
                    if any(shared):
                        shared = shared[0]
                        # Add the closest three coordinates from both lines.
                        sorted_closest = sorted(l, key=lambda x: (abs(x[0] - shared[0]) + abs(x[1] - shared[1])))
                        # TODO: Change this to also only extract 3 closest to intersection from line.
                        tile_coords.append([p for p in line] + [p for p in sorted_closest[:3] if p not in line])  
                        tile_names.append("bomb")
                        tile_colours.append(self.board[0, line[0][0], line[0][1]])
                        if len(l) < 6:  # Remove the other line if shorter than 3 after extracting bomb.
                            lines.remove(l)
                        else:
                            for c in sorted_closest[:3]:  # Remove the coordinates that were taken for the bomb
                                l.remove(c)
                        break  # Stop searching after finding one intersection. This should break out of the for loop.
            # Check for normals. This happens even if the lines are longer than 3 but there are no matching specials.
            elif len(line) >= 3:
                tile_names.append("normal")
                tile_coords.append(line)
                tile_colours.append(self.board[0, line[0][0], line[0][1]])

        return tile_coords, tile_names, tile_colours


    def move(self, coord1: Tuple[int, int], coord2: Tuple[int, int]) -> Tuple[int, int, int, int, bool]:
        """High-level entry point for each move. This function checks if the move is legal and effective, then executes the move.

        Args:
            coord1 (Tuple[int, int]): Tile used in move
            coord2 (Tuple[int, int]): Second tile used in move

        Raises:
            ValueError: If the move is illegal, an error is raised.

        Returns:
            Tuple[int, int, int, int, bool]: Number of tiles eliminated, whether it was a combination match, number of specials created, number of specials activated, whether a shuffle was performed.
        """
        self.num_specials_activated = 0
        self.num_new_specials = 0
        num_eliminations = 0
        is_combination_match = False
        shuffled = False

        if not self.is_move_legal(coord1, coord2):
            raise ValueError(f"Invalid move: {coord1}, {coord2}")

        if not is_move_effective(self.board, coord1, coord2):
            return num_eliminations, is_combination_match, self.num_new_specials, self.num_specials_activated, shuffled
        
        swap_coords(self.board, coord1, coord2)
        ## Combination match ##
        has_two_specials = self.board[1, coord1[0], coord1[1]] not in [0,1] and self.board[1, coord2[0], coord2[1]] not in [0,1]
        has_one_colourless_special = self.board[1, coord1[0], coord1[1]] < 0 or self.board[1, coord2[0], coord2[1]] < 0
        if has_two_specials or has_one_colourless_special:
            is_combination_match = True
            self.combination_match(coord1, coord2)
            num_eliminations += self.flat_size - np.count_nonzero(self.board[1])
            self.gravity()
            self.refill()
        
        ## Colour matching ##
        has_match = True
        while has_match:
            match_locs, match_types, match_colours = self.detect_colour_matches()
            if len(match_locs) == 0:
                has_match = False
            else:
                self.resolve_colour_matches(match_locs, match_types, match_colours)
                num_eliminations += self.flat_size - np.count_nonzero(self.board[1])
                self.gravity()
                self.refill()
        
        num_eliminations += self.num_new_specials # New specials are always placed in empty cells and this reduces count of eliminations.

        # Ensure the new board is playable.
        line_matches = []
        num_line_matches = 0
        while not self.possible_move() or num_line_matches > 0:
            if num_line_matches > 0:
                self.remove_colour_lines(line_matches) # You don't get extra points for matches that happen due to shuffling.
            else:
                shuffled = True
                self.shuffle()

            line_matches = self.get_colour_lines()
            num_line_matches = len(line_matches)

        # assert self.possible_move()
        # assert self.get_colour_lines() == []
        return num_eliminations, is_combination_match, self.num_new_specials, self.num_specials_activated, shuffled

    def resolve_colour_matches(
            self, 
            match_locs: List[List[Tuple[int, int]]], 
            match_types: List[str], 
            match_colours:List[int]
            ) -> None:
        """The main loop for processing a batch of colour matches. This function eliminates tiles, activates specials and creates new specials.
            Note: This function assumes there are matches.

        Args:
            match_locs (List[List[Tuple[int, int]]]): List of match locations. Each match location is a list of coordinates that are part of the match.
            match_types (List[str]): List of match types ordered in the same way as match_locs.
            match_colours (List[int]): The colour of a tile in each match. Used to track what colour new specials should be.
        """
        special_creation_q = []
        # Extract the special creation position first since the loop below deletes tiles so we cannot check colours for determining special pos.
        taken_positions = set()
        for i in range(len(match_locs)):
            if match_types[i] != "normal":
                special_creation_coord = self.get_special_creation_pos(match_locs[i], taken_pos=taken_positions, straight_match=match_types[i] != "bomb")
                taken_positions.add(special_creation_coord)
                special_creation_q.append((special_creation_coord, match_types[i], match_colours[i]))

        # Activate specials and delete tiles.
        for i in range(len(match_locs)):
            match_coords = match_locs[i]
            self.resolve_colour_match(match_coords)

        # Create new specials.
        for i in range(len(special_creation_q)):
            self.create_special(*special_creation_q[i])

    def get_special_creation_pos(self, coords: List[Tuple[int, int]], taken_pos:set, straight_match: Optional[bool]=True) -> Tuple[int, int]:
        """Given a set of coordinates return the position of the special tile that should be placed
        The position should be as close to the center as possible but should not already be special.

        Args:
            coords (List[Tuple[int, int]]): The coordinates in the match that created the special.
            taken_pos (set): Set of coordinates that are already pending special creation and so cannot be new special positions.
            straight_match (Optional[bool]): Whether the match is straight (not a bomb) or not. Defaults to True.

        """        
        valid_coords = [c for c in coords if c not in taken_pos]
        
        if not straight_match:
            # Get the corner coords and find closest valid coord
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            corner = (max(xs, key=xs.count), max(ys, key=ys.count)) 
            if corner in valid_coords:
                return corner
            else:
                chosen_c = sorted(valid_coords, key=lambda x: (x[0] - corner[0]) ** 2 + (x[1] - corner[1]) ** 2)[0]
                return chosen_c

        # For straight matches get the center of the coords
        sorted_coords = sorted(valid_coords, key=lambda x: (x[0], x[1]))        
        if len(sorted_coords) % 2 == 0:
            chosen_c = sorted_coords[len(sorted_coords) // 2 - 1]
            return chosen_c
        chosen_c = sorted_coords[len(sorted_coords) // 2]
        return chosen_c

    def resolve_colour_match(self, match_coords: List[Tuple[int, int]]) -> None:
        """Resolving a single match. This function eliminates normal tiles and activates special tiles.

        Args:
            match_coords (List[Tuple[int, int]]): List of coordinates that are part of the match.
        """
        
        for coord in match_coords:
            if self.board[1, coord[0], coord[1]] not in [0, 1]:
                self.activate_special(coord, self.board[1, coord[0], coord[1]], self.board[0, coord[0], coord[1]])
            else:
                self.board[:, coord[0], coord[1]] = 0 # Delete the normal tiles.   
    
    def activate_special(self, coord: Tuple[int, int], tile_type: int, tile_colour: int, is_combination_match: Optional[bool] = False) -> None:
        """Used in the move loop for when a special has been chosen to activate passively (as opposed to when a combination match occurs). 
        So the special tile has been _hit_ by another activation or it is involved in a colour match.

        Args:
            coord (Tuple[int, int]): Coordinate at which to activate the special.
            tile_type (int): Type of special.
            tile_colour (int): Colour of the special tile.
            is_combination_match (Optional[bool]): Whether the special is being activated as part of a combination match. Defaults to False.

        Raises:
            ValueError: If the type of tile being activated is not valid.
        """
        
        # For an empty board we can skip this.
        if np.all(self.board[0] == 0):
            return
        
        if tile_type in [0, 1]: 
            raise ValueError(f"Invalid type of tile given: {tile_type}")
        
        special_r, special_c = coord
        # Delete special
        self.board[:, special_r, special_c] = 0

        if not is_combination_match: # Avoid double counting activated specials.
            self.num_specials_activated += 1
        
        # vertical laser
        if tile_type == 2:
            for row in range(self.num_rows):
                if self.board[1, row, special_c] not in [0, 1]:
                    self.activate_special((row, special_c), self.board[1, row, special_c], self.board[0, row, special_c])
                else:
                    self.board[:, row, special_c] = 0

        # horizontal_laser
        elif tile_type == 3:
            for col in range(self.num_cols):
                if self.board[1, special_r, col] not in [0, 1]:
                    self.activate_special((special_r, col), self.board[1, special_r, col], self.board[0, special_r, col])
                else:
                    self.board[:, special_r, col] = 0   
        # bomb
        elif tile_type == 4:
            min_r = max(coord[0] - 1, 0)
            max_r = min(coord[0] + 1, self.num_rows - 1)
            min_c = max(coord[1] - 1, 0)
            max_c = min(coord[1] + 1, self.num_cols - 1)

            for i in range(min_r, max_r + 1):
                for j in range(min_c, max_c + 1):
                    if self.board[1, i, j] not in [0, 1]:
                        self.activate_special((i, j), self.board[1, i, j], self.board[0, i, j])
                    else:
                        self.board[:, i, j] = 0 
        # cookie
        elif tile_type == -1:
            # If the most common colour is 0 then we'd be deleting nothing.
            mask = self.board[0] != 0
            if mask.sum() == 0:
                return

            counts = np.bincount(self.board[0][mask])
            most_common_colour = np.argmax(counts)

            # Delete all normal tiles of the chosen colour.
            colour_mask = self.board[0] == most_common_colour
            normal_mask = self.board[1] == 1
            mask = colour_mask & normal_mask
            self.board[0, mask] = 0
            self.board[1, mask] = 0
            
            # Activate all specials of the chosen colour.
            special_type_mask = self.board[1] > 1
            mask = colour_mask & special_type_mask
            r_idcs, c_idcs = np.where(mask)
            
            for i in range(len(r_idcs)):
                r, c = r_idcs[i], c_idcs[i]
                if self.board[1, r, c] not in [0, 1]:
                    self.activate_special((r, c), self.board[1, r, c], self.board[0, r, c])
        else: 
            raise ValueError(f"Tile type: {tile_type}, tile colour: {tile_colour} is an invalid special tile.")

    def possible_move(self, grid: Optional[np.ndarray] = None):
        """Wrapper function that checks if there is an effective move possible.
        Args: 
            grid (Optional[np.ndarray]): Optionally, a grid can be passed in to check for this function. Otherwise, self.board is used.

        """
        if grid is None:
            grid = self.board
        for (coord1, coord2) in self.action_to_coords:
            if is_move_effective(grid, coord1, coord2):
                return True
        return False
    

    def create_special(self, coord: Tuple[int, int], special_type: str, tile_colour: int) -> None:
        """This function creates a special tile at the location specified by the match_coords.
        We don't check if the special is valid under the board specification because that should be checked in process_colour_lines.
        Args:
            coord (Tuple[int, int]): Coordinates to pick from.
            special_type (str): The type of new special tile to create
            tile_colour (int): The colour of the special tile.
        """

        # assert self.board[0, coord[0], coord[1]] == 0, (coord, self.board[0])
        # assert self.board[1, coord[0], coord[1]] == 0, (coord, self.board[0])
        self.num_new_specials += 1

        if special_type == "cookie":
            tile_type = -1
        elif special_type == "vertical_laser":
            tile_type = 2
        elif special_type == "horizontal_laser":
            tile_type = 3
        elif special_type == "bomb":
            tile_type = 4
        else:
            raise ValueError(f"Invalid match type for creating special: {special_type}")
        
        self.board[0, coord[0], coord[1]] = tile_colour
        self.board[1, coord[0], coord[1]] = tile_type
        

    def combination_match(self, coord1: Tuple[int, int], coord2: Tuple[int, int]) -> None:
        """
        This function executes a combination match. This usually creates a more powerful effect than activating each special tile alone.
        This occurs when both coordinates are special tiles, or one is a colourless special.

        Args:
            coord1 (Tuple[int, int]): Coordinates in combination.
            coord2 (Tuple[int, int]): Coordinates in combination.
        """
        self.num_specials_activated += 2 # in all cases but cookie + normal.
        # Get the types and colours of each tile.
        tile_type1, tile_colour1 = self.board[1, coord1[0], coord1[1]], self.board[0, coord1[0], coord1[1]]
        tile_type2, tile_colour2 = self.board[1, coord2[0], coord2[1]], self.board[0, coord2[0], coord2[1]]
        
        # Cookie + cookie
        if tile_type1 == tile_type2 == -1:
            self.board[:] = 0

        # Cookie + normal deletes all normal tiles of the same colour and activates all specials of same colour.
        elif tile_type1 == -1 and tile_type2 == 1 or tile_type1 == 1 and tile_type2 == -1:
            if tile_type1 == 1: # Deal with reverse order.
                tile_type1, tile_type2 = tile_type2, tile_type1
                tile_colour1, tile_colour2 = tile_colour2, tile_colour1
                coord1, coord2 = coord2, coord1

            # Delete cookie.
            self.board[:, coord1[0], coord1[1]] = 0
            # Delete normal tile
            self.board[:, coord1[0], coord1[1]] = 0

            # Delete all normal tiles of same colour
            colour_mask = self.board[0] == tile_colour2
            normal_mask = self.board[1] == 1
            mask = colour_mask & normal_mask
            self.board[0, mask] = 0
            self.board[1, mask] = 0

            # Activate all specials of same colour.
            special_type_mask = self.board[1] > 1
            mask = colour_mask & special_type_mask
            self.activate_specials_in_mask(mask, is_combination_match=True)
            self.num_specials_activated -= 1 # Correct the previous +2.

        # Cookie + vertical laser/horizontal laser/bomb -> convert all normals of same colour to the special type then activate them.
        elif (tile_type1 == -1 and tile_type2 >= 2) or (tile_type1 >=2  and tile_type2 == -1):
            if tile_type2 == -1:
                tile_type1, tile_type2 = tile_type2, tile_type1
                tile_colour1, tile_colour2 = tile_colour2, tile_colour1
                coord1, coord2 = coord2, coord1
            
            # Delete cookie.
            self.board[:, coord1[0], coord1[1]] = 0
            
            # Convert all normal tiles of same colour to special.
            colour_mask = self.board[0] == tile_colour2
            normal_mask = self.board[1] == 1
            mask = colour_mask & normal_mask
            self.board[1, mask] = tile_type2
            
            # Activate all specials of same colour.
            self.activate_specials_in_mask(colour_mask, is_combination_match=True)
                    
        # vertical laser + vertical laser or horizontal laser + horizontal laser or vertical laser + horizontal laser
        elif tile_type1 == tile_type2 == 2 or tile_type1 == tile_type2 == 3 or (tile_type1 == 2 and tile_type2 == 3) or (tile_type1 == 3 and tile_type2 == 2):
            # Delete both tiles.
            self.board[:, coord1[0], coord1[1]] = 0
            self.board[:, coord2[0], coord2[1]] = 0
            
            # Do a vertical and horizontal laser at the same time at the topmost leftmost coordinate.
            r = min(coord1[0], coord2[0])
            c = min(coord1[1], coord2[1])
            
            # Activate a vertical and then horizontal laser.
            self.activate_special((r, c), 2, tile_colour1, is_combination_match=True) # Tile colour doesn't matter here
            self.activate_special((r, c), 3, tile_colour1, is_combination_match=True) # Tile colour doesn't matter here
            
        # vertical laser/horizontal laser + bomb
        elif tile_type1 == 4 and 2 <= tile_type2 <= 3 or tile_type2 == 4 and 2 <= tile_type1 <= 3:
            # Delete the combination tiles.
            self.board[:, coord1[0], coord1[1]] = 0
            self.board[:, coord2[0], coord2[1]] = 0

            r = min(coord1[0], coord2[0])
            c = min(coord1[1], coord2[1])
            # Get the three rows and cols centred at laser coordinate.
            min_r = max(r - 1, 0)
            max_r = min(r + 1, self.num_rows - 1)
            min_c = max(c - 1, 0)
            max_c = min(c + 1, self.num_cols - 1)
            
            # Activate horizontal lasers
            for i in range(min_r, max_r + 1):
                self.activate_special((i, c), 3, tile_colour2, is_combination_match=True) # Tile colour doesn't matter here

            # Activate vertical lasers.
            for j in range(min_c, max_c + 1):
                self.activate_special((r, j), 2, tile_colour2, is_combination_match=True) # Tile colour doesn't matter here

        # bomb + bomb             
        elif tile_type1 == tile_type2 == 4:
            # Delete bombs
            self.board[:, coord1[0], coord1[1]] = 0
            self.board[:, coord2[0], coord2[1]] = 0

            # Get 5x5 grid
            central_r = min(coord1[0], coord2[0])
            central_c = min(coord1[1], coord2[1])

            min_r = max(central_r - 2, 0)
            max_r = min(central_r + 2, self.num_rows - 1)
            min_c = max(central_c - 2, 0)
            max_c = min(central_c + 2, self.num_cols - 1)
            
            # Iterate through activation area.
            for i in range(min_r, max_r + 1):
                for j in range(min_c, max_c + 1):
                    if self.board[1, i, j] == 1:
                        self.board[:, i, j] = 0
                    elif self.board[1, i, j] != 0:
                        self.activate_special((i, j), self.board[1, i, j], self.board[0, i, j], is_combination_match=True)

    def activate_specials_in_mask(self, mask, is_combination_match:bool):
        r_idcs, c_idcs = np.where(mask)
        for i in range(len(r_idcs)):
            r, c = r_idcs[i], c_idcs[i]
            if self.board[1, r, c] not in [0, 1]:
                self.activate_special((r, c), self.board[1, r, c], self.board[0, r, c], is_combination_match)


@njit
def swap_coords(board: np.ndarray, coord1: Tuple[int, int], coord2: Tuple[int, int]) -> None: 
    board[0, coord1[0], coord1[1]], board[0, coord2[0], coord2[1]] = board[0, coord2[0], coord2[1]], board[0, coord1[0], coord1[1]]
    board[1, coord1[0], coord1[1]], board[1, coord2[0], coord2[1]] = board[1, coord2[0], coord2[1]], board[1, coord1[0], coord1[1]]


@njit
def is_move_effective(board: np.ndarray, coord1: Tuple[int, int], coord2: Tuple[int, int]) -> bool:
    """
    This function checks if the action actually does anything i.e. if the action achieves some form of matching.

    Args:
        board (np.ndarray): The board.
        coord1 (tuple): The first coordinate on grid corresponding to the action taken. This will always be above or to the left of the second coordinate below.
        coord2 (tuple): Second coordinate on grid corresponding to the action taken.

    Returns:
        bool: True iff action has an effect on the environment.
    """
    num_rows, num_cols = board.shape[1:]
    # Checks if both are special
    if (board[1, coord1[0], coord1[1]] not in [0, 1]) and (board[1, coord2[0], coord2[1]] not in [0, 1]):
        return True

    # At least one colourless special.
    if board[1, coord1[0], coord1[1]] < 0 or board[1, coord2[0], coord2[1]] < 0:
        return True

    # Extract a minimal grid around the coords to check for at least 3 match. This covers checking for Ls or Ts.
    r_min = max(0, min(coord1[0], coord2[0]) - 2)
    r_max = min(num_rows-1, max(coord1[0], coord2[0]) + 2)
    c_min = max(0, min(coord1[1], coord2[1]) - 2)
    c_max = min(num_cols-1, max(coord1[1], coord2[1]) + 2)

    # Swap the coordinates_ to see what happens.
    swap_coords(board, coord1, coord2)
    colour_slice = board[0, r_min:r_max + 1, c_min:c_max + 1]
    # Horizontal Matches
    if c_min + 2 <= c_max:
        # horizontal_slice = colour_slice  # Slice for horizontal comparison
        horizontal_matches = (colour_slice[:, :-2] == colour_slice[:, 1:-1]) & (colour_slice[:, 1:-1] == colour_slice[:, 2:])
        matching_mask = horizontal_matches & (board[1, r_min:r_max + 1, c_min + 2:c_max + 1] >= 0) # Check that the tile types are coloured.
        if matching_mask.any():
            # Swap back
            swap_coords(board, coord1, coord2)
            return True

    # Vertical Matches
    if r_min + 2 <= r_max:
        # vertical_slice = colour_slice  # Slice for vertical comparison
        vertical_matches = (colour_slice[:-2, :] == colour_slice[1:-1, :]) & (colour_slice[1:-1, :] == colour_slice[2:, :])
        matching_mask = vertical_matches & (board[1, r_min + 2:r_max + 1, c_min:c_max + 1] >= 0)
        if matching_mask.any():
            # Swap back
            swap_coords(board, coord1, coord2)
            return True
        
    swap_coords(board, coord1, coord2)
    return False
