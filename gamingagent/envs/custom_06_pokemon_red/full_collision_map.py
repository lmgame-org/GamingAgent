import numpy as np
from typing import Optional, Dict, Tuple, List, Set
import os


class LocationCollisionMap:
    """
    Handles making an automatically updating collision map of an area as the model paths through it.
    """
    
    def __init__(self, initial_collision_map: np.ndarray, initial_sprite_locations: Set[Tuple[int, int]], initial_coords: Tuple[int, int]):
        """
        Initialize the collision map.
        
        Args:
            initial_collision_map: A 9x10 player-centered collision map where 0 is impassable and 1 is passable
            initial_sprite_locations: Set of sprite coordinates (col, row)
            initial_coords: Player coordinates (col, row)
        """
        # initial_collision_map is a 9 x 10 player-centered collision map which is 0 is impassable and 1 otherwise
        # Internally we store an expanding map based on locations we've been, with -1 in unknown spots, 2 in sprite locations, 3 in player location, and otherwise 0/1 as well.
        # We just accept that moving NPC locations are going to be inaccurate.
        # Note that while player coords are column, row, by default what we get from the collision map tooling is row, column
        self.player_coords = initial_coords
        self.col_offset = initial_coords[0] - 4
        self.row_offset = initial_coords[1] - 4
        self.internal_map = -np.ones((10, 9), dtype=np.int8)  # We make our map the "proper" order that everything else is.
        
        for row in range(9):
            for col in range(10):
                if row == 4 and col == 4:  # player character
                    self.internal_map[col][row] = 3
                elif (col, row) in initial_sprite_locations:
                    self.internal_map[col][row] = 2
                else:
                    self.internal_map[col][row] = initial_collision_map[row][col]
        
        self.distances: Dict[Tuple[int, int], int] = {}

    def update_map(self, collision_map: np.ndarray, sprite_locations: Set[Tuple[int, int]], coords: Tuple[int, int]):
        """
        Update the collision map with new data.
        
        Args:
            collision_map: New 9x10 collision map
            sprite_locations: Set of sprite coordinates (col, row)
            coords: New player coordinates (col, row)
        """
        # Remove the previous player marker. Most convenient to do it right now.
        self.internal_map[self.player_coords[0] - self.col_offset][self.player_coords[1] - self.row_offset] = 1
        self.player_coords = coords

        new_min_col = coords[0] - 4
        new_min_row = coords[1] - 4
        new_max_col = coords[0] + 5
        new_max_row = coords[1] + 4
        cur_size = self.internal_map.shape
        
        # First check if we need to move the boundaries of the array. Numpy pad makes this easy!
        expand_col_front = self.col_offset - new_min_col if new_min_col < self.col_offset else 0
        expand_col_back = new_max_col - (self.col_offset + cur_size[0] - 1) 
        expand_col_back = expand_col_back if expand_col_back > 0 else 0
        expand_row_front = self.row_offset - new_min_row if new_min_row < self.row_offset else 0
        expand_row_back = new_max_row - (self.row_offset + cur_size[1] - 1)
        expand_row_back = expand_row_back if expand_row_back > 0 else 0
        
        self.internal_map = np.pad(
            self.internal_map, 
            pad_width=[(expand_col_front, expand_col_back), (expand_row_front, expand_row_back)], 
            constant_values=-1
        )

        self.col_offset = min(new_min_col, self.col_offset)
        self.row_offset = min(new_min_row, self.row_offset)

        # Now update the map
        local_col_offset = new_min_col - self.col_offset
        local_row_offset = new_min_row - self.row_offset
        
        for row in range(9):
            for col in range(10):
                if row == 4 and col == 4:  # player character
                    self.internal_map[col + local_col_offset][row + local_row_offset] = 3
                    continue
                    
                if (col, row) in sprite_locations:
                    self.internal_map[col + local_col_offset][row + local_row_offset] = 2
                else:
                    self.internal_map[col + local_col_offset][row + local_row_offset] = collision_map[row][col]
                    
        self.distances = self.compute_effective_distance_to_tiles()

    def compute_effective_distance_to_tiles(self) -> Dict[Tuple[int, int], int]:
        """
        Compute effective distance to all reachable tiles using flood fill.
        
        Returns:
            Dictionary mapping coordinates to distance from player
        """
        depth = 99
        visited_tiles = set([self.player_coords])
        cur_tiles = set([self.player_coords])
        distances: Dict[Tuple[int, int], int] = {}
        
        for d in range(depth):
            new_tiles: Set[Tuple[int, int]] = set()
            for tile in cur_tiles:
                candidate_tiles = (
                    (tile[0] + 1, tile[1]), 
                    (tile[0] - 1, tile[1]), 
                    (tile[0], tile[1] + 1), 
                    (tile[0], tile[1] - 1)
                )
                
                for candidate in candidate_tiles:
                    shifted_col = candidate[0] - self.col_offset
                    shifted_row = candidate[1] - self.row_offset
                    
                    if (shifted_col < 0 or shifted_row < 0 or 
                        shifted_col > self.internal_map.shape[0] - 1 or 
                        shifted_row > self.internal_map.shape[1] - 1):
                        continue
                        
                    if candidate in visited_tiles:
                        continue
                        
                    if self.internal_map[shifted_col][shifted_row] == 1:   # the only passable scenario
                        new_tiles.add(candidate)
                        distances[candidate] = d + 1
                    visited_tiles.add(candidate)
                    
            cur_tiles = new_tiles
            
        return distances

    def generate_buttons_to_coord(self, col: int, row: int) -> Optional[List[str]]:
        """
        Generate button sequence to navigate to a specific coordinate.
        
        Args:
            col: Target column
            row: Target row
            
        Returns:
            List of button names to reach the target, or None if unreachable
        """
        starting_distance = self.distances.get((col, row))
        if starting_distance is None:
            return None  # invalid
            
        distance = starting_distance
        button_list = []
        
        # Basically look for tiles that are labelled with successively lower numbers
        while distance > 0:
            # just pick whichever happens to work first.
            left = self.distances.get((col - 1, row))
            if (left and left == distance - 1) or (col - 1, row) == self.player_coords:
                button_list.append("right")
                col -= 1
                distance -= 1
                continue
                
            right = self.distances.get((col + 1, row))
            if (right and right == distance - 1) or (col + 1, row) == self.player_coords:
                button_list.append("left")
                distance -= 1
                col += 1
                continue
                
            up = self.distances.get((col, row - 1))
            if (up and up == distance - 1) or (col, row - 1) == self.player_coords:
                button_list.append("down")
                distance -= 1
                row -= 1
                continue
                
            down = self.distances.get((col, row + 1))
            if (down and down == distance - 1) or (col, row + 1) == self.player_coords:
                button_list.append("up")
                distance -= 1
                row += 1
                continue
                
            break  # No valid path found
        
        # now reverse and return
        button_list.reverse()
        return button_list
    
    @staticmethod
    def make_ascii_segment(input_str: str, width: int, col: int, row: int) -> str:
        """
        Output ascii map blocks of a consistent width, using a given input_str and local coordinates.
        
        Args:
            input_str: String to display
            width: Width of the segment
            col: Column coordinate
            row: Row coordinate
            
        Returns:
            Formatted string segment
        """
        base_str = f"{input_str}({col},{row})"
        # pads always at the end.
        if len(base_str) > width - 1:
            raise ValueError("Not enough space to fit this!")
        base_str += (width - 1 - len(base_str)) * " "
        return f"|{base_str}"

    def to_ascii(self, local_location_tracker: Optional[List[List[bool]]] = None, save_file_path: Optional[str] = None) -> str:
        """
        Convert the collision map to ASCII representation (model version).
        
        Args:
            local_location_tracker: Optional tracker for visited locations
            save_file_path: Optional path to save the collision map file
            
        Returns:
            ASCII representation of the map for the model
        """
        horizontal_labels = list(range(self.col_offset, self.col_offset + self.internal_map.shape[0]))

        row_width = 35
        horizontal_border = "       +" + "".join("Column " + str(x) + " " * (row_width - len(str(x)) - 7) for x in horizontal_labels) + "+"

        lines = []
        lines += [f"({self.col_offset}, {self.row_offset})", horizontal_border]
        
        for row_num, this_row in enumerate(self.internal_map.transpose()):  # transposing makes printing easier
            real_row = self.row_offset + row_num
            row = f"Row: {str(real_row) + ' ' * (2 - len(str(real_row)))}"
            
            for col_num, col in enumerate(this_row):
                real_col = self.col_offset + col_num
                
                if col == -1:
                    row += self.make_ascii_segment("Check here", row_width, real_col, real_row)
                elif col == 0:
                    row += self.make_ascii_segment("Impassable", row_width, real_col, real_row)
                elif col == 1: 
                    # Potentially place a distance marker:
                    row_piece = ""
                    distance = self.distances.get((real_col, real_row))
                    if distance:  # removes 0 and None
                        row_piece += "StepsToReach:" + str(distance) + " " * (4 - len(str(distance))) + " "
                        
                    if (local_location_tracker and real_col > -1 and real_row > -1 and 
                        real_col < len(local_location_tracker) and real_row < len(local_location_tracker[real_col]) and 
                        local_location_tracker[real_col][real_row]):
                        row_piece += "Explored"
                    else:
                        row_piece += "Passable"
                        
                    row += self.make_ascii_segment(row_piece, row_width, real_col, real_row)
                elif col == 2:
                    row += self.make_ascii_segment("NPC/Object", row_width, real_col, real_row)
                elif col == 3:
                    row += self.make_ascii_segment("PLAYER", row_width, real_col, real_row)
                    
            row += f"|{str(real_row)}"
            lines.append(row)
            
        lines.append(horizontal_border + f"({self.col_offset + self.internal_map.shape[0] - 1}, {self.row_offset + self.internal_map.shape[1] - 1})")

        # Join all lines with newlines
        output = "\n".join(lines)
        
        # Save to file if path provided
        if save_file_path:
            os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
            human_readable = self.to_human_readable_ascii(local_location_tracker)
            with open(save_file_path, "w", encoding="utf-8") as fw:
                fw.write(human_readable)
                fw.write("\n\n=== DETAILED MAP ===\n\n")
                fw.write(output)
        
        return output

    def to_human_readable_ascii(self, local_location_tracker: Optional[List[List[bool]]] = None) -> str:
        """
        Convert the collision map to human-readable ASCII representation.
        
        Args:
            local_location_tracker: Optional tracker for visited locations
            
        Returns:
            Human-readable ASCII representation of the map
        """
        horizontal_labels = list(range(self.col_offset, self.col_offset + self.internal_map.shape[0]))
        horizontal_border_human = "       +" + "".join(str(x) + " " * (4 - len(str(x))) for x in horizontal_labels) + "+"

        lines_human = []
        
        # Add legend to human version
        if local_location_tracker:
            lines_human.extend([
                "",
                "Legend:",
                "██ - Wall/Obstacle",
                "Numbers - How many tiles away this tile is to reach",
                "SS - Sprite",
                "PP - Player Character",
                "xx - AVOID GOING HERE - Already Explored",
                "uu - CHECK HERE: Blank = Unknown/Unvisited"
            ])
        else:
            lines_human.extend([
                "",
                "Legend:",
                "██ - Wall/Obstacle",
                "Numbers - How many tiles away this tile is to reach",
                "SS - Sprite",
                "PP - Player Character",
                "uu - Blank = Unknown/Unvisited"
            ])

        lines_human += [f"({self.col_offset}, {self.row_offset})", horizontal_border_human]
        
        for row_num, this_row in enumerate(self.internal_map.transpose()):
            real_row = self.row_offset + row_num
            row_human = f"Row: {str(real_row) + ' ' * (2 - len(str(real_row)))}|"
            
            for col_num, col in enumerate(this_row):
                real_col = self.col_offset + col_num
                
                if col == -1:
                    row_human += " uu "
                elif col == 0:
                    row_human += " ██ "
                elif col == 1: 
                    row_piece_human = ""
                    distance = self.distances.get((real_col, real_row))
                    
                    if distance:
                        row_piece_human = str(distance) + " " * (4 - len(str(distance)))
                        
                    if (local_location_tracker and real_col > -1 and real_row > -1 and 
                        real_col < len(local_location_tracker) and real_row < len(local_location_tracker[real_col]) and 
                        local_location_tracker[real_col][real_row]):
                        if not row_piece_human:
                            row_human += " xx "
                    else:
                        if not row_piece_human:
                            row_human += " ·· "
                            
                    row_human += row_piece_human
                elif col == 2:
                    row_human += " SS "
                elif col == 3:
                    row_human += " PP "
                    
            row_human += f"|{str(real_row)}"
            lines_human.append(row_human)
            
        lines_human.append(horizontal_border_human + f"({self.col_offset + self.internal_map.shape[0] - 1}, {self.row_offset + self.internal_map.shape[1] - 1})")

        return "\n".join(lines_human)



    @classmethod
    def load_from_file(cls, file_path: str) -> Optional[str]:
        """
        Load collision map representation from a text file.
        
        Args:
            file_path: Path to load the file from
            
        Returns:
            String representation of the map, or None if file doesn't exist
        """
        try:
            with open(file_path, "r", encoding="utf-8") as fr:
                return fr.read()
        except FileNotFoundError:
            return None
