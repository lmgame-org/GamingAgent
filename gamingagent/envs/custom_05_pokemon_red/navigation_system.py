import logging
from typing import Dict, List, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)

class NavigationSystem:
    def __init__(self):
        self.collision_maps: Dict[str, np.ndarray] = {}  # Maps location name to collision map
        self.location_labels: Dict[str, Dict[Tuple[int, int], str]] = {}  # Maps location to coordinate labels
        self.distance_maps: Dict[str, np.ndarray] = {}  # Maps location name to distance map
        
    def update_collision_map(self, location: str, collision_map: np.ndarray) -> None:
        """Update the collision map for a location."""
        self.collision_maps[location] = collision_map
        # Update distance map when collision map changes
        self._update_distance_map(location)
        
    def _update_distance_map(self, location: str) -> None:
        """Update the distance map for a location using BFS."""
        if location not in self.collision_maps:
            return
            
        collision_map = self.collision_maps[location]
        height, width = collision_map.shape
        distance_map = np.full((height, width), -1)  # -1 indicates unreachable
        
        # Find player position (PP)
        player_pos = None
        for y in range(height):
            for x in range(width):
                if collision_map[y, x] == 'P':
                    player_pos = (x, y)
                    break
            if player_pos:
                break
                
        if not player_pos:
            return
            
        # BFS to calculate distances
        queue = [(player_pos, 0)]  # (position, distance)
        visited = {player_pos}
        
        while queue:
            (x, y), dist = queue.pop(0)
            distance_map[y, x] = dist
            
            # Check all 4 directions
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                new_x, new_y = x + dx, y + dy
                if (0 <= new_x < width and 0 <= new_y < height and 
                    collision_map[new_y, new_x] != '#' and 
                    (new_x, new_y) not in visited):
                    queue.append(((new_x, new_y), dist + 1))
                    visited.add((new_x, new_y))
                    
        self.distance_maps[location] = distance_map
        
    def add_location_label(self, location: str, coords: Tuple[int, int], label: str) -> None:
        """Add a label to a specific coordinate in a location."""
        if location not in self.location_labels:
            self.location_labels[location] = {}
        self.location_labels[location][coords] = label
        
    def get_ascii_map(self, location: str) -> str:
        """Generate ASCII representation of the map with distances and labels."""
        if location not in self.collision_maps:
            return "Location not found"
            
        collision_map = self.collision_maps[location]
        distance_map = self.distance_maps.get(location, np.zeros_like(collision_map))
        labels = self.location_labels.get(location, {})
        
        height, width = collision_map.shape
        ascii_map = []
        
        for y in range(height):
            row = []
            for x in range(width):
                cell = collision_map[y, x]
                if cell == 'P':  # Player
                    row.append('PP')
                elif cell == '#':  # Wall
                    row.append('##')
                elif cell == 'x':  # Explored
                    row.append('xx')
                else:
                    # Add distance if available
                    dist = distance_map[y, x]
                    if dist >= 0:
                        row.append(f"{dist:02d}")
                    else:
                        row.append('  ')
            ascii_map.append(''.join(row))
            
        # Add labels
        for (x, y), label in labels.items():
            if 0 <= y < height and 0 <= x < width:
                ascii_map[y] = ascii_map[y][:x*2] + label + ascii_map[y][x*2+len(label):]
                
        return '\n'.join(ascii_map)
        
    def find_path(self, location: str, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Find a path from start to goal using A* algorithm."""
        if location not in self.collision_maps:
            return []
            
        collision_map = self.collision_maps[location]
        height, width = collision_map.shape
        
        def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
            
        def get_neighbors(pos: Tuple[int, int]) -> List[Tuple[int, int]]:
            x, y = pos
            neighbors = []
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                new_x, new_y = x + dx, y + dy
                if (0 <= new_x < width and 0 <= new_y < height and 
                    collision_map[new_y, new_x] != '#'):
                    neighbors.append((new_x, new_y))
            return neighbors
            
        # A* implementation
        open_set = {start}
        closed_set = set()
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}
        
        while open_set:
            current = min(open_set, key=lambda x: f_score.get(x, float('inf')))
            
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path
                
            open_set.remove(current)
            closed_set.add(current)
            
            for neighbor in get_neighbors(current):
                if neighbor in closed_set:
                    continue
                    
                tentative_g_score = g_score[current] + 1
                
                if neighbor not in open_set:
                    open_set.add(neighbor)
                elif tentative_g_score >= g_score.get(neighbor, float('inf')):
                    continue
                    
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                
        return []  # No path found 