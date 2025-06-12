import logging
from typing import Dict, List, Tuple, Optional
from .navigation_system import NavigationSystem
from tools.serving.api_manager import APIManager

logger = logging.getLogger(__name__)

NAVIGATION_PROMPT = """Your job is to provide navigation advice for another model playing Pokemon Red.

You will be given a navigation goal, an text_based map of the area, and a list of locations that have been labeled by the model.

Read the text_based map VERY carefully.

It is important to understand the grid system used on the map and for the label list:

1. The top-left corner of the location is at 0, 0
2. The first number is the column number, increasing horizontally to the right.
3. The second number is the row number, increasing vertically downward.

Some example reasoning: If the top left of the text_based map is at (3, 38), then we are at least 38 units away from the top of the map. This is
relevant when looking for exits on the north or left of the map.

#### SPECIAL NAVIGATION INSTRUCTIONS WHEN TRYING TO REACH A LOCATION #####
Pay attention to the following procedure when trying to reach a specific location (if you know the coordinates).
1. Inspect the text_based map
2. Find where your destination is on the map using the coordinate system (column, row) and see if it is labeled with a number.
    2a. If not, instead find a nearby location labeled with a number
3. Trace a path from there back to the player character (PP) following the numbers on the map, in descending order.
    3a. So if your destination is numbered 20, then 19, 18...descending all the way to 1 and then PP.
4. Navigate via the REVERSE of this path.
###########################################

Avoid suggesting pathing into Explored Areas (marked with x). This is very frequently the wrong way!

Provide navigation directions to the other model that are very specific, explaining where to go point by point. For example:

Example 1: "You have not yet explored the northeast corner, and it may be worth looking there. Reach there by first heading east to (17, 18), then south to (17, 28) then east to (29, 28), then straight north all the way to (29, 10)."
Example 2: "Based on my knowledge of Pokemon Red, the exit from this area should be in the northwest corner. Going straight north or west from here is a dead-end. Instead, go south to (10, 19), then east to (21, 19), then north to (21, 9) where there is an explored path which may lead to progress."

You may use your existing knowledge of Pokemon Red but otherwise stick scrupulously to what is on the map. Do not hallucinate extra details.

Tip on using the navigate_to tool: Use it frequently to path quickly, but note that it will not take you offscreen.
"""

class NavigationAssistant:
    def __init__(self, navigation_system: NavigationSystem, model_name: str = "claude-3-7-sonnet-latest", vllm_url: Optional[str] = None, modal_url: Optional[str] = None):
        self.navigation_system = navigation_system
        self.model_name = model_name
        
        # Initialize API manager
        self.api_manager = APIManager(
            game_name="pokemon_red",
            vllm_url=vllm_url,
            modal_url=modal_url
        )
        
    def get_navigation_advice(self, location: str, navigation_goal: str) -> str:
        """Get navigation advice for the current location and goal."""
        # Get ASCII map with distances and labels
        ascii_map = self.navigation_system.get_ascii_map(location)
        
        # Get labels for the location
        labels = self.navigation_system.location_labels.get(location, {})
        labels_text = "No labels yet."
        if labels:
            labels_text = '\n'.join(f"({x}, {y}): {label}" for (x, y), label in labels.items())
            
        # Create prompt for LLM
        prompt = f"""Here is a map of the current location:

Current location: {location}

{ascii_map}

Remember, higher numbers in the first coordinate are to the RIGHT. Higher numbers in the second coordinate are DOWN.

Here are some labels:

{labels_text}

Here is the current navigation goal:

{navigation_goal}
"""
        
        # Get advice from LLM using API manager
        advice, _ = self.api_manager.text_only_completion(
            model_name=self.model_name,
            system_prompt=NAVIGATION_PROMPT,
            prompt=prompt
        )
        
        return advice
        
    def auto_path_to_location(self, location: str, goal_coords: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Find and return a path to a specific location."""
        # Get player position from collision map
        collision_map = self.navigation_system.collision_maps.get(location)
        if not collision_map:
            return []
            
        # Find player position
        player_pos = None
        for y in range(collision_map.shape[0]):
            for x in range(collision_map.shape[1]):
                if collision_map[y, x] == 'P':
                    player_pos = (x, y)
                    break
            if player_pos:
                break
                
        if not player_pos:
            return []
            
        # Find path using A*
        return self.navigation_system.find_path(location, player_pos, goal_coords) 