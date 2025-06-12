import os
import json
import time
import datetime
import re
import numpy as np
from typing import Dict, Tuple, List, Optional
from .core_module import CoreModule, GameTrajectory, Observation
import logging as logger

class MemoryModule(CoreModule):
    """
    A lightweight memory module: 
        1. stores the most recent N turns in a GameTrajectory deque.
        2. synthesises reflections with an LLM.
        3. stores navigation-related information like maps and explored areas (for Pokemon Red game only).
    """

    def __init__(self,
                 model_name: str = "claude-3-7-sonnet-latest",
                 cache_dir: str = "cache",
                 system_prompt: str = "",
                 prompt: str = "",
                 max_memory: int = 10):

        super().__init__(
            module_name="memory_module",
            model_name=model_name,
            system_prompt=system_prompt,
            prompt=prompt,
            cache_dir=cache_dir,
        )

        self.max_memory = max_memory
        
        # Navigation-related memory (for Pokemon Red game only)
        self.location_maps = {}  # Store maps for each location
        self.location_labels = {}  # Store labels for each location
        self.explored_areas = set()  # Store explored coordinates
        self.navigation_history = []  # Store navigation history

    def _load_trajectory(self) -> None:
        """Load and return trajectory entries (as already‑stringified lines) from disk."""
        
        trajectory = GameTrajectory(max_length=self.max_memory)
        if os.path.exists(self.module_file):
            try:
                with open(self.module_file, "r") as f:
                    entries = json.load(f)

                # keep only the last maxlen lines and push them into the deque
                for e in entries[-self.max_memory:]:
                    # expect the entry to have been stored as a ready‑to‑print line
                    if isinstance(e, str):
                        trajectory.add(e)
            except Exception as exc:
                print(f"[MemoryModule] failed to load trajectory: {exc}")
        else:
            print("trajectory entries do not exist.")
        
        return trajectory

    def _append_to_log(self, line: str) -> None:
        """
        Persist *just the printable line* per update.
        That keeps the on‑disk structure flat and forward‑compatible.
        """
        try:
            if os.path.exists(self.module_file):
                with open(self.module_file, "r") as f:
                    data = json.load(f)
            else:
                data = []

            data.append(line)
            with open(self.module_file, "w") as f:
                json.dump(data[-self.max_memory:], f, indent=2)
        except Exception as exc:
            print(f"[MemoryModule] failed to write log: {exc}")

    def _reflect(self,
                prev_context: str,
                current_state: str) -> str:
        """
        Ask the LLM to write a reflection given the running context string.
        """
        formatted_prompt = self.prompt.format(
            prev_context=prev_context or "None",
            current_observation=current_state,
        )

        raw = self.api_manager.text_only_completion(
            model_name=self.model_name,
            system_prompt=self.system_prompt,
            prompt=formatted_prompt,
            thinking=False,
            reasoning_effort=self.reasoning_effort,
            token_limit=self.token_limit,
        )
        # returned API response should be a tuple
        actual_raw_text = raw[0]
        # extract "reflection:" section if present
        m = re.search(
            r'(?:^|\n)(?:#\s*)?reflection:(.+?)(?=\n(?:#\s*)?[a-zA-Z]+:|$)',
            actual_raw_text, # Use the extracted text
            re.DOTALL | re.IGNORECASE,
        )
        return (m.group(1).strip() if m else actual_raw_text.strip()) or "No valid reflection produced."

    def process_observation(self,
                        observation: Observation,
                        game_state: dict) -> str:
        """
        Main entry point called by the agent each turn.
        Generates reflection and pushes a compact line into the trajectory.

        Args:
            observation: The new game observation
            
        Returns:
            processed_observation: An updated observation with processed data
        """

        """
        `-->` represents conversion performed by memory module
        game_trajctory |-- [obs_i, action_i]  |--> reflection


        (inspired by LMAct)
        Maybe we can add demonstrations as well
        """
        prev_context = observation.game_trajectory.get() or ""
        if observation.game_trajectory.background is None and observation.trajectory_includes_background:
            observation.game_trajectory.set_background(observation.get_background() or "Background not available.")

        reflection = self._reflect(
            prev_context=prev_context,
            current_state=str(game_state),
        )

        ts = datetime.datetime.now().isoformat(timespec="seconds")
        game_state.pop("img_path")
        
        if "processed_visual_description" in game_state and game_state["processed_visual_description"] is None:
            game_state.pop("processed_visual_description")
            
        # reflection excluded from game trajectory
        # reflection will be extracted by the reasoning module
        line = (
            f"##Turn Hash\n[{ts}]\n"
            f"###Obs\n{game_state}\n"
        )
        #f"###Reflection\n{reflection}\n"

        # add to dequeue
        observation.game_trajectory.add(line)
        # disk persistence
        self._append_to_log(line)

        observation.reflection = reflection

        return observation

    def update_action_memory(self,
                    observation: Observation,
                    action: str | None,
                    thought: str | None) -> str:
        """
        Main entry point called by the agent each turn.
        Generates reflection and pushes a compact line into the trajectory.

        Args:
            observation: The new game observation
            
        Returns:
            processed_observation: An updated observation with processed data
        """

        # build a single printable entry line
        ts = datetime.datetime.now().isoformat(timespec="seconds")
        line = (
            f"###Action\n{action}\n"
            f"###Thought\n{thought}\n"
        )

        # add to dequeue
        observation.game_trajectory.add(line)
        # disk persistence
        self._append_to_log(line)

        return observation

    def get_memory_summary(self, observation) -> dict[str, str]:
        """
        Provide the reasoning module with:
          • up‑to‑N past lines (already formatted by GameTrajectory)
          • no extra metadata dance
        """
        past = observation.game_trajectory.get() or "No previous game states available."
        latest = observation.game_trajectory.trajectory[-1] if observation.game_trajectory.trajectory else "N/A"

        return {
            "game_trajectory": past,
            "current_state": latest,   # includes (obs, action, thought)
            "reflection": latest.split("Reflection:", 1)[-1].strip()
                         if "Reflection:" in latest else "N/A",
        }

    def _parse_response(self, response):
        """
        Parse the reflection response from the LLM.
        
        Args:
            response (str): The raw response from the LLM
            
        Returns:
            dict: Parsed reflection data
        """
        
        if not response:
            return {"reflection": "No reflection generated."}
        
        # Try to extract reflection from structured format first
        reflection_match = re.search(r'(?:^|\n)(?:#\s*)?reflection:(.+?)(?=(?:\n(?:#\s*)?[a-zA-Z]+:)|$)', 
                                    response, re.DOTALL | re.IGNORECASE)
        
        if reflection_match:
            # Extract the reflection content from the pattern match
            reflection = reflection_match.group(1).strip()
        else:
            # If no structured format found, use the entire response
            reflection = response.strip()
            
        return {
            "reflection": reflection
        }

    # Navigation-related memory (for Pokemon Red game only)
    def update_navigation_memory(self, location: str, collision_map: np.ndarray, labels: Dict[Tuple[int, int], str] = None) -> None:
        """Update navigation-related memory."""
        self.location_maps[location] = collision_map
        if labels:
            if location not in self.location_labels:
                self.location_labels[location] = {}
            self.location_labels[location].update(labels)
    
    # Navigation-related memory (for Pokemon Red game only)
    def add_explored_area(self, location: str, coords: Tuple[int, int]) -> None:
        """Mark an area as explored."""
        self.explored_areas.add((location, coords))
    
    # Navigation-related memory (for Pokemon Red game only)
    def add_navigation_history(self, action: str, location: str, coords: Tuple[int, int]) -> None:
        """Add a navigation action to history."""
        self.navigation_history.append({
            'action': action,
            'location': location,
            'coords': coords,
            'timestamp': time.time()
        })
    
    # Navigation-related memory (for Pokemon Red game only)
    def get_location_map(self, location: str) -> Optional[np.ndarray]:
        """Get the collision map for a location."""
        return self.location_maps.get(location)
    
    # Navigation-related memory (for Pokemon Red game only)
    def get_location_labels(self, location: str) -> Dict[Tuple[int, int], str]:
        """Get the labels for a location."""
        return self.location_labels.get(location, {})
    
    # Navigation-related memory (for Pokemon Red game only)
    def is_explored(self, location: str, coords: Tuple[int, int]) -> bool:
        """Check if an area has been explored."""
        return (location, coords) in self.explored_areas
    
    # Navigation-related memory (for Pokemon Red game only)
    def get_navigation_history(self, limit: int = 10) -> List[dict]:
        """Get recent navigation history."""
        return self.navigation_history[-limit:]

    def read_map_name(self, memory) -> str:
        """Read the current map name from memory."""
        try:
            # Map name is stored at 0xD35E
            map_id = memory[0xD35E]
            # Map names are stored in a table starting at 0x4A000
            map_name_ptr = 0x4A000 + (map_id * 2)
            name_length = memory[map_name_ptr]
            name_bytes = memory[map_name_ptr + 1:map_name_ptr + 1 + name_length]
            return ''.join(chr(b) for b in name_bytes)
        except Exception as e:
            logger.error(f"Error reading map name: {e}")
            return None

    def read_inventory(self, memory) -> List[str]:
        """Read the current inventory from memory."""
        try:
            # Inventory starts at 0xD31E
            inventory = []
            for i in range(20):  # Max 20 items
                item_id = memory[0xD31E + i]
                if item_id == 0:
                    break
                item_name = self.get_item_name(memory, item_id)
                if item_name:
                    inventory.append(item_name)
            return inventory
        except Exception as e:
            logger.error(f"Error reading inventory: {e}")
            return []

    def read_party(self, memory) -> List[str]:
        """Read the current Pokemon party from memory."""
        try:
            # Party starts at 0xD163
            party = []
            for i in range(6):  # Max 6 Pokemon
                pokemon_id = memory[0xD163 + i]
                if pokemon_id == 0:
                    break
                pokemon_name = self.get_pokemon_name(memory, pokemon_id)
                if pokemon_name:
                    party.append(pokemon_name)
            return party
        except Exception as e:
            logger.error(f"Error reading party: {e}")
            return []

    def read_quest_state(self, memory) -> Dict[str, str]:
        """Read the current quest/objective states from memory."""
        try:
            # Quest flags start at 0xD7F1
            quest_state = {}
            
            # Check if player has received starter
            if memory[0xD7F1] & 0x01:
                quest_state["Get Starter Pokemon"] = "completed"
            else:
                quest_state["Get Starter Pokemon"] = "in_progress"
                
            # Check if player has received Pokedex
            if memory[0xD7F1] & 0x02:
                quest_state["Get Pokedex"] = "completed"
            else:
                quest_state["Get Pokedex"] = "in_progress"
                
            # Check if player has received first badge
            if memory[0xD7F1] & 0x04:
                quest_state["Get First Badge"] = "completed"
            else:
                quest_state["Get First Badge"] = "in_progress"
                
            return quest_state
        except Exception as e:
            logger.error(f"Error reading quest state: {e}")
            return {}

    def read_game_progress(self, memory) -> Dict[str, str]:
        """Read overall game progress metrics from memory."""
        try:
            progress = {}
            
            # Number of badges
            badges = memory[0xD356]
            progress["Badges"] = str(badges)
            
            # Number of Pokemon seen
            seen = sum(1 for b in memory[0xD2A7:0xD2A7+19] if b > 0)
            progress["Pokemon Seen"] = str(seen)
            
            # Number of Pokemon caught
            caught = sum(1 for b in memory[0xD2A7:0xD2A7+19] if b > 0)
            progress["Pokemon Caught"] = str(caught)
            
            # Current money
            money = (memory[0xD347] << 16) | (memory[0xD348] << 8) | memory[0xD349]
            progress["Money"] = str(money)
            
            return progress
        except Exception as e:
            logger.error(f"Error reading game progress: {e}")
            return {}

    def get_item_name(self, memory, item_id: int) -> Optional[str]:
        """Get the name of an item from its ID."""
        try:
            # Item names are stored in a table starting at 0x4B000
            item_name_ptr = 0x4B000 + (item_id * 2)
            name_length = memory[item_name_ptr]
            name_bytes = memory[item_name_ptr + 1:item_name_ptr + 1 + name_length]
            return ''.join(chr(b) for b in name_bytes)
        except Exception as e:
            logger.error(f"Error getting item name: {e}")
            return None

    def get_pokemon_name(self, memory, pokemon_id: int) -> Optional[str]:
        """Get the name of a Pokemon from its ID."""
        try:
            # Pokemon names are stored in a table starting at 0x4C000
            pokemon_name_ptr = 0x4C000 + (pokemon_id * 2)
            name_length = memory[pokemon_name_ptr]
            name_bytes = memory[pokemon_name_ptr + 1:pokemon_name_ptr + 1 + name_length]
            return ''.join(chr(b) for b in name_bytes)
        except Exception as e:
            logger.error(f"Error getting Pokemon name: {e}")
            return None
