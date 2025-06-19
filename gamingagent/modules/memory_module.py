import os
import json
import time
import datetime
import re
from .core_module import CoreModule, GameTrajectory, Observation

class MemoryModule(CoreModule):
    """
    A lightweight memory module: 
        1. stores the most recent N turns in a GameTrajectory deque.
        2. synthesises reflections with an LLM.
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

        self.max_memory=max_memory

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

    def process_observation(self, observation: Observation) -> str:
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
        game_state = observation.get_perception_summary()

        prev_context = observation.game_trajectory.get() or ""
        if observation.game_trajectory.background is None and observation.trajectory_includes_background:
            observation.game_trajectory.set_background(observation.get_background() or "Background not available.")

        reflection = self._reflect(
            prev_context=prev_context,
            current_state=str(game_state),
        )

        # observation = self.update_observation_memory(
        #     observation=observation,
        #     game_state=game_state
        # )
        observation = self.update_observation_memory(
            observation=observation,
        )
        observation.reflection = reflection

        return observation

    def update_observation_memory(self, observation: Observation) -> str:
        game_state = observation.get_perception_summary()

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
