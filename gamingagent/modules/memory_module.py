import os
import json
import time
import datetime
import re
from .core_module import CoreModule, GameTrajectory

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

        self.trajectory = GameTrajectory(max_length=max_memory)
        self._load_trajectory()

    def _load_trajectory(self) -> None:
        """Reload trajectory entries (as already‑stringified lines) from disk."""
        
        if os.path.exists(self.module_file):
            try:
                with open(self.module_file, "r") as f:
                    entries = json.load(f)

                # keep only the last maxlen lines and push them into the deque
                for e in entries[-self.trajectory.history_length:]:
                    # expect the entry to have been stored as a ready‑to‑print line
                    if isinstance(e, str):
                        self.trajectory.add(e)
            except Exception as exc:
                print(f"[MemoryModule] failed to load trajectory: {exc}")
        else:
            print("memory isn't reloaded as trajectory entries do not exist.")

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
                json.dump(data[-self.trajectory.history_length:], f, indent=2)
        except Exception as exc:
            print(f"[MemoryModule] failed to write log: {exc}")

    def _reflect(self,
                 prev_context: str,
                 current_state: str,
                 last_action: str | None,
                 last_thought: str | None) -> str:
        """
        Ask the LLM to write a reflection given the running context string.
        """
        formatted_prompt = self.prompt.format(
            prev_context=prev_context or "None",
            current_observation=current_state,
            last_action=str(last_action) if last_action else "None",
            last_thought=str(last_thought) if last_thought else "None",
        )

        raw = self.api_manager.text_only_completion(
            model_name=self.model_name,
            system_prompt=self.system_prompt,
            prompt=formatted_prompt,
            thinking=False,
            reasoning_effort=self.reasoning_effort,
            token_limit=self.token_limit,
        )

        # extract "reflection:" section if present
        m = re.search(
            r'(?:^|\n)(?:#\s*)?reflection:(.+?)(?=\n(?:#\s*)?[a-zA-Z]+:|$)',
            raw,
            re.DOTALL | re.IGNORECASE,
        )
        return (m.group(1).strip() if m else raw.strip()) or "No valid reflection produced."

    def update_memory(self,
                      game_state: str,
                      action: str | None = None,
                      thought: str | None = None) -> str:
        """
        Main entry point called by the agent each turn.
        Generates reflection and pushes a compact line into the trajectory.
        """
        prev_context = self.trajectory.get() or ""
        reflection = self._reflect(
            prev_context=prev_context,
            current_state=str(game_state),
            last_action=action,
            last_thought=thought,
        )

        # build a single printable entry line
        ts = datetime.datetime.now().isoformat(timespec="seconds")
        line = (
            f"[{ts}] "
            f"Obs: {game_state} | "
            f"Action: {action} | Thought: {thought} | "
            f"Reflection: {reflection}"
        )

        # add to dequeue
        self.trajectory.add(line)
        # disk persistence
        self._append_to_log(line)

        return reflection

    def get_memory_summary(self) -> dict[str, str]:
        """
        Provide the reasoning module with:
          • up‑to‑N past lines (already formatted by GameTrajectory)
          • no extra metadata dance
        """
        past = self.trajectory.get() or "No previous game states available."
        latest = self.trajectory.trajectory[-1] if self.trajectory.trajectory else ""

        return {
            "game_trajectory": past,
            "current_state": latest,   # includes obs/action/thought
            "reflection": latest.split("Reflection:", 1)[-1].strip()
                         if "Reflection:" in latest else "",
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
