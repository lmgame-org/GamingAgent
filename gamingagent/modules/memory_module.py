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
                reflection_system_prompt: str = "",
                reflection_prompt: str = "",
                summary_system_prompt: str = "",
                summary_prompt: str = "",
                max_memory: int = 10,
                token_limit: int = 100000, 
                use_reflection: bool = True,
                use_summary: bool = False,
                vllm_url=None,
                modal_url=None):

        print(f"memory module token limit: {token_limit}")

        super().__init__(
            module_name="memory_module",
            model_name=model_name,
            system_prompt=reflection_system_prompt,
            prompt=reflection_prompt,
            token_limit=token_limit,
            cache_dir=cache_dir,
            vllm_url=vllm_url,
            modal_url=modal_url
        )

        self.max_memory = max_memory
        self.use_reflection = use_reflection
        self.use_summary = use_summary
        self.summary_system_prompt = summary_system_prompt
        self.summary_prompt = summary_prompt
        self.current_summary = ""  # Store the current summary

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
                        # Check if this entry contains a summary and restore it
                        if e.startswith("##TRAJECTORY SUMMARY\n"):
                            summary_content = e.replace("##TRAJECTORY SUMMARY\n", "").strip()
                            if summary_content:
                                self.current_summary = summary_content
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

    def _summarize(self, game_trajectory: str) -> str:
        """
        Generate a summary of the game trajectory when it exceeds max_memory length.
        """
        if not self.summary_prompt or not self.use_summary:
            return ""
            
        formatted_prompt = self.summary_prompt.format(
            game_trajectory=game_trajectory,
            previous_summary=self.current_summary or "No previous summary."
        )

        try:
            print(f"[MemoryModule] Generating summary...")
            
            raw = self.api_manager.text_only_completion(
                model_name=self.model_name,
                system_prompt=self.summary_system_prompt,
                prompt=formatted_prompt,
                thinking=False,
                reasoning_effort=self.reasoning_effort,
                token_limit=self.token_limit,
            )

            # returned API response should be a tuple
            actual_raw_text = raw[0] if raw and len(raw) > 0 else ""
   
            # Clean and validate the response
            summary = actual_raw_text.strip() if actual_raw_text else ""
            
            # Check if we got a valid summary (not empty and not an error message)
            if summary and len(summary) > 10 and "no valid summary" not in summary.lower():
                print(f"[MemoryModule] Successfully generated summary. Length: {len(summary)} chars")
                return summary
            else:
                print(f"[MemoryModule] Generated invalid summary: '{summary[:100]}...'")
                # Return a basic fallback summary
                fallback_summary = f"FALLBACK SUMMARY: Game trajectory contained {len(game_trajectory)} characters of gameplay data. Previous summary: {self.current_summary[:200] if self.current_summary else 'None'}..."
                return fallback_summary
                
        except Exception as e:
            print(f"[MemoryModule] Error generating summary: {e}")
            # Return a basic fallback summary
            fallback_summary = f"FALLBACK SUMMARY: Game trajectory contained {len(game_trajectory)} characters of gameplay data. Previous summary: {self.current_summary[:200] if self.current_summary else 'None'}..."
            print(f"[MemoryModule] Using fallback summary due to error.")
            return fallback_summary

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

        if self.use_reflection:
            reflection = self._reflect(
                prev_context=prev_context,
                current_state=str(game_state),
            )
        else:
            reflection = None

        observation = self.update_observation_memory(
            observation=observation,
        )
        observation.reflection = reflection if self.use_reflection else None

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

        # Get current trajectory content for summarization
        current_trajectory = observation.game_trajectory.get() or ""
        char_len = len(current_trajectory)
        est_tokens = char_len // 3

        # Check if we need to summarize before adding new entry
        if self.use_summary and len(observation.game_trajectory.trajectory) >= self.max_memory or (est_tokens > 10_000 and "o3" in self.model_name):
            # Trigger summarisation if > 10 000 tokens

            # Only attempt summarization if we have substantial content
            if len(current_trajectory.strip()) > 50:  # Ensure we have meaningful content
                print(f"[MemoryModule] Trajectory reached length: {len(observation.game_trajectory.trajectory)}.")
                print(f"[MemoryModule] Current trajectory length: {char_len} chars")
                print(f"[MemoryModule] Trajectory ≈ {est_tokens:,} tokens — summarizing…")
                
                # Generate summary
                new_summary = self._summarize(current_trajectory)
                if new_summary and new_summary != "":
                    self.current_summary = new_summary
                    
                    # Clear the trajectory and replace with summary
                    observation.game_trajectory.trajectory.clear()
                    
                    # Clear the disk file as well since we're starting fresh
                    try:
                        with open(self.module_file, "w") as f:
                            json.dump([], f, indent=2)
                    except Exception as exc:
                        print(f"[MemoryModule] failed to clear log file: {exc}")
                    
                    # Add summary as the first entry
                    summary_line = f"##TRAJECTORY SUMMARY\n{self.current_summary}\n\n"
                    observation.game_trajectory.add(summary_line)
                    
                    # Persist summary to disk
                    self._append_to_log(summary_line)
                    
                    print(f"[MemoryModule] Successfully generated and saved summary. Length: {len(self.current_summary)} chars")
                else:
                    print(f"[MemoryModule] Failed to generate valid summary after all retries. Keeping existing trajectory.")
            else:
                print(f"[MemoryModule] Insufficient trajectory content ({len(current_trajectory)} chars) for summarization. Skipping.")

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
            "reflection": observation.reflection if hasattr(observation, 'reflection') and observation.reflection else "N/A",
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
