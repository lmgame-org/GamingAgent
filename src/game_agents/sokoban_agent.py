import os
import json
import re
import time
from typing import Any, Dict, Optional
from .base_agent import BaseGameAgent
from tools.serving.api_manager import APIManager
from src.prompts.sokoban import get_sokoban_prompt

class SokobanAgent(BaseGameAgent):
    """
    Sokoban agent: each worker calls API or reads state and returns raw `api_response`.
    The inherited `step` method aggregates these into `thought`, extracts `action`, and returns
    the final result dict with keys `thought`, `action`, `reward`, and `done`.
    Game-specific parameters (e.g., level) are provided via `game_config`.
    Uses `session_dir` (from BaseGameAgent) for all caching.
    """

    def __init__(
        self,
        game_name: str,
        model_name: str,
        modality: str,
        api_provider: str,
        thinking: bool = False,
        session_dir: Optional[str] = None,
        game_config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            game_name=game_name,
            model_name=model_name,
            modality=modality,
            api_provider=api_provider,
            thinking=thinking,
            session_dir=session_dir,
            game_config=game_config
        )
        cfg = game_config or {}
        self.level = cfg.get("level", 1)
        self.prev_response = cfg.get("prev_response", "")
        self.timestamp = cfg.get("timestamp", time.strftime("%Y%m%d_%H%M%S"))
        if not self.session_dir:
            raise ValueError("session_dir must be provided for caching")

        # Initialize API manager with context
        self.api_manager = APIManager(
            game_name="sokoban",
            base_cache_dir="cache",
            session_dir=self.session_dir,
            model_name=model_name,
            modality=modality,
            api_provider=api_provider,
            thinking=thinking,
            timestamp=self.timestamp
        )

        # Register workers: must return dict with key 'api_response'
        self.add_worker("vision_worker", self._vision_worker)
        self.add_worker("reasoning_worker", self._reasoning_worker)

    
    def _vision_worker(self,*args, **kwargs):
        matrix_path = os.path.join(self.cache_dir, "game_state.json")
        if not os.path.exists(matrix_path):
            return "No board available."

        with open(matrix_path, "r") as f:
            matrix = json.load(f)

        item_map = {
            '#': 'Wall', '@': 'Worker', '$': 'Box', '?': 'Dock',
            '*': 'Box on Dock', ' ': 'Empty'
        }

        table = ["ID  | Item Type    | Position", "-" * 36]
        item_id = 1
        for row_idx, row in enumerate(matrix):
            for col_idx, cell in enumerate(row):
                item_type = item_map.get(cell, 'Unknown')
                table.append(f"{item_id:<3} | {item_type:<12} | ({col_idx}, {row_idx})")
                item_id += 1

        return "\n".join(table)


    def _reasoning_worker(self, *args, **kwargs) -> Dict[str, Any]:
        """Builds prompt, calls API, and returns raw response"""
        screenshot_path = os.path.join(self.session_dir, "sokoban_screenshot.png")
        prompt = get_sokoban_prompt("", "")  # can include prev_response if tracked
        if self.modality == "text-only":
            resp, _ = self.api_manager.text_only_completion(
                model_name=self.model_name,
                system_prompt="",
                prompt=prompt,
                thinking=self.thinking
            )
        else:
            resp, _ = self.api_manager.vision_text_completion(
                model_name=self.model_name,
                system_prompt="",
                prompt=prompt,
                image_path=screenshot_path,
                thinking=self.thinking
            )
        return {"api_response": resp}
    
    def _run_reasoning(self):
        levels_path = os.path.join(self.cache_dir, "levels_dim.json")
        screenshot_path = os.path.join(self.cache_dir, "sokoban_screenshot.png")

        with open(levels_path, "r") as f:
            dims = json.load(f)

        level_dims = dims.get(f"level_{self.level}")
        if not level_dims:
            raise ValueError(f"No level info for level {self.level}")

        rows, cols = level_dims["rows"], level_dims["cols"]

        _, _, cropped_img = get_annotate_img(
            screenshot_path, 0, 0, 0, 0, rows, cols, cache_dir=self.cache_dir
        )

        board_str = self._read_board()
        prompt = get_sokoban_prompt(self.prev_response, board_str)

        base64_image = encode_image(cropped_img) if "o3-mini" not in self.model_name else None
        start_time = time.time()

        # Choose API
        provider = self.provider
        model = self.model_name
        if provider == "anthropic" and self.modality == "text-only":
            response = anthropic_text_completion(self.system_prompt, model, prompt, thinking=True)
        elif provider == "anthropic":
            response = anthropic_completion(self.system_prompt, model, base64_image, prompt, thinking=True)
        elif provider == "openai" and "o3" in model and self.modality == "text-only":
            response = openai_text_reasoning_completion(self.system_prompt, model, prompt)
        elif provider == "openai":
            response = openai_completion(self.system_prompt, model, base64_image, prompt)
        elif provider == "gemini" and self.modality == "text-only":
            response = gemini_text_completion(self.system_prompt, model, prompt)
        elif provider == "gemini":
            response = gemini_completion(self.system_prompt, model, base64_image, prompt)
        elif provider == "deepseek":
            response = deepseek_text_reasoning_completion(self.system_prompt, model, prompt)
        else:
            raise NotImplementedError(f"Unsupported provider: {provider}")

        latency = time.time() - start_time
        print(f"[SokobanAgent] Model responded in {latency:.2f}s")

        # Parse model response
        pattern = r"move:\s*(\w+),\s*thought:\s*(.*)"
        matches = re.findall(pattern, response, re.IGNORECASE)
        results = [{"move": m.strip().lower(), "thought": t.strip()} for m, t in matches]

        for item in results:
            log_output("sokoban_worker", f"move: {item['move']}, thought: {item['thought']}", "sokoban", mode="a")

        return results
