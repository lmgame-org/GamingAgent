import os
import json
import re
import time
from src.game_agents.base_agent import BaseGameAgent
from tools.utils import encode_image, get_annotate_img, log_output
from tools.serving.api_providers import (
    anthropic_completion, anthropic_text_completion,
    openai_completion, openai_text_reasoning_completion,
    gemini_completion, gemini_text_completion,
    deepseek_text_reasoning_completion
)
from src.prompts.sokoban import get_sokoban_prompt

class SokobanAgent(BaseGameAgent):
    def __init__(self, model_name, provider, level=1, modality="vision-text"):
        super().__init__()
        self.model_name = model_name
        self.provider = provider
        self.level = level
        self.modality = modality
        self.cache_dir = "cache/sokoban"
        self.prev_response = ""
        self.system_prompt = ""

    def worker(self, task: str, **kwargs):
        if task == "vision":
            return self._read_board()

        elif task == "reason":
            return self._run_reasoning()

        elif task == "memory":
            return self.prev_response

        else:
            raise ValueError(f"[SokobanAgent] Unknown worker task: {task}")

    def step(self):
        response = self.worker("reason")
        if not response:
            print("[SokobanAgent] No move returned.")
            return None

        move = response[0]["move"]
        thought = response[0]["thought"]
        self.prev_response = f"move: {move}, thought: {thought}"
        print(f"[SokobanAgent] move: {move} | thought: {thought}")
        return move
    



    #------ worker details-----------#
    def _read_board(self):
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
