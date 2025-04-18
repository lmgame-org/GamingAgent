import os
import re
import time
import json
import datetime
from src.game_agents.base_agent import BaseGameAgent
from tools.utils import get_annotate_img, encode_image, log_output
from tools.serving.api_providers import (
    anthropic_completion, anthropic_text_completion,
    openai_completion, openai_text_reasoning_completion,
    gemini_completion, gemini_text_completion,
    deepseek_text_reasoning_completion
)
import pyautogui

from src.prompts.tile_2048 import get_2048_prompt, get_2048_read_prompt

class Tile2048Agent(BaseGameAgent):
    def __init__(self, model_name, provider, modality="vision-text"):
        super().__init__()
        self.model_name = model_name
        self.provider = provider
        self.modality = modality
        self.cache_dir = "cache/2048"
        self.prev_response = ""
        self.system_prompt = ""
        self.datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    def worker(self, task: str, **kwargs):
        if task == "vision":
            return self._read_board()
        elif task == "reason":
            return self._run_reasoning()
        elif task == "memory":
            return self.prev_response
        else:
            raise ValueError(f"[Tile2048Agent] Unknown task: {task}")


    def step(self):
        result = self.worker("reason")
        if not result:
            print("[Tile2048Agent] No result from model.")
            return None

        move = result[0]["move"]
        thought = result[0]["thought"]
        self.prev_response = f"move: {move}, thought: {thought}"
        print(f"[Tile2048Agent] move: {move}, thought: {thought}")
        return move

    def _run_reasoning(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        screenshot_path = os.path.join(self.cache_dir, "temp_screenshot.png")
        cropped_path = os.path.join(self.cache_dir, "annotated_crop.png")

        # Capture screenshot and annotate
        
        pyautogui.screenshot(screenshot_path)

        _, _, cropped_img = get_annotate_img(
            screenshot_path,
            crop_left=0, crop_right=0, crop_top=0, crop_bottom=0,
            grid_rows=4, grid_cols=4,
            cache_dir=self.cache_dir,
            black=True
        )

        base64_image = encode_image(cropped_img) if "o3-mini" not in self.model_name else None

        prompt = get_2048_prompt(self.prev_response, "<injected board text>")

        # Choose provider
        start_time = time.time()
        model = self.model_name
        provider = self.provider
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
            raise NotImplementedError(f"[2048Agent] Unsupported provider: {provider}")

        latency = time.time() - start_time
        print(f"[2048Agent] Model responded in {latency:.2f}s")

        pattern = r"move:\s*(\w+),\s*thought:\s*(.*)"
        matches = re.findall(pattern, response, re.IGNORECASE)
        results = [{"move": m.strip().lower(), "thought": t.strip()} for m, t in matches]

        for item in results:
            log_output("tile_2048_agent", f"move: {item['move']}, thought: {item['thought']}", "2048", mode="a")

        return results
    
    def _read_board(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        screenshot_path = os.path.join(self.cache_dir, "temp_screenshot.png")

        pyautogui.screenshot(screenshot_path)

        _, _, cropped_img = get_annotate_img(
            screenshot_path,
            crop_left=0, crop_right=0, crop_top=0, crop_bottom=0,
            grid_rows=4, grid_cols=4,
            cache_dir=self.cache_dir,
            black=True
        )

        base64_image = encode_image(cropped_img) if "o3-mini" not in self.model_name else None

        # Prompt the model to extract board layout
        prompt = get_2048_read_prompt()
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
            raise NotImplementedError(f"[2048Agent] Unsupported provider: {provider}")

        # Optional: log it
        log_output("tile_2048_read_worker", f"Board extracted:\n{response.strip()}", "2048", mode="a")
        return response.strip()
