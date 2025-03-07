import time
import os
import pyautogui
import numpy as np
import re
import json
import pywinctl

from tools.utils import encode_image, log_output, extract_python_code
from tools.serving.api_providers import anthropic_completion, openai_completion, gemini_completion

CACHE_DIR = "cache/space_invaders_cache"

def log_action(move, thought, latency):
    """
    Record the thought process of AI
    """
    log_file_path = os.path.join(CACHE_DIR, "space_invaders_moves.log")
    
    log_entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Action: {move}, Thought: {thought}, Latency: {latency:.2f} sec\n"
    
    try:
        with open(log_file_path, "a") as log_file:
            log_file.write(log_entry)
    except Exception as e:
        print(f"[ERROR] Failed to write log entry: {e}")

def get_game_window():
    """
    Capture the game window precisely
    """
    all_windows = pywinctl.getAllTitles()

    for title in all_windows:
        if "Space Invaders" in title:  
            game_window = pywinctl.getWindowsWithTitle(title)[0] 
            print(f"[INFO] Find Space Invaders Window: Position({game_window.left}, {game_window.top}), Size({game_window.width}x{game_window.height})")
            return game_window 

    print("[ERROR] Can't find game window！")
    return None


def space_invaders_worker(system_prompt, api_provider, model_name, crop_top=50, crop_bottom=400, crop_left=50, crop_right=700, prev_response=""):
    """
    1) Capture the screenshot of current game state
    2) Analyze the screenshot and design the optimal strategy
    3) Control the weapon
    """

    game_window = get_game_window()
    if not game_window:
        print("[ERROR] Can't find game window.")
        return None
    x, y, width, height = game_window.left, game_window.top, game_window.width, game_window.height
    screenshot = pyautogui.screenshot(region=(x, y, width, height))

    # Save to the cache
    os.makedirs(CACHE_DIR, exist_ok=True)
    screenshot_path = os.path.join(CACHE_DIR, "screenshot.png")
    screenshot.save(screenshot_path)
    base64_image = encode_image(screenshot_path)

    # Generate the prompt for AI to analyze
    prompt = f"""
You are an expert Space Invaders player. Your goal is to analyze the game state and determine the best action.

## GAME RULES ##
- You control a cannon at the bottom of the screen.
- Your goal is to destroy all aliens while avoiding enemy bullets.
- You can move left/right and shoot. You can only have one bullet on the screen at a time.

## ACTION PRIORITIES ##
**Avoid danger first (highest priority)**
   - If an **enemy bullet is about to hit you**, move left or right immediately to avoid it.
   - If aliens are too close to the bottom, move left or right **to stay alive**.

**Find the best shooting position**
   - If **you are not aligned with the closest alien**, move left or right to position yourself for a clean shot.
   - Do **not** move randomly—move **only if it helps you shoot better**.

**Shoot strategically**
   - If **there is no immediate threat** and you are in a **good shooting position**, shoot the closest alien directly above you.
   - **If a UFO appears, prioritize shooting it** for high points.
   - **You can only have one bullet on the screen at a time**, so wait for the previous shot to clear before shooting again.

## STRICT OUTPUT FORMAT ##
- Respond **ONLY** with a valid JSON object, **without any markdown formatting**.
- Example valid response:
{{
  "action": "move_left" | "move_right" | "shoot" | "stay",
  "thought": "Brief explanation of the move."
}}
- **DO NOT INCLUDE:** "```json", explanations, or additional text.
"""


    # Call API
    start_time = time.time()
    
    if api_provider == "anthropic":
        response = anthropic_completion(system_prompt, model_name, base64_image, prompt)
    elif api_provider == "openai":
        response = openai_completion(system_prompt, model_name, base64_image, prompt)
    elif api_provider == "gemini":
        response = gemini_completion(system_prompt, model_name, base64_image, prompt)
    else:
        raise NotImplementedError(f"API provider: {api_provider} is not supported.")

    latency = time.time() - start_time
    response = response.strip().replace("```json", "").replace("```", "").strip()
    print(f"AI Response: {response}")

    # Parse the json response
    try:
        if not response.strip():
            print("[ERROR] AI returned an empty response. Skipping this response.")
            return json.dumps({"action": "stay", "thought": "AI response was invalid, defaulting to stay."}) 
        action_data = json.loads(response)
        action = action_data["action"]
        thought = action_data["thought"]
    except Exception as e:
        print(f"[ERROR] Failed to parse AI response: {e}")
        return json.dumps({"action": "stay", "thought": "AI response was invalid, defaulting to stay."})

    print(f"AI Decision: {action} | Reason: {thought}")

    # Perform the action in the game
    if action == "start":
        print("[INFO] Pressing Enter to start the game...")
        pyautogui.press("enter")  # Press Enter to start the game
        time.sleep(3)  # Wait for game to load
        print("[INFO] Enter key pressed. Waiting for game to start.")

    elif action == "move_left":
        print("[INFO] Move to the left")
        pyautogui.keyDown("left")
        time.sleep(0.3)  
        pyautogui.keyUp("left")

    elif action == "move_right":
        print("[INFO] Move to the right")
        pyautogui.keyDown("right")
        time.sleep(0.3)
        pyautogui.keyUp("right")

    elif action == "shoot":
        print("[INFO] Fire！")
        pyautogui.keyDown("space")
        time.sleep(0.2)  
        pyautogui.keyUp("space")

    elif action == "stay":
        print("[INFO] Stay")
        pass 

    
    log_action(action, thought, latency)

    
