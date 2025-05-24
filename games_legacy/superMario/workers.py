import time
import os
import pyautogui
import numpy as np

from tools.utils import encode_image, log_output, extract_python_code
from tools.serving.api_providers import anthropic_completion, openai_completion, gemini_completion

def worker_short(thread_id, offset, system_prompt, api_provider, model_name):
    """
    Worker function for short-term (1 second) motion control.
    1) Sleeps 'offset' seconds before starting (to stagger starts).
    2) Continuously takes screenshots, calls Anthropic with streaming output, logs latency, executes returned code, etc.
    """
    all_response_time = []

    time.sleep(offset)
    print(f"[Thread {thread_id} - SHORT] Starting after {offset}s delay...")

    short_prompt = (
        f"Analyze the current game state and generate PyAutoGUI code to control Mario "
        "for the next 1 second.\n"
        "Mario's position most likely has moved forward when the generated code gets to execute.\n"
        "Your objective is to avoid obstacles, enemies, and hazards.\n"

        "### General Controls:\n"
        "- Press 'Enter' to start the game ONLY IF the game hasn't started.\n"
        "  Otherwise the game will be paused.\n"
        "- Press the right arrow to move forward.\n"
        "- Press 'x' along with right/left arrow to jump over obstacles or gaps. Be very careful with gaps, do lopped jumps if necessary.\n\n"

        "### Strategies and Caveats:\n"
        "- Whenever a gap is detected, AVOID jumping over the gap. Only do small position adjustments to prepare for big jump.\n"
        "- If an obstacle or enemy is near, move/jump left to dodge.\n"
        "- If an enemy is detected, do one big jump ONLY IF very confident, ortherwise do consecutive short jumps.\n"
        "- If in doubt, take a more defensive approaches like moving to the left (move back).\n"
        "- Sleep and do nothing if no obvious danger."

        "### Output Format:\n"
        "- Output ONLY the Python code for PyAutoGUI commands.\n"
        "- Include brief comments for each action.\n"
    )

    try:
        while True:
            screen_width, screen_height = pyautogui.size()
            region = (0, 0, screen_width, screen_height)
            screenshot = pyautogui.screenshot(region=region)

            thread_folder = f"cache/mario/thread_{thread_id}"
            os.makedirs(thread_folder, exist_ok=True)

            screenshot_path = os.path.join(thread_folder, "screenshot.png")
            screenshot.save(screenshot_path)

            base64_image = encode_image(screenshot_path)

            start_time = time.time()

            if api_provider == "anthropic":
                generated_code_str = anthropic_completion(system_prompt, model_name, base64_image, short_prompt)
            elif api_provider == "openai":
                generated_code_str = openai_completion(system_prompt, model_name, base64_image, short_prompt)
            elif api_provider == "gemini":
                generated_code_str = gemini_completion(system_prompt, model_name, base64_image, short_prompt)
            else:
                raise NotImplementedError(f"API provider: {api_provider} is not supported.")

            end_time = time.time()
            latency = end_time - start_time
            all_response_time.append(latency)

            print(f"[Thread {thread_id} - SHORT] Request latency: {latency:.2f}s")
            avg_latency = np.mean(all_response_time)
            print(f"[Thread {thread_id} - SHORT] Latencies: {all_response_time}")
            print(f"[Thread {thread_id} - SHORT] Average latency: {avg_latency:.2f}s")

            print(f"\n[Thread {thread_id} - SHORT] --- Generation (Streaming) ---\n{generated_code_str}\n")

            clean_code = extract_python_code(generated_code_str)
            log_output(thread_id, f"[Thread {thread_id} - SHORT] Python code to be executed:\n{clean_code}\n", "mario")
            print(f"[Thread {thread_id} - SHORT] Python code to be executed:\n{clean_code}\n")

            try:
                exec(clean_code)
            except Exception as e:
                print(f"[Thread {thread_id} - SHORT] Error executing code: {e}")

    except KeyboardInterrupt:
        print(f"[Thread {thread_id} - SHORT] Interrupted by user. Exiting...")

def worker_long(thread_id, offset, system_prompt, api_provider, model_name):
    """
    Worker function for long-term (2 seconds) motion control.
    1) Sleeps 'offset' seconds before starting (to stagger starts).
    2) Continuously takes screenshots, calls Anthropic with streaming output, logs latency, executes returned code, etc.
    """
    all_response_time = []

    time.sleep(offset)
    print(f"[Thread {thread_id} - LONG] Starting after {offset}s delay...")

    long_prompt = (
        f"Analyze the current game state and generate PyAutoGUI code to control Mario "
        "for the next 2 seconds.\n"
        "Mario's position most likely has moved forward when the generated code gets to execute.\n"
        "Your objective is to make progress while avoiding obstacles, enemies, and hazards.\n"

        "### General Controls:\n"
        "- Press 'Enter' to start the game ONLY IF the game hasn't started.\n"
        "  Otherwise the game will be paused.\n"
        "- Press the right arrow to move forward.\n"
        "- Press 'x' along with right/left arrow to jump over obstacles or gaps. Be very careful with gaps, do lopped jumps if necessary.\n\n"

        "### Strategies and Caveats:\n"
        "- Don't move too fast, as unseen enemies may appear from off-screen.\n"
        "- If an obstacle or enemy is near, move forward in small increments and be ready to jump.\n"
        "- Avoid walking forward without jumping as Mario can run into off-screen enemies.\n"
        "- If a gap is detected, make sure to leave room for acceleration and then jump. Otherwise, move left first to get more space for acceleration.\n"
        "- If in doubt, take a more defensive approaches like moving to the left (move back).\n"
        "- Secondary goal: only if very safe, collect as many question mark blocks as possible.\n\n"

        "### Output Format:\n"
        "- Output ONLY the Python code for PyAutoGUI commands.\n"
        "- Include brief comments for each action.\n"
    )

    try:
        while True:
            screen_width, screen_height = pyautogui.size()
            region = (0, 0, screen_width, screen_height)
            screenshot = pyautogui.screenshot(region=region)

            thread_folder = f"cache/mario/thread_{thread_id}"
            os.makedirs(thread_folder, exist_ok=True)

            screenshot_path = os.path.join(thread_folder, "screenshot.png")
            screenshot.save(screenshot_path)

            base64_image = encode_image(screenshot_path)

            start_time = time.time()

            if api_provider == "anthropic":
                generated_code_str = anthropic_completion(system_prompt, model_name, base64_image, long_prompt)
            elif api_provider == "openai":
                generated_code_str = openai_completion(system_prompt, model_name, base64_image, long_prompt)
            elif api_provider == "gemini":
                generated_code_str = gemini_completion(system_prompt, model_name, base64_image, long_prompt)
            else:
                raise NotImplementedError(f"API provider: {api_provider} is not supported.")

            end_time = time.time()
            latency = end_time - start_time
            all_response_time.append(latency)

            print(f"[Thread {thread_id} - LONG] Request latency: {latency:.2f}s")
            avg_latency = np.mean(all_response_time)
            print(f"[Thread {thread_id} - LONG] Latencies: {all_response_time}")
            print(f"[Thread {thread_id} - LONG] Average latency: {avg_latency:.2f}s")

            print(f"\n[Thread {thread_id} - LONG] --- Generation (Streaming) ---\n{generated_code_str}\n")

            clean_code = extract_python_code(generated_code_str)
            log_output(thread_id, f"[Thread {thread_id} - LONG] Python code to be executed:\n{clean_code}\n", "mario")
            print(f"[Thread {thread_id} - LONG] Python code to be executed:\n{clean_code}\n")

            try:
                exec(clean_code)
            except Exception as e:
                print(f"[Thread {thread_id} - LONG] Error executing code: {e}")

    except KeyboardInterrupt:
        print(f"[Thread {thread_id} - LONG] Interrupted by user. Exiting...")