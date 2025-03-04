import time
import os
import pyautogui
import numpy as np

from tools.utils import encode_image, log_output, extract_python_code, read_log_to_string
from tools.serving.api_providers import anthropic_completion, openai_completion, gemini_completion

def worker_tetris(
    thread_id,
    offset,
    system_prompt,
    api_provider,
    model_name,
    plan_seconds,
):
    """
    A single Tetris worker that plans moves for 'plan_seconds'.
    1) Sleeps 'offset' seconds before starting (to stagger starts).
    2) Continuously:
        - Captures a screenshot
        - Calls the LLM with a Tetris prompt that includes 'plan_seconds'
        - Extracts the Python code from the LLM output
        - Executes the code with `exec()`
    """
    all_response_time = []

    time.sleep(offset)
    print(f"[Thread {thread_id}] Starting after {offset}s delay... (Plan: {plan_seconds} seconds)")

    tetris_prompt_template = """
Analyze the current Tetris board state and generate PyAutoGUI code to control Tetris 
for the next {plan_seconds} second(s).

## General Tetris Controls (example keybinds).
- left: move piece left by 1 grid unit.
- right: move piece right by 1 grid unit.
- up: rotate piece clockwise once.
- down: accelerated drop （will award more points, use ONLY IF you are very confident).

## Game Physics.
- The game is played on a 10x20 grid.
- Blocks fall at a rate of approximately 1 grid unit per second.
- Pressing the down key moves the block down by 1 grid unit.

## Strategies and Caveats.
### Code generation and latency:
- In generated code, only consider the current block OR the next block. Don't not generate code to control both.
- At the time the code is executed, 3~5 seconds have elapsed. The game might have moved on to the next block if the stack is high.
- If the stack is high, most likely you are controlling the "next" block due to latency.

## Planning:
- Prioritize keeping the stack flat. Balance the two sides.
- Place pieces in a way that leaves open columns for future pieces to fit naturally.
- Never create holes that cannot be filled—these are spaces that, due to their location, cannot be naturally occupied by falling pieces without clearing lines first.
- If you see a chance to clear lines, rotate and move the block to correct positions.
- Position the current block with the next block in mind: Plan ahead so that future pieces can be placed efficiently without causing gaps.
- DO NOT rotate and quickly move the block again once it's position has been decided.
- The entire sequence of key presses should be feasible within {plan_seconds} second(s).

## Lessons learned:
{experience_summary}

## Output Format:
- Output ONLY the Python code for PyAutoGUI commands, e.g. `pyautogui.press("left")`.
- Include brief comments for each action.
- Do not print anything else besides these Python commands.
    """

    iter_counter = 0
    try:
        while True:
            # Read information passed from the speculator cache
            try:
                experience_summary = read_log_to_string(f"cache/tetris/thread_{thread_id}/planner/experience_summary.log")
            except Exception as e:
                experience_summary = "- No lessons learned so far."
            
            print(f"-------------- experience summary --------------\n{experience_summary}\n")

            tetris_prompt = tetris_prompt_template.format(plan_seconds=plan_seconds, experience_summary=experience_summary)

            # Capture the screen
            screen_width, screen_height = pyautogui.size()
            region = (0, 0, screen_width // 64 * 18, screen_height // 64 * 40)
            screenshot = pyautogui.screenshot(region=region)

            # Create a unique folder for this thread's cache
            thread_folder = f"cache/tetris/thread_{thread_id}"
            os.makedirs(thread_folder, exist_ok=True)
            iter_folder = os.path.join(thread_folder, f"iter_{iter_counter}")
            os.makedirs(iter_folder, exist_ok=True)
            
            screenshot_path = os.path.join(iter_folder, "screenshot.png")
            screenshot.save(screenshot_path)

            # Encode the screenshot
            base64_image = encode_image(screenshot_path)

            start_time = time.time()
            try:
                if api_provider == "anthropic":
                    generated_code_str = anthropic_completion(system_prompt, model_name, base64_image, tetris_prompt)
                elif api_provider == "openai":
                    generated_code_str = openai_completion(system_prompt, model_name, base64_image, tetris_prompt)
                elif api_provider == "gemini":
                    generated_code_str = gemini_completion(system_prompt, model_name, base64_image, tetris_prompt)
                else:
                    raise NotImplementedError(f"API provider: {api_provider} is not supported.")
            except Exception as e:
                print(f"[Thread {thread_id}] Error executing code: {e}")

            end_time = time.time()
            latency = end_time - start_time
            all_response_time.append(latency)

            print(f"[Thread {thread_id}] Request latency: {latency:.2f}s")
            avg_latency = np.mean(all_response_time)
            print(f"[Thread {thread_id}] Latencies: {all_response_time}")
            print(f"[Thread {thread_id}] Average latency: {avg_latency:.2f}s\n")

            print(f"[Thread {thread_id}] --- API output ---\n{generated_code_str}\n")

            # Extract Python code for execution
            clean_code = extract_python_code(generated_code_str)
            log_output(thread_id, f"[Thread {thread_id}] Python code to be executed:\n{clean_code}\n", "tetris", f"iter_{iter_counter}")
            print(f"[Thread {thread_id}] Python code to be executed:\n{clean_code}\n")

            try:
                exec(clean_code)
            except Exception as e:
                print(f"[Thread {thread_id}] Error executing code: {e}")
            
            iter_counter += 1

    except KeyboardInterrupt:
        print(f"[Thread {thread_id}] Interrupted by user. Exiting...")
