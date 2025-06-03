import time
import os
import pyautogui
import openai
import numpy as np
import concurrent.futures

from tools.utils import encode_image, log_output, extract_python_code

from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

system_prompt = (
        "You are an intelligent Super Mario gameplay agent that controls Mario, search for and execute optimal path given each game state. Prioritize survival over speed."
    )

def worker_short(thread_id, offset):
    """
    Worker function for short-term (1 second) motion control.
    Continuously takes a screenshot, sends it (with a system prompt) to OpenAI,
    and executes the generated PyAutoGUI code.
    """
    all_response_time = []
    time.sleep(offset)
    print(f"[Thread {thread_id} - SHORT] Starting after {offset}s delay...")

    # This prompt is now moved to the system message.
    short_prompt = (
        f"Analyze the current game state and generate PyAutoGUI code to control Mario "
        "for the next 1 second.\n"
        "Mario's position most likely has moved forward when the generated code gets to execute.\n"
        "Your objective is to avoid obstacles, enemies, and hazards.\n"

        "### General Controls:\n"
        "- Press 'Enter' to start the game ONLY IF the game hasn't started.\n"
        "  Otherwise the game will be paused.\n"
        "- Press the right arrow to move forward.\n"
        "- Press 'X' along with right/left arrow to jump over obstacles or gaps. Be very careful with gaps, do lopped jumps if necessary.\n\n"

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
            # Take screenshot of the full screen
            screen_width, screen_height = pyautogui.size()
            region = (0, 0, screen_width, screen_height)
            screenshot = pyautogui.screenshot(region=region)

            # Save screenshot temporarily and encode it
            thread_folder = f"cache/thread_{thread_id}"
            os.makedirs(thread_folder, exist_ok=True)
            screenshot_path = os.path.join(thread_folder, "screenshot.png")
            screenshot.save(screenshot_path)
            base64_screenshot = encode_image(screenshot_path)

            # Messages: system prompt now contains the text instructions; user message contains only the image.
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_screenshot}"
                            },
                        },
                        {
                            "type": "text",
                            "text": short_prompt
                        },
                    ],
                }
            ]

            start_time = time.time()

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0,
                max_tokens=1024,
            )

            generated_code_str = response.choices[0].message.content

            end_time = time.time()
            latency = end_time - start_time
            all_response_time.append(latency)

            print(f"[Thread {thread_id} - SHORT] Request latency: {latency:.2f}s")
            avg_latency = np.mean(all_response_time)
            print(f"[Thread {thread_id} - SHORT] Latencies: {all_response_time}")
            print(f"[Thread {thread_id} - SHORT] Average latency: {avg_latency:.2f}s")

            print(f"\n[Thread {thread_id} - SHORT] --- Generation ---\n{generated_code_str}\n")

            clean_code = extract_python_code(generated_code_str)
            log_output(thread_id, f"[Thread {thread_id} - SHORT] Python code:\n{clean_code}")
            print(f"[Thread {thread_id} - SHORT] Python code to be executed:\n{clean_code}\n")

            try:
                exec(clean_code)
            except Exception as e:
                print(f"[Thread {thread_id} - SHORT] Error executing code: {e}")

    except KeyboardInterrupt:
        print(f"[Thread {thread_id} - SHORT] Interrupted by user. Exiting...")

def worker_long(thread_id, offset):
    """
    Worker function for long-term (2-3 seconds) motion control.
    Continuously takes a screenshot, sends it (with a system prompt) to OpenAI,
    and executes the generated PyAutoGUI code.
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
        "- Press 'X' along with right/left arrow to jump over obstacles or gaps. Be very careful with gaps, do lopped jumps if necessary.\n\n"

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
            # Capture full screen screenshot
            screen_width, screen_height = pyautogui.size()
            region = (0, 0, screen_width, screen_height)
            screenshot = pyautogui.screenshot(region=region)

            # Save and encode the screenshot
            thread_folder = f"cache/thread_{thread_id}"
            os.makedirs(thread_folder, exist_ok=True)
            screenshot_path = os.path.join(thread_folder, "screenshot.png")
            screenshot.save(screenshot_path)
            base64_screenshot = encode_image(screenshot_path)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_screenshot}"
                            },
                        },
                        {
                            "type": "text",
                            "text": long_prompt
                        },
                    ],
                }
            ]

            start_time = time.time()

            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=messages,
                temperature=0,
                max_tokens=1024,
            )

            generated_code_str = response.choices[0].message.content

            end_time = time.time()
            latency = end_time - start_time
            all_response_time.append(latency)

            print(f"[Thread {thread_id} - LONG] Request latency: {latency:.2f}s")
            avg_latency = np.mean(all_response_time)
            print(f"[Thread {thread_id} - LONG] Latencies: {all_response_time}")
            print(f"[Thread {thread_id} - LONG] Average latency: {avg_latency:.2f}s")

            print(f"\n[Thread {thread_id} - LONG] --- Generation ---\n{generated_code_str}\n")

            clean_code = extract_python_code(generated_code_str)
            log_output(thread_id, f"[Thread {thread_id} - LONG] Python code:\n{clean_code}")
            print(f"[Thread {thread_id} - LONG] Python code to be executed:\n{clean_code}\n")

            try:
                exec(clean_code)
            except Exception as e:
                print(f"[Thread {thread_id} - LONG] Error executing code: {e}")

    except KeyboardInterrupt:
        print(f"[Thread {thread_id} - LONG] Interrupted by user. Exiting...")

def main():
    """
    Spawns a number of short-term and long-term workers in parallel.
    """
    num_threads = 14
    offsets = [i * 0.5 for i in range(num_threads)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        for i in range(num_threads):
            if i % 2 == 0:
                executor.submit(worker_long, i, offsets[i])
            else:
                executor.submit(worker_short, i, offsets[i])

        try:
            while True:
                time.sleep(0.25)
        except KeyboardInterrupt:
            print("\nMain thread interrupted. Exiting all threads...")

if __name__ == "__main__":
    main()
