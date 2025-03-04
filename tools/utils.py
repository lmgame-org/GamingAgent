import time
import os
import pyautogui
import base64
import anthropic
import numpy as np
import concurrent.futures
import re


def encode_image(image_path):
    """
    Read a file from disk and return its contents as a base64-encoded string.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def log_output(thread_id, log_text, game, alias=None):
    """
    Logs output
    """
    # alias has to be string
    if alias:
        assert isinstance(alias, str), f"Expected {str}, got {type(alias)}"
        thread_folder = f"cache/{game}/thread_{thread_id}/{alias}"
    else:
        thread_folder = f"cache/{game}/thread_{thread_id}"
    os.makedirs(thread_folder, exist_ok=True)
    
    log_path = os.path.join(thread_folder, "output.log")
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(log_text + "\n\n")

def extract_python_code(content):
    """
    Extracts Python code from the assistant's response.
    - Detects code enclosed in triple backticks (```python ... ```)
    - If no triple backticks are found, returns the raw content.
    """
    match = re.search(r"```python\s*(.*?)\s*```", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return content.strip()

def extract_planning_prompt(generated_str): 
    """ Searches for a segment in generated_str of the form:
    ```planning prompt
    {text to be extracted}
    ```

    and returns the text inside. If it doesn’t find it, returns an empty string.
    """
    pattern = r"```planning prompt\s*(.*?)\s*```"
    match = re.search(pattern, generated_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def read_log_to_string(log_path):
    """
    Reads the log file and returns its content as a string.
    """
    assert os.path.exists(log_path), "Log file {log_path} does not exist."

    with open(log_path, "r", encoding="utf-8") as file:
        log_content = file.read()
    
    return log_content

def find_iteration_dirs(base_path):
    """
    Returns a list of tuples (iteration_number, iteration_path), sorted by iteration_number.
    Only includes directories matching the pattern iter_#.
    """
    print("Traversing root directory...")
    iteration_dirs = []

    for item in os.listdir(base_path):
        full_path = os.path.join(base_path, item)
        for iter_dir in os.listdir(full_path):
            iter_match = re.match(r"iter_(\d+)$", iter_dir)  # Extract number from iter_#
            if iter_match:
                print(iter_dir)
                iter_num = int(iter_match.group(1))  # Convert the extracted number to int
                iter_full_path = os.path.join(full_path, iter_dir)
                iteration_dirs.append((iter_num, iter_full_path))
            else:
                continue

    # Sort by iteration number (ascending)
    iteration_dirs.sort(key=lambda x: x[0])

    return iteration_dirs

def build_iteration_content(iteration_dirs, memory_size):
    """
    Given a sorted list of (iter_num, path) for all Tetris iterations,
    select only the latest `memory_size` iterations.
    For each iteration, locate the PNG and LOG file,
    extract base64 image and generated code, and build a single string
    that includes 'generated code for STEP <iter_num>:' blocks.
    Returns a tuple (steps_content, list_image_base64).
    """
    print("building iteration content...")
    total_iterations = len(iteration_dirs)
    relevant_iterations = iteration_dirs[-memory_size:] if total_iterations > memory_size else iteration_dirs
    steps_content = []
    list_image_base64 = []

    for (iter_num, iter_path) in relevant_iterations:
        png_file = None
        log_file = None

        # Identify .png and .log
        for f in os.listdir(iter_path):
            f_lower = f.lower()
            if f_lower.endswith(".png"):
                png_file = os.path.join(iter_path, f)
            elif f_lower.endswith(".log"):
                log_file = os.path.join(iter_path, f)

        # Encode img
        image_base64 = ""
        if png_file and os.path.isfile(png_file):
            image_base64 = encode_image(png_file)
            list_image_base64.append(image_base64)

        # Extract generated code
        code_snippets = ""
        if log_file and os.path.isfile(log_file):
            with open(log_file, "r", encoding="utf-8") as lf:
                log_content = lf.read()
                code_snippets = extract_python_code(log_content).strip()

        # Build the block
        block = f"Generated code for STEP {iter_num}:\n{code_snippets}"
        steps_content.append(block)

    # Join all iteration blocks into one string
    return steps_content, list_image_base64