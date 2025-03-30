import time
import numpy as np
import concurrent.futures
import argparse
from collections import deque, Counter

import os
import json
import re
import pyautogui

from games.game_2048.workers import game_2048_worker
from tools.utils import str2bool
from collections import Counter

CACHE_DIR = "cache/phoenix"
os.makedirs(CACHE_DIR, exist_ok=True)

# System prompt remains constant
system_prompt = (
    "You are an expert AI agent specialized in solving text-based detective games with optimal logic. "
    "Your goal is to detect contradictions in witness statements and uncover the hidden truth beneath the surface."
)

def main():
    parser = argparse.ArgumentParser(description="sokoban AI Agent")
    parser.add_argument("--api_provider", type=str, default="anthropic", help="API provider to use.")
    parser.add_argument("--model_name", type=str, default="claude-3-7-sonnet-20250219", help="LLM model name.")
    parser.add_argument("--modality", type=str, default="vision-text", choices=["text-only", "vision-text"],
                        help="modality used.")
    parser.add_argument("--thinking", type=str, default=True, help="Whether to use deep thinking.")
    parser.add_argument("--num_threads", type=int, default=1, help="Number of parallel threads to launch.")
    args = parser.parse_args()

    prev_responses = deque(maxlen=1)

    try:
        while True:

    except KeyboardInterrupt:
        print("\nStopped by user.")

if __name__ == "__main__":
    main()