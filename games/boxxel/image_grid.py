import time
import os
import pyautogui
import numpy as np

from tools.utils import encode_image, log_output, extract_python_code, get_annotate_img
cache = "cache/boxxel"
get_annotate_img("games/boxxel/sokoban_level_1.png", crop_left=450, crop_right=575, crop_top=60, crop_bottom=170, grid_rows=8, grid_cols=8, cache_dir = cache)
