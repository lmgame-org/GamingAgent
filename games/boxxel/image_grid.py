import time
import os
import pyautogui
import numpy as np

from tools.utils import encode_image, log_output, extract_python_code, get_annotate_img


screen_width, screen_height = pyautogui.size()
region = (0, 0, screen_width, screen_height)
    
screenshot = pyautogui.screenshot(region=region)

screenshot_path = "boxxel_screenshot.png"


screenshot.save(screenshot_path)

get_annotate_img(screenshot_path, crop_left=450, crop_right=575, crop_top=60, crop_bottom=170, grid_rows=8, grid_cols=8)
