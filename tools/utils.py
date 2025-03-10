import time
import os
import pyautogui
import base64
import anthropic
import numpy as np
import concurrent.futures
import re
import cv2
import numpy as np
import json



def encode_image(image_path):
    """
    Read a file from disk and return its contents as a base64-encoded string.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def log_output(thread_id, log_text, game, mode="w"):
    """
    Logs output to `cache/thread_{thread_id}/output.log`
    """
    thread_folder = f"cache/{game}/thread_{thread_id}"
    os.makedirs(thread_folder, exist_ok=True)
    
    log_path = os.path.join(thread_folder, "output.log")
    with open(log_path, mode, encoding="utf-8") as log_file:
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


def preprocess_image(image_path, crop_left=0, crop_right=0, crop_top=0, crop_bottom=0, cache_dir=None):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Image '{image_path}' not found or unreadable.")
        exit()
    
    # Crop image to remove left, right, top, and bottom sides
    height, width = image.shape[:2]
    new_x_start = crop_left
    new_x_end = width - crop_right
    new_y_start = crop_top
    new_y_end = height - crop_bottom
    cropped_image = image[new_y_start:new_y_end, new_x_start:new_x_end]
    
    # Save cropped image for debugging
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cropped_debug_path = os.path.join(cache_dir, "cropped_debug.png")
    else:
        cropped_debug_path = "cropped_debug.png"
    cv2.imwrite(cropped_debug_path, cropped_image)
    # print(f"Cropped image saved as {cropped_debug_path}")
    
    return image, cropped_image, new_x_start, new_y_start

def generate_grid(image, grid_rows, grid_cols):
    height, width = image.shape[:2]
    cell_width = width // grid_cols
    cell_height = height // grid_rows
    
    vertical_lines = [i * cell_width for i in range(grid_cols + 1)]
    horizontal_lines = [i * cell_height for i in range(grid_rows + 1)]
    
    return vertical_lines, horizontal_lines

def annotate_with_grid(image, vertical_lines, horizontal_lines, x_offset, y_offset, alpha=0.5, enable_digit_label = True, line_thickness = 1, black = False):
    """Annotates the image with semi-transparent gray grid cell numbers."""
    grid_annotations = []
    
    # Create a copy of the image to overlay transparent text
    overlay = image.copy()

    for row in range(len(horizontal_lines) - 1):
        for col in range(len(vertical_lines) - 1):
            x = (vertical_lines[col] + vertical_lines[col + 1]) // 2
            y = (horizontal_lines[row] + horizontal_lines[row + 1]) // 2
            cell_id = row * (len(vertical_lines) - 1) + col + 1
            grid_annotations.append({'id': cell_id, 'x': x + x_offset, 'y': y + y_offset})
            
            if enable_digit_label:
                # Draw semi-transparent text on the overlay
                text = str(cell_id)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.4
                thickness = 1
                text_color = (255, 255, 255)  # Gray color
            
                cv2.putText(overlay, text, (x - 10, y + 10), font, font_scale, text_color, thickness, cv2.LINE_AA)
            
            # Draw green grid rectangle
            if black:
                cv2.rectangle(image, (vertical_lines[col], horizontal_lines[row]), 
                            (vertical_lines[col + 1], horizontal_lines[row + 1]), (0, 0, 0), line_thickness)
            else:
                cv2.rectangle(image, (vertical_lines[col], horizontal_lines[row]), 
                            (vertical_lines[col + 1], horizontal_lines[row + 1]), (0, 255, 0), line_thickness)

    # Blend the overlay with the original image
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    return image, grid_annotations


def save_grid_annotations(grid_annotations, cache_dir=None):
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        output_file = os.path.join(cache_dir, 'grid_annotations.json')
    else:
        output_file = 'grid_annotations.json'
    
    with open(output_file, 'w') as file:
        json.dump(grid_annotations, file, indent=4)
    return output_file

def get_annotate_img(image_path, crop_left=50, crop_right=50, crop_top=50, crop_bottom=50, grid_rows=9, grid_cols=9, output_image='annotated_grid.png', cache_dir=None, enable_digit_label=True, line_thickness=1, black=False):
    original_image, cropped_image, x_offset, y_offset = preprocess_image(image_path, crop_left, crop_right, crop_top, crop_bottom, cache_dir)
    vertical_lines, horizontal_lines = generate_grid(cropped_image, grid_rows, grid_cols)
    annotated_cropped_image, grid_annotations = annotate_with_grid(cropped_image, vertical_lines, horizontal_lines, x_offset, y_offset, enable_digit_label=enable_digit_label, line_thickness=line_thickness, black=black)
    grid_annotation_path = save_grid_annotations(grid_annotations, cache_dir)
    
    # Place the annotated cropped image back onto the original image
    original_image[y_offset:y_offset + annotated_cropped_image.shape[0], x_offset:x_offset + annotated_cropped_image.shape[1]] = annotated_cropped_image
    
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        output_image_path = os.path.join(cache_dir, output_image)
        annotated_cropped_image_path = os.path.join(cache_dir, 'annotated_cropped_image.png')
    else:
        output_image_path = output_image
        annotated_cropped_image_path = 'annotated_cropped_image.png'
    
    cv2.imwrite(output_image_path, original_image)
    cv2.imwrite(annotated_cropped_image_path, annotated_cropped_image)

    return output_image_path, grid_annotation_path, annotated_cropped_image_path