"""
Utility functions for image processing:
1. Scaling images up to a maximum dimension
2. Drawing coordinate grids on images
"""

import cv2
import os
import numpy as np

def scale_image_up(image_path, maximum_scale=1500):
    """
    Scales an image up to a maximum dimension while maintaining aspect ratio.
    
    Args:
        image_path (str): Path to the input image
        maximum_scale (int): Maximum dimension for both width and height (default: 1800)
        
    Returns:
        str: Path to the scaled image
    """
    # Read the input image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Unable to read the image at {image_path}")
    
    # Get current dimensions
    height, width = image.shape[:2]
    
    # Calculate scale factor to fit within maximum_scale
    scale_factor = min(maximum_scale / width, maximum_scale / height)
    
    # Only scale up if necessary
    if scale_factor > 1:
        # Calculate new dimensions
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # Resize the image
        scaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Create output path
        file_name = os.path.basename(image_path)
        file_dir = os.path.dirname(image_path)
        name, ext = os.path.splitext(file_name)
        output_path = os.path.join(file_dir, f"{name}_scaled{ext}")
        
        # Save the scaled image
        cv2.imwrite(output_path, scaled_image)
        print(f"Scaled image from {width}x{height} to {new_width}x{new_height}")
        return output_path
    
    # If no scaling was needed, return original path
    return image_path

def draw_grid_on_image(observation, grid_dim=(5, 5)):
    """
    Draws a coordinate grid on an image from an observation.
    
    Args:
        observation: The observation object containing img_path
        grid_dim (tuple): Grid dimensions as (rows, cols) (default: (5, 5))
        
    Returns:
        observation: Updated observation with new image path containing grid
    """
    import copy
    
    # Create a copy of the observation to avoid modifying the original
    new_observation = copy.deepcopy(observation)
    
    # Get the image path from observation
    image_path = observation.img_path
    if image_path is None:
        print("Warning: No image path found in observation. Returning original observation.")
        return new_observation
    
    # Read the input image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Unable to read the image at {image_path}. Returning original observation.")
        return new_observation
    
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Calculate cell dimensions
    cell_height = height // grid_dim[0]
    cell_width = width // grid_dim[1]
    
    # Create a copy of the image to draw on
    grid_image = image.copy()
    
    # Draw horizontal lines
    for i in range(grid_dim[0] + 1):
        y = i * cell_height
        cv2.line(grid_image, (0, y), (width, y), (0, 255, 0), 2)
    
    # Draw vertical lines
    for i in range(grid_dim[1] + 1):
        x = i * cell_width
        cv2.line(grid_image, (x, 0), (x, height), (0, 255, 0), 2)
    
    # Add coordinate labels at the top-left corner of each cell
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5  # 3 times larger than original 0.5
    font_thickness = 2  # Slightly thicker for better visibility
    font_color = (255, 255, 255)  # White text
    
    # Add background rectangles for better visibility of text
    rect_color = (0, 0, 0)  # Black background
    rect_padding = 8  # Increased padding for larger text
    
    for row in range(grid_dim[0]):
        for col in range(grid_dim[1]):
            # Calculate top-left coordinate of the cell
            x = col * cell_width
            y = row * cell_height
            
            # Prepare the coordinate text
            coord_text = f"({col},{row})"
            
            # Get text size
            (text_width, text_height), _ = cv2.getTextSize(
                coord_text, font, font_scale, font_thickness
            )
            
            # Draw background rectangle
            cv2.rectangle(
                grid_image,
                (x + 2, y + 2),
                (x + text_width + 2 * rect_padding, y + text_height + 2 * rect_padding),
                rect_color,
                -1  # Filled rectangle
            )
            
            # Add text
            cv2.putText(
                grid_image, 
                coord_text, 
                (x + rect_padding, y + text_height + rect_padding), 
                font, 
                font_scale, 
                font_color, 
                font_thickness
            )
    
    # Create output path
    file_name = os.path.basename(image_path)
    file_dir = os.path.dirname(image_path)
    name, ext = os.path.splitext(file_name)
    output_path = os.path.join(file_dir, f"{name}_grid{ext}")
    
    # Save the image with grid
    cv2.imwrite(output_path, grid_image)
    print(f"Added {grid_dim[0]}x{grid_dim[1]} grid to image, saved as {output_path}")
    
    # Update the observation with the new image path
    new_observation.img_path = output_path
    
    return new_observation

def convert_numpy_to_python(item):
    """Recursively converts numpy arrays and numpy scalar types in a data structure to Python lists and base types."""
    if isinstance(item, np.ndarray):
        return item.tolist()
    elif isinstance(item, dict):
        return {k: convert_numpy_to_python(v) for k, v in item.items()} # Note: recursive call uses new public name
    elif isinstance(item, list):
        return [convert_numpy_to_python(i) for i in item] # Note: recursive call uses new public name
    elif isinstance(item, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                        np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(item)
    elif isinstance(item, (np.float_, np.float16, np.float32, np.float64)):
        return float(item)
    elif isinstance(item, np.bool_):
        return bool(item)
    return item
