import cv2
import numpy as np
import json

def preprocess_image(image_path, crop_left=0, crop_right=0, crop_top=0, crop_bottom=0):
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
    cv2.imwrite("cropped_debug.png", cropped_image)
    print("Cropped image saved as cropped_debug.png")
    
    return image, cropped_image, new_x_start, new_y_start

def generate_grid(image, grid_rows, grid_cols):
    height, width = image.shape[:2]
    cell_width = width // grid_cols
    cell_height = height // grid_rows
    
    vertical_lines = [i * cell_width for i in range(grid_cols + 1)]
    horizontal_lines = [i * cell_height for i in range(grid_rows + 1)]
    
    return vertical_lines, horizontal_lines

def annotate_with_grid(image, vertical_lines, horizontal_lines, x_offset, y_offset):
    grid_annotations = []
    
    for row in range(len(horizontal_lines) - 1):
        for col in range(len(vertical_lines) - 1):
            x = (vertical_lines[col] + vertical_lines[col + 1]) // 2
            y = (horizontal_lines[row] + horizontal_lines[row + 1]) // 2
            cell_id = row * (len(vertical_lines) - 1) + col + 1
            grid_annotations.append({'id': cell_id, 'x': x + x_offset, 'y': y + y_offset})
            
            # Draw black rectangle for better visibility
            cv2.rectangle(image, (x - 15, y - 15), (x + 15, y + 15), (0, 0, 0), -1)
            cv2.putText(image, str(cell_id), (x - 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.rectangle(image, (vertical_lines[col], horizontal_lines[row]), 
                          (vertical_lines[col + 1], horizontal_lines[row + 1]), (0, 255, 0), 1)
    
    return image, grid_annotations

def save_grid_annotations(grid_annotations, output_file='grid_annotations.json'):
    with open(output_file, 'w') as file:
        json.dump(grid_annotations, file, indent=4)
    print(f"Grid annotations saved to {output_file}")

def get_annotate_img(image_path, crop_left=50, crop_right=50, crop_top=50, crop_bottom=50, grid_rows=9, grid_cols=9, output_image='annotated_grid.png'):
    original_image, cropped_image, x_offset, y_offset = preprocess_image(image_path, crop_left, crop_right, crop_top, crop_bottom)
    vertical_lines, horizontal_lines = generate_grid(cropped_image, grid_rows, grid_cols)
    annotated_cropped_image, grid_annotations = annotate_with_grid(cropped_image, vertical_lines, horizontal_lines, x_offset, y_offset)
    save_grid_annotations(grid_annotations)
    
    # Place the annotated cropped image back onto the original image
    original_image[y_offset:y_offset + annotated_cropped_image.shape[0], x_offset:x_offset + annotated_cropped_image.shape[1]] = annotated_cropped_image
    
    cv2.imwrite(output_image, original_image)
    print(f"Annotated image saved as {output_image}")
    
    # cv2.imshow("Annotated Grid", original_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# Example usage:
get_annotate_img("candy-crush.png", crop_left=700, crop_right=800, crop_top=300, crop_bottom=300, grid_rows=7, grid_cols=7)
