# import cv2
# import numpy as np
# import json

# def preprocess_image(image_path, crop_left=0, crop_right=0, crop_top=0, crop_bottom=0):
#     image = cv2.imread(image_path)
#     if image is None:
#         print(f"Error: Image '{image_path}' not found or unreadable.")
#         exit()
    
#     # Crop image to remove left, right, top, and bottom sides
#     height, width = image.shape[:2]
#     new_x_start = crop_left
#     new_x_end = width - crop_right
#     new_y_start = crop_top
#     new_y_end = height - crop_bottom
#     cropped_image = image[new_y_start:new_y_end, new_x_start:new_x_end]
    
#     # Save cropped image for debugging
#     cv2.imwrite("cropped_debug.png", cropped_image)
#     print("Cropped image saved as cropped_debug.png")
    
#     gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     edges = cv2.Canny(blurred, 50, 150)
    
#     return image, cropped_image, edges, new_x_start, new_y_start

# def detect_candies(image, edges, x_offset, y_offset):
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     detected_candies = []
    
#     for idx, contour in enumerate(contours, start=1):
#         x, y, w, h = cv2.boundingRect(contour)
#         center_x, center_y = x + w//2, y + h//2
        
#         # Adjust the x and y coordinates based on the crop
#         adjusted_x = center_x + x_offset
#         adjusted_y = center_y + y_offset
#         detected_candies.append({'id': idx, 'x': adjusted_x, 'y': adjusted_y, 'w': w, 'h': h})
    
#     return detected_candies

# def annotate_candies(image, detected_candies):
#     for candy in detected_candies:
#         x, y, w, h, cid = candy['x'], candy['y'], candy['w'], candy['h'], candy['id']
#         cv2.rectangle(image, (x - w//2, y - h//2), (x + w//2, y + h//2), (0, 255, 0), 2)
#         cv2.putText(image, f"{cid}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
#     return image

# def save_coordinates(detected_candies, output_file='candy_coordinates.json'):
#     with open(output_file, 'w') as file:
#         json.dump(detected_candies, file, indent=4)
#     print(f"Coordinates saved to {output_file}")

# def main(image_path, crop_left=50, crop_right=50, crop_top=50, crop_bottom=50, output_image='annotated_candy_crush.png'):
#     original_image, cropped_image, edges, x_offset, y_offset = preprocess_image(image_path, crop_left, crop_right, crop_top, crop_bottom)
#     detected_candies = detect_candies(cropped_image, edges, x_offset, y_offset)
#     annotated_image = annotate_candies(original_image, detected_candies)
#     save_coordinates(detected_candies)
    
#     cv2.imwrite(output_image, annotated_image)
#     print(f"Annotated image saved as {output_image}")
    
#     # cv2.imshow("Annotated Candies", annotated_image)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()

# # Example usage:
# main("candy-crush.png", crop_left=700, crop_right=800, crop_top=300, crop_bottom=300)


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
    
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    return image, cropped_image, edges, new_x_start, new_y_start

def detect_candies(image, edges, x_offset, y_offset, area_threshold=500):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_candies = []
    
    for idx, contour in enumerate(contours, start=1):
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < area_threshold:
            continue  # Skip small detections
        
        center_x, center_y = x + w//2, y + h//2
        
        # Adjust the x and y coordinates based on the crop
        adjusted_x = center_x + x_offset
        adjusted_y = center_y + y_offset
        detected_candies.append({'id': idx, 'x': adjusted_x, 'y': adjusted_y, 'w': w, 'h': h})
    
    return detected_candies

def annotate_candies(image, detected_candies):
    for candy in detected_candies:
        x, y, w, h, cid = candy['x'], candy['y'], candy['w'], candy['h'], candy['id']
        cv2.rectangle(image, (x - w//2, y - h//2), (x + w//2, y + h//2), (0, 255, 0), 2)
        cv2.putText(image, f"{cid}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    return image

def save_coordinates(detected_candies, output_file='candy_coordinates.json'):
    with open(output_file, 'w') as file:
        json.dump(detected_candies, file, indent=4)
    print(f"Coordinates saved to {output_file}")

def main(image_path, crop_left=50, crop_right=50, crop_top=50, crop_bottom=50, area_threshold=500, output_image='annotated_candy_crush.png'):
    original_image, cropped_image, edges, x_offset, y_offset = preprocess_image(image_path, crop_left, crop_right, crop_top, crop_bottom)
    detected_candies = detect_candies(cropped_image, edges, x_offset, y_offset, area_threshold)
    annotated_image = annotate_candies(original_image, detected_candies)
    save_coordinates(detected_candies)
    
    cv2.imwrite(output_image, annotated_image)
    print(f"Annotated image saved as {output_image}")
    
    # cv2.imshow("Annotated Candies", annotated_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# Example usage:
main("candy-crush.png", crop_left=700, crop_right=800, crop_top=300, crop_bottom=300, area_threshold=9000)
