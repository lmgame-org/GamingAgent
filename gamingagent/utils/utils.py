import base64
import io
from PIL import Image
import numpy as np

def encode_image(image_array: np.ndarray) -> str:
    """Convert a numpy array to a base64 encoded image string.
    
    Args:
        image_array: A numpy array representing an image
        
    Returns:
        A base64 encoded string of the image
    """
    # Convert numpy array to PIL Image
    if len(image_array.shape) == 3:
        # If the array is already in RGB format
        image = Image.fromarray(image_array)
    else:
        # If the array is in grayscale
        image = Image.fromarray(image_array.astype(np.uint8))
    
    # Convert to bytes
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    
    # Encode to base64
    return base64.b64encode(buffered.getvalue()).decode('utf-8') 