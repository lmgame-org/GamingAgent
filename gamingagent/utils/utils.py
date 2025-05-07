import base64
import io
from PIL import Image
import numpy as np
import json

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

def convert_to_json_serializable(obj):
    """Convert potentially non-serializable types (like NumPy) to JSON serializable types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(i) for i in obj]
    elif hasattr(obj, 'isoformat'): # Handle datetime objects
        return obj.isoformat()
    # Add other type conversions if needed (e.g., sets)
    try:
        # Check if obj is serializable, if not, convert to string representation
        json.dumps(obj)
        return obj
    except (TypeError, OverflowError):
        return str(obj) # Fallback to string representation

# Example usage:
# data = {'count': np.int64(10), 'values': np.array([1.0, 2.5]), 'nested': {'flag': np.bool_(True)}}
# serializable_data = convert_to_json_serializable(data)
# print(json.dumps(serializable_data)) 