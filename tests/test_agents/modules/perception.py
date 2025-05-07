import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import json
from datetime import datetime

# Cache directories and file paths
# CACHE_DIR = os.path.join("cache", "2048_experiments", datetime.now().strftime("%Y%m%d_%H%M%S"))
# BOARD_IMG_PATH = os.path.join(CACHE_DIR, "board_latest.png")
# os.makedirs(CACHE_DIR, exist_ok=True)

class PerceptionModule:
    def __init__(self, model_name="claude-3-7-sonnet-latest", cache_dir=None, board_img_path=None):
        """
        Initialize the Perception Module for analyzing 2048 game states.
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.board_img_path = board_img_path
        global CACHE_DIR, BOARD_IMG_PATH
        CACHE_DIR = self.cache_dir
        BOARD_IMG_PATH = self.board_img_path

        
    def analyze_board(self, observation, info=None):
        """
        Analyze the current 2048 board using primarily info['board'].
        
        Args:
            observation: The game observation (only used as fallback)
            info: Information dictionary from environment, containing the board
            
        Returns:
            dict: A dictionary containing the analyzed game state
        """
        try:
            # Primarily use info['board'] as the most reliable source
            if info is not None and 'board' in info:
                board = info['board']
                print("Using board from info dictionary")
            else:
                # Fallback to observation if info not available
                if len(observation.shape) == 3 and observation.shape[2] > 4:
                    # Handle one-hot encoded observation
                    board = self._convert_onehot_to_board(observation)
                    print("Using one-hot encoded observation")
                else:
                    # Assume it's a 2D board
                    board = observation
                    print("Using direct 2D board observation")
            
            # Create a simple board image
            try:
                self._create_board_image(board)
                print(f"Board image saved to {BOARD_IMG_PATH}")
            except Exception as e:
                print(f"Error creating board image: {e}")
            
            # Create simple perception data and convert NumPy types to Python native types
            perception_data = self._create_perception_data(board)
            return self._convert_numpy_types(perception_data)
                
        except Exception as e:
            print(f"Error in perception module: {e}")
            return self._create_fallback_response(observation)
    
    def _convert_onehot_to_board(self, observation):
        """Convert one-hot encoded observation to a 2D board array."""
        board = np.zeros((4, 4), dtype=np.int32)
        
        for i in range(4):
            for j in range(4):
                for power in range(16):
                    if observation[i, j, power] == 1:
                        board[i, j] = power
                        break
        
        return board
    
    def _create_perception_data(self, board):
        """Create simple perception data from the board."""
        highest_power = int(np.max(board))
        highest_value = 0 if highest_power == 0 else 2**highest_power
        
        # Count empty spaces
        empty_spaces = []
        for row in range(4):
            for col in range(4):
                if board[row][col] == 0:
                    empty_spaces.append({"row": row, "col": col})
        
        # Create visual board (actual 2048 values)
        visual_board = []
        for row in board:
            visual_row = []
            for cell in row:
                if cell == 0:
                    visual_row.append(0)
                else:
                    visual_row.append(2**cell)
            visual_board.append(visual_row)
        
        return {
            "board": board.tolist(),
            "visual_board": visual_board,
            "highest_tile_power": highest_power,
            "highest_tile": highest_value,
            "empty_spaces": empty_spaces,
            "best_moves": ["up", "left", "down", "right"],
            "game_state": {
                "analysis": f"Board has {len(empty_spaces)} empty spaces. Highest tile: {highest_value} (2^{highest_power}).",
                "strategy": "Keep highest tiles in a corner and maintain merge paths."
            }
        }
    
    def _create_board_image(self, board):
        """Create a simple board image and save it."""
        size = 400
        cell_size = size // 4
        padding = cell_size // 10
        
        # Create image with background
        img = Image.new('RGB', (size, size), (250, 248, 239))
        draw = ImageDraw.Draw(img)
        
        # Simple color mapping for tiles
        colors = {
            0: (205, 193, 180),      # Empty
            2: (238, 228, 218),      # 2
            4: (237, 224, 200),      # 4
            8: (242, 177, 121),      # 8
            16: (245, 149, 99),      # 16
            32: (246, 124, 95),      # 32
            64: (246, 94, 59),       # 64
            128: (237, 207, 114),    # 128
            256: (237, 204, 97),     # 256
            512: (237, 200, 80),     # 512
            1024: (237, 197, 63),    # 1024
            2048: (237, 194, 46),    # 2048
        }
        
        # Text colors
        dark_text = (119, 110, 101)  # For small values
        light_text = (249, 246, 242) # For large values
        
        try:
            # Use default font
            font = ImageFont.load_default()
            
            # Draw each cell
            for row in range(4):
                for col in range(4):
                    # Get power value and convert to actual 2048 value
                    power = int(board[row][col])
                    value = 0 if power == 0 else 2**power
                    
                    # Calculate position
                    x0 = col * cell_size + padding
                    y0 = row * cell_size + padding
                    x1 = (col + 1) * cell_size - padding
                    y1 = (row + 1) * cell_size - padding
                    
                    # Draw cell background - use simple rectangle without radius
                    cell_color = colors.get(value, (60, 58, 50))
                    draw.rectangle([x0, y0, x1, y1], fill=cell_color)
                    
                    # Skip text for empty cells
                    if value == 0:
                        continue
                    
                    # Choose text color based on value
                    text_color = light_text if value > 4 else dark_text
                    
                    # Simple text positioning
                    text = str(value)
                    text_x = x0 + cell_size//4
                    text_y = y0 + cell_size//4
                    
                    # Draw text
                    draw.text((text_x, text_y), text, fill=text_color, font=font)
            
            # Save the image
            img.save(BOARD_IMG_PATH)
            
        except Exception as e:
            print(f"Error creating board image: {e}")
            # Create minimal image on error
            Image.new('RGB', (400, 400), (250, 248, 239)).save(BOARD_IMG_PATH)
    
    def _create_fallback_response(self, observation):
        """Create a minimal response if there's an error."""
        try:
            # Convert observation if possible
            if len(observation.shape) == 3 and observation.shape[2] > 4:
                board = self._convert_onehot_to_board(observation)
            else:
                board = observation
                
            highest_power = int(np.max(board))
            highest_value = 0 if highest_power == 0 else 2**highest_power
                
            return {
                "board": board.tolist(),
                "highest_tile_power": highest_power,
                "highest_tile": highest_value,
                "empty_spaces": [],
                "best_moves": ["up", "right", "down", "left"],
                "game_state": {
                    "analysis": f"Fallback analysis. Highest tile: {highest_value} (2^{highest_power}).",
                    "strategy": "Try all available moves."
                }
            }
        except:
            # Ultimate fallback
            print("Creating empty board as ultimate fallback")
            return {
                "board": [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                "highest_tile_power": 0,
                "highest_tile": 0,
                "empty_spaces": [],
                "best_moves": ["up", "right", "down", "left"],
                "game_state": {
                    "analysis": "Error in perception.",
                    "strategy": "Try all available moves."
                }
            }
    
    def _convert_numpy_types(self, obj):
        """Convert NumPy types to Python native types to avoid JSON serialization errors."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return self._convert_numpy_types(obj.tolist())
        elif isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(i) for i in obj]
        else:
            return obj 