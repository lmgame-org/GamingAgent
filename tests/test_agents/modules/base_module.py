import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import json
from datetime import datetime
from tools.serving import APIManager

# Cache directories and file paths
CACHE_DIR = os.path.join("cache", "2048_experiments", datetime.now().strftime("%Y%m%d_%H%M%S"))
BOARD_IMG_PATH = os.path.join(CACHE_DIR, "board_latest.png")
os.makedirs(CACHE_DIR, exist_ok=True)

class Base_module:
    """
    A simplified module that directly processes observation images and returns actions.
    This module skips separate perception and memory stages used in the full pipeline.
    """
    def __init__(self, model_name="claude-3-7-sonnet-latest", reasoning_effort="high", thinking=True):
        """
        Initialize the Base Module for direct action planning.
        
        Args:
            model_name (str): Name of the model to use for reasoning
            reasoning_effort (str): Reasoning effort level for compatible models
            thinking (bool): Whether to enable thinking mode for compatible models
        """
        self.model_name = model_name
        self.reasoning_effort = reasoning_effort
        self.thinking = thinking
        self.api_manager = APIManager(game_name="2048")
        self.last_action = None
        
        # System prompt with strict output instructions
        self.system_prompt = """You are an intelligent AI player playing the 2048 game. Your goal is to make strategic moves to combine tiles and reach the highest possible tile value.

IMPORTANT: You MUST format your response using EXACTLY these lines:
thought: [Your reasoning about the game state]
move: [move]

Where [move] must be one of: "up", "down", "left", or "right".
Do not include # or any other prefix. Start directly with "thought:" followed by your analysis."""
        
        # Prompt for direct observation analysis and action planning
        self.action_prompt = """2048 Game Quick Guide:
Primary Goal: Combine like tiles to create tiles with higher values.
Ultimate Goal: Create a tile with the value 2048 or higher.

Game Mechanics:
- The game is played on a 4x4 grid.
- Each move (up, down, left, right) shifts all tiles in that direction.
- Tiles with the same value that collide during a move combine into a single tile with twice the value.
- After each move, a new tile (2 or 4) appears in a random empty cell.
- The game ends when there are no valid moves left.

Action Space:
You must select one of these 4 moves:
- up: Shift all tiles upward
- down: Shift all tiles downward
- left: Shift all tiles to the left
- right: Shift all tiles to the right

Key Strategies:
1. Build a stable structure - Keep your highest value tiles in a corner.
2. Maintain a clear path - Always have a direction where you can combine tiles.
3. Chain reactions - Set up sequences of merges that can happen in a single move.
4. Look ahead - Think about the consequences of your moves 2-3 steps ahead.
5. Building patterns - Common patterns include:
   - Snake/Zig-zag pattern: Arrange tiles in decreasing order in a zigzag.
   - Corner anchoring: Keep the highest tile in a corner and build around it.

Avoid:
- Getting high-value tiles stuck in the middle of the board
- Creating scattered small values that block potential merges
- Making moves that could lead to grid lock

Your response format should contain:
1. thought: [Your reasoning about the game state]
2. move: [move]

Example responses:
- thought: I see several opportunities to merge tiles. Moving right would combine the two 4s in row 2, creating an 8. This also maintains my highest tiles on the right edge.
  move: right

- thought: The board is getting crowded. Moving up would consolidate several tiles and create a clear path at the bottom for new tiles.
  move: up

- thought: I need to maintain my corner strategy. Moving down would keep my 64 and 32 tiles anchored in the bottom-right corner and potentially create merge opportunities.
  move: down"""

    def _create_board_image(self, board_matrix, size=400):
        """
        Create a visualization of the 2048 board.
        
        Args:
            board_matrix: 2D numpy array of the game board
            size: Size of the output image in pixels
            
        Returns:
            PIL Image representing the board
        """
        cell_size = size // 4
        padding = cell_size // 10
        
        # Create a new image with a beige background
        img = Image.new('RGB', (size, size), (250, 248, 239))
        draw = ImageDraw.Draw(img)
        
        # Color mapping for different tile values
        colors = {
            0: (205, 193, 180),      # Empty cell
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
            4096: (60, 58, 50),      # 4096
            8192: (39, 37, 33)       # 8192
        }
        
        # Font colors: dark for small values, light for larger values
        dark_text = (119, 110, 101)
        light_text = (249, 246, 242)
        
        # Try to load a font (fallback to default if not available)
        try:
            font = ImageFont.load_default()
        except:
            font = None
        
        # Draw each cell
        for row in range(4):
            for col in range(4):
                # Get the power of 2 value from the board
                power = int(board_matrix[row][col])
                
                # Convert to actual value (2^power, where 0 stays 0)
                value = 0 if power == 0 else 2**power
                
                # Calculate cell position
                x0 = col * cell_size + padding
                y0 = row * cell_size + padding
                x1 = (col + 1) * cell_size - padding
                y1 = (row + 1) * cell_size - padding
                
                # Draw cell background
                cell_color = colors.get(value, (60, 58, 50))  # Default to dark color for large values
                draw.rectangle([x0, y0, x1, y1], fill=cell_color)
                
                # Skip text for empty cells
                if value == 0:
                    continue
                
                # Choose text color based on value
                text_color = light_text if value > 4 else dark_text
                
                # Draw the value
                text = str(value)
                
                # Simple text positioning
                text_x = x0 + cell_size//4
                text_y = y0 + cell_size//4
                
                # Draw text
                if font:
                    draw.text((text_x, text_y), text, fill=text_color, font=font)
        
        return img

    def _convert_rgb_to_board(self, rgb_array):
        """
        Convert RGB image from env.render() to a 2048 board matrix.
        For the base module, we work directly with the RGB image itself
        without trying to extract the actual board values.
        
        Args:
            rgb_array: RGB image from env.render() 
            
        Returns:
            2D numpy array representation of the board (with powers of 2)
        """
        # For the base module, we just use a placeholder board since
        # we're working directly with the RGB image for the vision model
        return np.zeros((4, 4), dtype=np.int32)
        
    def process_observation(self, observation, info=None):
        """
        Process the RGB observation directly to plan the next action.
        The base module works with RGB images from env.render() but also uses info dictionary if available.
        
        Args:
            observation: The game observation (RGB image)
            info: Additional information about the game state
            
        Returns:
            dict: A dictionary containing move and thought
        """
        try:
            # Ensure we have a valid RGB image
            if len(observation.shape) != 3 or observation.shape[2] != 3:
                print("ERROR: Base module requires RGB image from env.render()")
                # Return a default action if we don't have a proper image
                return {
                    "move": "up",
                    "thought": "Error: Expected RGB image but received different input"
                }
            
            # Process the RGB image directly
            print("Base module: Processing RGB image observation")
            
            # Save the RGB image directly for the API call
            img = Image.fromarray(observation)
            img.save(BOARD_IMG_PATH)
            print(f"Saved RGB frame to {BOARD_IMG_PATH}")
            
            # Get board information from info if available
            board_features = ""
            if info is not None and 'board' in info:
                board = info['board']
                # Convert the board values to actual 2048 values for display
                visual_board = []
                for row in board:
                    visual_row = []
                    for cell in row:
                        if cell == 0:
                            visual_row.append(0)
                        else:
                            visual_row.append(2**cell)
                    visual_board.append(visual_row)
                
                # Get max tile information
                max_tile_power = int(info.get('max', 0))
                max_tile = 2 ** max_tile_power if max_tile_power > 0 else 0
                
                board_features = f"Current board:\n{visual_board}\n"
                board_features += f"Highest tile: {max_tile} (2^{max_tile_power})\n"
                board_features += f"Empty spaces: {sum(1 for row in board for cell in row if cell == 0)} out of 16\n"
            else:
                board_features = "The board is shown in the image. Please analyze the 2048 board and choose the best move."
            
            # Create a user prompt focused on image analysis
            user_prompt = f"""{self.action_prompt}

Please analyze the 2048 board in the image and determine the best move.

{board_features}

Key considerations:
- Look for opportunities to merge similar tiles
- Maintain your highest tiles in a corner
- Keep space for new tiles to appear
- Avoid trapping high-value tiles in the middle

IMPORTANT - FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
thought: [your analysis here]
move: [move]

Where [move] must be one of: "up", "down", "left", or "right".
Do NOT use # or any other prefix. Start directly with "thought:" followed by your analysis.
"""
            
            # Make API call for reasoning with the image
            print("Making API call with vision model on RGB frame")
            response, _ = self.api_manager.vision_text_completion(
                model_name=self.model_name,
                system_prompt=self.system_prompt,
                prompt=user_prompt,
                image_path=BOARD_IMG_PATH,
                thinking=self.thinking,
                reasoning_effort=self.reasoning_effort,
                token_limit=100000
            )
            
            # Parse the response to extract the move
            result = self._parse_response(response)
            print(f"Base module decided move: {result['move']}")
            return result
            
        except Exception as e:
            print(f"Error in Base Module: {e}")
            # Return a default action on error
            return {
                "move": "skip",
                "thought": f"Error occurred: {str(e)}"
            }
    
    def _parse_response(self, response):
        """
        Parse the response to extract thought and move.
        
        Args:
            response (str): Response from the model
            
        Returns:
            dict: Dictionary with thought and move
        """
        move = None
        thought = None
        
        # Look for thought: and move: in the response (with or without # prefix)
        lines = response.strip().split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Match both "thought:" and "# thought:" patterns
            if line.startswith("thought:") or line.startswith("# thought:"):
                prefix_len = line.find("thought:") + len("thought:")
                
                # If this is the last line, just use it
                if i == len(lines) - 1:
                    thought = line[prefix_len:].strip()
                else:
                    # If not the last line, collect all lines until we hit a move: line
                    thought_lines = []
                    thought_lines.append(line[prefix_len:].strip())
                    
                    j = i + 1
                    while j < len(lines) and not (lines[j].strip().startswith("move:") or lines[j].strip().startswith("# move:")):
                        thought_lines.append(lines[j].strip())
                        j += 1
                    
                    thought = " ".join(thought_lines).strip()
            
            # Match both "move:" and "# move:" patterns  
            elif line.startswith("move:") or line.startswith("# move:"):
                prefix_len = line.find("move:") + len("move:")
                move_text = line[prefix_len:].strip()
                
                # Expected format: move: up/down/left/right
                move = move_text.strip().lower()
                
                # Validate move
                valid_moves = ["up", "down", "left", "right"]
                if move not in valid_moves:
                    print(f"Warning: Invalid move '{move}', defaulting to 'up'")
                    move = "up"
        
        # If parsing failed, use default values
        if move is None or thought is None:
            print(f"Failed to parse response: {response}")

        if move is None:
            move = "up"
            print(f"Failed to parse move from response: {response}")
        
        if thought is None:
            thought = "No explicit thought provided in response"
            print(f"Failed to parse thought from response: {response}")
        
        return {
            "move": move,
            "thought": thought
        }

    def _convert_onehot_to_board(self, observation):
        """Convert one-hot encoded observation to a 2D board array."""
        # One-hot encoded observation has shape (4, 4, 16)
        # Each position in the 16 depth dimension represents a power of 2 (0 to 15)
        board = np.zeros((4, 4), dtype=np.int32)
        
        # For each cell position
        for i in range(4):
            for j in range(4):
                # Find which power of 2 is represented (if any)
                for power in range(16):
                    if observation[i, j, power] == 1:
                        board[i, j] = power
                        break
        
        return board
        
    def _extract_board_features(self, board):
        """Extract textual features from the board for the prompt."""
        highest_power = int(np.max(board))
        highest_value = 0 if highest_power == 0 else 2**highest_power
        
        # Count empty spaces
        empty_spaces = np.sum(board == 0)
        
        # Find potential merges
        horizontal_merges = 0
        vertical_merges = 0
        
        # Check horizontal merges
        for row in range(4):
            for col in range(3):
                if board[row][col] != 0 and board[row][col] == board[row][col+1]:
                    horizontal_merges += 1
                    
        # Check vertical merges
        for row in range(3):
            for col in range(4):
                if board[row][col] != 0 and board[row][col] == board[row+1][col]:
                    vertical_merges += 1
        
        # Create a readable board representation with actual values (2^power)
        board_text = "Board:\n"
        for row in board:
            row_values = []
            for cell in row:
                if cell == 0:
                    row_values.append("0".center(6))
                else:
                    row_values.append(str(2**cell).center(6))
            board_text += "|" + "|".join(row_values) + "|\n"
        
        # Add summary statistics
        features = f"{board_text}\n"
        features += f"Highest tile: {highest_value}\n"
        features += f"Empty spaces: {empty_spaces} out of 16\n"
        features += f"Potential merges: {horizontal_merges} horizontal, {vertical_merges} vertical\n"
        
        return features 