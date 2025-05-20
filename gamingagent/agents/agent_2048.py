import os
import json
import numpy as np
import gymnasium as gym
from PIL import Image, ImageDraw, ImageFont

from gamingagent.agents.base_agent import BaseAgent
from gamingagent.modules import Observation, BaseModule, PerceptionModule, ReasoningModule

class TwentyFortyEightAgent(BaseAgent):
    """
    Agent implementation for the 2048 game.
    Demonstrates how to extend BaseAgent for a specific game.
    """
    
    def __init__(self, model_name="claude-3-7-sonnet-latest", config_path=None, 
                 harness=True, max_memory=20, cache_dir=None):
        """
        Initialize the 2048 agent.
        
        Args:
            model_name (str): Name of the model to use for inference
            config_path (str, optional): Path to config file with prompts
            harness (bool): If True, uses perception-memory-reasoning pipeline;
                           If False, uses base module only
            max_memory (int): Maximum number of memory entries to store
            cache_dir (str, optional): Custom cache directory path
        """
        # Custom modules specific to 2048
        custom_modules = {
            "base_module": TwentyFortyEightBaseModule,
            "perception_module": TwentyFortyEightPerceptionModule,
            "reasoning_module": TwentyFortyEightReasoningModule
        }
        
        # Initialize base agent
        super().__init__(
            game_name="2048",
            model_name=model_name,
            config_path=config_path,
            harness=harness,
            max_memory=max_memory,
            cache_dir=cache_dir,
            custom_modules=custom_modules
        )
        
        # Game-specific variables
        self.move_to_action = {
            "up": 0,
            "right": 1, 
            "down": 2,
            "left": 3
        }
        
    def env_init(self, render_mode="rgb_array", size=4, max_pow=16, **kwargs):
        """Minimal implementation of env_init"""
        pass
        
    def reset(self, **kwargs):
        """Minimal implementation of reset"""
        pass
    
    def parse_move_to_action(self, move):
        """
        Convert move string to action index for the environment.
        
        Args:
            move (str): String representation of the move ('up', 'down', 'left', 'right')
            
        Returns:
            int: Action index or None if invalid
        """
        move_str = move.lower() if isinstance(move, str) else None
        
        # Return action index if move is valid
        if move_str in self.move_to_action:
            return self.move_to_action[move_str]
        
        # Return None for invalid moves
        print(f"Invalid move: {move}")
        return None
    
    def save_board_image(self, board, filename=None, size=400):
        """
        Create and save a visualization of the 2048 board.
        
        Args:
            board: 2D array representing the board
            filename: Optional filename (without extension)
            size: Image size in pixels
            
        Returns:
            str: Path to the saved image
        """
        if filename is None:
            step_count = len([f for f in os.listdir(self.observations_dir) if f.endswith('.png')])
            filename = f"board_{step_count:04d}"
            
        # Create full path
        filepath = os.path.join(self.observations_dir, f"{filename}.png")
        
        # Create and save the board image
        self._create_board_image(board, filepath, size)
        
        return filepath
    
    def _create_board_image(self, board, save_path, size=400):
        """
        Create a visualization of the 2048 board.
        
        Args:
            board: 2D array representing the board
            save_path: Path to save the image
            size: Size of the output image
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
                    
                    # Simple centering of text
                    text_width = len(text) * 8
                    text_x = x0 + (cell_size - text_width) // 2
                    text_y = y0 + (cell_size - 16) // 2
                    
                    draw.text((text_x, text_y), text, fill=text_color, font=font)
            
            # Save the image
            img.save(save_path)
            print(f"Saved board image to {save_path}")
            
        except Exception as e:
            print(f"Error creating board image: {e}")
            
        return save_path


# Game-specific module implementations

class TwentyFortyEightBaseModule(BaseModule):
    """2048-specific implementation of BaseModule."""
    
    def __init__(self, model_name="claude-3-7-sonnet-latest", cache_dir="cache",
                 system_prompt="", prompt=""):
        # 2048-specific default prompts if none provided
        if not system_prompt:
            system_prompt = """You are an AI playing the 2048 game. Your goal is to make strategic moves to combine tiles and reach the highest possible tile value.

IMPORTANT: You MUST format your response using EXACTLY these lines:
thought: [Your reasoning about the game state]
action: [move]

Where [move] must be one of: "up", "down", "left", or "right"."""

        if not prompt:
            prompt = """2048 Game Rules:
- The game is played on a 4x4 grid
- Each move (up, down, left, right) shifts all tiles in that direction
- Tiles with the same value that collide merge into a single tile with twice the value
- After each move, a new tile (2 or 4) appears in a random empty cell

Please analyze the game board shown in the image and determine the best move.
Consider:
1. Creating space for new tiles
2. Keeping high value tiles in a corner
3. Setting up future merges

Decide on the optimal move (up, down, left, or right) and explain your reasoning."""

        super().__init__(
            model_name=model_name,
            cache_dir=cache_dir,
            system_prompt=system_prompt,
            prompt=prompt
        )
        

class TwentyFortyEightPerceptionModule(PerceptionModule):
    """2048-specific implementation of PerceptionModule."""
    
    def __init__(self, model_name="claude-3-7-sonnet-latest", observation=None, 
                 cache_dir="cache", system_prompt="", prompt=""):
        super().__init__(
            model_name=model_name,
            observation=observation,
            cache_dir=cache_dir,
            system_prompt=system_prompt,
            prompt=prompt
        )
    
    def process_observation(self, observation):
        """Process 2048 observation to extract board state."""
        # Store the original observation
        self.observation = observation
        
        # Create a new observation to return
        self.new_observation = Observation()
        
        # Handle image observation
        if observation.img_path:
            self.img_path = observation.img_path
            self.new_observation.img_path = observation.img_path
            
            # In a full implementation, we would extract the board state from the image
            # For this example, we'll just pass along the image path
        
        # Handle symbolic observation (e.g., direct board state)
        if observation.symbolic_representation is not None:
            self.symbolic_representation = observation.symbolic_representation
            
            # If symbolic_representation is a numpy array (direct board state)
            if isinstance(self.symbolic_representation, np.ndarray):
                board = self.symbolic_representation
                
                # Analyze the board
                perception_data = self._analyze_board(board)
                self.new_observation.symbolic_representation = perception_data
        
        return self.new_observation
    
    def _analyze_board(self, board):
        """Analyze the 2048 board state to extract features."""
        # Count empty spaces
        empty_spaces = []
        for row in range(len(board)):
            for col in range(len(board[0])):
                if board[row][col] == 0:
                    empty_spaces.append((row, col))
        
        # Find highest tile
        highest_power = int(np.max(board))
        highest_value = 0 if highest_power == 0 else 2 ** highest_power
        
        # Convert to visual board (actual 2048 values)
        visual_board = []
        for row in board:
            visual_row = []
            for cell in row:
                if cell == 0:
                    visual_row.append(0)
                else:
                    visual_row.append(2 ** int(cell))
            visual_board.append(visual_row)
            
        # Create structured perception data
        perception_data = {
            "symbolic_representation": str(visual_board),
            "game_state_details": {
                "highest_tile": highest_value,
                "highest_tile_power": highest_power,
                "empty_spaces": len(empty_spaces),
                "empty_space_locations": empty_spaces,
                "board_size": f"{len(board)}x{len(board[0])}",
                "board_analysis": self._get_board_analysis(board, empty_spaces, highest_power)
            }
        }
        
        return perception_data
    
    def _get_board_analysis(self, board, empty_spaces, highest_power):
        """Generate a simple analysis of the board state."""
        if len(empty_spaces) > 10:
            crowding = "Board has plenty of empty spaces"
        elif len(empty_spaces) > 5:
            crowding = "Board has a moderate number of empty spaces"
        else:
            crowding = "Board is getting crowded"
            
        if highest_power >= 11:  # 2048 or higher
            progress = "Excellent progress with high tiles"
        elif highest_power >= 8:  # 256 or higher
            progress = "Good progress with medium-high tiles"
        else:
            progress = "Early game with low tiles"
            
        return f"{crowding}. {progress}."


class TwentyFortyEightReasoningModule(ReasoningModule):
    """2048-specific implementation of ReasoningModule."""
    
    def __init__(self, model_name="claude-3-7-sonnet-latest", cache_dir="cache",
                 system_prompt="", prompt=""):
        # 2048-specific default prompts if none provided
        if not system_prompt:
            system_prompt = """You are an AI playing the 2048 game. Your goal is to make strategic moves to combine tiles and reach the highest possible tile value.

IMPORTANT: You MUST format your response using EXACTLY these lines:
thought: [Your detailed analysis]
action: [move]

Where [move] must be one of: "up", "down", "left", or "right"."""

        if not prompt:
            prompt = """{context}

Based on the current game state and your memory of previous moves, determine the optimal next move.

Key strategies for 2048:
1. Keep your highest tile in a corner
2. Maintain a chain of decreasing values from that corner
3. Ensure there are always empty cells for new tiles
4. Look ahead to see how tiles will merge and shift

Choose the best move (up, down, left, or right) and explain your thought process in detail."""

        super().__init__(
            model_name=model_name,
            cache_dir=cache_dir,
            system_prompt=system_prompt,
            prompt=prompt
        )
    
    def plan_action(self, perception_data, memory_summary, img_path=None):
        """Plan the next action for 2048 game."""
        # Implementation of abstract method from parent class
        return super().plan_action(perception_data, memory_summary, img_path)
    
    def _parse_response(self, response):
        """Parse LLM response to extract action and thought."""
        if not response:
            return {"action": "up", "thought": "No response received"}
        
        # Initialize result with defaults
        result = {
            "action": None,
            "thought": ""
        }
        
        # Use regex to extract thought and action
        import re
        
        # Match patterns like "thought:", "# thought:", "Thought:", etc.
        thought_pattern = r'(?:^|\n)(?:#\s*)?thought:(.+?)(?=(?:\n(?:#\s*)?action:)|$)'
        action_pattern = r'(?:^|\n)(?:#\s*)?action:(.+?)(?=(?:\n(?:#\s*)?thought:)|$)'
        
        # Find thought section (case insensitive)
        thought_match = re.search(thought_pattern, response, re.DOTALL | re.IGNORECASE)
        if thought_match:
            result["thought"] = thought_match.group(1).strip()
        
        # Find action section (case insensitive)
        action_match = re.search(action_pattern, response, re.DOTALL | re.IGNORECASE)
        if action_match:
            result["action"] = action_match.group(1).strip()
        
        # If no structured format was found, treat the whole response as thought
        if not result["thought"] and not result["action"]:
            result["thought"] = response.strip()
            # Try to infer action from text
            for action in ["up", "down", "left", "right"]:
                if action in response.lower():
                    result["action"] = action
                    break
            if not result["action"]:
                result["action"] = "up"  # Default action
        
        # Ensure we have a valid action
        if not result["action"] or result["action"].lower() not in ["up", "down", "left", "right"]:
            result["action"] = "up"  # Default to up if no valid action found
            
        return result 