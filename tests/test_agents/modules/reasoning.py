import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tools.serving import APIManager

class ReasoningModule:
    def __init__(self, model_name="claude-3-7-sonnet-latest", reasoning_effort="high", thinking=True):
        """
        Initialize the Reasoning Module for action planning in 2048.
        
        Args:
            model_name (str): Name of the model to use for reasoning
            reasoning_effort (str): Level of reasoning effort (low, medium, high)
            thinking (bool): Whether to enable thinking mode
        """
        self.model_name = model_name
        self.reasoning_effort = reasoning_effort
        self.thinking = thinking
        self.api_manager = APIManager(game_name="2048")
        
        self.system_prompt = """You are an intelligent AI player playing the 2048 game. Your goal is to make strategic moves to combine tiles and reach the highest possible tile value.

IMPORTANT: You MUST format your response using EXACTLY these lines:
thought: [Your reasoning about the game state]
move: [move]

Where [move] must be one of: "up", "down", "left", or "right".
Do not include # or any other prefix. Start directly with "thought:" followed by your analysis."""
        
        self.action_prompt = """2048 Game Quick Guide:
Primary Goal: Combine like tiles to create tiles with higher values.
Ultimate Goal: Create a tile with the value 2048 or higher.

Game Mechanics:
- The game is played on a 4x4 grid.
- Each move (up, down, left, right) shifts all tiles in that direction.
- Tiles with the same value that collide during a move combine into a single tile with twice the value.
- After each move, a new tile (2 or 4) appears in a random empty cell.
- The game ends when there are no valid moves left.

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
1. thought: [Your reasoning about the game state and planned action]
2. move: [move]

Example responses:
- thought: I see several opportunities to merge tiles. Moving right would combine the two 4s in row 2, creating an 8. This also maintains my highest tiles on the right edge.
  move: right

- thought: The board is getting crowded. Moving up would consolidate several tiles and create a clear path at the bottom for new tiles.
  move: up

- thought: I need to maintain my corner strategy. Moving down would keep my 64 and 32 tiles anchored in the bottom-right corner and potentially create merge opportunities.
  move: down

Focus on making strategic decisions that maximize your score and maintain a well-organized board."""

    async def plan_action(self, current_perception, memory_summary, img_path, max_retries=3):
        """
        Plan the next action based on current perception and board image.
        Uses both the structured perception data and visual board image
        to make the most informed decision.
        
        Args:
            current_perception (dict): Current perceived game state from perception module
            memory_summary (list): Summary of past game states
            img_path (str): Path to the current board image
            max_retries (int): Maximum number of retry attempts
            
        Returns:
            dict: A dictionary containing move and thought
        """
        import asyncio
        
        try:
            # Check if the image file exists, create a fallback image if not
            if not os.path.exists(img_path):
                print(f"Warning: Board image not found at {img_path}, creating fallback image")
                self._create_fallback_image(current_perception, img_path)
            
            # Convert NumPy values to Python native types to avoid JSON serialization errors
            current_perception = self._convert_numpy_types(current_perception)
            memory_summary = self._convert_numpy_types(memory_summary)
                
            # Prepare memory context for the prompt
            memory_context = self._prepare_memory_context(memory_summary)
            
            # Create combined user prompt with perception data and reference to image
            user_prompt = f"""{self.action_prompt}

I'll provide both perception data and a visual board image to help you decide the best move.

PERCEPTION DATA:
{json.dumps(current_perception, indent=2)}

MEMORY CONTEXT:
{memory_context}

The board image is also provided for visual reference. Please analyze both the perception data and the board image to determine the best move.

Based on this combined information, what move should be taken next?

IMPORTANT - FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
thought: [your analysis here]
move: [move]

Where [move] must be one of: "up", "down", "left", or "right".
Do NOT use # or any other prefix. Start directly with "thought:" followed by your analysis.
"""
            
            # Implement retry mechanism
            retry_count = 0
            result = None
            
            while result is None and retry_count < max_retries:
                if retry_count > 0:
                    print(f"Retry attempt {retry_count}/{max_retries} for reasoning module...")
                    await asyncio.sleep(2)  # Short delay before retry
                
                # Use both the structured data and the image for the API call
                print(f"Reasoning module making API call with both perception data and image from {img_path}")
                
                # For deepseek models, use text_completion instead of vision_text_completion
                if "deepseek" in self.model_name.lower():
                    response, _ = await self.api_manager.text_completion(
                        model_name=self.model_name,
                        system_prompt=self.system_prompt,
                        prompt=user_prompt,
                        thinking=self.thinking,
                        reasoning_effort=self.reasoning_effort,
                        token_limit=100000
                    )
                elif "grok" in self.model_name.lower():
                    response, _ = await self.api_manager.text_completion(
                        model_name=self.model_name,
                        system_prompt=self.system_prompt,
                        prompt=user_prompt,
                        token_limit=100000,
                        reasoning_effort=self.reasoning_effort,
                    )
                elif "gemini-2.5-flash" in self.model_name.lower():
                    # Fix for Gemini models - don't use await since the function likely returns a tuple directly
                    # This avoids the "object tuple can't be used in 'await' expression" error
                    response, _ = self.api_manager.text_completion(
                        model_name=self.model_name,
                        system_prompt=self.system_prompt,
                        prompt=user_prompt,
                        token_limit=100000,
                    )
                else:
                    response, _ = self.api_manager.vision_text_completion(
                        model_name=self.model_name,
                        system_prompt=self.system_prompt,
                        prompt=user_prompt,
                        image_path=img_path,
                        thinking=self.thinking,
                        reasoning_effort=self.reasoning_effort,
                        token_limit= 100000
                    )



                await asyncio.sleep(3)
                
                print(f"Model: {self.model_name}")
                print(f"Reasoning module response: {response}")
                # Parse the response
                result = self._parse_response(response)
                
                # Check if we got a valid result
                if result is None or 'move' not in result or result['move'] == "skip":
                    retry_count += 1
                    result = None  # Reset to None for the next iteration
                else:
                    print(f"Reasoning module decided move: {result['move']}")
                    break
            
            # If all retries failed, return a fallback action
            if result is None or 'move' not in result:
                print("All reasoning attempts failed. Using fallback action.")
                return {
                    "move": "skip",
                    "thought": "Fallback action after failed reasoning attempts"
                }
                
            return result
            
        except Exception as e:
            print(f"Error in reasoning module: {e}")
            # Return a skip action on error
            return {
                "move": "skip",
                "thought": f"Error occurred in reasoning: {str(e)}"
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

    def _prepare_memory_context(self, memory_summary):
        """
        Prepare a concise summary of memory for the prompt.
        
        Args:
            memory_summary (list): List of memory entries
            
        Returns:
            str: A concise summary of memory
        """
        if not memory_summary:
            return "No memory of past states available."
            
        # Take up to the last 3 memory entries to keep context concise
        recent_memory = memory_summary[-3:]
        
        # Create a summary string
        summary_parts = []
        for idx, entry in enumerate(recent_memory):
            timestamp = entry.get("timestamp", "unknown_time")
            game_state = entry.get("game_state", {})
            last_action = entry.get("last_action", None)
            reflection = entry.get("reflection", None)
            
            summary = f"State {len(memory_summary) - len(recent_memory) + idx + 1}/{len(memory_summary)}:\n"
            
            # Extract key information
            if "highest_tile" in game_state:
                summary += f"- Highest tile: {game_state['highest_tile']}\n"
            
            if "empty_spaces" in game_state:
                summary += f"- Empty spaces: {len(game_state['empty_spaces'])}\n"
                
            if "best_moves" in game_state:
                summary += f"- Suggested moves: {', '.join(game_state['best_moves'])}\n"
                
            if last_action:
                summary += f"- Last action: {last_action}\n"
                
            if reflection:
                summary += f"- Reflection: {reflection}\n"
            
            summary_parts.append(summary)
            
        return "\n".join(summary_parts)
            
    def _parse_response(self, response):
        """
        Parse the reasoning response to extract thought and move.
        
        Args:
            response (str): Response from the reasoning model
            
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
                    move = "skip"
        
        # If parsing failed, use default values
        if move is None or thought is None:
            print(f"Failed to parse response: {response}")

        if move is None:
            move = "skip"
            print(f"Failed to parse move from response: {response}")
        
        if thought is None:
            thought = "No explicit thought provided in response"
            print(f"Failed to parse thought from response: {response}")
        
        return {
            "move": move,
            "thought": thought
        }

    def _create_fallback_image(self, perception_data, img_path):
        """Create a basic board image as fallback if the original is missing."""
        try:
            # Get board data from perception
            if "visual_board" in perception_data:
                board = perception_data["visual_board"]
            elif "board" in perception_data:
                # Convert power values to actual tile values
                board = []
                for row in perception_data["board"]:
                    visual_row = []
                    for cell in row:
                        visual_row.append(0 if cell == 0 else 2**cell)
                    board.append(visual_row)
            else:
                # Create empty board
                board = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
            
            # Create image
            size = 400
            cell_size = size // 4
            padding = 5
            
            # Create new image with background
            img = Image.new('RGB', (size, size), (250, 248, 239))
            draw = ImageDraw.Draw(img)
            
            # Draw cells
            for row in range(4):
                for col in range(4):
                    # Calculate position
                    x0 = col * cell_size + padding
                    y0 = row * cell_size + padding
                    x1 = (col + 1) * cell_size - padding
                    y1 = (row + 1) * cell_size - padding
                    
                    # Get value
                    value = board[row][col]
                    
                    # Draw cell
                    if value == 0:
                        color = (205, 193, 180)  # Empty cell
                    else:
                        color = (237, 224, 200)  # Default for non-empty
                    
                    draw.rectangle([x0, y0, x1, y1], fill=color)
                    
                    # Add text for non-empty cells
                    if value > 0:
                        # Use default font
                        font = ImageFont.load_default()
                        draw.text((x0 + cell_size//4, y0 + cell_size//4), 
                                 str(value), fill=(119, 110, 101))
            
            # Save the image
            img.save(img_path)
            print(f"Created fallback image at {img_path}")
            
        except Exception as e:
            print(f"Error creating fallback image: {e}")
            # Create absolute minimal image
            try:
                Image.new('RGB', (400, 400), (250, 248, 239)).save(img_path)
            except:
                pass 