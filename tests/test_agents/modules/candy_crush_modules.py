import numpy as np
from collections import OrderedDict
import json
from tools.serving import APIManager
from datetime import datetime
import os
import time
import asyncio
from PIL import Image

class CandyCrushPerceptionModule:
    """Module for analyzing Candy Crush game state."""
    
    def __init__(self, model_name="claude-3-7-sonnet-latest"):
        """
        Initialize the Perception Module for Candy Crush.
        
        Args:
            model_name (str): Name of the model to use
        """
        self.model_name = model_name
    
    def analyze_board(self, observation, info):
        """
        Analyze the current Candy Crush board state.
        
        Args:
            observation: The observation from the environment
            info: Additional info from the environment
            
        Returns:
            dict: A dictionary containing the analyzed game state
        """
        try:
            # Extract the board from the observation or info
            if isinstance(info, dict) and 'board' in info:
                board = info['board']
            else:
                board = observation
            
            # Convert board to a more readable format if needed
            if len(board.shape) == 3 and board.shape[0] <= 2:
                # Raw board format
                tile_colors = board[0]
            elif len(board.shape) == 3 and board.shape[0] > 2:
                # One-hot encoded format
                tile_colors = self._decode_ohe_board(board)
            else:
                tile_colors = board
            
            # Find potential matches
            horizontal_matches = self._find_horizontal_matches(tile_colors)
            vertical_matches = self._find_vertical_matches(tile_colors)
            
            # Count empty spaces (color value 0)
            empty_spaces = []
            for r in range(tile_colors.shape[0]):
                for c in range(tile_colors.shape[1]):
                    if tile_colors[r, c] == 0:
                        empty_spaces.append((r, c))
            
            # Find highest color value (excluding 0)
            non_zero_colors = tile_colors[tile_colors > 0]
            highest_color = non_zero_colors.max() if len(non_zero_colors) > 0 else 0
            
            # Calculate board statistics
            color_counts = {}
            for color in range(1, int(highest_color) + 1):
                color_counts[str(color)] = int(np.sum(tile_colors == color))
            
            # Put together the perception data
            perception_data = {
                "board": tile_colors.tolist(),
                "board_shape": tile_colors.shape,
                "empty_spaces": empty_spaces,
                "num_empty_spaces": len(empty_spaces),
                "highest_color": int(highest_color),
                "color_counts": color_counts,
                "horizontal_matches": horizontal_matches,
                "vertical_matches": vertical_matches,
                "potential_matches": horizontal_matches + vertical_matches,
                "num_potential_matches": len(horizontal_matches) + len(vertical_matches)
            }
            
            return perception_data
            
        except Exception as e:
            print(f"Error in perception module: {e}")
            # Return a minimal valid structure on error
            return {
                "board": [],
                "board_shape": (0, 0),
                "empty_spaces": [],
                "num_empty_spaces": 0,
                "highest_color": 0,
                "color_counts": {},
                "horizontal_matches": [],
                "vertical_matches": [],
                "potential_matches": [],
                "num_potential_matches": 0,
                "error": str(e)
            }
    
    def _decode_ohe_board(self, ohe_board, num_colours=4):
        """
        Decode a one-hot encoded board back to a 2D array of color indices.
        
        Args:
            ohe_board: One-hot encoded board
            num_colours: Number of colors in the game
            
        Returns:
            2D array with color indices
        """
        # Get board dimensions
        if len(ohe_board.shape) == 3:
            decoded_board = np.zeros((ohe_board.shape[1], ohe_board.shape[2]), dtype=int)
            
            # For each position, find which color channel has a 1
            for r in range(ohe_board.shape[1]):
                for c in range(ohe_board.shape[2]):
                    for color in range(num_colours):
                        if color < ohe_board.shape[0] and ohe_board[color, r, c] == 1:
                            decoded_board[r, c] = color + 1  # +1 because colors start at 1
                            break
            
            return decoded_board
        
        # If the shape is not as expected, return the input
        return ohe_board
    
    def _find_horizontal_matches(self, board):
        """
        Find potential horizontal matches by simulating swaps.
        
        Args:
            board: 2D array with color indices
            
        Returns:
            list: List of potential horizontal matches
        """
        matches = []
        rows, cols = board.shape
        
        for r in range(rows):
            for c in range(cols - 1):  # No need to check the last column
                # Skip empty cells
                if board[r, c] == 0 or board[r, c+1] == 0:
                    continue
                
                # Simulate swapping with the right neighbor
                simulated_board = board.copy()
                simulated_board[r, c], simulated_board[r, c+1] = simulated_board[r, c+1], simulated_board[r, c]
                
                # Check for matches after the swap
                match_found = self._check_for_match(simulated_board, r, c)
                match_found |= self._check_for_match(simulated_board, r, c+1)
                
                if match_found:
                    # Record the potential match
                    matches.append({
                        "coord1": (r, c),
                        "coord2": (r, c+1),
                        "direction": "horizontal",
                        "colors": (int(board[r, c]), int(board[r, c+1]))
                    })
        
        return matches
    
    def _find_vertical_matches(self, board):
        """
        Find potential vertical matches by simulating swaps.
        
        Args:
            board: 2D array with color indices
            
        Returns:
            list: List of potential vertical matches
        """
        matches = []
        rows, cols = board.shape
        
        for r in range(rows - 1):  # No need to check the last row
            for c in range(cols):
                # Skip empty cells
                if board[r, c] == 0 or board[r+1, c] == 0:
                    continue
                
                # Simulate swapping with the below neighbor
                simulated_board = board.copy()
                simulated_board[r, c], simulated_board[r+1, c] = simulated_board[r+1, c], simulated_board[r, c]
                
                # Check for matches after the swap
                match_found = self._check_for_match(simulated_board, r, c)
                match_found |= self._check_for_match(simulated_board, r+1, c)
                
                if match_found:
                    # Record the potential match
                    matches.append({
                        "coord1": (r, c),
                        "coord2": (r+1, c),
                        "direction": "vertical",
                        "colors": (int(board[r, c]), int(board[r+1, c]))
                    })
        
        return matches
    
    def _check_for_match(self, board, r, c):
        """
        Check if the tile at position (r, c) is part of a match of 3 or more.
        
        Args:
            board: 2D array with color indices
            r: Row index
            c: Column index
            
        Returns:
            bool: True if a match is found, False otherwise
        """
        rows, cols = board.shape
        color = board[r, c]
        
        # Skip empty cells
        if color == 0:
            return False
        
        # Check horizontal match
        count = 1
        # Check left
        for i in range(c-1, -1, -1):
            if board[r, i] == color:
                count += 1
            else:
                break
        # Check right
        for i in range(c+1, cols):
            if board[r, i] == color:
                count += 1
            else:
                break
        if count >= 3:
            return True
        
        # Check vertical match
        count = 1
        # Check up
        for i in range(r-1, -1, -1):
            if board[i, c] == color:
                count += 1
            else:
                break
        # Check down
        for i in range(r+1, rows):
            if board[i, c] == color:
                count += 1
            else:
                break
        if count >= 3:
            return True
        
        return False

class CandyCrushMemoryModule:
    """Memory module for tracking Candy Crush game state history."""
    
    def __init__(self, memory_file=None, max_memory=10, model_name="claude-3-7-sonnet-latest"):
        """
        Initialize the Memory Module for tracking game state history.
        
        Args:
            memory_file (str): Path to the memory JSON file
            max_memory (int): Maximum number of game states to remember
            model_name (str): Name of the model to use for reflections
        """
        self.memory_file = memory_file
        self.max_memory = max_memory
        self.memory = []
        self.model_name = model_name
        self.api_manager = APIManager(game_name="candy_crush")
        
        # Create the memory file directory if it doesn't exist
        if memory_file:
            os.makedirs(os.path.dirname(memory_file), exist_ok=True)
            
            # Load existing memory if available
            self.load_memory()
        
    def load_memory(self):
        """Load memory from the memory file if it exists."""
        try:
            if self.memory_file and os.path.exists(self.memory_file):
                with open(self.memory_file, 'r') as f:
                    memory_data = json.load(f)
                    # Limit to max_memory entries
                    self.memory = memory_data[-self.max_memory:] if len(memory_data) > self.max_memory else memory_data
        except Exception as e:
            print(f"Error loading memory: {e}")
            self.memory = []
            
    def save_memory(self):
        """Save the current memory to the memory file."""
        try:
            if self.memory_file:
                with open(self.memory_file, 'w') as f:
                    json.dump(self.memory, f, indent=2)
        except Exception as e:
            print(f"Error saving memory: {e}")
    
    def generate_reflection(self, current_perception, last_action):
        """
        Generate a reflection on the current state by comparing it with previous states.
        
        Args:
            current_perception (dict): The current perceived game state
            last_action (tuple): The previous action taken ((r1, c1), (r2, c2))
            
        Returns:
            str: A reflection on the current state and how it relates to previous states and actions
        """
        try:
            # If there are not enough previous states, return a default reflection
            if len(self.memory) < 1:
                return "Not enough history to generate a meaningful reflection."
            
            # Get the previous state
            previous_state = self.memory[-1]["game_state"]
            previous_action = self.memory[-1].get("last_action", None)
            
            # Extract key information from current perception
            current_num_matches = current_perception.get('num_potential_matches', 0)
            previous_num_matches = previous_state.get('num_potential_matches', 0)
            
            # Create prompt for reflection
            system_prompt = """You are an analytical assistant for a Candy Crush AI agent. Generate a brief, insightful reflection on the game state changes and the effectiveness of recent actions. Focus on strategic insights and patterns that would help the agent make better decisions."""
            
            user_prompt = f"""Please analyze the following Candy Crush game states and actions to generate a brief reflection:

Previous Game State:
{json.dumps(previous_state, indent=2)}

Previous Action: {str(previous_action)}

Current Game State:
{json.dumps(current_perception, indent=2)}

Last Action: {str(last_action)}

Focus your reflection on:
1. How the game state changed after the last action
2. Whether the action was effective (created matches or opened opportunities)
3. The current state of potential matches
4. Any strategic insights for future actions

Keep your reflection under 100 words and focus only on the most important insights."""
            
            # Make the API call for reflection
            response, _ = self.api_manager.text_completion(
                model_name=self.model_name,
                system_prompt=system_prompt,
                prompt=user_prompt,
                reasoning_effort="medium",
                thinking=False
            )
            
            return response.strip()
            
        except Exception as e:
            print(f"Error generating reflection: {str(e)}")
            return f"No reflection available due to error: {str(e)[:50]}"
    
    def add_game_state(self, game_state, action=None, timestamp=None):
        """
        Add a new game state to memory.
        
        Args:
            game_state (dict): The perceived game state to add
            action (tuple, optional): Action taken in the previous state ((r1, c1), (r2, c2))
            timestamp (float, optional): Timestamp for the game state
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Generate reflection if we have at least one previous state
        reflection = None
        if len(self.memory) > 0:
            reflection = self.generate_reflection(game_state, action)
            
        # Add timestamp, action and reflection to the game state
        memory_entry = {
            "timestamp": timestamp,
            "game_state": game_state,
            "last_action": action,
            "reflection": reflection
        }
        
        # Add to memory
        self.memory.append(memory_entry)
        
        # Keep only the last max_memory entries
        if len(self.memory) > self.max_memory:
            self.memory = self.memory[-self.max_memory:]
        
        # Save updated memory
        self.save_memory()
        
    def get_memory_summary(self):
        """
        Get a summary of the memory for the reasoning module.
        
        Returns:
            list: List of memory entries
        """
        return self.memory

class CandyCrushReasoningModule:
    """Module for planning next actions in Candy Crush."""
    
    def __init__(self, model_name="claude-3-7-sonnet-latest"):
        """
        Initialize the Reasoning Module for action planning.
        
        Args:
            model_name (str): Name of the model to use for reasoning
        """
        self.model_name = model_name
        self.api_manager = APIManager(game_name="candy_crush")
        self.system_prompt = """You are an AI assistant playing Candy Crush. Your goal is to find the best move that will create matches of 3 or more identical candies.

I want your response to be formatted as follows:
thought: [Your reasoning about the game state]
move: ((row1, col1), (row2, col2))

The move should be a valid swap between two adjacent tiles that creates a match."""
    
    async def plan_action(self, current_perception, memory_summary, img_path=None, max_retries=3):
        """
        Plan the next action based on current perception and memory.
        
        Args:
            current_perception (dict): Current perceived game state
            memory_summary (list): Summary of past game states
            img_path (str, optional): Path to the current board image (not used in this implementation)
            max_retries (int): Maximum number of retry attempts if response is empty
            
        Returns:
            dict: A dictionary containing move and thought
        """
        try:
            # Extract board representation from perception data
            board = current_perception.get('board', [])
            potential_matches = current_perception.get('potential_matches', [])
            
            # Create a text representation of the board
            board_repr = "Current board state:\n"
            for row in board:
                board_repr += " ".join(str(int(col)) for col in row) + "\n"
            
            # Create a text representation of potential matches
            matches_repr = "Potential matches found by perception module:\n"
            for i, match in enumerate(potential_matches):
                matches_repr += f"{i+1}. Swap {match['coord1']} with {match['coord2']} ({match['direction']} match, colors: {match['colors']})\n"
            
            # Prepare memory context (use the last 10 entries at most)
            memory_context = self._prepare_memory_context(memory_summary)
            
            # Compose the prompt
            user_prompt = f"""Analyze this Candy Crush game state and find the best move.

{board_repr}

Recent game history:
{memory_context}

Your task is to select the best move based on the current board state and available matches.
Consider the following criteria:
1. Prefer moves that create matches of 4 or 5 candies over matches of 3
2. Prefer moves that might create cascading matches
3. Avoid moves that have been tried recently without success

Rules:
1. You can only swap adjacent candies (up, down, left, right - no diagonals)
2. A valid move must create at least one match of 3 or more identical candies
3. Matches can be horizontal or vertical lines (not diagonal)
4. Prefer moves that create matches of 4 or 5 candies over matches of 3
5. Consider moves that might create cascading matches

Format your response exactly as follows:
thought: [Your reasoning about which move is best and why]
move: ((row1, col1), (row2, col2))

Coordinates must be integers representing valid positions on the board. The move must be adjacent tiles."""
            
            # Implement retry mechanism
            retry_count = 0
            response = None
            parsed_response = None
            
            while (not parsed_response or 'move' not in parsed_response) and retry_count < max_retries:
                if retry_count > 0:
                    print(f"Retry attempt {retry_count}/{max_retries} for reasoning module...")
                    await asyncio.sleep(2)  # Short delay before retry
                
                # Use text completion, not vision
                response, _ = self.api_manager.text_completion(
                    model_name=self.model_name,
                    system_prompt=self.system_prompt,
                    prompt=user_prompt,
                    thinking=True,
                    reasoning_effort="high",
                    token_limit=100000
                )
                
                # Parse the response
                parsed_response = self._parse_response(response)
                retry_count += 1
                
                if not parsed_response or 'move' not in parsed_response:
                    continue
                    
                # Verify the move is valid (adjacent tiles)
                move = parsed_response['move']
                if move and len(move) == 2:
                    (r1, c1), (r2, c2) = move
                    # Check if adjacent
                    if self._is_adjacent(r1, c1, r2, c2):
                        break
                    else:
                        print(f"Invalid move: {move} - tiles not adjacent")
                        parsed_response = None
            
            # If all retries failed, find a fallback move
            if not parsed_response or 'move' not in parsed_response:
                # Use the first valid match from perception if available
                if potential_matches:
                    first_match = potential_matches[0]
                    fallback_move = (first_match['coord1'], first_match['coord2'])
                    return {
                        "move": fallback_move,
                        "thought": "Fallback to first perceived match after failed reasoning attempts"
                    }
                else:
                    # Last resort: random adjacent tiles from the board
                    rows = len(board)
                    cols = len(board[0]) if rows > 0 else 0
                    # Middle of the board if possible
                    mid_r, mid_c = rows // 2, cols // 2
                    fallback_move = ((mid_r, mid_c), (mid_r, mid_c + 1))
                    return {
                        "move": fallback_move,
                        "thought": "Fallback to random move after failed reasoning attempts"
                    }
            
            return parsed_response
            
        except Exception as e:
            print(f"Error in reasoning module: {e}")
            # Return a default action on error
            return {
                "move": ((0, 0), (0, 1)),
                "thought": f"Error occurred in reasoning: {str(e)}"
            }
    
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
            
        # Take up to the last 10 memory entries to keep context concise
        recent_memory = memory_summary[-10:] if len(memory_summary) > 10 else memory_summary
        
        # Create a summary string
        summary_parts = []
        for i, entry in enumerate(recent_memory):
            timestamp = entry.get("timestamp", "unknown_time")
            last_action = entry.get("last_action", None)
            reflection = entry.get("reflection", None)
            
            entry_summary = f"Move {len(memory_summary) - len(recent_memory) + i + 1}:\n"
            if last_action:
                entry_summary += f"- Action: Swapped {last_action[0]} with {last_action[1]}\n"
            if reflection:
                entry_summary += f"- Reflection: {reflection}\n"
            
            summary_parts.append(entry_summary)
            
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
        
        # Split response by lines
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            
            # Extract thought
            if line.startswith("thought:"):
                thought = line[len("thought:"):].strip()
            
            # Extract move
            elif line.startswith("move:"):
                move_text = line[len("move:"):].strip()
                try:
                    # Expected format: ((row1, col1), (row2, col2))
                    import re
                    # Extract coordinates from the move text
                    match = re.search(r'\(\((\d+),\s*(\d+)\),\s*\((\d+),\s*(\d+)\)\)', move_text)
                    if match:
                        r1, c1, r2, c2 = map(int, match.groups())
                        move = ((r1, c1), (r2, c2))
                except Exception as e:
                    print(f"Error parsing move: {e}")
                    move = None
        
        # If parsing failed, use default values
        if move is None:
            print(f"Failed to parse move from response: {response}")
        
        if thought is None:
            thought = "No explicit thought provided in response"
            print(f"Failed to parse thought from response: {response}")
        
        return {
            "move": move,
            "thought": thought
        }
    
    def _is_adjacent(self, r1, c1, r2, c2):
        """Check if two tiles are adjacent."""
        # Adjacent means exactly one coordinate differs by exactly 1
        r_diff = abs(r1 - r2)
        c_diff = abs(c1 - c2)
        return (r_diff == 1 and c_diff == 0) or (r_diff == 0 and c_diff == 1)


class CandyCrushBaseModule:
    """A simplified module that directly processes observation images and returns actions."""
    
    def __init__(self, model_name="claude-3-7-sonnet-latest"):
        """
        Initialize the Base Module for direct action planning.
        
        Args:
            model_name (str): Name of the model to use for reasoning
        """
        self.model_name = model_name
        self.api_manager = APIManager(game_name="candy_crush")
        self.last_action = None
        
        # System prompt with strict output instructions
        self.system_prompt = """You are an AI assistant playing Candy Crush. Your goal is to find the best move that will create matches of 3 or more identical candies.

I want your response to be formatted as follows:
thought: [Your reasoning about the game state]
move: ((row1, col1), (row2, col2))

The move should be a valid swap between two adjacent tiles that creates a match."""
        
        # For direct observation analysis and action planning
        self.action_prompt = """Analyze the current Candy Crush board state shown in the image.

Your task is to identify and suggest the best move by finding two adjacent candy tiles to swap that will create a match of 3 or more identical candies.

Rules:
1. You can only swap adjacent candies (up, down, left, right - no diagonals)
2. A valid move must create at least one match of 3 or more identical candies
3. Matches can be horizontal or vertical lines (not diagonal)
4. Prefer moves that create matches of 4 or 5 candies over matches of 3
5. Consider moves that might create cascading matches

Based on your analysis, identify the coordinates of two adjacent candies to swap.
Coordinates are in (row, column) format, with (0,0) at the top-left corner.

Format your response exactly as follows:
thought: [Your detailed reasoning about which move is best and why]
move: ((row1, col1), (row2, col2))

Make sure your coordinates are integers representing valid positions on the visible board.
The candies to swap MUST be adjacent."""
    
    def process_observation(self, observation, info=None):
        """
        Process the observation directly to plan the next action.
        
        Args:
            observation: The game observation (image or board state)
            info: Additional info from the environment
            
        Returns:
            dict: A dictionary containing move and thought
        """
        try:
            # Base module should only use images
            # Get the image path from info if provided (from main script)
            img_path = info.get('img_path', None) if isinstance(info, dict) else None
            
            # If we have a valid observation with image data
            if isinstance(observation, np.ndarray) and len(observation.shape) == 3 and observation.shape[2] == 3:
                # Create a user prompt for image analysis
                user_prompt = f"""Analyze this Candy Crush board image and find the best move.

{self.action_prompt}

Based on the current game state in the image, what is the best move you can make?"""
                
                # Make API call with vision model
                response, _ = self.api_manager.vision_text_completion(
                    model_name=self.model_name,
                    system_prompt=self.system_prompt,
                    prompt=user_prompt,
                    image_path=img_path,
                    thinking=True,
                    reasoning_effort="high",
                    token_limit=100000
                )
                
                # Parse the response
                action_plan = self._parse_response(response)
                
                # Store this action for the next iteration
                self.last_action = action_plan.get("move")
                
                return action_plan
            else:
                # If no valid image observation, return a simple default action
                print("Base module received no valid image. Using default action.")
                return {
                    "move": ((0, 0), (0, 1)),
                    "thought": "No valid image observation available"
                }
            
        except Exception as e:
            print(f"Error in Base_module: {e}")
            # Return a default action on error
            return {
                "move": ((0, 0), (0, 1)),
                "thought": f"Error occurred in Base_module: {str(e)}"
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
        
        # Split response by lines
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            
            # Extract thought
            if line.startswith("thought:"):
                thought = line[len("thought:"):].strip()
            
            # Extract move
            elif line.startswith("move:"):
                move_text = line[len("move:"):].strip()
                try:
                    # Expected format: ((row1, col1), (row2, col2))
                    import re
                    # Extract coordinates from the move text
                    match = re.search(r'\(\((\d+),\s*(\d+)\),\s*\((\d+),\s*(\d+)\)\)', move_text)
                    if match:
                        r1, c1, r2, c2 = map(int, match.groups())
                        move = ((r1, c1), (r2, c2))
                except Exception as e:
                    print(f"Error parsing move: {e}")
                    move = None
        
        # If parsing failed, use default values
        if move is None:
            print(f"Failed to parse move from response: {response}")
            # Default to a simple move if parsing fails
            move = ((0, 0), (0, 1))
        
        if thought is None:
            thought = "No explicit thought provided in response"
            print(f"Failed to parse thought from response: {response}")
        
        return {
            "move": move,
            "thought": thought
        }
