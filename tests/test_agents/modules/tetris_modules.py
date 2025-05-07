import numpy as np
from collections import OrderedDict
import json
from tools.serving import APIManager
from datetime import datetime
import os
import time
import asyncio
from PIL import Image

# Tetris action space mapping
tetris_actions = {
    "move_left": 0,
    "move_right": 1,
    "move_down": 2,
    "rotate_clockwise": 3,
    "rotate_counterclockwise": 4,
    "hard_drop": 5,
    "swap": 6,
    "no_operation": 7
}

# Default fallback action
DEFAULT_ACTION = [7]  # no_operation

class TetrisPerceptionModule:
    """Module for analyzing Tetris game state."""
    
    def __init__(self, model_name="claude-3-7-sonnet-latest"):
        """Initialize the Perception Module for Tetris."""
        self.model_name = model_name
    
    def analyze_board(self, observation, info=None):
        """
        Analyze the Tetris board state.
        
        Args:
            observation: The game observation
            info: Additional information from the environment
            
        Returns:
            dict: A dictionary containing board analysis
        """
        try:
            # Start with basic structure even if observation is None
            perception_data = {
                'board_shape': None,
                'board_text': 'Unable to analyze board',
                'active_piece': None,
                'active_position': None,
                'rotated_board_texts': [],
                'potential_states': []
            }
            
            if observation is None:
                print("Warning: Received None observation in perception module")
                return perception_data
                
            # Extract the board and active piece info
            board = None
            active_piece = None
            active_position = None
            
            # Try to parse the observation structure
            try:
                # Handle various observation formats
                if isinstance(observation, dict):
                    # Dict format observation
                    board = observation.get('board', None)
                    if 'active_piece' in observation:
                        active_piece = observation['active_piece']
                    if 'active_position' in observation:
                        active_position = observation['active_position']
                elif isinstance(observation, np.ndarray):
                    # Direct board array
                    board = observation
                    # Active piece info might be in 'info'
                    if isinstance(info, dict):
                        active_piece = info.get('active_piece', None)
                        active_position = info.get('active_position', None)
            except Exception as parse_error:
                print(f"Error parsing observation: {parse_error}")
            
            # Update perception with whatever we were able to extract
            perception_data['board_shape'] = board.shape if board is not None and hasattr(board, 'shape') else None
            
            # Convert board to text representation for the agent
            if board is not None:
                board_text = self._board_to_text(board)
                perception_data['board_text'] = board_text
                
                # Generate rotated board texts for decision making
                rotated_board_texts = self._generate_rotated_boards(board)
                perception_data['rotated_board_texts'] = rotated_board_texts
            
            # Store active piece info
            perception_data['active_piece'] = active_piece
            perception_data['active_position'] = active_position
            
            # Generate potential board states for each possible move
            if board is not None and active_piece is not None and active_position is not None:
                try:
                    potential_states = self._generate_potential_states(board, active_piece, active_position)
                    perception_data['potential_states'] = potential_states
                except Exception as state_error:
                    print(f"Error generating potential states: {state_error}")
            
            return perception_data
        
        except Exception as e:
            print(f"Error in perception module: {e}")
            return {
                'board_shape': None,
                'board_text': 'Error in perception module',
                'active_piece': None,
                'active_position': None,
                'rotated_board_texts': [],
                'potential_states': []
            }
    
    def _board_to_text(self, board):
        """Convert board array to text representation."""
        if board is None:
            return "No board available"
        
        try:
            if not isinstance(board, np.ndarray):
                try:
                    board = np.array(board)
                except:
                    return "Cannot convert board to array"
            
            # Check if board has expected shape
            if len(board.shape) != 2:
                # Try to extract a 2D board from a 3D representation
                if len(board.shape) == 3 and board.shape[0] > 0:
                    board = board[0]  # Try to use the first channel/layer
                else:
                    return f"Board has unexpected shape: {board.shape}"
            
            rows, cols = board.shape
            
            # Create header
            board_str = "  " + " ".join([str(i % 10) for i in range(cols)]) + "\n"
            board_str += "  " + "-" * (cols * 2 - 1) + "\n"
            
            # Create rows
            for r in range(rows):
                board_str += f"{r % 10}|"
                for c in range(cols):
                    cell = board[r, c]
                    if cell == 0:
                        board_str += " ."
                    else:
                        board_str += f" {cell}"
                board_str += "\n"
            
            return board_str
        except Exception as e:
            return f"Error converting board to text: {str(e)}"
    
    def _generate_rotated_boards(self, board):
        """
        Generate text representations of the board for different rotations.
        
        Args:
            board: The current board state
            
        Returns:
            list: List of text representations for different board rotations
        """
        try:
            if board is None:
                return ["No board available"]
            
            # Convert to numpy array if not already
            if not isinstance(board, np.ndarray):
                try:
                    board = np.array(board)
                except:
                    return ["Cannot convert board to array"]
            
            # If board has unexpected shape, just return the text of the original
            if len(board.shape) != 2:
                # Try to extract a 2D board from a 3D representation
                if len(board.shape) == 3 and board.shape[0] > 0:
                    board = board[0]  # Try to use the first layer
                else:
                    return [self._board_to_text(board)]
            
            # Generate text representations for each rotation
            rotated_texts = []
            
            # Original board
            rotated_texts.append(self._board_to_text(board))
            
            return rotated_texts
        except Exception as e:
            print(f"Error generating rotated boards: {e}")
            return ["Error generating rotated boards"]
    
    def _generate_potential_states(self, board, active_piece, active_position):
        """Generate potential board states based on rotations and movements."""
        # This would simulate dropping the active piece with different rotations and positions
        # For now, return a placeholder as this requires detailed Tetris game logic
        return []

class TetrisMemoryModule:
    """Memory module for tracking Tetris game state history."""
    
    def __init__(self, memory_file=None, max_memory=10, model_name="claude-3-7-sonnet-latest"):
        """Initialize the Memory Module for tracking game state history."""
        self.memory_file = memory_file
        self.max_memory = max_memory
        self.memory = []
        self.model_name = model_name
        self.api_manager = APIManager(game_name="tetris")
        
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
    
    async def generate_reflection(self, current_perception, last_action):
        """
        Generate a reflection on the current game state based on the memory.
        
        Args:
            current_perception: Current perception data
            last_action: The last action taken
            
        Returns:
            str: A reflection on the current game state
        """
        try:
            if not current_perception:
                return "No perception data available for reflection."
            
            # Extract board information
            board_text = current_perception.get('board_text', "No board text available")
            
            # Construct the memory context from recent memory entries
            memory_context = ""
            if self.memory:
                # Get last few memory entries (limiting to avoid token overflow)
                recent_entries = self.memory[-5:]
                
                for i, entry in enumerate(recent_entries):
                    memory_context += f"Memory {i+1}:\n"
                    memory_context += f"Timestamp: {entry.get('timestamp', 'unknown')}\n"
                    memory_context += f"Board:\n{entry.get('board_text', 'unknown')}\n"
                    
                    if 'action' in entry and entry['action'] is not None:
                        memory_context += f"Action: {entry['action']}\n"
                    
                    if 'reflection' in entry and entry['reflection'] is not None:
                        memory_context += f"Reflection: {entry['reflection']}\n"
                    
                    memory_context += "\n"
            
            # Create the reflection prompt
            reflection_prompt = f"""
You are the memory module of a Tetris-playing AI. Your task is to reflect on the current game state and provide insights.

Recent memory entries:
{memory_context}

Current board state:
{board_text}

Last action taken: {last_action if last_action is not None else "None"}

Please provide a brief reflection (2-3 sentences) on the current game state. Consider:
1. How has the board changed since previous states?
2. Was the last move effective?
3. What areas of the board need attention?
4. Any strategic insights for future moves?

Reflection:
"""
            
            # Get reflection from API
            response, _ = await self.api_manager.text_completion(
                model_name=self.model_name,
                system_prompt="You are the memory module of a Tetris-playing AI that provides concise strategic reflections.",
                prompt=reflection_prompt,
                thinking=True,
                max_tokens=500,
                token_limit=4000
            )
            
            return response.strip()
            
        except Exception as e:
            print(f"Error generating reflection: {e}")
            return None
    
    def add_game_state(self, game_state, action=None, timestamp=None):
        """
        Add a new game state to memory.
        
        Args:
            game_state (dict): The perceived game state to add
            action (list, optional): Action sequence taken in the previous state
            timestamp (float, optional): Timestamp for the game state
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Generate reflection if we have at least one previous state
        reflection = None
        # Note: We'll call the async reflection function outside this synchronous method
        # Reflection will be added later
            
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
    
    async def update_reflection(self, current_perception, last_action):
        """
        Update the latest memory entry with a reflection.
        
        Args:
            current_perception: Current perception data
            last_action: The last action taken
            
        Returns:
            bool: True if reflection was successfully added, False otherwise
        """
        try:
            if not self.memory:
                return False
                
            # Generate the reflection
            reflection = await self.generate_reflection(current_perception, last_action)
            
            # Update the most recent memory entry with this reflection
            if reflection and self.memory:
                self.memory[-1]['reflection'] = reflection
                self.save_memory()
                return True
            
            return False
        except Exception as e:
            print(f"Error updating reflection: {e}")
            return False
    
    def get_memory_summary(self):
        """
        Get a summary of the memory for the reasoning module.
        
        Returns:
            list: List of memory entries
        """
        return self.memory

class TetrisReasoningModule:
    """Module for planning actions in Tetris."""
    
    def __init__(self, model_name="claude-3-7-sonnet-latest"):
        """Initialize the Reasoning Module for action planning."""
        self.model_name = model_name
        self.api_manager = APIManager(game_name="tetris")
        self.system_prompt = """You are an AI assistant playing Tetris. Your goal is to plan a sequence of actions to optimally place the current piece.

I want your response to be formatted as follows:
thought: [Your reasoning about the game state]
action_sequence: [List of action indices]

Your action sequence should include all necessary rotations, movements, and drops to place the current piece."""
    
    async def plan_action(self, current_perception, memory_summary, img_path=None, max_retries=3):
        """
        Plan the next action sequence based on current perception and memory.
        
        Args:
            current_perception (dict): Current perceived game state
            memory_summary (list): Summary of past game states
            img_path (str, optional): Path to the current board image (not used in this implementation)
            max_retries (int): Maximum number of retry attempts if response is empty
            
        Returns:
            dict: A dictionary containing action_sequence and thought
        """
        try:
            # Handle None or empty perception data
            if not current_perception:
                print("Warning: Empty perception data provided to reasoning module")
                return {
                    "action_sequence": DEFAULT_ACTION,
                    "thought": "No perception data available"
                }
                
            # Extract board representation from perception data
            board_text = current_perception.get('board_text', "No board text available")
            rotated_board_texts = current_perception.get('rotated_board_texts', [])
            
            # Prepare rotation representations for the prompt
            rotation_info = "Rotated board representations:\n\n"
            for i, rot_text in enumerate(rotated_board_texts):
                if i == 0:
                    rotation_info += f"Rotation 0 (original):\n{rot_text}\n\n"
                else:
                    rotation_info += f"Rotation {i} ({i*90} degrees counterclockwise):\n{rot_text}\n\n"
            
            # Prepare memory context
            memory_context = self._prepare_memory_context(memory_summary)
            
            # Create the action prompt with game mechanics explanation
            tetris_mechanics = """
Tetris Game Mechanics:
1. The game board is a grid where pieces (tetrominoes) fall from the top.
2. Your goal is to clear lines by filling all cells in a row, which removes that row.
3. The game ends if the pieces stack up to the top of the board.

Available Actions:
0: move_left - Move piece one cell to the left
1: move_right - Move piece one cell to the right
2: move_down - Move piece one cell down
3: rotate_clockwise - Rotate piece 90 degrees clockwise
4: rotate_counterclockwise - Rotate piece 90 degrees counterclockwise
5: hard_drop - Drop piece to the bottom instantly
6: swap - Swap with held piece (if available)
7: no_operation - Do nothing

Tetris Piece Rotations:
I-piece (line):
Rotation 0: IIII
Rotation 1: I
           I
           I
           I

O-piece (square):
Only one rotation: OO
                   OO

T-piece:
Rotation 0:  T
            TTT
Rotation 1: T
            TT
            T
Rotation 2: TTT
             T
Rotation 3:  T
            TT
            T

L-piece:
Rotation 0:   L
             LLL
Rotation 1: L
            L
            LL
Rotation 2: LLL
            L
Rotation 3: LL
             L
             L

J-piece:
Rotation 0: J
            JJJ
Rotation 1: JJ
            J
            J
Rotation 2: JJJ
              J
Rotation 3:  J
             J
             JJ

S-piece:
Rotation 0:  SS
            SS
Rotation 1: S
            SS
             S

Z-piece:
Rotation 0: ZZ
             ZZ
Rotation 1:  Z
            ZZ
            Z

Guidelines for planning:
1. Start with any necessary rotations (action 3 or 4)
2. Then position the piece horizontally (actions 0 and 1)
3. Finally, either let it drop gradually (action 2) or use hard drop (action 5)
4. Your sequence should be complete until the piece is placed
"""
            
            # Compose the prompt
            user_prompt = f"""Analyze this Tetris game state and plan an optimal action sequence.

Current Board State:
{board_text}

{rotation_info}

Recent Game History:
{memory_context}

{tetris_mechanics}

Your task is to plan a complete action sequence that will:
1. Rotate the active piece to the optimal orientation
2. Move it to the optimal position
3. Drop it into place

The sequence should be a list of action indices (0-7) that completely handles the current piece.
Example: [3, 3, 1, 1, 2, 2, 2, 2, 2] (rotate twice, move right twice, then move down until placed)

Format your response exactly as follows:
thought: [Your detailed reasoning about the optimal placement and why]
action_sequence: [list of action indices]

Make sure your action sequence is comprehensive and includes all steps needed."""
            
            # Implement retry mechanism
            retry_count = 0
            response = None
            parsed_response = None
            
            while (not parsed_response or 'action_sequence' not in parsed_response) and retry_count < max_retries:
                if retry_count > 0:
                    print(f"Retry attempt {retry_count}/{max_retries} for reasoning module...")
                    await asyncio.sleep(2)  # Short delay before retry
                
                # Use text completion for reasoning
                response, _ = await self.api_manager.text_completion(
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
                
                if not parsed_response or 'action_sequence' not in parsed_response:
                    continue
            
            # If all retries failed, use a fallback sequence
            if not parsed_response or 'action_sequence' not in parsed_response:
                # Use the default no_operation action
                return {
                    "action_sequence": DEFAULT_ACTION,
                    "thought": "Fallback action sequence after failed reasoning attempts"
                }
            
            return parsed_response
            
        except Exception as e:
            print(f"Error in reasoning module: {e}")
            # Return a default action sequence on error
            return {
                "action_sequence": DEFAULT_ACTION,
                "thought": f"Error occurred in reasoning: {str(e)}"
            }
    
    def _prepare_memory_context(self, memory_summary):
        """Prepare a concise summary of memory for the prompt."""
        if not memory_summary:
            return "No memory of past states available."
            
        # Take up to the last 10 memory entries to keep context concise
        recent_memory = memory_summary[-10:] if len(memory_summary) > 10 else memory_summary
        
        # Create a summary string
        summary_parts = []
        for i, entry in enumerate(recent_memory):
            timestamp = entry.get("timestamp", "unknown_time")
            game_state = entry.get("game_state", {})
            last_action = entry.get("last_action", None)
            reflection = entry.get("reflection", None)
            
            entry_summary = f"State {len(memory_summary) - len(recent_memory) + i + 1}:\n"
            if last_action:
                entry_summary += f"- Last action sequence: {last_action}\n"
            if reflection:
                entry_summary += f"- Reflection: {reflection}\n"
            
            summary_parts.append(entry_summary)
            
        return "\n".join(summary_parts)
    
    def _parse_response(self, response):
        """Parse the reasoning response to extract thought and action_sequence."""
        action_sequence = None
        thought = None
        
        # Split response by lines
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            
            # Extract thought
            if line.startswith("thought:"):
                thought = line[len("thought:"):].strip()
            
            # Extract action sequence
            elif line.startswith("action_sequence:"):
                action_text = line[len("action_sequence:"):].strip()
                try:
                    # Expected format: [0, 1, 2, 3, 4, 5]
                    # Extract action sequence using eval (careful with this in production)
                    action_sequence = eval(action_text)
                    # Validate that all actions are within range
                    if not all(0 <= a <= 7 for a in action_sequence):
                        print(f"Invalid action in sequence: {action_sequence}")
                        action_sequence = None
                except Exception as e:
                    print(f"Error parsing action sequence: {e}")
                    action_sequence = None
        
        # If parsing failed, use default values
        if action_sequence is None:
            print(f"Failed to parse action sequence from response: {response}")
        
        if thought is None:
            thought = "No explicit thought provided in response"
            print(f"Failed to parse thought from response: {response}")
        
        return {
            "action_sequence": action_sequence,
            "thought": thought
        }

class TetrisBaseModule:
    """A simplified module that directly processes observation images and returns actions."""
    
    def __init__(self, model_name="claude-3-7-sonnet-latest"):
        """Initialize the Base Module for direct action planning."""
        self.model_name = model_name
        self.api_manager = APIManager(game_name="tetris")
        self.last_action = None
        
        # System prompt with strict output instructions
        self.system_prompt = """You are an AI assistant playing Tetris. Your goal is to plan a sequence of actions to optimally place the current piece.

I want your response to be formatted as follows:
thought: [Your reasoning about the game state]
action_sequence: [List of action indices]

Your action sequence should include all necessary rotations, movements, and drops to place the current piece."""
    
    async def process_observation(self, observation, info=None):
        """
        Process the observation directly to plan the next action sequence.
        
        Args:
            observation: The game observation (image)
            info: Additional info from the environment
            
        Returns:
            dict: A dictionary containing action_sequence and thought
        """
        try:
            # Base module uses images
            # Get the image path from info if provided
            img_path = info.get('img_path', None) if isinstance(info, dict) else None
            
            # If we have a valid observation with image data
            if isinstance(observation, np.ndarray) and len(observation.shape) == 3:
                # Create a user prompt for image analysis
                tetris_mechanics = """
Tetris Game Mechanics:
1. The game board is a grid where pieces (tetrominoes) fall from the top.
2. Your goal is to clear lines by filling all cells in a row, which removes that row.
3. The game ends if the pieces stack up to the top of the board.

Available Actions:
0: move_left - Move piece one cell to the left
1: move_right - Move piece one cell to the right
2: move_down - Move piece one cell down
3: rotate_clockwise - Rotate piece 90 degrees clockwise
4: rotate_counterclockwise - Rotate piece 90 degrees counterclockwise
5: hard_drop - Drop piece to the bottom instantly
6: swap - Swap with held piece (if available)
7: no_operation - Do nothing

Tetris Piece Rotations:
I-piece (line):
Rotation 0: IIII
Rotation 1: I
           I
           I
           I

O-piece (square):
Only one rotation: OO
                   OO

T-piece:
Rotation 0:  T
            TTT
Rotation 1: T
            TT
            T
Rotation 2: TTT
             T
Rotation 3:  T
            TT
            T

L-piece:
Rotation 0:   L
             LLL
Rotation 1: L
            L
            LL
Rotation 2: LLL
            L
Rotation 3: LL
             L
             L

J-piece:
Rotation 0: J
            JJJ
Rotation 1: JJ
            J
            J
Rotation 2: JJJ
              J
Rotation 3:  J
             J
             JJ

S-piece:
Rotation 0:  SS
            SS
Rotation 1: S
            SS
             S

Z-piece:
Rotation 0: ZZ
             ZZ
Rotation 1:  Z
            ZZ
            Z

Guidelines for planning:
1. Start with any necessary rotations (action 3 or 4)
2. Then position the piece horizontally (actions 0 and 1)
3. Finally, either let it drop gradually (action 2) or use hard drop (action 5)
4. Your sequence should be complete until the piece is placed
"""
                
                user_prompt = f"""Analyze this Tetris screenshot and plan an optimal action sequence.

{tetris_mechanics}

Your task is to:
1. Analyze the board state and identify the current falling piece
2. Plan a complete action sequence that will:
   - Rotate the active piece to the optimal orientation
   - Move it to the optimal position
   - Drop it into place

The sequence should be a list of action indices (0-7) that completely handles the current piece.
Example: [3, 3, 1, 1, 2, 2, 2, 2, 2] (rotate twice, move right twice, then move down until placed)

Format your response exactly as follows:
thought: [Your detailed reasoning about the optimal placement and why]
action_sequence: [list of action indices]

Make sure your action sequence is comprehensive and includes all steps needed."""
                
                # Make API call with vision model
                response, _ = await self.api_manager.vision_text_completion(
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
                self.last_action = action_plan.get("action_sequence")
                
                return action_plan
            else:
                # If no valid image observation, return a simple default sequence
                print("Base module received no valid image. Using default action sequence.")
                return {
                    "action_sequence": DEFAULT_ACTION,
                    "thought": "No valid image observation available"
                }
            
        except Exception as e:
            print(f"Error in Base_module: {e}")
            # Return a default action sequence on error
            return {
                "action_sequence": DEFAULT_ACTION,
                "thought": f"Error occurred in Base_module: {str(e)}"
            }
    
    def _parse_response(self, response):
        """Parse the response to extract thought and action_sequence."""
        action_sequence = None
        thought = None
        
        # Split response by lines
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            
            # Extract thought
            if line.startswith("thought:"):
                thought = line[len("thought:"):].strip()
            
            # Extract action sequence
            elif line.startswith("action_sequence:"):
                action_text = line[len("action_sequence:"):].strip()
                try:
                    # Expected format: [0, 1, 2, 3, 4, 5]
                    # Extract action sequence using eval (careful with this in production)
                    action_sequence = eval(action_text)
                    # Validate that all actions are within range
                    if not all(0 <= a <= 7 for a in action_sequence):
                        print(f"Invalid action in sequence: {action_sequence}")
                        action_sequence = None
                except Exception as e:
                    print(f"Error parsing action sequence: {e}")
                    action_sequence = None
        
        # If parsing failed, use default values
        if action_sequence is None:
            print(f"Failed to parse action sequence from response: {response}")
            # Default to no_operation if parsing fails
            action_sequence = DEFAULT_ACTION
        
        if thought is None:
            thought = "No explicit thought provided in response"
            print(f"Failed to parse thought from response: {response}")
        
        return {
            "action_sequence": action_sequence,
            "thought": thought
        }
