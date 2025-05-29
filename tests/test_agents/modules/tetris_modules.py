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
    
    def __init__(self, model_name="claude-3-7-sonnet-latest", observation=None ):
        """Initialize the Perception Module for Tetris."""
        self.model_name = model_name
        self.observation = observation
        self.board = observation['board']
        self.active_piece = observation['active_tetromino_mask']
        self.held_piece = observation['holder']
        self.next_pieces = observation['queue']

    def convert_board_to_symbol(self,board):
        """Return a simplified text version of the board for debugging"""
        # Create a copy to visualize, skipping border and showing only the game area
        # Assuming borders are at indices [0-3, 14-17] horizontally and [20-23] vertically
        vis_board = board[0:20, 4:14].copy()
        
        # Replace numbers with symbols for better visualization
        symbols = {0: '.', 1: '#', 2: 'I', 3: 'O', 4: 'T', 5: 'S', 6: 'Z', 7: 'J', 8: 'L'}
        
        rows = []
        for row in vis_board:
            rows.append(''.join([symbols.get(cell, str(cell)) for cell in row]))
        
        return '\n'.join(rows)
    
    def get_board(self):
        """Return the board as a numpy array"""
        return self.convert_board_to_symbol(self.board)
    
    def extract_active_tetromino(self, board, active_mask):
        """Extract the active tetromino from the board using the mask"""
        # Use element-wise multiplication to extract the active tetromino
        active_piece = board * active_mask
        
        # Find the bounds of the active piece
        non_zero = np.nonzero(active_mask)
        if len(non_zero[0]) == 0:
            return None, None, None, None
        
        min_row, max_row = np.min(non_zero[0]), np.max(non_zero[0])
        min_col, max_col = np.min(non_zero[1]), np.max(non_zero[1])
        
        # Extract the piece and its position
        piece = active_piece[min_row:max_row+1, min_col:max_col+1]
        
        return piece, min_row, min_col, active_piece
    
    def get_piece_id(self,piece):
        """Identify the piece type from the piece matrix"""
        unique_values = np.unique(piece)
        unique_values = unique_values[unique_values > 0]  # Ignore empty cells
        if len(unique_values) == 0:
            return None
        return unique_values[0] 
    
    def rotate_piece(self, piece):
        """Rotate a tetromino piece clockwise using np.rot90, matching the gymnasium environment"""
        # Simply use np.rot90 with k=1 for clockwise rotation
        # This matches the Tetris gymnasium environment's rotate function
        return np.rot90(piece, k=1)
    
    def generate_rotated_and_dropped_boards(self):
        """
        Generate potential boards with the active piece rotated and dropped
        using simple rotation and collision detection
        
        Args:
            board: The game board
            active_mask: Mask of the active tetromino
            active_orientation: Not used in this simpler implementation
        
        Returns:
            List of dictionaries with potential board states
        """
        board = self.board
        active_mask = self.active_piece
        # Extract the active tetromino
        piece, row, col, active_piece_on_board = self.extract_active_tetromino(board, active_mask)
        if piece is None:
            return []
        
        # Get the piece ID (type)
        piece_id = self.get_piece_id(piece)
        if piece_id is None:
            return []
        
        # Determine max rotations based on piece type
        # I, S, Z have 2 rotations
        # T, L, J have 4 rotations
        # O has 1 rotation (no change)
        max_rotations = 1  # Default for O piece
        if piece_id in [2]:  # I piece
            max_rotations = 2
        elif piece_id in [5, 6]:  # S and Z pieces
            max_rotations = 2
        elif piece_id in [4, 7, 8]:  # T, J, L pieces
            max_rotations = 4
        
        # Clean the board by removing the active piece
        clean_board = board.copy()
        clean_board[active_mask > 0] = 0
        
        results = []
        
        # First, add the original position (no rotation, no drop)
        original_board = clean_board.copy()
        
        # Place the original piece on the board
        for r in range(piece.shape[0]):
            for c in range(piece.shape[1]):
                if piece[r, c] > 0:
                    if (row+r < original_board.shape[0] and 
                        col+c < original_board.shape[1]):
                        original_board[row+r, col+c] = piece[r, c]
        
        # Add original position board
        results.append({
            'rotation': 0,
            'board': original_board,
            'row': row,
            'col': col,
            'description': "Original position"
        })
        
        # Skip rotation for O piece (ID 3)
        if piece_id == 3:
            return results
        
        # For each possible rotation (starting from 1)
        current_piece = piece.copy()
        for rotation in range(1, max_rotations):
            # Rotate the piece (apply rotation multiple times for higher rotation states)
            current_piece = self.rotate_piece(current_piece)
            
            # Try wall kicks for rotation (simple implementation)
            # Each tuple is (col_offset, row_offset)
            wall_kicks = [(0, 0), (-1, 0), (1, 0), (0, -1)]
            
            # Try each wall kick until a valid rotation is found
            rotation_applied = False
            for col_offset, row_offset in wall_kicks:
                new_row = row + row_offset
                new_col = col + col_offset
                
                # Check if rotation is valid at this position
                valid = True
                for r in range(current_piece.shape[0]):
                    for c in range(current_piece.shape[1]):
                        if current_piece[r, c] > 0:
                            board_r = new_row + r
                            board_c = new_col + c
                            
                            # Check bounds and collision
                            if (board_r < 0 or board_r >= board.shape[0] or
                                board_c < 0 or board_c >= board.shape[1] or
                                clean_board[board_r, board_c] > 0):
                                valid = False
                                break
                    if not valid:
                        break
                
                if valid:
                    # Found a valid rotation, now try to drop it one row
                    can_drop = True
                    drop_row = new_row + 1
                    
                    # Check if the piece can be dropped
                    for r in range(current_piece.shape[0]):
                        for c in range(current_piece.shape[1]):
                            if current_piece[r, c] > 0:
                                board_r = drop_row + r
                                board_c = new_col + c
                                
                                # Check bounds and collision
                                if (board_r >= board.shape[0] or
                                    clean_board[board_r, board_c] > 0):
                                    can_drop = False
                                    break
                        if not can_drop:
                            break
                    
                    # Create board with rotated piece
                    rotated_board = clean_board.copy()
                    
                    # Place rotated piece at the appropriate position (dropped if possible)
                    final_row = drop_row if can_drop else new_row
                    for r in range(current_piece.shape[0]):
                        for c in range(current_piece.shape[1]):
                            if current_piece[r, c] > 0:
                                if (final_row+r < rotated_board.shape[0] and 
                                    new_col+c < rotated_board.shape[1]):
                                    rotated_board[final_row+r, new_col+c] = current_piece[r, c]
                    
                    # Add to results
                    results.append({
                        'rotation': rotation,
                        'board': rotated_board,
                        'row': final_row,
                        'col': new_col,
                        'description': f"Rotation {rotation}" + (" + Drop" if can_drop else "")
                    })
                    
                    # We found a valid rotation, mark as applied and break out of wall kicks loop
                    rotation_applied = True
                    break
            
            # If we couldn't apply this rotation, we might still want to continue to the next rotation state
            if not rotation_applied:
                continue
        
        return results
    
    def get_next_pieces(self):
        """Get the next piece from the queue"""
        return self.piece_to_symbol(self.next_pieces)
    
    def piece_to_symbol(self,pieces):
        """Convert piece numbers to symbols and return formatted string"""
        symbols = {0: '.', 2: 'I', 3: 'O', 4: 'T', 5: 'S', 6: 'Z', 7: 'J', 8: 'L'}
        if isinstance(pieces, np.ndarray):
            return '\n'.join([''.join([symbols.get(int(cell), str(cell)) for cell in row]) for row in pieces])
        return symbols.get(int(pieces), str(pieces))
    
    def get_perception_data(self):
        """Get the perception data"""
        return {
            "board": self.get_board(),
            "next_pieces": self.get_next_pieces(),
            "potential_states": [self.convert_board_to_symbol(state['board']) for state in self.generate_rotated_and_dropped_boards()],
        }
        
    

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
                print(f"Memory loaded successfully: {len(self.memory)} entries")
            else:
                if self.memory_file:
                    print(f"Memory file does not exist: {self.memory_file}")
                else:
                    print("No memory file specified")
        except Exception as e:
            print(f"Error loading memory: {e}")
            self.memory = []
            
    def save_memory(self):
        """Save the current memory to the memory file."""
        try:
            if self.memory_file:
                print(f"Saving memory to file: {self.memory_file}")
                with open(self.memory_file, 'w') as f:
                    json.dump(self.memory, f, indent=2)
                print(f"Memory saved successfully: {len(self.memory)} entries")
        except Exception as e:
            print(f"Error saving memory: {e}")
    
    async def generate_reflection(self, current_perception, last_action):
        """
        Generate a reflection on the current game state based on the memory.
        
        Args:
            current_perception: Current perception data (simplified to {'board_text': ..., 'next_pieces_text': ...})
            last_action: The last action taken
            
        Returns:
            str: A reflection on the current game state
        """
        try:
            if not current_perception:
                return "No perception data available for reflection."
            
            # Construct the memory context from recent memory entries
            memory_context = ""
            if self.memory:
                print(f"Using {len(self.memory)} memory entries for reflection context")
                # Get last few memory entries (respecting self.max_memory, up to 5 if max_memory is 5)
                # The self.memory already respects max_memory due to how add_game_state works
                recent_entries = self.memory # Use all available memory up to max_memory
                if len(self.memory) > 1: # if there is at least one previous state to compare to current
                    recent_entries = self.memory[:-1] # all but the most recent (which is before current_perception)
                elif len(self.memory) == 1: # only one entry, which is previous state
                     recent_entries = self.memory
                else: # no memory yet, this case should be caught by outer if self.memory, but for safety
                    recent_entries = []

                for i, entry in enumerate(reversed(recent_entries)): # Show most recent past states first
                    memory_context += f"Past State {i+1} (from {len(recent_entries)-i} steps ago):\n"
                    memory_context += f"Timestamp: {entry.get('timestamp', 'unknown')}\n"
                    
                    game_state_in_memory = entry.get('game_state', {})
                    past_board_text = game_state_in_memory.get('board_text', 'Unknown past board')
                    past_next_pieces_text = game_state_in_memory.get('next_pieces_text', 'Unknown past next pieces')
                    
                    memory_context += f"Board:\n{past_board_text}\n"
                    memory_context += f"Next Pieces:\n{past_next_pieces_text}\n"
                    
                    if 'last_action' in entry and entry['last_action'] is not None:
                        memory_context += f"Action taken before this state: {entry['last_action']}\n"
                    
                    if 'reflection' in entry and entry['reflection'] is not None:
                        memory_context += f"Reflection on this state: {entry['reflection']}\n"
                    memory_context += "-----\n"
            else:
                memory_context = "No past states in memory.\n"
            
            # Create the reflection prompt
            reflection_prompt = f"""
You are the memory module of a Tetris-playing AI. Your task is to reflect on the current game state and provide insights to help the AI improve.

{memory_context}

Current State:
Board:
{current_perception.get('board_text', "No current board text available")}
Next Pieces:
{current_perception.get('next_pieces_text', "No current next pieces text available")}

Last action taken (that led to this Current State): {last_action if last_action is not None else "Game Start or Unknown"}

Please provide a brief reflection (2-3 sentences) on the situation. Consider:
1. How has the board changed from the most recent past state (if any) due to the last action?
2. Was the last action effective in improving the board state or setting up future plays?
3. What are the immediate opportunities or dangers on the current board?
4. Are there any strategic considerations based on the current board and upcoming pieces?

Reflection:"""
            
            # Get reflection from API
            response, _ =  self.api_manager.text_completion(
                model_name=self.model_name,
                system_prompt="You are the memory module of a Tetris-playing AI that provides concise strategic reflections.",
                prompt=reflection_prompt,
                thinking=False,
                reasoning_effort="medium",
                token_limit=100000
            )
            
            return response.strip()
            
        except Exception as e:
            print(f"Error generating reflection: {e}")
            return "Reflection error: Could not generate insights."
    
    async def add_game_state(self, game_state, action=None, timestamp=None):
        """
        Add a new game state to memory.
        
        Args:
            game_state (dict): The perceived game state to add (simplified to {'board_text': ..., 'next_pieces_text': ...})
            action (list, optional): Action sequence taken that LED to this game_state
            timestamp (float, optional): Timestamp for the game state
        """
        if timestamp is None:
            timestamp = time.time()
        
        # The reflection should be about the *transition* to the new game_state based on the *previous* state and *action* taken.
        # So, we generate reflection *before* adding the new state, using the new state as 'current_perception'.
        reflection = None
        if self.memory: # If there's at least one state in memory (which would be the previous state)
            reflection = await self.generate_reflection(game_state, action) # game_state is the new state, action is what led to it
        elif not self.memory and action is None: # Very first state of the game
             reflection = "Initial game state. No prior actions or board states to compare. Ready to play!"

        memory_entry = {
            "timestamp": timestamp,
            "game_state": game_state, # This is the simplified dict now
            "last_action": action, # Action that led to this game_state
            "reflection": reflection # Reflection on the transition to this state, or initial state reflection
        }
        
        self.memory.append(memory_entry)
        if len(self.memory) > self.max_memory:
            self.memory.pop(0) # Remove the oldest entry to maintain max_memory
        
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
            
            # Handle both potential_states and rotated_board_states
            rotated_states = current_perception.get('rotated_board_states', current_perception.get('potential_states', []))
            
            # Prepare rotation representations for the prompt
            rotation_info = "Rotation Information:\n\n"
            
            if rotated_states:
                for i, state in enumerate(rotated_states):
                    # Check if it's a dict with the expected structure
                    if isinstance(state, dict) and 'description' in state and 'board' in state:
                        # This is the new format from generate_rotated_and_dropped_boards
                        description = state['description']
                        board = state['board']
                        # Convert the board to text if needed
                        if isinstance(board, np.ndarray):
                            # Use a simple string representation
                            board_lines = []
                            rows, cols = board.shape
                            for r in range(rows):
                                row_str = ' '.join([str(int(board[r, c])) for c in range(cols)])
                                board_lines.append(row_str)
                            board_text = '\n'.join(board_lines)
                        else:
                            board_text = str(board)
                        
                        rotation_info += f"{description}:\n{board_text}\n\n"
                    else:
                        # Fallback for any other format
                        rotation_info += f"Rotation {i}:\n{state}\n\n"
            else:
                rotation_info += "No rotation information available.\n"
            
            # Add queue information to the prompt
            queue_info = "Next Pieces Queue:\n"
            piece_queue_symbols = current_perception.get('piece_queue_symbols', [])
            piece_queue_texts = current_perception.get('piece_queue_texts', [])
            
            if piece_queue_symbols:
                queue_info += f"Symbols: {piece_queue_symbols}\n\n"
                
                if piece_queue_texts:
                    for i, piece_text in enumerate(piece_queue_texts):
                        queue_info += f"Next Piece {i+1}:\n{piece_text}\n\n"
            else:
                queue_info += "No upcoming pieces information available.\n"
            
            # Prepare memory context
            memory_context = self._prepare_memory_context(memory_summary)

            board_text = current_perception.get('board_text', "No board text available")
            queue_info = current_perception.get('next_pieces', "No queue info available")
            
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
5: hard_drop - Drop piece to the bottom instantly (RECOMMENDED for final placement)
6: swap - Swap with held piece (if available)
7: no_operation - Do nothing

Examples of common action sequences:
- [3, 1, 1, 5] → Rotate once clockwise, move right twice, then hard drop
- [3, 3, 0, 0, 5] → Rotate twice clockwise, move left twice, then hard drop
- [1, 1, 1, 5] → Move right three times, then hard drop
- [0, 0, 0, 3, 5] → Move left three times, rotate once, then hard drop

Tetris Piece Rotations (achieved with rotate_clockwise action):
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
3. Finally, use hard drop (action 5) for final placement
4. Consider the next pieces in queue when planning your current move
"""
            
            # Compose the prompt
            user_prompt = f"""Analyze this Tetris game state and plan an optimal action sequence.

Current Board State:
{board_text}
^ This shows the current playfield where '.' are empty spaces and '#' are filled cells. The current falling piece is shown in its position.

Rotation Information:
{rotation_info}
^ These show how your piece will look after rotating. To achieve any of these rotations, use the 'rotate_clockwise' action (3) the required number of times.

Next Pieces Queue:
{queue_info}
^ These are the upcoming pieces that will appear after the current piece is placed. Plan your current move with these in mind.

Recent Game History:
{memory_context}

{tetris_mechanics}

Your task is to plan a complete action sequence that will:
1. Rotate the active piece to the optimal orientation
2. Move it to the optimal position
3. Drop it into place (preferably using hard_drop action 5 after positioning)
4. Consider the upcoming pieces in the queue for better long-term placement.

The sequence should be a list of action indices (0-7) that completely handles the current piece.
Example: [3, 3, 1, 1, 5] (rotate twice, move right twice, then hard drop)

Format your response exactly as follows:
thought: [Your detailed reasoning about the optimal placement, including consideration of the queue, and why]
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
    
    def _prepare_memory_context(self, memory_summary):
        """Prepare a concise summary of memory for the prompt."""
        if not memory_summary:
            return "No memory of past states available."
            
        # Take up to the last 5-10 memory entries to keep context concise
        # The number can be adjusted based on typical context window and usefulness
        recent_memory = memory_summary[-5:] 
        
        summary_parts = []
        for i, entry in enumerate(recent_memory):
            # entry structure is assumed to be: 
            # {"timestamp": ..., "game_state": {"board_text": ..., "next_pieces_text": ...}, 
            #  "last_action": ..., "reflection": ...}
            game_state = entry.get("game_state", {})
            board_text = game_state.get("board_text", "N/A")
            next_pieces_text = game_state.get("next_pieces_text", "N/A")
            last_action_taken = entry.get("last_action", "N/A")
            reflection = entry.get("reflection", "No reflection.")
            
            entry_summary = f"Past State {i+1} (most recent is last):"
            entry_summary += f"  Board: {board_text}"
            entry_summary += f"  Next Pieces: {next_pieces_text}"
            entry_summary += f"  Action leading to this state: {last_action_taken}"
            entry_summary += f"  Reflection on this state: {reflection}"
            summary_parts.append(entry_summary)
            
        return "\n".join(summary_parts)

    async def process_observation(self, observation, info=None, memory_summary=None):
        """
        Process the observation directly to plan the next action sequence.
        
        Args:
            observation: The game observation (image)
            info: Additional info from the environment
            memory_summary: Summary of past game states
            
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
                # Try to get queue from info if available (info might not always have it for base module)
                piece_queue_text = "Piece queue: Not available" 
                if isinstance(info, dict) and 'next_piece_queue' in info and 'piece_queue_symbols' in info:
                    # Assuming info is now enriched by the main script if it's base mode
                    symbols = info.get('piece_queue_symbols', [])
                    piece_queue_text = f"Upcoming pieces (queue): {symbols}\n\n"
                    
                    # Add detailed piece representations if available
                    if 'piece_queue_texts' in info and info['piece_queue_texts']:
                        for i, piece_text_item in enumerate(info['piece_queue_texts']):
                            piece_queue_text += f"Next Piece {i+1}:\n{piece_text_item}\n\n"

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
5: hard_drop - Drop piece to the bottom instantly (RECOMMENDED for final placement)
6: swap - Swap with held piece (if available)
7: no_operation - Do nothing

Examples of common action sequences:
- [3, 1, 1, 5] → Rotate once clockwise, move right twice, then hard drop
- [3, 3, 0, 0, 5] → Rotate twice clockwise, move left twice, then hard drop
- [1, 1, 1, 5] → Move right three times, then hard drop
- [0, 0, 0, 3, 5] → Move left three times, rotate once, then hard drop

Tetris Piece Rotations (achieved with rotate_clockwise action):
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
3. Finally, use hard drop (action 5) for final placement
4. Consider the next pieces in queue when planning your current move
5. Review recent game history (if provided) to inform your strategy and avoid repeating mistakes.
"""
                memory_context_for_prompt = "No recent game history available." # Default
                if memory_summary:
                    memory_context_for_prompt = self._prepare_memory_context(memory_summary)

                user_prompt = f"""Analyze this Tetris screenshot and plan an optimal action sequence.

{piece_queue_text}

Recent Game History:
{memory_context_for_prompt}

{tetris_mechanics}

Your task is to:
1. Analyze the board state and identify the current falling piece from the image.
2. Plan a complete action sequence that will:
   - Rotate the active piece to the optimal orientation.
   - Move it to the optimal position.
   - Drop it into place.
3. Consider the upcoming pieces in the queue (provided above as text, if available) for better long-term placement.
4. Leverage insights from Recent Game History to make strategic decisions.

The sequence should be a list of action indices (0-7) that completely handles the current piece.
Example: [3, 3, 1, 1, 5] (rotate twice, move right twice, then hard drop)

Format your response exactly as follows:
thought: [Your detailed reasoning about the optimal placement, considering the upcoming pieces and game history if available, and why]
action_sequence: [list of action indices]

Make sure your action sequence is comprehensive and includes all steps needed."""
                
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