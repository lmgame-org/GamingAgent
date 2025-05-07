import cv2
import gymnasium as gym
import numpy as np
import time

from tetris_gymnasium.envs.tetris import Tetris

def print_board(board):
    """Print a simplified text version of the board for debugging"""
    # Create a copy to visualize, skipping border and showing only the game area
    # Assuming borders are at indices [0-3, 14-17] horizontally and [20-23] vertically
    vis_board = board[0:20, 4:14].copy()
    
    # Replace numbers with symbols for better visualization
    symbols = {0: '.', 1: '#', 2: 'I', 3: 'O', 4: 'T', 5: 'S', 6: 'Z', 7: 'J', 8: 'L'}
    
    for row in vis_board:
        print(''.join([symbols.get(cell, str(cell)) for cell in row]))

def piece_to_symbol(pieces):
    """Convert piece numbers to symbols and return formatted string"""
    symbols = {0: '.', 2: 'I', 3: 'O', 4: 'T', 5: 'S', 6: 'Z', 7: 'J', 8: 'L'}
    if isinstance(pieces, np.ndarray):
        return '\n'.join([''.join([symbols.get(int(cell), str(cell)) for cell in row]) for row in pieces])
    return symbols.get(int(pieces), str(pieces))

def extract_active_tetromino(board, active_mask):
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

def get_piece_id(piece):
    """Identify the piece type from the piece matrix"""
    unique_values = np.unique(piece)
    unique_values = unique_values[unique_values > 0]  # Ignore empty cells
    if len(unique_values) == 0:
        return None
    return unique_values[0]  # Return the first non-zero value

def rotate_piece(piece, piece_id):
    """Rotate a tetromino piece clockwise using np.rot90, matching the gymnasium environment"""
    # Simply use np.rot90 with k=1 for clockwise rotation
    # This matches the Tetris gymnasium environment's rotate function
    return np.rot90(piece, k=1)

def generate_rotated_and_dropped_boards(board, active_mask, active_orientation=0):
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
    # Extract the active tetromino
    piece, row, col, active_piece_on_board = extract_active_tetromino(board, active_mask)
    if piece is None:
        return []
    
    # Get the piece ID (type)
    piece_id = get_piece_id(piece)
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
        current_piece = rotate_piece(current_piece, piece_id)
        
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

if __name__ == "__main__":
    env = gym.make("tetris_gymnasium/Tetris", render_mode="human")
    obs, info = env.reset(seed=42)
    
    terminated = False
    all_rewards = 0
    step_count = 0
    
    while not terminated:  # Add step limit to avoid infinite loops
        # Just call render without trying to print its return value
        env.render()
        
        # Get current game state information
        board = obs['board']
        active_piece = obs['active_tetromino_mask']
        held_piece = obs['holder']
        next_pieces = obs['queue']
        
        # Print a simplified board representation
        print("\n--- Current Board State ---")
        print_board(board)
        # print(f"Active Piece: {active_piece}")
        # print(f"Held Piece: {held_piece}")
        # print("Next Pieces:")
        # print(piece_to_symbol(next_pieces))
        print("---------------------------\n")
        
        # Generate and display potential boards with rotated and dropped pieces
        potential_boards = generate_rotated_and_dropped_boards(board, active_piece)
        for i, result in enumerate(potential_boards):
            print(f"\n--- Potential Move {i+1}: {result['description']} ---")
            print_board(result['board'])
            print("---------------------------\n")
        
        # Take a random action
        action = env.action_space.sample()
        time.sleep(1)
        print(f"Taking action: {action}")
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(3)
        
        # Print relevant information
        print(f"Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
        if 'lines_cleared' in info:
            print(f"Lines cleared: {info['lines_cleared']}")
        
        all_rewards += reward
        step_count += 1
        
        # Wait to see movement
        key = cv2.waitKey(100)
        
        # Allow early exit with 'q' key
        if key == ord('q'):
            break
    
    print(f"Total rewards: {all_rewards}")
    print(f"Steps taken: {step_count}")
    print("Game Over!")
