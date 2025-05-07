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

def get_rotated_boards(board, active_piece):
    """Get all possible rotated positions of the current piece on the board"""
    if not isinstance(active_piece, np.ndarray):
        return []
    
    # Find the current position of the piece
    piece_pos = np.where(active_piece != 0)
    if len(piece_pos[0]) == 0:
        return []
        
    min_row, max_row = piece_pos[0].min(), piece_pos[0].max()
    min_col, max_col = piece_pos[1].min(), piece_pos[1].max()
    
    # Get the piece type from the board where the active piece is
    piece_type = board[min_row:max_row+1, min_col:max_col+1][active_piece[min_row:max_row+1, min_col:max_col+1] != 0][0]
    
    # Create a mask with the actual piece type instead of 1s
    piece_area = np.zeros_like(active_piece[min_row:max_row+1, min_col:max_col+1])
    piece_area[active_piece[min_row:max_row+1, min_col:max_col+1] != 0] = piece_type
    
    # Determine number of unique rotations based on piece type
    num_rotations = {
        2: 2,  # I piece: horizontal and vertical
        3: 1,  # O piece: only one rotation
        4: 4,  # T piece: four unique rotations
        5: 2,  # S piece: two unique rotations
        6: 2,  # Z piece: two unique rotations
        7: 4,  # J piece: four unique rotations
        8: 4   # L piece: four unique rotations
    }
    
    rotations = num_rotations.get(piece_type, 1)
    rotated_boards = []
    current = piece_area.copy()
    
    # Generate all rotations
    for i in range(rotations):
        # Create a copy of the board
        board_copy = board.copy()
        
        # Rotate the piece
        if i > 0:
            current = np.rot90(current, k=-1)
        
        # Place the rotated piece on the board
        piece_height, piece_width = current.shape
        for r in range(piece_height):
            for c in range(piece_width):
                if current[r, c] != 0:
                    board_copy[min_row + r, min_col + c] = current[r, c]
        
        rotated_boards.append(board_copy)
    
    return rotated_boards

def piece_to_symbol(pieces):
    """Convert piece numbers to symbols and return formatted string"""
    symbols = {0: '.', 2: 'I', 3: 'O', 4: 'T', 5: 'S', 6: 'Z', 7: 'J', 8: 'L'}
    if isinstance(pieces, np.ndarray):
        return '\n'.join([''.join([symbols.get(int(cell), str(cell)) for cell in row]) for row in pieces])
    return symbols.get(int(pieces), str(pieces))

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
        print("---------------------------")
        print(f"Active Piece: {active_piece}")
        print(f"Held Piece: {held_piece}")
        print("Next Pieces:")
        print(piece_to_symbol(next_pieces))
        print("---------------------------")
        print("\nPossible Rotated Positions:")
        rotated_boards = get_rotated_boards(board, active_piece)
        for i, rotated_board in enumerate(rotated_boards):
            print(f"\nRotation {i} (clockwise):")
            print_board(rotated_board)
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
