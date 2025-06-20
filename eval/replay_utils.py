import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tempfile
import shutil
import subprocess
from typing import Optional, Dict, List, Tuple, Union
import ast
import re
import cv2

# Default seconds per frame for videos
DEFAULT_SECONDS_PER_FRAME = 1.0

# --- Game-specific Constants and Parsers ---

# Tetris Constants
TETRIS_COLORS = {
    '.': (0, 0, 0),          # Empty space - black
    'S': (0, 255, 0),        # S-piece - green
    'Z': (255, 0, 0),        # Z-piece - red
    'I': (0, 255, 255),      # I-piece - cyan
    'O': (255, 255, 0),      # O-piece - yellow
    'T': (128, 0, 128),      # T-piece - purple
    'L': (255, 165, 0),      # L-piece - orange
    'J': (0, 0, 255),        # J-piece - blue
}

# Sokoban Constants
SOKOBAN_ASSET_DIR = "gamingagent/envs/custom_02_sokoban/assets/images"

def load_sokoban_asset_image(path, size):
    """Load and resize a Sokoban asset image"""
    if not os.path.exists(path):
        return None
    try:
        img = Image.open(path).convert("RGBA")
        return img.resize(size, Image.Resampling.LANCZOS)
    except Exception:
        return None

def parse_tetris_textual_board(text_rep: str) -> Optional[List[List[str]]]:
    """Parse Tetris textual representation from the Board: section"""
    if not text_rep:
        return None
    
    try:
        # Look for the "Board:" section
        lines = text_rep.split('\n')
        board_lines = []
        in_board_section = False
        
        for line in lines:
            if line.strip().startswith('Board:'):
                in_board_section = True
                continue
            elif in_board_section:
                # Stop when we hit the explanation line or empty line
                if line.strip().startswith('(') or line.strip().startswith('Next Pieces:') or line.strip() == '':
                    if line.strip().startswith('('):
                        break  # Found explanation line
                    continue
                # This is a board line
                if len(line) >= 10:  # Tetris board is typically 10 wide
                    # Take exactly 10 characters for the board
                    board_row = list(line[:10])
                    board_lines.append(board_row)
        
        if board_lines:
            return board_lines
        else:
            return None
            
    except Exception as e:
        print(f"Error parsing tetris board: {e}")
        return None

def visualize_tetris_frame(board: List[List[str]], extra_info: str = "", config_info: Dict = None) -> Image.Image:
    """Create a visual representation of a Tetris board with square layout"""
    if not board:
        return None
    
    rows, cols = len(board), len(board[0])
    
    # Calculate game board dimensions
    cell_size = 25
    border_size = 2
    board_width = cols * cell_size + (cols + 1) * border_size
    board_height = rows * cell_size + (rows + 1) * border_size
    
    # Create square frame with info panel on the right
    info_panel_width = 300
    total_width = board_width + info_panel_width
    frame_size = max(total_width, board_height, 600)  # Minimum 600px square
    
    img = Image.new('RGB', (frame_size, frame_size), (40, 40, 40))  # Dark gray background
    draw = ImageDraw.Draw(img)
    
    # Calculate board position (centered vertically on left side)
    board_x_offset = 20
    board_y_offset = (frame_size - board_height) // 2
    
    # Draw the tetris board
    for row in range(rows):
        for col in range(cols):
            cell_value = board[row][col]
            color = TETRIS_COLORS.get(cell_value, (128, 128, 128))
            
            x1 = board_x_offset + col * (cell_size + border_size) + border_size
            y1 = board_y_offset + row * (cell_size + border_size) + border_size
            x2 = x1 + cell_size
            y2 = y1 + cell_size
            
            draw.rectangle([x1, y1, x2, y2], fill=color, outline=(255, 255, 255), width=1)
    
    # Draw info panel on the right
    info_x_start = board_width + 40
    info_y_start = 30
    
    try:
        # Load larger fonts by scaling default font
        from PIL import ImageFont
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
        
        # Try to create larger fonts - if this fails, fallback to default
        try:
            # For systems with truetype fonts available
            font_large = ImageFont.truetype("arial.ttf", 24)  # 2x larger
            font_small = ImageFont.truetype("arial.ttf", 20)   # 2x larger
        except:
            try:
                # For Linux systems
                font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
                font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
            except:
                # Fallback to default but still usable
                font_large = ImageFont.load_default()
                font_small = ImageFont.load_default()
    except:
        font_large = None
        font_small = None
    
    # Draw title
    draw.text((info_x_start, info_y_start), "TETRIS", fill=(255, 255, 0), font=font_large)
    current_y = info_y_start + 40
    
    # Draw config information
    if config_info:
        config_lines = [
            f"Game: {config_info.get('game_name', 'Unknown')}",
            f"Model: {config_info.get('model_name', 'Unknown')}",
            f"Harness: {config_info.get('harness', 'Unknown')}"
        ]
        
        for line in config_lines:
            draw.text((info_x_start, current_y), line, fill=(200, 200, 200), font=font_small)
            current_y += 25
        
        current_y += 20
    
    # Draw step information
    if extra_info:
        draw.text((info_x_start, current_y), "GAME STATUS", fill=(255, 255, 0), font=font_large)
        current_y += 35
        
        lines = extra_info.split('\n')
        for line in lines:
            if line.strip():
                draw.text((info_x_start, current_y), line, fill=(255, 255, 255), font=font_small)
                current_y += 25
    
    # Draw border around game board
    draw.rectangle([board_x_offset, board_y_offset, 
                   board_x_offset + board_width, board_y_offset + board_height], 
                   outline=(255, 255, 255), width=2)
    
    return img

def parse_2048_textual_board(text_board_str: str) -> Optional[np.ndarray]:
    """Parse 2048 textual representation: [[0, 1, 0, 0], [0, 0, 2, 0], ...]"""
    if not text_board_str:
        return None
    try:
        board_list = ast.literal_eval(text_board_str)
        if isinstance(board_list, list) and all(isinstance(row, list) for row in board_list):
            return np.array(board_list)
    except:
        pass
    return None

def visualize_2048_frame(board: np.ndarray, extra_info: str = "", config_info: Dict = None) -> Image.Image:
    """Create a visual representation of a 2048 board with authentic 2048 styling"""
    if board is None:
        return None
    
    rows, cols = board.shape
    
    # 2048 board styling - authentic look
    board_size = 400
    cell_size = board_size // 4
    padding = cell_size // 10
    
    # Create square frame with info panel on the right
    info_panel_width = 300
    total_width = board_size + info_panel_width + 40  # Extra spacing
    frame_size = max(total_width, board_size + 100, 600)  # Minimum size
    
    # Create image with beige background (typical 2048 background)
    img = Image.new('RGB', (frame_size, frame_size), (250, 248, 239))
    draw = ImageDraw.Draw(img)
    
    # Calculate board position (centered with some offset)
    board_x_offset = 20
    board_y_offset = (frame_size - board_size) // 2
    
    # Authentic 2048 color mapping
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
        8192: (60, 58, 50)       # 8192
    }
    
    # Text colors
    dark_text = (119, 110, 101)  # For small values (2, 4)
    light_text = (249, 246, 242) # For large values (8+)
    
    # Draw the game board background
    draw.rectangle([board_x_offset, board_y_offset, 
                   board_x_offset + board_size, board_y_offset + board_size], 
                   fill=(187, 173, 160))  # 2048 board background
    
    # Try to load better fonts
    try:
        base_font_size = cell_size // 3
        
        # Try common font paths
        potential_fonts = [
            "arial.ttf",
            "Arial.ttf",
            "DejaVuSans-Bold.ttf",
            "LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/System/Library/Fonts/SFNSDisplay-Bold.otf",  # macOS
            "C:/Windows/Fonts/Arial.ttf",  # Windows
            "C:/Windows/Fonts/ArialBd.ttf",  # Windows Bold
        ]
        
        font = None
        for font_name in potential_fonts:
            try:
                font = ImageFont.truetype(font_name, base_font_size)
                break
            except (OSError, IOError):
                continue
                
        # Fall back to default if no font found
        if font is None:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()
        base_font_size = 20
    
    # Draw each cell with authentic 2048 styling
    for row in range(rows):
        for col in range(cols):
            # The textual representation already contains actual tile values (2, 4, 8, etc.)
            # NOT power values, so no conversion needed
            value = int(board[row, col])
            
            # Calculate cell position
            x0 = board_x_offset + col * cell_size + padding
            y0 = board_y_offset + row * cell_size + padding
            x1 = board_x_offset + (col + 1) * cell_size - padding
            y1 = board_y_offset + (row + 1) * cell_size - padding
            
            # Draw cell background
            cell_color = colors.get(value, (60, 58, 50))  # Default to dark color for large values
            draw.rectangle([x0, y0, x1, y1], fill=cell_color)
            
            # Skip text for empty cells
            if value == 0:
                continue
            
            # Choose text color based on value
            text_color = light_text if value > 4 else dark_text
            
            # Draw the value text
            text = str(value)
            
            # Adjust font size based on number length
            font_size = base_font_size
            if len(text) == 3:
                font_size = int(base_font_size * 0.8)
            elif len(text) >= 4:
                font_size = int(base_font_size * 0.65)
            
            # Get font with correct size
            adjusted_font = None
            for font_name in potential_fonts:
                try:
                    adjusted_font = ImageFont.truetype(font_name, font_size)
                    break
                except (OSError, IOError):
                    continue
            
            if adjusted_font:
                current_font = adjusted_font
            else:
                current_font = font
            
            # Get text size
            if hasattr(current_font, 'getbbox'):
                # For newer PIL versions
                bbox = current_font.getbbox(text)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            elif hasattr(current_font, 'getsize'):
                # For older PIL versions
                text_width, text_height = current_font.getsize(text)
            else:
                # Fallback estimation
                text_width = len(text) * font_size // 2
                text_height = font_size
            
            # Calculate center of cell
            cell_center_x = (x0 + x1) // 2
            cell_center_y = (y0 + y1) // 2
            
            # Calculate text position for perfect centering
            text_x = cell_center_x - text_width // 2
            text_y = cell_center_y - text_height // 2 - (cell_size // 15)
            
            # Draw the text
            draw.text((text_x, text_y), text, fill=text_color, font=current_font)
            
            # For larger numbers, draw the text slightly bolder
            if value >= 8:
                draw.text((text_x+1, text_y), text, fill=text_color, font=current_font)
    
    # Draw info panel on the right
    info_x_start = board_size + 60
    info_y_start = board_y_offset  # Align with board position instead of fixed 30
    
    try:
        # Load fonts for info panel
        try:
            font_large = ImageFont.truetype("arial.ttf", 28)
            font_medium = ImageFont.truetype("arial.ttf", 22)
            font_small = ImageFont.truetype("arial.ttf", 18)
        except:
            try:
                font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
                font_medium = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
                font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
            except:
                font_large = ImageFont.load_default()
                font_medium = ImageFont.load_default()
                font_small = ImageFont.load_default()
    except:
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Draw title with 2048 styling
    draw.text((info_x_start, info_y_start), "2048", fill=(119, 110, 101), font=font_large)
    current_y = info_y_start + 45
    
    # Draw config information
    if config_info:
        config_lines = [
            f"Game: {config_info.get('game_name', 'Unknown')}",
            f"Model: {config_info.get('model_name', 'Unknown')}",
            f"Harness: {config_info.get('harness', 'Unknown')}"
        ]
        
        for line in config_lines:
            draw.text((info_x_start, current_y), line, fill=(119, 110, 101), font=font_small)
            current_y += 22
        
        current_y += 25
    
    # Draw game status information
    if extra_info:
        draw.text((info_x_start, current_y), "GAME STATUS", fill=(119, 110, 101), font=font_medium)
        current_y += 30
        
        lines = extra_info.split('\n')
        for line in lines:
            if line.strip():
                # Clean up the line and make it more readable
                clean_line = line.strip()
                if clean_line.startswith('Score:') or clean_line.startswith('Move:') or clean_line.startswith('Best:'):
                    draw.text((info_x_start, current_y), clean_line, fill=(119, 110, 101), font=font_small)
                    current_y += 22
                elif len(clean_line) < 40:  # Don't show very long lines
                    draw.text((info_x_start, current_y), clean_line, fill=(119, 110, 101), font=font_small)
                    current_y += 22
    
    return img

def parse_sokoban_textual_board(text_rep: str) -> Optional[Dict[str, List[Tuple[int, int]]]]:
    """Parse Sokoban textual representation from ID/Item Type/Position format"""
    if not text_rep:
        return None
    
    try:
        # Parse the specific format: "ID | Item Type | Position"
        lines = text_rep.strip().split('\n')
        
        elements = {
            'walls': [],
            'empty': [],
            'boxes': [],
            'worker': [],
            'docks': [],
            'box_on_dock': [],
            'worker_on_dock': []
        }
        
        max_row, max_col = 0, 0
        
        # First pass: collect all positions and their types
        all_positions = {}  # position -> item_type
        
        for line in lines:
            if '|' not in line or line.startswith('ID') or line.startswith('---'):
                continue
                
            parts = [part.strip() for part in line.split('|')]
            if len(parts) != 3:
                continue
                
            try:
                item_id = int(parts[0])
                item_type = parts[1].lower()
                position_str = parts[2]
                
                # Parse position (row, col) from string like "(2, 3)"
                position_match = re.search(r'\((\d+),\s*(\d+)\)', position_str)
                if not position_match:
                    continue
                    
                row, col = int(position_match.group(1)), int(position_match.group(2))
                max_row = max(max_row, row)
                max_col = max(max_col, col)
                
                # Store position and type
                if (row, col) not in all_positions:
                    all_positions[(row, col)] = []
                all_positions[(row, col)].append(item_type)
                    
            except (ValueError, IndexError):
                continue
        
        # Second pass: categorize based on combined information
        for (row, col), item_types in all_positions.items():
            # Check for combinations first
            has_worker = any('worker' in item_type for item_type in item_types)
            has_dock = any('dock' in item_type for item_type in item_types)
            has_box = any('box' in item_type and 'dock' not in item_type for item_type in item_types)
            has_box_on_dock = any('box on dock' in item_type for item_type in item_types)
            has_unknown = any('unknown' in item_type for item_type in item_types)
            
            if has_box_on_dock:
                elements['box_on_dock'].append((row, col))
            elif has_unknown:
                # "Unknown" in sokoban typically means worker is on a dock
                elements['worker_on_dock'].append((row, col))
            elif has_worker and has_dock:
                # Worker is standing on a dock (explicit case)
                elements['worker_on_dock'].append((row, col))
            elif has_worker and not has_dock:
                elements['worker'].append((row, col))
            elif has_box and not has_dock:
                elements['boxes'].append((row, col))
            elif has_dock and not has_worker and not has_box:
                elements['docks'].append((row, col))
            elif any('wall' in item_type for item_type in item_types):
                elements['walls'].append((row, col))
            else:
                elements['empty'].append((row, col))
        
        if max_row == 0 and max_col == 0:
            return None
            
        elements['board_size'] = (max_row + 1, max_col + 1)
        return elements
        
    except Exception as e:
        print(f"Error parsing sokoban textual representation: {e}")
        return None

def visualize_sokoban_frame(elements: Dict[str, List[Tuple[int, int]]], extra_info: str = "", config_info: Dict = None) -> Image.Image:
    """Create a visual representation of a Sokoban board using assets"""
    if not elements or 'board_size' not in elements:
        return None
    
    rows, cols = elements['board_size']
    tile_size = 64  # Increased from 48 to 64 for even larger game board
    
    # Calculate board dimensions
    board_width = cols * tile_size
    board_height = rows * tile_size
    
    # Create square frame with info panel on the right
    info_panel_width = 350  # Increased for better proportion
    total_width = board_width + info_panel_width + 50  # Extra padding
    frame_size = max(total_width, board_height + 120, 900)  # Increased minimum size to 900px
    
    img = Image.new('RGB', (frame_size, frame_size), (200, 200, 200))  # Back to original light gray
    
    # Calculate board position (centered vertically on left side)
    board_x_offset = 40  # Increased padding
    board_y_offset = (frame_size - board_height) // 2
    
    # Load Sokoban assets
    asset_paths = {
        "wall": os.path.join(SOKOBAN_ASSET_DIR, "wall.png"),
        "floor": os.path.join(SOKOBAN_ASSET_DIR, "floor.png"),
        "box": os.path.join(SOKOBAN_ASSET_DIR, "box.png"),
        "box_on_target": os.path.join(SOKOBAN_ASSET_DIR, "box_docked.png"),
        "player": os.path.join(SOKOBAN_ASSET_DIR, "worker.png"),
        "player_on_target": os.path.join(SOKOBAN_ASSET_DIR, "worker_dock.png"),
        "target": os.path.join(SOKOBAN_ASSET_DIR, "dock.png"),
    }
    
    assets = {k: load_sokoban_asset_image(p, (tile_size, tile_size)) for k, p in asset_paths.items()}
    
    # Draw the sokoban board
    for row in range(rows):
        for col in range(cols):
            x0 = board_x_offset + col * tile_size
            y0 = board_y_offset + row * tile_size
            
            # Draw floor first as base
            if assets["floor"]:
                img.paste(assets["floor"], (x0, y0), assets["floor"] if assets["floor"].mode == 'RGBA' else None)
            
            # Determine what to draw at this position
            pos = (row, col)
            asset_to_draw = None
            
            if pos in elements['walls']:
                asset_to_draw = assets["wall"]
            elif pos in elements['box_on_dock']:
                # Draw target first, then box on top
                if assets["target"]:
                    img.paste(assets["target"], (x0, y0), assets["target"] if assets["target"].mode == 'RGBA' else None)
                asset_to_draw = assets["box_on_target"]
            elif pos in elements['boxes']:
                asset_to_draw = assets["box"]
            elif pos in elements['worker_on_dock']:
                # Draw target first, then worker on top
                if assets["target"]:
                    img.paste(assets["target"], (x0, y0), assets["target"] if assets["target"].mode == 'RGBA' else None)
                # Try to use worker_on_target asset, fallback to regular worker if it doesn't work well
                asset_to_draw = assets["player_on_target"]
                if not asset_to_draw:
                    # Fallback to regular worker if worker_dock.png is missing or problematic
                    asset_to_draw = assets["player"]
            elif pos in elements['worker']:
                asset_to_draw = assets["player"]
            elif pos in elements['docks']:
                asset_to_draw = assets["target"]
            # If it's just empty, the floor is already drawn
            
            if asset_to_draw:
                img.paste(asset_to_draw, (x0, y0), asset_to_draw if asset_to_draw.mode == 'RGBA' else None)
    
    # Draw info panel on the right
    info_x_start = board_width + 80  # Increased spacing from board
    info_y_start = board_y_offset  # Align with board position instead of fixed 50
    
    try:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
        try:
            font_large = ImageFont.truetype("arial.ttf", 32)  # Increased font size further
            font_small = ImageFont.truetype("arial.ttf", 24)  # Increased font size further
        except:
            try:
                font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 32)
                font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
            except:
                font_large = ImageFont.load_default()
                font_small = ImageFont.load_default()
    except:
        font_large = None
        font_small = None
    
    draw = ImageDraw.Draw(img)
    
    # Draw title (back to brown color for light background)
    draw.text((info_x_start, info_y_start), "SOKOBAN", fill=(139, 69, 19), font=font_large)
    current_y = info_y_start + 50
    
    # Draw config information
    if config_info:
        config_lines = [
            f"Game: {config_info.get('game_name', 'Unknown')}",
            f"Model: {config_info.get('model_name', 'Unknown')}",
            f"Harness: {config_info.get('harness', 'Unknown')}"
        ]
        
        for line in config_lines:
            draw.text((info_x_start, current_y), line, fill=(60, 60, 60), font=font_small)  # Dark gray text for light background
            current_y += 30
        
        current_y += 30
    
    # Draw step information
    if extra_info:
        draw.text((info_x_start, current_y), "GAME STATUS", fill=(139, 69, 19), font=font_large)  # Brown title
        current_y += 45
        
        lines = extra_info.split('\n')
        for line in lines:
            if line.strip():
                draw.text((info_x_start, current_y), line, fill=(60, 60, 60), font=font_small)  # Dark gray text
                current_y += 30
    
    # Draw border around game board (dark border for light background)
    draw.rectangle([board_x_offset - 3, board_y_offset - 3, 
                   board_x_offset + board_width + 3, board_y_offset + board_height + 3], 
                   outline=(80, 80, 80), width=4)  # Dark gray border, even thicker
    
    return img

def parse_candy_crush_textual_board(text_rep: str) -> Optional[List[List[str]]]:
    """Parse candy crush textual representation to extract board"""
    if not text_rep:
        return None
    
    try:
        lines = text_rep.strip().split('\n')
        board = []
        
        for line in lines:
            line = line.strip()
            if '|' in line and any(char.isdigit() for char in line.split('|')[0]):
                # This is a board row like "0| R R C P G P R C"
                parts = line.split('|', 1)
                if len(parts) == 2:
                    row_data = parts[1].strip().split()
                    if row_data:  # Only add non-empty rows
                        board.append(row_data)
        
        # Validate board dimensions (should be 8x8 for candy crush)
        if len(board) == 8 and all(len(row) == 8 for row in board):
            return board
        elif board:  # Return whatever board we found, even if not 8x8
            print(f"Warning: Candy crush board dimensions are {len(board)}x{len(board[0]) if board else 0}, expected 8x8")
            return board
        else:
            return None
            
    except Exception as e:
        print(f"Error parsing candy crush textual representation: {e}")
        return None

def visualize_candy_crush_frame(board: List[List[str]], extra_info: str = "", config_info: Dict = None) -> Image.Image:
    """Create a visual representation of a Candy Crush board with square layout"""
    if not board:
        return None
    
    rows, cols = len(board), max(len(row) for row in board)
    
    # Calculate game board dimensions
    cell_size = 35
    border_size = 2
    board_width = cols * cell_size + (cols + 1) * border_size
    board_height = rows * cell_size + (rows + 1) * border_size
    
    # Create square frame with info panel on the right
    info_panel_width = 300
    total_width = board_width + info_panel_width
    frame_size = max(total_width, board_height, 600)  # Minimum 600px square
    
    img = Image.new('RGB', (frame_size, frame_size), (0, 0, 0))  # Black background
    draw = ImageDraw.Draw(img)
    
    # Calculate board position (centered vertically on left side)
    board_x_offset = 20
    board_y_offset = (frame_size - board_height) // 2
    
    candy_colors = {
        'R': (255, 0, 0),       # Red
        'G': (0, 255, 0),       # Green
        'B': (0, 0, 255),       # Blue
        'Y': (255, 255, 0),     # Yellow
        'P': (128, 0, 128),     # Purple
        'O': (255, 165, 0),     # Orange
        '_': (64, 64, 64),      # Empty - dark gray
    }
    
    # Draw candy crush board
    for row in range(rows):
        for col in range(len(board[row])):
            cell_value = board[row][col]
            color = candy_colors.get(cell_value, (200, 200, 200))
            
            x1 = board_x_offset + col * (cell_size + border_size) + border_size
            y1 = board_y_offset + row * (cell_size + border_size) + border_size
            x2 = x1 + cell_size
            y2 = y1 + cell_size
            
            # Draw candy as circle for filled cells
            if cell_value != '_':
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                radius = cell_size // 3
                draw.ellipse([center_x-radius, center_y-radius, center_x+radius, center_y+radius], 
                           fill=color, outline=(255, 255, 255), width=2)
            else:
                draw.rectangle([x1, y1, x2, y2], fill=color, outline=(128, 128, 128), width=1)
    
    # Draw info panel on the right
    info_x_start = board_width + 40
    info_y_start = board_y_offset  # Align with board position instead of fixed 30
    
    try:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
        try:
            font_large = ImageFont.truetype("arial.ttf", 24)
            font_small = ImageFont.truetype("arial.ttf", 20)
        except:
            try:
                font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
                font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
            except:
                font_large = ImageFont.load_default()
                font_small = ImageFont.load_default()
    except:
        font_large = None
        font_small = None
    
    # Draw title
    draw.text((info_x_start, info_y_start), "CANDY CRUSH", fill=(255, 255, 0), font=font_large)
    current_y = info_y_start + 40
    
    # Draw config information
    if config_info:
        config_lines = [
            f"Game: {config_info.get('game_name', 'Unknown')}",
            f"Model: {config_info.get('model_name', 'Unknown')}",
            f"Harness: {config_info.get('harness', 'Unknown')}"
        ]
        
        for line in config_lines:
            draw.text((info_x_start, current_y), line, fill=(200, 200, 200), font=font_small)
            current_y += 25
        
        current_y += 20
    
    # Draw step information
    if extra_info:
        draw.text((info_x_start, current_y), "GAME STATUS", fill=(255, 255, 0), font=font_large)
        current_y += 35
        
        lines = extra_info.split('\n')
        for line in lines:
            if line.strip():
                draw.text((info_x_start, current_y), line, fill=(255, 255, 255), font=font_small)
                current_y += 25
    
    # Draw border around game board
    draw.rectangle([board_x_offset, board_y_offset, 
                   board_x_offset + board_width, board_y_offset + board_height], 
                   outline=(255, 255, 255), width=2)
    
    return img

# --- Main Video Generation Functions ---

def extract_image_paths_from_jsonl(episode_log_path: str) -> List[str]:
    """Extract image paths from episode log for Pokemon Red"""
    image_paths = []
    
    with open(episode_log_path, 'r') as f:
        for line in f:
            try:
                step_data = json.loads(line.strip())
                if 'agent_observation' in step_data:
                    obs = step_data['agent_observation']
                    if isinstance(obs, str):
                        obs = json.loads(obs)
                    
                    img_path = obs.get('img_path', '')
                    if img_path:
                        # Convert to original image path by replacing with _original suffix
                        original_img_path = img_path.replace('.png', '_original.png')
                        if os.path.exists(original_img_path):
                            image_paths.append(original_img_path)
                        elif os.path.exists(img_path):
                            # Fallback to regular image if original doesn't exist
                            image_paths.append(img_path)
            except Exception as e:
                print(f"Error parsing line in episode log: {e}")
                continue
    
    return image_paths

def generate_video_from_pokemon_red_images(
    episode_log_path: str,
    output_path: str,
    fps: float = 1.0,
    cleanup_frames: bool = True
) -> bool:
    """Generate video from Pokemon Red original screenshots with scaling"""
    
    # Extract image paths
    print(f"Extracting image paths from {episode_log_path}")
    image_paths = extract_image_paths_from_jsonl(episode_log_path)
    
    if not image_paths:
        print("No image paths found in episode log")
        return False
    
    print(f"Found {len(image_paths)} image files")
    
    # Scale up images using temporary files
    print("Scaling up images to 1500px maximum dimension...")
    scaled_image_paths = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            for i, original_path in enumerate(image_paths):
                if not os.path.exists(original_path):
                    print(f"Warning: Image not found: {original_path}")
                    continue
                
                # Read the original image
                image = cv2.imread(original_path)
                if image is None:
                    print(f"Warning: Could not read image: {original_path}")
                    continue
                
                # Get current dimensions
                height, width = image.shape[:2]
                
                # Calculate scale factor to fit within 1500px maximum
                maximum_scale = 1500
                scale_factor = min(maximum_scale / width, maximum_scale / height)
                
                # Only scale up if necessary
                if scale_factor > 1:
                    # Calculate new dimensions
                    new_width = int(width * scale_factor)
                    new_height = int(height * scale_factor)
                    
                    # Resize the image
                    scaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                    
                    # Save to temporary file
                    temp_path = os.path.join(temp_dir, f"scaled_frame_{i:04d}.png")
                    cv2.imwrite(temp_path, scaled_image)
                    scaled_image_paths.append(temp_path)
                    
                    if i == 0:  # Print scaling info for first image
                        print(f"Scaled images from {width}x{height} to {new_width}x{new_height}")
                else:
                    # If no scaling needed, copy to temp directory for consistency
                    temp_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
                    shutil.copy2(original_path, temp_path)
                    scaled_image_paths.append(temp_path)
            
            if not scaled_image_paths:
                print("No valid images found for video creation")
                return False
            
            print(f"Successfully processed {len(scaled_image_paths)} images")
            
            # Create video from scaled images
            print(f"Creating video at {output_path}")
            success = create_video_from_frames(scaled_image_paths, output_path, fps)
            
            return success
            
        except ImportError:
            print("Error: OpenCV (cv2) is required for image scaling. Install with: pip install opencv-python")
            return False
        except Exception as e:
            print(f"Error during image scaling: {e}")
            return False

def extract_textual_representations_from_jsonl(episode_log_path: str) -> List[Tuple[str, Dict]]:
    """Extract textual representations and metadata from episode log"""
    representations = []
    
    with open(episode_log_path, 'r') as f:
        for line in f:
            try:
                step_data = json.loads(line.strip())
                if 'agent_observation' in step_data:
                    obs = step_data['agent_observation']
                    if isinstance(obs, str):
                        obs = json.loads(obs)
                    
                    textual_rep = obs.get('textual_representation', '')
                    
                    # Extract metadata
                    metadata = {
                        'step': step_data.get('step', 0),
                        'action': step_data.get('agent_action', ''),
                        'reward': step_data.get('reward', 0),
                        'perf_score': step_data.get('perf_score', 0),
                        'info': step_data.get('info', {})
                    }
                    
                    representations.append((textual_rep, metadata))
            except Exception as e:
                print(f"Error parsing line in episode log: {e}")
                continue
    
    return representations

def generate_frames_from_textual_representations(
    representations: List[Tuple[str, Dict]], 
    game_name: str,
    output_dir: str,
    config_info: Dict = None
) -> List[str]:
    """Generate frame images from textual representations"""
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Normalize game names to standard format
    game_name_mapping = {
        "twenty_forty_eight": "2048",
        "twentyfortyeight": "2048", 
        "2048": "2048",
        "tetris": "tetris",
        "sokoban": "sokoban",
        "candy_crush": "candy_crush",
        "candycrush": "candy_crush",
        "pokemon_red": "pokemon_red"
    }
    
    normalized_game_name = game_name_mapping.get(game_name.lower(), game_name)
    
    frame_paths = []
    
    # Initialize cumulative reward counter
    total_reward = 0.0
    
    for i, (text_rep, metadata) in enumerate(representations):
        try:
            # Ensure reward exists and update cumulative total
            current_reward = metadata.get('reward', 0.0)
            if current_reward is None:
                current_reward = 0.0
            total_reward += current_reward
            
            # Parse board based on normalized game type
            if normalized_game_name == "tetris":
                board = parse_tetris_textual_board(text_rep)
                if board:
                    extra_info = f"Step: {metadata['step']}\nAction: {metadata['action']}\nCurrent Reward: {current_reward:.1f}\nTotal Reward: {total_reward:.1f}"
                    frame_img = visualize_tetris_frame(board, extra_info, config_info)
                else:
                    print(f"Could not parse tetris board for step {i}")
                    continue
                    
            elif normalized_game_name == "2048":
                # Parse the textual representation which is a dictionary string
                try:
                    # The textual representation is a string containing a dictionary
                    text_dict = ast.literal_eval(text_rep)
                    if isinstance(text_dict, dict) and 'board' in text_dict:
                        board_list = text_dict['board']
                        if isinstance(board_list, list) and all(isinstance(row, list) for row in board_list):
                            board = np.array(board_list)
                            extra_info = f"Step: {metadata['step']}\nAction: {metadata['action']}\nCurrent Reward: {current_reward:.1f}\nTotal Reward: {total_reward:.1f}"
                            # Add game info from the text dict if available
                            if 'highest_tile' in text_dict:
                                extra_info += f"\nHighest: {text_dict['highest_tile']}"
                            if metadata['info'].get('score'):
                                extra_info += f"\nGame Score: {metadata['info'].get('score', 0)}"
                            frame_img = visualize_2048_frame(board, extra_info, config_info)
                        else:
                            continue
                    else:
                        continue
                except Exception as e:
                    print(f"Error parsing 2048 textual representation: {e}")
                    continue
                    
            elif normalized_game_name == "sokoban":
                elements = parse_sokoban_textual_board(text_rep)
                if elements:
                    # For sokoban, keep both reward and perf_score if needed
                    current_perf_score = metadata.get('perf_score', 0.0)
                    if current_perf_score is None:
                        current_perf_score = 0.0
                    extra_info = f"Step: {metadata['step']}\nAction: {metadata['action']}\nCurrent Reward: {current_reward:.1f}\nTotal Reward: {total_reward:.1f}"
                    if current_perf_score != 0:
                        extra_info += f"\nPerf Score: {current_perf_score:.1f}"
                    frame_img = visualize_sokoban_frame(elements, extra_info, config_info)
                else:
                    print(f"Could not parse sokoban board for step {i}")
                    continue
                    
            elif normalized_game_name == "candy_crush":
                board = parse_candy_crush_textual_board(text_rep)
                if board:
                    extra_info = f"Step: {metadata['step']}\nAction: {metadata['action']}\nCurrent Reward: {current_reward:.1f}\nTotal Reward: {total_reward:.1f}"
                    frame_img = visualize_candy_crush_frame(board, extra_info, config_info)
                else:
                    continue
            elif normalized_game_name == "pokemon_red":
                # For Pokemon Red, we skip textual processing and use image-based approach
                # This will be handled by extract_image_paths_from_jsonl function
                continue
            else:
                print(f"Unsupported game: {game_name} (normalized: {normalized_game_name})")
                continue
            
            # Save frame
            if frame_img:
                frame_path = os.path.join(output_dir, f"frame_{i:04d}.png")
                frame_img.save(frame_path)
                frame_paths.append(frame_path)
                
        except Exception as e:
            print(f"Error generating frame {i}: {e}")
            continue
    
    return frame_paths

def create_video_from_frames(frame_paths: List[str], output_path: str, fps: float = 1.0) -> bool:
    """Create video from frame images using ffmpeg"""
    if not frame_paths:
        print("No frames to create video from")
        return False
    
    # Create a temporary directory with symlinks for ffmpeg
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create symlinks with consecutive numbering for ffmpeg
        for i, frame_path in enumerate(frame_paths):
            temp_frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
            try:
                if os.name == 'nt':  # Windows
                    shutil.copy2(frame_path, temp_frame_path)
                else:  # Unix-like
                    os.symlink(os.path.abspath(frame_path), temp_frame_path)
            except Exception as e:
                print(f"Error creating temp frame {i}: {e}")
                shutil.copy2(frame_path, temp_frame_path)
        
        # Use ffmpeg to create video
        input_pattern = os.path.join(temp_dir, "frame_%04d.png")
        cmd = [
            'ffmpeg', '-y',  # -y to overwrite output file
            '-r', str(fps),  # input framerate
            '-i', input_pattern,
            '-vcodec', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-r', str(fps),  # output framerate
            output_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"Video created successfully: {output_path}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error creating video with ffmpeg: {e}")
            print(f"Command: {' '.join(cmd)}")
            print(f"Stderr: {e.stderr}")
            return False
        except FileNotFoundError:
            print("ffmpeg not found. Please install ffmpeg to create videos.")
            return False

def generate_video_from_textual_logs(
    episode_log_path: str,
    game_name: str,
    output_path: str,
    fps: float = 1.0,
    cleanup_frames: bool = True,
    config_info: Dict = None
) -> bool:
    """Main function to generate video from episode logs using textual representations"""
    
    # Special handling for Pokemon Red - use original images instead of textual representations
    if game_name.lower() == "pokemon_red":
        print(f"Pokemon Red detected - using image-based video generation")
        return generate_video_from_pokemon_red_images(
            episode_log_path, output_path, fps, cleanup_frames
        )
    
    # Extract textual representations
    print(f"Extracting textual representations from {episode_log_path}")
    representations = extract_textual_representations_from_jsonl(episode_log_path)
    
    if not representations:
        print("No textual representations found in episode log")
        return False
    
    print(f"Found {len(representations)} textual representations")
    
    # Create temporary directory for frames
    frames_dir = output_path.replace('.mp4', '_frames')
    
    # Generate frames
    print(f"Generating frames for {game_name}")
    frame_paths = generate_frames_from_textual_representations(
        representations, game_name, frames_dir, config_info
    )
    
    if not frame_paths:
        print("No frames were generated")
        return False
    
    print(f"Generated {len(frame_paths)} frames")
    
    # Create video
    print(f"Creating video at {output_path}")
    success = create_video_from_frames(frame_paths, output_path, fps)
    
    # Cleanup frames if requested
    if cleanup_frames and os.path.exists(frames_dir):
        try:
            shutil.rmtree(frames_dir)
            print("Cleaned up temporary frame files")
        except Exception as e:
            print(f"Warning: Could not clean up frames directory: {e}")
    
    return success