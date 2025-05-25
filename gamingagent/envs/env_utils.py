import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
from typing import Optional, Dict, List, Any

# Color mapping for different tile values (extended from user's example)
COLORS = {
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
    4096: (60, 58, 50),      # 4096 (using a default dark for very high values)
    8192: (60, 58, 50)       # 8192 (using a default dark for very high values)
}

DARK_TEXT_COLOR = (119, 110, 101)  # For small values (2, 4)
LIGHT_TEXT_COLOR = (249, 246, 242) # For large values (8+)

POTENTIAL_FONTS = [
    "arial.ttf", "Arial.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    # "/System/Library/Fonts/SFNSDisplay-Bold.otf",  # macOS example
    # "C:/Windows/Fonts/Arial.ttf",  # Windows example
    # "C:/Windows/Fonts/ArialBd.ttf" # Windows Bold example
]

def _get_font(font_name_list, desired_size, default_font_func):
    """Helper to load a font."""
    font = None
    for font_name in font_name_list:
        try:
            font = ImageFont.truetype(font_name, desired_size)
            return font
        except (OSError, IOError):
            continue
    return default_font_func(size=desired_size)


def create_board_image_2048(board_powers: np.ndarray, save_path: str, size: int = 400, perf_score: Optional[float] = None) -> None:
    """Create a visualization of the 2048 board, incorporating new styling and perf_score display."""
    if not isinstance(board_powers, np.ndarray):
        print(f"[create_board_image_2048] Warning: board_powers is not a numpy array. Got {type(board_powers)}. Attempting to convert.")
        try:
            board_powers = np.array(board_powers, dtype=int)
        except ValueError as e:
            print(f"[create_board_image_2048] Error: Could not convert board_powers to numpy array: {e}. Cannot create image.")
            return
            
    if board_powers.shape != (4, 4):
        print(f"[create_board_image_2048] Error: board_powers does not have shape (4,4). Actual shape: {board_powers.shape}. Cannot create image.")
        return

    cell_size = size // 4
    padding = cell_size // 10

    img = Image.new('RGB', (size, size), (250, 248, 239)) 
    draw = ImageDraw.Draw(img)

    base_font_size = cell_size // 3
    perf_score_font_size = max(15, size // 25)

    main_font = _get_font(POTENTIAL_FONTS, base_font_size, ImageFont.load_default)
    perf_score_display_font = _get_font(POTENTIAL_FONTS, perf_score_font_size, ImageFont.load_default)
    
    if main_font is None: # Fallback for main font
        main_font = ImageFont.load_default()
        print("[create_board_image_2048] Main font not found from potential_fonts. Using PIL default.")
    if perf_score_display_font is None: # Fallback for perf_score font
        perf_score_display_font = ImageFont.load_default(size=perf_score_font_size) # PIL default doesn't always respect size arg well for load_default()
        print("[create_board_image_2048] Perf score font not found from potential_fonts. Using PIL default.")


    draw.rectangle([0, 0, size, size], fill=(187, 173, 160))

    for r_idx in range(4):
        for c_idx in range(4):
            power = int(board_powers[r_idx, c_idx])
            value = 0 if power == 0 else 2**power
            
            x0 = c_idx * cell_size + padding
            y0 = r_idx * cell_size + padding
            x1 = (c_idx + 1) * cell_size - padding
            y1 = (r_idx + 1) * cell_size - padding
            
            cell_color = COLORS.get(value, (60, 58, 50)) 
            draw.rectangle([x0, y0, x1, y1], fill=cell_color, outline=(187,173,160), width=padding//4 if padding > 4 else 1) # Add thin outline
            
            if value == 0:
                continue
            
            text_content = str(value)
            current_text_color = LIGHT_TEXT_COLOR if value > 4 else DARK_TEXT_COLOR
            
            current_font_size = base_font_size
            if len(text_content) == 3:
                current_font_size = int(base_font_size * 0.8)
            elif len(text_content) >= 4:
                current_font_size = int(base_font_size * 0.65)
            
            final_font_for_tile = main_font
            if current_font_size != base_font_size:
                final_font_for_tile = _get_font(POTENTIAL_FONTS, current_font_size, ImageFont.load_default)
                if final_font_for_tile is None: # Fallback
                    final_font_for_tile = ImageFont.load_default(size=current_font_size)


            text_bbox = draw.textbbox((0,0), text_content, font=final_font_for_tile)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            cell_center_x = (x0 + x1) // 2
            cell_center_y = (y0 + y1) // 2
            
            text_x = cell_center_x - text_width // 2
            text_y = cell_center_y - text_height // 2 - (padding // 3) # Adjusted for better vertical centering with textbbox

            draw.text((text_x, text_y), text_content, fill=current_text_color, font=final_font_for_tile)

    if perf_score is not None:
        score_text_content = f"Perf: {perf_score:.2f}"
        score_display_text_color = (10, 10, 10) 
        
        # Position at top-left with small padding
        text_bbox_score = draw.textbbox((0,0), score_text_content, font=perf_score_display_font)
        score_text_width = text_bbox_score[2] - text_bbox_score[0]
        # score_text_height = text_bbox_score[3] - text_bbox_score[1]

        # Draw a small semi-transparent background for the score text for better visibility
        bg_padding = padding // 2
        bg_rect_x0 = padding // 2 - bg_padding
        bg_rect_y0 = padding // 2 - bg_padding
        bg_rect_x1 = padding // 2 + score_text_width + bg_padding
        bg_rect_y1 = padding // 2 + (text_bbox_score[3]-text_bbox_score[1]) + bg_padding
        
        # Create a temporary surface for alpha blending if possible, otherwise solid
        try:
            score_bg_img = Image.new('RGBA', (size,size))
            score_bg_draw = ImageDraw.Draw(score_bg_img)
            score_bg_draw.rectangle([bg_rect_x0,bg_rect_y0,bg_rect_x1,bg_rect_y1], fill=(200,200,200,180)) # semi-transparent white
            img.paste(score_bg_img, (0,0), score_bg_img)
        except: # Fallback for systems without good RGBA paste support or errors
             draw.rectangle([bg_rect_x0,bg_rect_y0,bg_rect_x1,bg_rect_y1], fill=(220,220,220)) # solid light grey

        draw.text((padding // 2, padding // 2), score_text_content, fill=score_display_text_color, font=perf_score_display_font)


    try:
        save_dir = os.path.dirname(save_path)
        if save_dir: 
            os.makedirs(save_dir, exist_ok=True)
        img.save(save_path)
    except Exception as e:
        print(f"[create_board_image_2048] Error saving 2048 board image to {save_path}: {e}") 


# Define a default text color here if not passed or found in mapping for info panel text
DEFAULT_INFO_TEXT_COLOR = (220, 220, 220) 
FALLBACK_PIECE_COLOR_UTIL = (255, 0, 255) # Magenta, if a piece ID is not in the map
FALLBACK_EMPTY_COLOR_UTIL = (0,0,0) # Black for empty, if 0 not in map

def create_board_image_tetris(
    board: np.ndarray, 
    save_path: str, 
    pixel_color_mapping: Dict[int, List[int]],
    all_tetromino_objects: Optional[List[Any]] = None, # New parameter
    score: int = 0, 
    lines: int = 0, 
    level: int = 0,
    next_pieces_ids: Optional[List[int]] = None,
    held_piece_id: Optional[int] = None,
    perf_score: Optional[float] = None,
    img_width: int = 300, 
    info_panel_width: int = 150 
) -> None:
    """Create a visualization of the Tetris board and game info, using provided color mapping."""
    if not isinstance(board, np.ndarray):
        print(f"[create_board_image_tetris] Warning: board is not a numpy array. Got {type(board)}. Cannot create image.")
        return

    board_height_cells, board_width_cells = board.shape
    if board_height_cells == 0 or board_width_cells == 0:
        print(f"[create_board_image_tetris] Error: Board has zero dimension: {board.shape}. Cannot create image.")
        return
    
    if not pixel_color_mapping:
        print(f"[create_board_image_tetris] Error: pixel_color_mapping is empty or None. Cannot determine piece colors.")
        return
    
    empty_color_val_id = 0 # Assuming 0 is the ID for empty space, used for comparison
    # empty_color_rgb = pixel_color_mapping.get(empty_color_val_id, FALLBACK_EMPTY_COLOR_UTIL) 

    cell_size = img_width // board_width_cells
    img_height = cell_size * board_height_cells
    total_width = img_width + info_panel_width

    img = Image.new('RGB', (total_width, img_height), (50, 50, 50))
    draw = ImageDraw.Draw(img)

    # Fonts (same as before)
    try:
        font_size_main = max(15, cell_size // 2)
        font_main = _get_font(POTENTIAL_FONTS, font_size_main, ImageFont.load_default)
        font_size_info = max(12, info_panel_width // 10)
        font_info = _get_font(POTENTIAL_FONTS, font_size_info, ImageFont.load_default)
    except Exception as e:
        print(f"[create_board_image_tetris] Font loading error: {e}. Using default font.")
        font_main = ImageFont.load_default()
        font_info = ImageFont.load_default()

    # Draw Tetris Board (same as before)
    for r_idx in range(board_height_cells):
        for c_idx in range(board_width_cells):
            cell_id_on_board = int(board[r_idx, c_idx])
            color_tuple = tuple(pixel_color_mapping.get(cell_id_on_board, FALLBACK_PIECE_COLOR_UTIL))
            x0 = c_idx * cell_size
            y0 = r_idx * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            draw.rectangle([x0, y0, x1, y1], fill=color_tuple, outline=(80, 80, 80), width=1)
    
    draw.rectangle([img_width, 0, total_width, img_height], fill=(70, 70, 70))
    info_text_color_tuple = tuple(DEFAULT_INFO_TEXT_COLOR)
    info_text_x = img_width + 10
    current_y = 10
    draw.text((info_text_x, current_y), f"Score: {score}", fill=info_text_color_tuple, font=font_info)
    current_y += font_info.getbbox("A")[3] + 5 
    draw.text((info_text_x, current_y), f"Lines: {lines}", fill=info_text_color_tuple, font=font_info)
    current_y += font_info.getbbox("A")[3] + 5
    draw.text((info_text_x, current_y), f"Level: {level}", fill=info_text_color_tuple, font=font_info)
    current_y += font_info.getbbox("A")[3] + 15

    # --- Draw Held Piece Shape ---
    draw.text((info_text_x, current_y), "Held:", fill=info_text_color_tuple, font=font_info)
    current_y += font_info.getbbox("A")[3] + 2
    mini_shape_cell_size_pil = 3 # Small cell size for PIL drawing
    shape_area_start_x_pil = info_text_x + 5
    
    if held_piece_id is not None and held_piece_id != empty_color_val_id and all_tetromino_objects:
        held_tetromino_obj = next((t for t in all_tetromino_objects if t.id == held_piece_id), None)
        if held_tetromino_obj:
            color_rgb = held_tetromino_obj.color_rgb
            matrix = held_tetromino_obj.matrix
            shape_matrix = (matrix > 0).astype(np.uint8)
            for r_idx, row in enumerate(shape_matrix):
                for c_idx, cell in enumerate(row):
                    if cell == 1:
                        x0_shape = shape_area_start_x_pil + c_idx * mini_shape_cell_size_pil
                        y0_shape = current_y + r_idx * mini_shape_cell_size_pil
                        draw.rectangle(
                            [x0_shape, y0_shape, x0_shape + mini_shape_cell_size_pil, y0_shape + mini_shape_cell_size_pil],
                            fill=tuple(color_rgb), outline=(90,90,90) # Slightly lighter outline for shapes
                        )
            current_y += (shape_matrix.shape[0] * mini_shape_cell_size_pil) + 10 # Adjust Y based on actual shape height
        else:
            draw.text((shape_area_start_x_pil, current_y), "?", fill=info_text_color_tuple, font=font_info)
            current_y += font_info.getbbox("A")[3] + 5
    else:
        draw.text((shape_area_start_x_pil, current_y), "-", fill=info_text_color_tuple, font=font_info)
        current_y += font_info.getbbox("A")[3] + 5
    current_y += 5 # Extra spacing before "Next:"

    # --- Draw Next Pieces Shapes ---
    draw.text((info_text_x, current_y), "Next:", fill=info_text_color_tuple, font=font_info)
    current_y += font_info.getbbox("A")[3] + 2
    if next_pieces_ids and all_tetromino_objects:
        for i, piece_id_val in enumerate(next_pieces_ids[:3]): 
            if piece_id_val != empty_color_val_id:
                next_tetromino_obj = next((t for t in all_tetromino_objects if t.id == piece_id_val), None)
                if next_tetromino_obj:
                    color_rgb = next_tetromino_obj.color_rgb
                    matrix = next_tetromino_obj.matrix
                    shape_matrix = (matrix > 0).astype(np.uint8)
                    for r_idx, row in enumerate(shape_matrix):
                        for c_idx, cell in enumerate(row):
                            if cell == 1:
                                x0_shape = shape_area_start_x_pil + c_idx * mini_shape_cell_size_pil
                                y0_shape = current_y + r_idx * mini_shape_cell_size_pil
                                draw.rectangle(
                                    [x0_shape, y0_shape, x0_shape + mini_shape_cell_size_pil, y0_shape + mini_shape_cell_size_pil],
                                    fill=tuple(color_rgb), outline=(90,90,90)
                                )
                    current_y += (shape_matrix.shape[0] * mini_shape_cell_size_pil) + 7 # Y advance for this piece + small gap
                else:
                    draw.text((shape_area_start_x_pil, current_y), "?", fill=info_text_color_tuple, font=font_info)
                    current_y += font_info.getbbox("A")[3] + 5 
            else:
                draw.text((shape_area_start_x_pil, current_y), "-", fill=info_text_color_tuple, font=font_info)
                current_y += font_info.getbbox("A")[3] + 5
    else:
        draw.text((shape_area_start_x_pil, current_y), "-", fill=info_text_color_tuple, font=font_info)
        current_y += font_info.getbbox("A")[3] + 5

    # Perf score (same as before)
    if perf_score is not None:
        score_text_content = f"Perf: {perf_score:.2f}"
        text_bbox_score = draw.textbbox((0,0), score_text_content, font=font_info)
        score_text_width = text_bbox_score[2] - text_bbox_score[0]
        score_text_height = text_bbox_score[3] - text_bbox_score[1]
        perf_score_y = img_height - score_text_height - 10
        perf_score_x = img_width + (info_panel_width - score_text_width) // 2 
        draw.text((perf_score_x, perf_score_y), score_text_content, fill=info_text_color_tuple, font=font_info)

    try:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        img.save(save_path)
    except Exception as e:
        print(f"[create_board_image_tetris] Error saving Tetris board image to {save_path}: {e}") 