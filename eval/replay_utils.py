import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
# import imageio # No longer used for video writing
import tempfile
import shutil
import statistics # For median calculation
import subprocess # For calling ffmpeg
from typing import Optional

# Default seconds per frame for videos (1 FPS)
DEFAULT_SECONDS_PER_FRAME = 1.0


def create_board_image_2048(board_powers: np.ndarray, save_path: str, size: int = 400, perf_score: Optional[float] = None) -> None:
    """Create a visualization of the 2048 board, incorporating new styling and perf_score display."""
    cell_size = size // 4
    padding = cell_size // 10

    img = Image.new('RGB', (size, size), (250, 248, 239)) # Overall image background (can be overridden by grid bg)
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
        8192: (60, 58, 50)       # 8192
    }
    
    dark_text_color = (119, 110, 101)
    light_text_color = (249, 246, 242)

    font = None
    perf_score_display_font = None
    base_font_size = cell_size // 3
    perf_score_font_size = max(15, size // 25)

    potential_fonts = [
        "arial.ttf", "Arial.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]

    for font_name in potential_fonts:
        try:
            if font is None:
                font = ImageFont.truetype(font_name, base_font_size)
            if perf_score_display_font is None:
                 perf_score_display_font = ImageFont.truetype(font_name, perf_score_font_size)
            if font and perf_score_display_font:
                break 
        except (OSError, IOError):
            continue
    
    if font is None:
        font = ImageFont.load_default()
        print("[ReplayUtils] Main font not found. Using PIL default.")
    if perf_score_display_font is None:
        perf_score_display_font = ImageFont.load_default(size=perf_score_font_size)
        print("[ReplayUtils] Perf score font not found. Using PIL default.")

    draw.rectangle([0, 0, size, size], fill=(187, 173, 160))

    for r_idx in range(4):
        for c_idx in range(4):
            power = int(board_powers[r_idx, c_idx])
            value = 0 if power == 0 else 2**power
            
            x0 = c_idx * cell_size + padding
            y0 = r_idx * cell_size + padding
            x1 = (c_idx + 1) * cell_size - padding
            y1 = (r_idx + 1) * cell_size - padding
            
            cell_color = colors.get(value, (60, 58, 50)) 
            draw.rectangle([x0, y0, x1, y1], fill=cell_color)
            
            if value == 0:
                continue
            
            text_content = str(value)
            current_text_color = light_text_color if value > 4 else dark_text_color
            
            current_font_size = base_font_size
            if len(text_content) == 3:
                current_font_size = int(base_font_size * 0.8)
            elif len(text_content) >= 4:
                current_font_size = int(base_font_size * 0.65)
            
            final_font_for_tile = font
            if current_font_size != base_font_size:
                temp_font_found = False
                for font_name in potential_fonts:
                    try:
                        final_font_for_tile = ImageFont.truetype(font_name, current_font_size)
                        temp_font_found = True
                        break
                    except (OSError, IOError):
                        continue
                if not temp_font_found:
                    final_font_for_tile = ImageFont.load_default(size=current_font_size)
            
            text_width, text_height = 0, 0
            try:
                if hasattr(final_font_for_tile, 'getbbox'):
                    bbox = final_font_for_tile.getbbox(text_content)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                elif hasattr(final_font_for_tile, 'getsize'):
                    text_width, text_height = final_font_for_tile.getsize(text_content)
                else:
                    text_width = len(text_content) * current_font_size // 2
                    text_height = current_font_size
            except Exception as e:
                 print(f"[ReplayUtils] Error getting text size: {e}. Using fallback.")
                 text_width = len(text_content) * current_font_size // 2
                 text_height = current_font_size
            
            cell_center_x = (x0 + x1) // 2
            cell_center_y = (y0 + y1) // 2
            text_x = cell_center_x - text_width // 2
            text_y = cell_center_y - text_height // 2 - (cell_size // 20)
            
            draw.text((text_x, text_y), text_content, fill=current_text_color, font=final_font_for_tile)
            if value >= 8:
                draw.text((text_x + 1, text_y), text_content, fill=current_text_color, font=final_font_for_tile)

    if perf_score is not None:
        score_text_content = f"Step Score: {perf_score:.2f}"
        score_display_text_color = (10, 10, 10)
        score_pos_x = padding 
        score_pos_y = padding // 2 
        try:
            draw.text((score_pos_x, score_pos_y), score_text_content, fill=score_display_text_color, font=perf_score_display_font)
        except Exception as e:
            print(f"[ReplayUtils] Error drawing perf_score on image: {e}")

    try:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        img.save(save_path)
    except Exception as e:
        print(f"[ReplayUtils] Error saving 2048 board image to {save_path}: {e}")


def generate_2048_median_score_replay(
    game_perf_json_path: str, 
    model_name_prefix: str, 
    game_display_name: str, # e.g. "2048" as in game_perf.json
    harness_status_key: str, # e.g. "harness_false"
    video_output_base_dir: str,
    seconds_per_frame: float = DEFAULT_SECONDS_PER_FRAME # Renamed and default updated
):
    """
    Generates an MP4 video replay for the median-scoring episode of a given model 
    in the 2048 game, using data from game_perf.json.
    Frames are generated using step_infos (board and step_score).
    Video is created using ffmpeg.
    """
    print(f"Attempting to generate median score MP4 replay for {model_name_prefix} in {game_display_name} ({harness_status_key}).")

    # Check for ffmpeg first
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        print("[ReplayUtils] ffmpeg found.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[ReplayUtils] Error: ffmpeg is not installed or not found in PATH. Video generation aborted.")
        print("[ReplayUtils] Please install ffmpeg: sudo apt-get install ffmpeg (on Debian/Ubuntu) or brew install ffmpeg (on macOS)")
        return

    try:
        with open(game_perf_json_path, 'r') as f:
            all_game_perf_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: game_perf.json not found at {game_perf_json_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {game_perf_json_path}")
        return

    try:
        model_data = all_game_perf_data.get(game_display_name, {}).get(model_name_prefix, {}).get(harness_status_key, {})
        if not model_data or "episodes_data" not in model_data:
            print(f"No data found for {model_name_prefix} - {game_display_name} - {harness_status_key} in {game_perf_json_path}")
            return

        episodes_data = model_data["episodes_data"]
        if not episodes_data:
            print(f"No episodes found for {model_name_prefix} - {game_display_name} - {harness_status_key}.")
            return

        # Sort episodes by total_episode_perf_score to find the median
        # If even number of episodes, statistics.median will average the two middle scores.
        # We need to pick one actual episode.
        scores = [ep["total_episode_perf_score"] for ep in episodes_data]
        if not scores:
            print("No scores available to find median.")
            return
            
        median_score_value = statistics.median(scores)
        
        # Find all episodes that match the median score
        median_episodes = [ep for ep in episodes_data if ep["total_episode_perf_score"] == median_score_value]
        
        if not median_episodes: # Should not happen if scores list was not empty
             # Fallback: if median is an average, pick the one closest (lower bias)
            episodes_data.sort(key=lambda ep: ep["total_episode_perf_score"])
            if len(episodes_data) % 2 == 0 and len(episodes_data) > 0 : # Even number
                median_episode_idx = len(episodes_data) // 2 -1 # pick the lower of the two middle ones
            elif len(episodes_data) > 0: # Odd number
                median_episode_idx = len(episodes_data) // 2
            else: # No episodes
                print(f"No episodes to select for median replay generation for {model_name_prefix}, {harness_status_key}.")
                return
            median_episode = episodes_data[median_episode_idx]
            print(f"Could not find exact median score match. Using episode with score: {median_episode['total_episode_perf_score']}")
        else:
            median_episode = median_episodes[0] # Pick the first one if multiple have the exact median score
            print(f"Selected median episode with score: {median_episode['total_episode_perf_score']}")


        episode_id_str = median_episode.get("episode_id", "unknown_ep")
        actual_median_score = median_episode["total_episode_perf_score"]
        
        step_infos = median_episode.get("step_infos")
        if not step_infos:
            print(f"No step_infos found for median episode {episode_id_str} of {model_name_prefix}.")
            return

        temp_dir = tempfile.mkdtemp()
        frame_files = []
        print(f"Generating frames in temporary directory: {temp_dir}")

        cumulative_step_score = 0.0 # Initialize cumulative score

        for idx, step_info in enumerate(step_infos):
            board_raw = step_info.get("board")
            current_step_score = step_info.get("step_score", 0.0) # Get current step's score
            cumulative_step_score += current_step_score # Add to cumulative

            if board_raw is None:
                print(f"Warning: 'board' key not found in step_info {idx} for episode {episode_id_str}. Skipping frame.")
                continue
            try:
                board_powers = np.array(board_raw, dtype=int)
                if board_powers.shape != (4, 4):
                    print(f"Warning: Board in step_info {idx} (ep {episode_id_str}) does not have shape (4,4). Actual: {board_powers.shape}. Skipping frame.")
                    continue
            except Exception as e:
                print(f"Warning: Could not convert board in step_info {idx} (ep {episode_id_str}) to numpy array: {e}. Skipping frame.")
                continue
            
            frame_path = os.path.join(temp_dir, f"frame_{idx:04d}.png")
            # Pass the cumulative_step_score to be displayed on the image
            create_board_image_2048(board_powers, frame_path, perf_score=cumulative_step_score)
            frame_files.append(frame_path)

        if not frame_files:
            print(f"No frames generated for median episode {episode_id_str} of {model_name_prefix}. Video not created.")
            shutil.rmtree(temp_dir)
            return

        harness_short = "hT" if "true" in harness_status_key else "hF"
        safe_model_name = "".join(c if c.isalnum() or c in ['-', '_'] else '_' for c in model_name_prefix)
        
        video_name = f"{game_display_name.replace('_', '')}_{safe_model_name}_{harness_short}_median_ep{episode_id_str}_score{actual_median_score:.0f}.mp4" # Changed to .mp4
        
        model_video_dir = os.path.join(video_output_base_dir, safe_model_name)
        os.makedirs(model_video_dir, exist_ok=True)
        output_video_path = os.path.join(model_video_dir, video_name)

        print(f"Compiling {len(frame_files)} frames into MP4 video: {output_video_path} at {1/seconds_per_frame:.2f} FPS.")
        
        # ffmpeg command
        # -r: framerate (input/output based on context)
        # -framerate: specific input framerate
        # -i: input pattern
        # -c:v: video codec
        # -pix_fmt: pixel format, yuv420p is widely compatible for mp4
        # -y: overwrite output without asking
        framerate = 1.0 / seconds_per_frame
        ffmpeg_cmd = [
            "ffmpeg",
            "-framerate", str(framerate),
            "-i", os.path.join(temp_dir, "frame_%04d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-y", # Overwrite output
            output_video_path
        ]

        try:
            print(f"[ReplayUtils] Executing ffmpeg command: {' '.join(ffmpeg_cmd)}")
            process = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)
            print("[ReplayUtils] ffmpeg stdout:")
            print(process.stdout)
            print(f"Median score replay MP4 video saved to {output_video_path}")
        except subprocess.CalledProcessError as e:
            print(f"[ReplayUtils] Error creating MP4 video with ffmpeg:")
            print(f"[ReplayUtils] Command: {' '.join(e.cmd)}")
            print(f"[ReplayUtils] Return code: {e.returncode}")
            print(f"[ReplayUtils] stdout:{e.stdout}")
            print(f"[ReplayUtils] stderr:{e.stderr}")
            print(f"Frames are available in {temp_dir}")
            return # Keep temp_dir for debugging if ffmpeg fails
        except FileNotFoundError: # Should have been caught by the initial check, but as a safeguard
            print("[ReplayUtils] Error: ffmpeg not found during video compilation. This should not happen after initial check.")
            print(f"Frames are available in {temp_dir}")
            return
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                print(f"Cleaned up temporary directory: {temp_dir}")

    except Exception as e:
        print(f"An unexpected error occurred in generate_2048_median_score_replay for {model_name_prefix}, {harness_status_key}: {e}")
        # Clean up temp dir if it exists and an error occurs mid-process
        if 'temp_dir' in locals() and os.path.exists(temp_dir) and os.path.isdir(temp_dir): # Ensure it's a directory
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory due to error: {temp_dir}")
