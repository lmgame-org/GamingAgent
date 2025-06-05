import os
import sys
import retro
import numpy as np
import cv2
import subprocess
import socket
import time

def render_to_video(bk2_file_path):
    """Render a .bk2 file to video using retro's official playback approach."""
    if not os.path.exists(bk2_file_path):
        print(f"Error: Recording file not found at {bk2_file_path}")
        return

    print(f"Rendering video from: {bk2_file_path}")
    try:
        # Get absolute paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        bk2_abs_path = os.path.abspath(bk2_file_path)
        
        # Add custom integration path
        print(f"Adding custom integration path: {script_dir}")
        retro.data.Integrations.add_custom_path(script_dir)
        
        # Load movie
        print("Loading movie...")
        movie = retro.Movie(bk2_abs_path)
        movie.step()  # Step once to get initial state
        
        # Create environment
        print("Creating environment...")
        env = retro.make(
            game=movie.get_game(),
            state=retro.State.NONE,
            use_restricted_actions=retro.Actions.ALL,
            players=movie.players,
            inttype=retro.data.Integrations.ALL
        )
        
        # Set initial state
        data = movie.get_state()
        env.initial_state = data
        env.reset()
        
        # Setup video recording
        output_path = os.path.splitext(bk2_abs_path)[0] + '.mp4'
        print(f"Creating video at: {output_path}")
        
        # Setup ffmpeg process
        video = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        video.bind(("127.0.0.1", 0))
        vr = video.getsockname()[1]
        
        input_vformat = [
            "-r", str(env.em.get_screen_rate()),
            "-s", "%dx%d" % env.observation_space.shape[1::-1],
            "-pix_fmt", "rgb24",
            "-f", "rawvideo",
            "-probesize", "32",
            "-thread_queue_size", "10000",
            "-i", f"tcp://127.0.0.1:{vr}?listen"
        ]
        
        output = [
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", "17",
            "-f", "mp4",
            "-pix_fmt", "yuv420p",
            output_path
        ]
        
        ffmpeg_proc = subprocess.Popen(
            ["ffmpeg", "-y", *input_vformat, *output],
            stdout=subprocess.PIPE
        )
        
        video.close()
        video = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # Connect to ffmpeg
        time.sleep(0.3)
        video.connect(("127.0.0.1", vr))
        
        # Process frames
        print("Processing frames...")
        frame_count = 0
        
        while True:
            if movie.step():
                keys = []
                for p in range(movie.players):
                    for i in range(env.num_buttons):
                        keys.append(movie.get_key(i, p))
            else:
                break
                
            display, reward, terminated, truncated, info = env.step(keys)
            frame_count += 1
            
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames...")
            
            try:
                video.sendall(bytes(display))
            except BrokenPipeError:
                break
                
            if terminated or truncated:
                break
        
        # Clean up
        video.close()
        ffmpeg_proc.terminate()
        env.close()
        
        print(f"\nVideo rendering completed. Output saved to: {output_path}")
        print(f"Total frames processed: {frame_count}")
        
    except Exception as e:
        print(f"Error during video rendering: {e}")
        import traceback
        traceback.print_exc()

def main():
    if len(sys.argv) != 2:
        print("Usage: python render_video.py <path_to_bk2_file>")
        print("Example: python render_video.py recordings/AceAttorney-GbAdvance-level1_1_5-000000.bk2")
        return

    bk2_file_path = sys.argv[1]
    render_to_video(bk2_file_path)

if __name__ == "__main__":
    main() 