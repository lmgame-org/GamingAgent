#!/usr/bin/env python3
"""
Video Generation Script for Gaming Agent Episodes

This script generates videos from episode logs using different methods:
- text: Reconstruct frames from textual_representation in episode logs
- image: Use saved images from episode logs (placeholder for future)
- replay: Use game-specific replay mechanisms (placeholder for future)
- retro: Use retro's official playback approach for retro games

Usage:
    python video_generation_script.py --agent_config_path <path> --episode_log_path <path> --method text [--output_path <path>] [--fps <fps>]

Example:
    python video_generation_script.py \
        --agent_config_path configs/agent_configs/gpt4o_mini.json \
        --episode_log_path runs_output/gpt4o_mini/2048/episode_001_log.json \
        --method text \
        --output_path videos/2048_episode_001.mp4 \
        --fps 2
"""

import argparse
import json
import os
import sys
import retro
import socket
import subprocess
import time
import yaml
from pathlib import Path
from typing import Dict, Any

# Add the current directory to Python path for imports
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

# Import video generation functions
from eval.replay_utils import generate_video_from_textual_logs

def playback_movie(
    emulator,
    movie,
    video_file=None,
    video_delay=0,
    record_audio=True,
):
    ffmpeg_proc = None
    if video_file:
        video = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        video.bind(("127.0.0.1", 0))
        vr = video.getsockname()[1]
        input_vformat = [
            "-r",
            str(emulator.em.get_screen_rate()),
            "-s",
            "%dx%d" % emulator.observation_space.shape[1::-1],
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "-probesize",
            "32",
            "-thread_queue_size",
            "10000",
            "-i",
            "tcp://127.0.0.1:%i?listen" % vr,
        ]
        if record_audio:
            audio = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            audio.bind(("127.0.0.1", 0))
            ar = audio.getsockname()[1]
            input_aformat = [
                "-ar",
                "%i" % emulator.em.get_audio_rate(),
                "-ac",
                "2",
                "-f",
                "s16le",
                "-probesize",
                "32",
                "-thread_queue_size",
                "60",
                "-i",
                "tcp://127.0.0.1:%i?listen" % ar,
            ]
        else:
            audio = None
            ar = None
            input_aformat = ["-an"]
        output = [
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-strict",
            "-2",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "17",
            "-f",
            "mp4",
            "-pix_fmt",
            "yuv420p",
            video_file,
        ]
        
        ffmpeg_proc = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                *input_vformat,  # Input params (video)
                *input_aformat,  # Input params (audio)
                *output,
            ],  # Output params
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        video.close()
        video = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if audio:
            audio.close()
            audio = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        audio_connected = False

        time.sleep(0.3)
        try:
            video.connect(("127.0.0.1", vr))
        except ConnectionRefusedError:
            video.close()
            if audio:
                audio.close()
            ffmpeg_proc.terminate()
            raise

    frames = 0
    wasDone = False

    def killprocs(*args, **kwargs):
        ffmpeg_proc.terminate()
        raise BrokenPipeError

    def waitprocs():
        if ffmpeg_proc:
            video.close()
            if audio:
                audio.close()
            ffmpeg_proc.wait()

    while True:
        if movie.step():
            keys = []
            for p in range(movie.players):
                for i in range(emulator.num_buttons):
                    keys.append(movie.get_key(i, p))
        elif video_delay < 0 and frames < -video_delay:
            keys = [0] * emulator.num_buttons
        else:
            break
        display, reward, terminated, truncated, info = emulator.step(keys)
        frames += 1
        
        if frames % 100 == 0:
            print(f"Processed {frames} frames...")

        try:
            if ffmpeg_proc and frames > video_delay:
                video.sendall(bytes(display))
                if audio:
                    sound = emulator.em.get_audio()
                    if not audio_connected:
                        time.sleep(0.2)
                        audio.connect(("127.0.0.1", ar))
                        audio_connected = True
                    if len(sound):
                        audio.sendall(bytes(sound))
        except BrokenPipeError:
            waitprocs()
            raise
        
        if (terminated or truncated) and not wasDone:
            frames = 0
        wasDone = terminated or truncated
    
    waitprocs()
    print(f"\nVideo rendering completed. Total frames processed: {frames}")


def render_retro_video(bk2_file_path: str, game_name: str, output_path: str) -> bool:
    """Render a .bk2 file to video using retro's official playback approach."""
    if not os.path.exists(bk2_file_path):
        print(f"Error: Recording file not found at {bk2_file_path}")
        return False

    print(f"Rendering video from: {bk2_file_path}")
    try:
        # Get absolute paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        GAMING_AGENT_DIR = os.path.dirname(script_dir)

        # Handle custom integrations. Ace Attorney has one.
        print(f"Setting up retro integration paths...")
        if game_name == 'ace_attorney':
            ace_attorney_dir = os.path.join(GAMING_AGENT_DIR, "gamingagent", "envs", "retro_02_ace_attorney")
            if os.path.exists(ace_attorney_dir):
                retro.data.Integrations.add_custom_path(ace_attorney_dir)
                print(f"Added custom integration path for Ace Attorney: {ace_attorney_dir}")

        # Add default retro integrations to find games like Super Mario Bros.
        retro.data.add_integrations(retro.data.Integrations.ALL)
        print("Added default retro integration paths.")

        # Load movie
        print("Loading movie...")
        movie = retro.Movie(bk2_file_path)
        movie.step()
        
        # Create environment
        print("Creating environment...")
        emulator = retro.make(
            game=movie.get_game(),
            state=retro.State.NONE,
            use_restricted_actions=retro.Actions.ALL,
            players=movie.players,
        )
        data = movie.get_state()
        emulator.initial_state = data
        emulator.reset()

        print(f"Creating video at: {output_path}")
        playback_movie(emulator, movie, video_file=output_path)
        
        emulator.close()
        return True
        
    except Exception as e:
        print(f"Error during video rendering: {e}")
        import traceback
        traceback.print_exc()
        return False

def load_agent_config(agent_config_path: str) -> Dict[str, Any]:
    """Load agent configuration from JSON or YAML file"""
    try:
        with open(agent_config_path, 'r') as f:
            if agent_config_path.endswith('.yaml') or agent_config_path.endswith('.yml'):
                return yaml.safe_load(f)
            else:
                return json.load(f)
    except Exception as e:
        raise ValueError(f"Error loading agent config: {e}")

def load_episode_log(episode_log_path: str) -> Dict[str, Any]:
    """Load episode log from JSON/JSONL file or validate .bk2 file"""
    try:
        if episode_log_path.endswith('.bk2'):
            # For .bk2 files, just check if it exists and is readable
            if not os.path.exists(episode_log_path):
                raise FileNotFoundError(f"Recording file not found: {episode_log_path}")
            # Try to open the file to verify it's readable
            with open(episode_log_path, 'rb') as f:
                # Just read a small chunk to verify file is readable
                f.read(1024)
            return {"valid": True}
        else:
            # For JSON/JSONL files, validate format
            with open(episode_log_path, 'r') as f:
                # Try to read first line to validate format
                first_line = f.readline().strip()
                if first_line:
                    json.loads(first_line)  # Validate JSON format
            return {"valid": True}
    except Exception as e:
        raise ValueError(f"Error loading episode log: {e}")

def extract_info_from_paths(agent_config_path: str, episode_log_path: str) -> Dict[str, str]:
    """Extract game_name, model_name, harness from paths and config"""
    agent_config = load_agent_config(agent_config_path)
    
    # Extract information from agent config
    # Handle both YAML and JSON formats
    if 'game_env' in agent_config:
        # YAML format
        game_name = agent_config['game_env'].get('name', 'unknown')
        model_name_full = agent_config['agent'].get('model_name', 'unknown')
        harness = agent_config['agent'].get('harness', False)
    else:
        # JSON format
        game_name = agent_config.get('game_name', 'unknown')
        model_name_full = agent_config.get('model_name', 'unknown')
        harness = agent_config.get('harness', False)
    
    # Clean up model name - take part after the slash if it exists
    if '/' in model_name_full:
        model_name = model_name_full.split('/')[-1]
    else:
        model_name = model_name_full
    
    return {
        'game_name': game_name,
        'model_name': model_name,
        'model_name_full': model_name_full,
        'harness': str(harness)
    }

def generate_default_output_path(episode_log_path: str, agent_config_path: str, method: str) -> str:
    """Generate a default output path based on input files"""
    episode_path = Path(episode_log_path)
    config_info = extract_info_from_paths(agent_config_path, episode_log_path)
    
    # Create filename: game_model_episode_method.mp4
    episode_name = episode_path.stem  # e.g., episode_001_log
    episode_num = episode_name.replace('_log', '').replace('episode_', '')
    
    filename = f"{config_info['game_name']}_{config_info['model_name']}_{episode_num}_{method}.mp4"
    return str(episode_path.parent / filename)

def validate_inputs(args: argparse.Namespace) -> None:
    """Validate input arguments"""
    if not os.path.exists(args.agent_config_path):
        raise FileNotFoundError(f"Agent config file not found: {args.agent_config_path}")
    
    if not os.path.exists(args.episode_log_path):
        raise FileNotFoundError(f"Episode log file not found: {args.episode_log_path}")
    
    if args.method not in ['text', 'image', 'replay', 'retro']:
        raise ValueError(f"Invalid method: {args.method}. Must be one of: text, image, replay, retro")
    
    if args.method in ['image', 'replay']:
        print(f"Warning: Method '{args.method}' is not yet implemented. Only 'text' and 'retro' methods are currently supported.")
    
    if args.fps <= 0:
        raise ValueError(f"FPS must be positive, got: {args.fps}")

def print_episode_info(episode_data: Dict[str, Any]) -> None:
    """Print information about the episode"""
    print("\n" + "="*50)
    print("EPISODE INFORMATION")
    print("="*50)
    for key, value in episode_data.items():
        print(f"{key}: {value}")
    print("="*50 + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="Generate videos from gaming agent episode logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with text method
  python video_generation_script.py \\
      --agent_config_path configs/agent.json \\
      --episode_log_path logs/episode_001.jsonl \\
      --method text

  # With custom output path and FPS
  python video_generation_script.py \\
      --agent_config_path configs/agent.json \\
      --episode_log_path logs/episode_001.jsonl \\
      --method text \\
      --output_path my_video.mp4 \\
      --fps 2

  # Using retro method for Ace Attorney
  python video_generation_script.py \\
      --agent_config_path configs/agent.json \\
      --episode_log_path logs/episode_001.bk2 \\
      --method retro
        """
    )
    
    parser.add_argument(
        '--agent_config_path',
        type=str,
        required=True,
        help='Path to agent configuration JSON file'
    )
    
    parser.add_argument(
        '--episode_log_path', 
        type=str,
        required=True,
        help='Path to episode log JSON/JSONL file'
    )
    
    parser.add_argument(
        '--method',
        type=str,
        required=True,
        choices=['text', 'image', 'replay', 'retro'],
        help='Video generation method: text (from textual_representation), image (from saved images), replay (from game replay), retro (from retro .bk2 file)'
    )
    
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='Output video path (default: auto-generated based on input files)'
    )
    
    parser.add_argument(
        '--fps',
        type=float,
        default=1.0,
        help='Frames per second for output video (default: 1.0)'
    )
    
    args = parser.parse_args()
    
    try:
        # Validate inputs
        validate_inputs(args)
        
        # Extract information from config and paths
        config_info = extract_info_from_paths(args.agent_config_path, args.episode_log_path)
        
        # Load episode log to validate
        episode_data = load_episode_log(args.episode_log_path)
        
        # Generate output path if not provided
        if args.output_path is None:
            args.output_path = generate_default_output_path(args.episode_log_path, args.agent_config_path, args.method)
        
        # Print episode information
        print_episode_info(config_info)
        
        print(f"Input Config: {args.agent_config_path}")
        print(f"Input Episode Log: {args.episode_log_path}")
        print(f"Output Video: {args.output_path}")
        print(f"Method: {args.method}")
        print(f"FPS: {args.fps}")
        print()
        
        # Check game type and provide appropriate messaging
        supported_games = ['tetris', '2048', 'candy_crush', 'sokoban', 'ace_attorney', 'super_mario_bros', 'pokemon_red']
        if config_info['game_name'].lower() in supported_games:
            print(f"✓ Detected {config_info['game_name']} game - proceeding with video generation")
        else:
            print(f"⚠ Warning: Game '{config_info['game_name']}' detected - video generation may not be optimal")
            print(f"Currently optimized for: {', '.join(supported_games)}")
        
        if args.method == 'text':
            print("Starting video generation from textual representations...")
            
            success = generate_video_from_textual_logs(
                episode_log_path=args.episode_log_path,
                game_name=config_info['game_name'],
                output_path=args.output_path,
                fps=args.fps,
                config_info=config_info  # Pass config info for display
            )
            
            if success:
                print(f"\n✓ Video generated successfully: {args.output_path}")
                
                # Print final summary
                print("\n" + "="*50)
                print("VIDEO GENERATION COMPLETE")
                print("="*50)
                print(f"Game: {config_info['game_name']}")
                print(f"Model: {config_info['model_name']}")
                print(f"Harness: {config_info['harness']}")
                print(f"Output: {args.output_path}")
                print("="*50)
            else:
                print("\n✗ Video generation failed")
                sys.exit(1)
        elif args.method == 'retro':
            if config_info['game_name'].lower() not in ['ace_attorney', 'super_mario_bros']:
                print(f"✗ Retro method is currently only supported for Ace Attorney and Super Mario Bros games")
                sys.exit(1)
                
            print("Starting video generation from retro recording...")
            success = render_retro_video(args.episode_log_path, config_info['game_name'].lower(), args.output_path)
            
            if success:
                print(f"\n✓ Video generated successfully: {args.output_path}")
                
                # Print final summary
                print("\n" + "="*50)
                print("VIDEO GENERATION COMPLETE")
                print("="*50)
                print(f"Game: {config_info['game_name']}")
                print(f"Model: {config_info['model_name']}")
                print(f"Harness: {config_info['harness']}")
                print(f"Output: {args.output_path}")
                print("="*50)
            else:
                print("\n✗ Video generation failed")
                sys.exit(1)
        else:
            print(f"Method '{args.method}' is not yet implemented")
            print("Currently supported methods: text, retro")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()