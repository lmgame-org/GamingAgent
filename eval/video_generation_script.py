#!/usr/bin/env python3
"""
Video Generation Script for Gaming Agent Episodes

This script generates videos from episode logs using different methods:
- text: Reconstruct frames from textual_representation in episode logs
- image: Use saved images from episode logs (placeholder for future)
- replay: Use game-specific replay mechanisms (placeholder for future)

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
from pathlib import Path
from typing import Dict, Any

# Add the current directory to Python path for imports
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

# Import video generation functions
from eval.replay_utils import generate_video_from_textual_logs

def load_agent_config(agent_config_path: str) -> Dict[str, Any]:
    """Load agent configuration from JSON file"""
    try:
        with open(agent_config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise ValueError(f"Error loading agent config: {e}")

def load_episode_log(episode_log_path: str) -> Dict[str, Any]:
    """Load episode log from JSON/JSONL file"""
    try:
        # For JSONL files, just check if it exists and is readable
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
    
    if args.method not in ['text', 'image', 'replay']:
        raise ValueError(f"Invalid method: {args.method}. Must be one of: text, image, replay")
    
    if args.method in ['image', 'replay']:
        print(f"Warning: Method '{args.method}' is not yet implemented. Only 'text' method is currently supported.")
    
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
        choices=['text', 'image', 'replay'],
        help='Video generation method: text (from textual_representation), image (from saved images), replay (from game replay)'
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
        supported_games = ['tetris', '2048', 'candy_crush', 'sokoban']
        if config_info['game_name'].lower() in supported_games:
            print(f"✓ Detected {config_info['game_name']} game - proceeding with video generation")
        else:
            print(f"⚠ Warning: Game '{config_info['game_name']}' detected - video generation may not be optimal")
            print("Currently optimized for: tetris, 2048, candy_crush, sokoban")
        
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
        else:
            print(f"Method '{args.method}' is not yet implemented")
            print("Currently supported methods: text")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
