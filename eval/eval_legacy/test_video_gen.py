#!/usr/bin/env python3

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from eval.replay_utils import generate_video_from_textual_logs
    print("✓ Successfully imported replay_utils")
except Exception as e:
    print(f"✗ Error importing replay_utils: {e}")
    sys.exit(1)

# Test the video generation with tetris example
agent_config_path = "eval/video_samples/20250601_170219_deepseek_R10528_tetris/agent_config.json"
episode_log_path = "eval/video_samples/20250601_170219_deepseek_R10528_tetris/episode_001_log.jsonl"
output_path = "test_tetris_video.mp4"

print("Testing video generation for tetris...")

try:
    success = generate_video_from_textual_logs(
        episode_log_path=episode_log_path,
        game_name="tetris",
        output_path=output_path,
        fps=1.0
    )
    
    if success:
        print("✓ Video generation completed successfully!")
        print(f"✓ Output video: {output_path}")
    else:
        print("✗ Video generation failed")
        
except Exception as e:
    print(f"✗ Error during video generation: {e}")
    import traceback
    traceback.print_exc() 