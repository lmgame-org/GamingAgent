#!/usr/bin/env python3
"""
Script to generate normalized data cache for faster visualization loading.

Usage:
    python generate_normalized_cache.py [input_file] [output_file]

Example:
    python generate_normalized_cache.py data/rank_data.json normalized_data.json
"""

import sys
import json
from data_visualization import generate_and_save_normalized_data, load_normalized_data

def main():
    # Default files
    input_file = "data/rank_data.json"  # Update this path as needed
    output_file = "normalized_data.json"
    
    # Handle command line arguments
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    try:
        # Load rank data
        print(f"Loading rank data from {input_file}...")
        with open(input_file, 'r') as f:
            rank_data = json.load(f)
        
        # Generate and save normalized data
        print("Generating normalized data...")
        saved_path = generate_and_save_normalized_data(rank_data, output_file)
        
        # Verify the saved data
        print("Verifying saved data...")
        cached_data = load_normalized_data(output_file)
        
        if cached_data:
            print(f"✅ Successfully generated normalized data cache!")
            print(f"📁 Saved to: {saved_path}")
            print(f"🎮 Games included: {list(cached_data['games'].keys())}")
            print(f"👥 Players included: {len(cached_data['players'])}")
            print(f"📅 Generated at: {cached_data['timestamp']}")
        else:
            print("❌ Failed to verify cached data")
            
    except FileNotFoundError:
        print(f"❌ Error: Could not find input file '{input_file}'")
        print("Please check the file path and try again.")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main() 