#!/usr/bin/env python3
"""
Example usage of the APIManager class for handling API calls.

This script demonstrates how to use the APIManager class to make API calls
with different input modalities (text-only, vision-only, vision-text, multi-image),
calculate costs, and log results.
"""

import os
import sys
import datetime
from api_manager import APIManager

def example_text_only_completion():
    """Example of using text-only completion."""
    # Initialize the API manager with info parameter
    api_manager = APIManager(
        game_name="ace_attorney",
        base_cache_dir="cache",
        info={
            "model_name": "gpt-4o",
            "modality": "text_only",
            "datetime": datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        }
    )
    
    # Set up parameters
    model_name = "gpt-4o"  # This can be overridden by the info parameter
    system_prompt = "You are a helpful assistant for the Phoenix Wright: Ace Attorney game."
    user_prompt = "What are the key evidence items in the Turnabout Sisters case?"
    
    # Make API call
    try:
        completion, costs = api_manager.text_only_completion(
            model_name=model_name,
            system_prompt=system_prompt,
            prompt=user_prompt,
            session_name="Turnabout_Sisters_text_example"
        )
        
        # Print results
        print("\n=== Text-Only Completion Example ===")
        print(f"Model: {model_name}")
        print(f"Input Tokens: {costs['prompt_tokens']}")
        print(f"Output Tokens: {costs['completion_tokens']}")
        print(f"Total Cost: ${float(costs['prompt_cost']) + float(costs['completion_cost']):.6f}")
        print("\nResponse:")
        print(completion)
        
    except Exception as e:
        print(f"Error in text-only completion example: {e}")

def example_vision_only_completion(image_path):
    """Example of using vision-only completion (image analysis without text prompt)."""
    # Initialize the API manager with info parameter
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    api_manager = APIManager(
        game_name="ace_attorney",
        base_cache_dir="cache",
        info={
            "model_name": "gpt-4o",
            "modality": "vision_only",
            "datetime": current_time
        }
    )
    
    # Set up parameters
    model_name = "gpt-4o"  # This can be overridden by the info parameter
    system_prompt = "You are a helpful assistant for the Phoenix Wright: Ace Attorney game. Analyze the visual elements of the scene."
    
    # Make API call
    try:
        completion, costs = api_manager.vision_only_completion(
            model_name=model_name,
            system_prompt=system_prompt,
            image_path=image_path,
            session_name="Turnabout_Sisters_vision_only_example"
        )
        
        # Print results
        print("\n=== Vision-Only Completion Example ===")
        print(f"Model: {model_name}")
        print(f"Total Input Tokens: {costs['prompt_tokens']}")
        print(f"Text Input Tokens: {costs['prompt_tokens'] - costs.get('image_tokens', 0)}")
        print(f"Image Input Tokens: {costs.get('image_tokens', 0)}")
        print(f"Output Tokens: {costs['completion_tokens']}")
        print(f"Total Cost: ${float(costs['prompt_cost']) + float(costs['completion_cost']):.6f}")
        print("\nResponse:")
        print(completion)
        
    except Exception as e:
        print(f"Error in vision-only completion example: {e}")

def example_vision_text_completion(image_path):
    """Example of using vision-text completion (image + specific text prompt)."""
    # Initialize the API manager with info parameter
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    api_manager = APIManager(
        game_name="ace_attorney",
        base_cache_dir="cache",
        info={
            "model_name": "gpt-4o",
            "modality": "vision_text",
            "datetime": current_time
        }
    )
    
    # Set up parameters
    model_name = "gpt-4o"  # This can be overridden by the info parameter
    system_prompt = "You are a helpful assistant for the Phoenix Wright: Ace Attorney game."
    user_prompt = "What's happening in this scene? Who are the characters and what emotions are they showing?"
    
    # Make API call
    try:
        completion, costs = api_manager.vision_text_completion(
            model_name=model_name,
            system_prompt=system_prompt,
            prompt=user_prompt,
            image_path=image_path,
            session_name="Turnabout_Sisters_vision_text_example"
        )
        
        # Print results
        print("\n=== Vision-Text Completion Example ===")
        print(f"Model: {model_name}")
        print(f"Total Input Tokens: {costs['prompt_tokens']}")
        print(f"Text Input Tokens: {costs['prompt_tokens'] - costs.get('image_tokens', 0)}")
        print(f"Image Input Tokens: {costs.get('image_tokens', 0)}")
        print(f"Output Tokens: {costs['completion_tokens']}")
        print(f"Total Cost: ${float(costs['prompt_cost']) + float(costs['completion_cost']):.6f}")
        print("\nResponse:")
        print(completion)
        
    except Exception as e:
        print(f"Error in vision-text completion example: {e}")

def example_multi_image_completion(image_paths):
    """Example of using multi-image completion."""
    # Initialize the API manager
    api_manager = APIManager(
        game_name="ace_attorney",
        base_cache_dir="cache"
    )
    
    # Set up parameters
    model_name = "claude-3-opus-20240229"  # Change to your preferred model
    system_prompt = "You are a helpful assistant for the Phoenix Wright: Ace Attorney game."
    user_prompt = "Compare these court scenes and explain the differences in the characters' reactions."
    
    # Create content list (one per image)
    list_content = [
        f"Scene {i+1}" for i in range(len(image_paths))
    ]
    
    # Make API call
    try:
        completion, costs = api_manager.multi_image_completion(
            model_name=model_name,
            system_prompt=system_prompt,
            prompt=user_prompt,
            list_content=list_content,
            list_image_paths=image_paths,
            session_name="Turnabout_Sisters_multi_image_example"
        )
        
        # Print results
        print("\n=== Multi-Image Completion Example ===")
        print(f"Model: {model_name}")
        print(f"Total Input Tokens: {costs['prompt_tokens']}")
        print(f"Text Input Tokens: {costs['prompt_tokens'] - costs.get('image_tokens', 0)}")
        print(f"Image Input Tokens: {costs.get('image_tokens', 0)}")
        print(f"Output Tokens: {costs['completion_tokens']}")
        print(f"Total Cost: ${float(costs['prompt_cost']) + float(costs['completion_cost']):.6f}")
        print("\nResponse:")
        print(completion)
        
    except Exception as e:
        print(f"Error in multi-image completion example: {e}")

def example_legacy_methods(image_path):
    """Example of using legacy methods for backward compatibility."""
    # Initialize the API manager
    api_manager = APIManager(
        game_name="ace_attorney",
        base_cache_dir="cache"
    )
    
    # Set up parameters
    model_name = "gpt-4o"  # Change to your preferred model
    system_prompt = "You are a helpful assistant for the Phoenix Wright: Ace Attorney game."
    user_prompt = "What do you see in this image?"
    
    # Make API calls using legacy methods
    try:
        # Legacy text completion
        text_completion, text_costs = api_manager.text_completion(
            model_name=model_name,
            system_prompt=system_prompt,
            prompt="Tell me about Phoenix Wright's background.",
            session_name="legacy_text_example"
        )
        
        # Legacy vision completion
        vision_completion, vision_costs = api_manager.vision_completion(
            model_name=model_name,
            system_prompt=system_prompt,
            prompt=user_prompt,
            image_path=image_path,
            session_name="legacy_vision_example"
        )
        
        print("\n=== Legacy Methods Example ===")
        print("Legacy methods are supported for backward compatibility, but using the new specific methods is recommended.")
        
    except Exception as e:
        print(f"Error in legacy methods example: {e}")

if __name__ == "__main__":
    # Check if required environment variables are set
    required_keys = {
        "gpt": "OPENAI_API_KEY",
        "claude": "ANTHROPIC_API_KEY", 
        "gemini": "GEMINI_API_KEY"
    }
    
    for provider, key in required_keys.items():
        if not os.getenv(key):
            print(f"Warning: {key} environment variable not set. {provider.capitalize()} models will not work.")
    
    # Run text-only completion example
    example_text_only_completion()
    
    # Run examples that require an image path
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if os.path.exists(image_path):
            # Run vision-only completion example (image analysis)
            example_vision_only_completion(image_path)
            
            # Run vision-text completion example (image + specific text prompt)
            example_vision_text_completion(image_path)
            
            # Run legacy methods example
            example_legacy_methods(image_path)
        else:
            print(f"Image not found: {image_path}")
            
    # Run multi-image example if multiple image paths are provided
    if len(sys.argv) > 2:
        image_paths = sys.argv[1:]
        all_exist = all(os.path.exists(path) for path in image_paths)
        if all_exist:
            example_multi_image_completion(image_paths)
        else:
            missing = [path for path in image_paths if not os.path.exists(path)]
            print(f"Some images not found: {missing}")
    
    print("\nLogs and API call details are saved in the following directory structure:")
    print(f"cache/{{game_name}}/{{model_name}}/{{modality}}/{{datetime}}_{{session_name}}/")
    print("\nEach session directory contains:")
    print("1. api_call.json - Complete API call details, conversation log, and cost information")
    print("2. {game_name}_api_costs.log - Detailed token and cost metrics for this session")
    print("\nA consolidated cost log is also maintained at:")
    print("cache/{game_name}/{game_name}_api_costs.log") 