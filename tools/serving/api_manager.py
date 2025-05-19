"""
API Manager for handling API calls to various providers.

This class provides a simplified approach to:
1. Make API calls to various providers (OpenAI, Anthropic, Gemini)
2. Support different modalities (text, vision, multi-image)
"""

# TODO: Optional functionality that could be added back:
# 1. API Cost Calculation:
#    - Track token usage for prompts and completions
#    - Calculate costs based on model pricing
#    - Support for different token counting methods per model
#    - Handle image token calculations
#
# 2. API Call Logging:
#    - Log all API calls with inputs/outputs
#    - Store structured JSON logs with timestamps
#    - Track costs and token usage over time
#    - Support session-based logging with custom directories
#    - Handle base64 image data in logs efficiently
#
# 3. Local Model Support:
#    - Add Ollama integration for local model inference
#    - Support for local model management (pull, list, delete)
#    - Handle local model parameters and configurations
#    - Implement fallback mechanisms for offline usage
#    - Support for local model streaming responses

import os
import base64
from typing import Dict, List, Optional, Union, Any

# Import API providers
from .api_providers import (
    anthropic_completion,
    anthropic_text_completion,
    anthropic_multiimage_completion,
    openai_completion,
    openai_text_completion,
    openai_multiimage_completion,
    gemini_completion,
    gemini_text_completion,
    gemini_multiimage_completion,
    together_ai_completion,
    together_ai_text_completion,
    together_ai_multiimage_completion,
    deepseek_text_reasoning_completion,
    xai_grok_completion,
)

class APIManager:
    """
    Simplified manager for API calls to various model providers.
    
    This class centralizes all API calls and provides a consistent interface
    for different modalities (text, vision, multi-image).
    """
    
    def __init__(
        self, 
        game_name: str = "default", 
        base_cache_dir: str = "cache"
    ):
        """
        Initialize the API Manager.
        
        Args:
            game_name (str): Name of the game/application (e.g., "ace_attorney")
            base_cache_dir (str): Base directory for all cache files
        """
        self.game_name = game_name
        self.base_cache_dir = base_cache_dir
        
        # Create base game directory
        # self.game_dir = os.path.join(self.base_cache_dir, self.game_name)
        # os.makedirs(self.game_dir, exist_ok=True)
    
    def _get_base64_from_path(self, image_path: str) -> str:
        """
        Convert image file to base64 string.
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            str: Base64-encoded image data
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            raise ValueError(f"Error reading image file: {e}")
    
    def vision_text_completion(
        self, 
        model_name: str, 
        system_prompt: str, 
        prompt: str, 
        image_path: Optional[str] = None,
        base64_image: Optional[str] = None, 
        temperature: float = 1,
        thinking: bool = False,
        reasoning_effort: str = "medium",
        token_limit: int = 30000
    ) -> str:
        """
        Make a combined vision-text completion API call.
        Both image and text are provided as input to the model.
        
        Args:
            model_name (str): Model name (e.g., "claude-3-opus-20240229")
            system_prompt (str): System prompt
            prompt (str): User prompt text
            image_path (str, optional): Path to image file
            base64_image (str, optional): Base64-encoded image data (alternative to image_path)
            temperature (float): Temperature parameter (0-1)
            thinking (bool): Whether to enable thinking mode (Anthropic models)
            reasoning_effort (str): Reasoning effort for O-series models ("low"|"medium"|"high")
            token_limit (int): Maximum number of tokens for the completion response
            
        Returns:
            str: Generated text
        """
        # Validate inputs
        if not (image_path or base64_image):
            raise ValueError("Either image_path or base64_image must be provided")
        
        # Get base64 image if path is provided
        if image_path and not base64_image:
            base64_image = self._get_base64_from_path(image_path)
        
        # Select appropriate API based on model name
        try:
            if "claude" in model_name.lower():
                completion = anthropic_completion(
                    system_prompt=system_prompt,
                    model_name=model_name,
                    base64_image=base64_image,
                    prompt=prompt,
                    thinking=thinking,
                    token_limit=token_limit
                )
            elif "gpt" in model_name.lower() or model_name.startswith("o"):
                completion = openai_completion(
                    system_prompt=system_prompt,
                    model_name=model_name,
                    base64_image=base64_image,
                    prompt=prompt,
                    temperature=temperature,
                    reasoning_effort=reasoning_effort,
                    token_limit=token_limit
                )
            elif "gemini" in model_name.lower():
                completion = gemini_completion(
                    system_prompt=system_prompt,
                    model_name=model_name,
                    base64_image=base64_image,
                    prompt=prompt,
                    token_limit=token_limit
                )
            elif "llama" in model_name.lower() or "meta" in model_name.lower():
                completion = together_ai_completion(
                    system_prompt=system_prompt,
                    model_name=model_name,
                    base64_image=base64_image,
                    prompt=prompt,
                    temperature=temperature,
                    token_limit=token_limit
                )
            else:
                raise ValueError(f"Unsupported model: {model_name}")
            
            return completion
            
        except Exception as e:
            raise ValueError(f"Error in vision-text completion API call: {e}")
    
    def text_only_completion(
        self,
        model_name: str,
        system_prompt: str,
        prompt: str,
        temperature: float = 1,
        thinking: bool = False,
        reasoning_effort: str = "medium",
        token_limit: int = 30000
    ) -> str:
        """
        Make a text-only completion API call.
        
        Args:
            model_name (str): Model name
            system_prompt (str): System prompt
            prompt (str): User prompt
            temperature (float): Temperature parameter (0-1)
            thinking (bool): Whether to enable thinking mode (Anthropic models)
            reasoning_effort (str): Reasoning effort for O-series models ("low"|"medium"|"high")
            token_limit (int): Maximum number of tokens for the completion response
            
        Returns:
            str: Generated text
        """
        # Select appropriate API based on model name
        try:
            if "claude" in model_name.lower():
                completion = anthropic_text_completion(
                    system_prompt=system_prompt,
                    model_name=model_name,
                    prompt=prompt,
                    thinking=thinking,
                    token_limit=token_limit
                )
            elif "gpt" in model_name.lower() or model_name.startswith("o"):
                completion = openai_text_completion(
                    system_prompt=system_prompt,
                    model_name=model_name,
                    prompt=prompt,
                    reasoning_effort=reasoning_effort,
                    token_limit=token_limit
                )
            elif "gemini" in model_name.lower():
                completion = gemini_text_completion(
                    system_prompt=system_prompt,
                    model_name=model_name,
                    prompt=prompt,
                    token_limit=token_limit
                )
            elif "llama" in model_name.lower() or "meta" in model_name.lower():
                completion = together_ai_text_completion(
                    system_prompt=system_prompt,
                    model_name=model_name,
                    prompt=prompt,
                    temperature=temperature,
                    token_limit=token_limit
                )
            elif "deepseek" in model_name.lower():
                completion = deepseek_text_reasoning_completion(
                    system_prompt=system_prompt,
                    model_name=model_name,
                    prompt=prompt,
                    token_limit=token_limit
                )
            elif "grok" in model_name.lower():
                completion = xai_grok_completion(
                    system_prompt=system_prompt,
                    model_name=model_name,
                    prompt=prompt,
                    token_limit=token_limit,
                    temperature=temperature,
                    reasoning_effort=reasoning_effort
                )
            else:
                raise ValueError(f"Unsupported model: {model_name}")
            
            return completion
            
        except Exception as e:
            raise ValueError(f"Error in text-only completion API call: {e}")
    
    def multi_image_completion(
        self,
        model_name: str,
        system_prompt: str,
        prompt: str,
        list_content: List[str],
        list_image_paths: Optional[List[str]] = None,
        list_image_base64: Optional[List[str]] = None,
        temperature: float = 1,
        reasoning_effort: str = "medium"
    ) -> str:
        """
        Make a multi-image completion API call.
        
        Args:
            model_name (str): Model name
            system_prompt (str): System prompt
            prompt (str): User prompt
            list_content (List[str]): List of text content corresponding to each image
            list_image_paths (List[str], optional): List of image file paths
            list_image_base64 (List[str], optional): List of base64-encoded image data
            temperature (float): Temperature parameter (0-1)
            reasoning_effort (str): Reasoning effort for O-series models ("low"|"medium"|"high")
            
        Returns:
            str: Generated text
        """
        # Validate inputs
        if not (list_image_paths or list_image_base64):
            raise ValueError("Either list_image_paths or list_image_base64 must be provided")
            
        if list_image_paths and not list_image_base64:
            # Convert image paths to base64
            list_image_base64 = []
            for image_path in list_image_paths:
                list_image_base64.append(self._get_base64_from_path(image_path))
        
        # Select appropriate API based on model name
        try:
            if "claude" in model_name.lower():
                completion = anthropic_multiimage_completion(
                    system_prompt=system_prompt,
                    model_name=model_name,
                    prompt=prompt,
                    list_content=list_content,
                    list_image_base64=list_image_base64
                )
            elif "gpt" in model_name.lower() or model_name.startswith("o"):
                completion = openai_multiimage_completion(
                    system_prompt=system_prompt,
                    model_name=model_name,
                    prompt=prompt,
                    list_content=list_content,
                    list_image_base64=list_image_base64,
                    reasoning_effort=reasoning_effort
                )
            elif "gemini" in model_name.lower():
                completion = gemini_multiimage_completion(
                    system_prompt=system_prompt,
                    model_name=model_name,
                    prompt=prompt,
                    list_content=list_content,
                    list_image_base64=list_image_base64
                )
            elif "llama" in model_name.lower() or "meta" in model_name.lower():
                completion = together_ai_multiimage_completion(
                    system_prompt=system_prompt,
                    model_name=model_name,
                    prompt=prompt,
                    list_content=list_content,
                    list_image_base64=list_image_base64,
                    temperature=temperature
                )
            else:
                raise ValueError(f"Unsupported model: {model_name}")
            
            return completion
            
        except Exception as e:
            raise ValueError(f"Error in multi-image completion API call: {e}") 