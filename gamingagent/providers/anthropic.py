# gamingagent/providers/anthropic.py

import os
import anthropic
from typing import List, Optional, Dict, Any

from .base_provider import BaseProvider

class AnthropicProvider(BaseProvider):
    """Provider implementation for Anthropic's Claude models."""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None, **kwargs):
        """
        Initialize Anthropic provider.
        
        Args:
            model_name: Name of the Claude model to use
            api_key: Anthropic API key (if not set in env)
            **kwargs: Additional configuration including:
                - max_tokens: Maximum tokens in response
                - temperature: Sampling temperature
        """
        super().__init__(model_name, api_key, **kwargs)
        
        # Get API key from env if not provided
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment or constructor")
            
        # Initialize client
        self.client = anthropic.Anthropic(api_key=self.api_key)
        
        # Set default parameters
        self.max_tokens = kwargs.get('max_tokens', 1024)
        self.temperature = kwargs.get('temperature', 0)
        
    def generate_with_images(self, prompt: str, images: List[str], **kwargs) -> str:
        """
        Generate text response based on prompt and images using Claude.
        
        Args:
            prompt: Text prompt for the model
            images: List of base64 encoded images
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        # Build message content
        content = []
        for image in images:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": image,
                },
            })
        content.append({
            "type": "text",
            "text": prompt
        })
        
        # Create messages structure
        messages = [{
            "role": "user",
            "content": content,
        }]
        
        # Get generation parameters
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        temperature = kwargs.get('temperature', self.temperature)
        
        # Make API call
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
        )
        
        return response.content[0].text
        
    def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Generate text response based on prompt only using Claude.
        
        Args:
            prompt: Text prompt for the model
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        # Create messages structure
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }]
        
        # Get generation parameters
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        temperature = kwargs.get('temperature', self.temperature)
        
        # Make API call
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
        )
        
        return response.content[0].text