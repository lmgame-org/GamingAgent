# gamingagent/providers/xai.py

import os
from typing import List, Optional, Dict, Any
from openai import OpenAI
from .base_provider import BaseProvider

class XAIProvider(BaseProvider):
    """Provider implementation for XAI/Grok models."""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None, **kwargs):
        """
        Initialize XAI provider.
        
        Args:
            model_name: Name of the XAI/Grok model to use
            api_key: XAI API key (if not set in env)
            **kwargs: Additional configuration including:
                - api_base: Custom API endpoint
                - max_tokens: Maximum tokens in response
                - temperature: Sampling temperature
        """
        super().__init__(model_name, api_key, **kwargs)
        
        self.api_key = api_key or os.getenv("XAI_API_KEY")
        if not self.api_key:
            raise ValueError("XAI_API_KEY not found in environment or constructor")
            
        # Set API configuration
        self.api_base = kwargs.get('api_base', 'https://api.x.ai/v1')
        self.max_tokens = kwargs.get('max_tokens', 1024)
        self.temperature = kwargs.get('temperature', 0)
        
        # Initialize OpenAI client with XAI configuration
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base
        )
        
    def generate_with_images(self, prompt: str, images: List[str], **kwargs) -> str:
        """
        Generate text response based on prompt and images using XAI/Grok.
        
        Args:
            prompt: Text prompt for the model
            images: List of base64 encoded images
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        messages = [{"role": "user", "content": []}]
        
        # Add text content
        messages[0]["content"].append({
            "type": "text",
            "text": prompt
        })
        
        # Add images to content if provided
        for image in images:
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image}"
                }
            })
            
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=kwargs.get('max_tokens', self.max_tokens),
            temperature=kwargs.get('temperature', self.temperature)
        )
        
        return response.choices[0].message.content
        
    def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Generate text response based on prompt only using XAI/Grok.
        
        Args:
            prompt: Text prompt for the model
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=kwargs.get('max_tokens', self.max_tokens),
            temperature=kwargs.get('temperature', self.temperature)
        )
        
        return response.choices[0].message.content