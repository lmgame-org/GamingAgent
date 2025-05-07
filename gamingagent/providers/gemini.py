# gamingagent/providers/gemini.py

import os
import google.generativeai as genai
from typing import List, Optional, Dict, Any
from .base_provider import BaseProvider

class GeminiProvider(BaseProvider):
    """Provider implementation for Google's Gemini models."""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None, **kwargs):
        """
        Initialize Gemini provider.
        
        Args:
            model_name: Name of the Gemini model to use (e.g., 'gemini-pro-vision')
            api_key: Gemini API key (if not set in env)
            **kwargs: Additional configuration including:
                - temperature: Sampling temperature
                - max_output_tokens: Maximum tokens in response
        """
        super().__init__(model_name, api_key, **kwargs)
        
        # Get API key from env if not provided
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment or constructor")
            
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        # Initialize model
        self.model = genai.GenerativeModel(model_name=self.model_name)
        
        # Set generation parameters
        self.temperature = kwargs.get('temperature', 0.0)
        self.max_output_tokens = kwargs.get('max_output_tokens', 2048)
        
    def generate_with_images(self, prompt: str, images: List[str], **kwargs) -> str:
        """
        Generate text response based on prompt and images using Gemini.
        
        Args:
            prompt: Text prompt for the model
            images: List of base64 encoded images
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        try:
            # Prepare content parts
            content = []
            
            # Add images
            for image in images:
                content.append({
                    "mime_type": "image/jpeg",
                    "data": image
                })
            
            # Add prompt
            content.append(prompt)
            
            # Generate response
            response = self.model.generate_content(
                content,
                generation_config=genai.types.GenerationConfig(
                    temperature=kwargs.get('temperature', self.temperature),
                    max_output_tokens=kwargs.get('max_output_tokens', self.max_output_tokens),
                    top_p=kwargs.get('top_p', 1.0),
                    top_k=kwargs.get('top_k', 32)
                )
            )
            
            # Check for valid response
            if not response or not hasattr(response, 'text'):
                print("Warning: Empty or invalid response from Gemini")
                return ""
                
            return response.text
            
        except Exception as e:
            print(f"Error in Gemini image generation: {e}")
            return ""
            
    def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Generate text response based on prompt only using Gemini.
        
        Args:
            prompt: Text prompt for the model
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=kwargs.get('temperature', self.temperature),
                    max_output_tokens=kwargs.get('max_output_tokens', self.max_output_tokens),
                    top_p=kwargs.get('top_p', 1.0),
                    top_k=kwargs.get('top_k', 32)
                )
            )
            
            # Check for valid response
            if not response or not hasattr(response, 'text'):
                print("Warning: Empty or invalid response from Gemini")
                return ""
                
            return response.text
            
        except Exception as e:
            print(f"Error in Gemini text generation: {e}")
            return ""
            
    def _handle_safety_ratings(self, response) -> None:
        """Log safety ratings if available."""
        if hasattr(response, 'safety_ratings'):
            for rating in response.safety_ratings:
                if rating.probability >= 0.5:  # Only log significant ratings
                    print(f"Safety concern: {rating.category} - {rating.probability}")