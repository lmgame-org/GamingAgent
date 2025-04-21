# gamingagent/providers/base_provider.py

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

class BaseProvider(ABC):
    """Base class for all LLM providers."""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None, **kwargs):
        """
        Initialize provider.
        
        Args:
            model_name: Name of the model to use
            api_key: API key for the provider (if not set in env)
            **kwargs: Additional provider-specific configuration
        """
        self.model_name = model_name
        self.api_key = api_key
        self.kwargs = kwargs
        
    @abstractmethod
    def generate_with_images(self, prompt: str, images: List[str], **kwargs) -> str:
        """
        Generate text response based on prompt and images.
        
        Args:
            prompt: Text prompt for the model
            images: List of base64 encoded images
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        pass
        
    @abstractmethod
    def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Generate text response based on prompt only.
        
        Args:
            prompt: Text prompt for the model
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        pass