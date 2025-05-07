# gamingagent/providers/ollama.py

import os
import requests
from typing import List, Optional, Dict, Any
from .base_provider import BaseProvider

class OllamaProvider(BaseProvider):
    """Provider implementation for Ollama models."""
    
    def __init__(self, model_name: str, endpoint: str = "http://localhost:11434", **kwargs):
        super().__init__(model_name, None, **kwargs)
        self.endpoint = endpoint
        self.temperature = kwargs.get('temperature', 0)
        
    def _call_ollama_api(self, prompt: str, images: Optional[List[str]] = None) -> str:
        """Make API call to Ollama endpoint."""
        url = f"{self.endpoint}/api/generate"
        
        # Prepare request data
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "temperature": self.temperature
        }
        
        if images:
            # Add images to request if supported by model
            data["images"] = images
            
        try:
            response = requests.post(url, json=data)
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            print(f"Ollama API error: {e}")
            return ""
            
    def generate_with_images(self, prompt: str, images: List[str], **kwargs) -> str:
        return self._call_ollama_api(prompt, images)
        
    def generate_text(self, prompt: str, **kwargs) -> str:
        return self._call_ollama_api(prompt)