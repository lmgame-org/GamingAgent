# gamingagent/providers/together.py

import os
from together import Together
from typing import List, Optional, Dict, Any
from .base_provider import BaseProvider

class TogetherProvider(BaseProvider):
    """Provider implementation for Together.ai models."""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None, **kwargs):
        super().__init__(model_name, api_key, **kwargs)
        self.api_key = api_key or os.getenv("TOGETHER_API_KEY")
        if not self.api_key:
            raise ValueError("TOGETHER_API_KEY not found")
            
        self.client = Together(api_key=self.api_key)
        self.temperature = kwargs.get('temperature', 0)
        
    def generate_with_images(self, prompt: str, images: List[str], **kwargs) -> str:
        messages = []
        for image in images:
            messages.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image}"}
            })
        messages.append({"type": "text", "text": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": messages}],
            temperature=kwargs.get('temperature', self.temperature)
        )
        
        return response.choices[0].message.content
        
    def generate_text(self, prompt: str, **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get('temperature', self.temperature)
        )
        
        return response.choices[0].message.content