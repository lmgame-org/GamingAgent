# gamingagent/providers/api_provider_manager.py

import os
from typing import Optional, Dict, Any
from .base_provider import BaseProvider
from .anthropic import AnthropicProvider
from .openai import OpenAIProvider
from .gemini import GeminiProvider
from .together import TogetherProvider
from .ollama import OllamaProvider
from .xai import XAIProvider

class APIProviderManager:
    """Manager class for handling different API providers."""
    
    def __init__(self):
        self._providers: Dict[str, BaseProvider] = {}
        self._default_provider = None
        
    def register_provider(self, name: str, provider: BaseProvider) -> None:
        """Register a new provider."""
        self._providers[name.lower()] = provider
        
    def get_provider(self, name: str) -> Optional[BaseProvider]:
        """Get a registered provider by name."""
        provider = self._providers.get(name.lower())
        if provider is None:
            raise ValueError(f"Provider '{name}' not found")
        return provider
        
    def set_default_provider(self, name: str) -> None:
        """Set the default provider."""
        if name.lower() in self._providers:
            self._default_provider = name.lower()
            
    @property
    def default_provider(self) -> Optional[BaseProvider]:
        """Get the default provider."""
        if self._default_provider:
            return self._providers[self._default_provider]
        return None
        
    @property
    def anthropic(self) -> Optional[AnthropicProvider]:
        """Get Anthropic provider."""
        return self.get_provider('anthropic')
        
    @property
    def openai(self) -> Optional[OpenAIProvider]:
        """Get OpenAI provider."""
        return self.get_provider('openai')
        
    @property
    def gemini(self) -> Optional[GeminiProvider]:
        """Get Gemini provider."""
        return self.get_provider('gemini')
        
    @property
    def together(self) -> Optional[TogetherProvider]:
        """Get Together provider."""
        return self.get_provider('together')
    
    @property
    def xai(self) -> Optional[XAIProvider]:
        """Get XAI provider."""
        return self.get_provider('xai')
        
    @property
    def ollama(self) -> Optional[OllamaProvider]:
        """Get Ollama provider."""
        return self.get_provider('ollama')
        
    def initialize_providers(self, **kwargs) -> None:
        """Initialize all available providers with their respective API keys."""
        # Anthropic
        if os.getenv("ANTHROPIC_API_KEY"):
            self.register_provider('anthropic', AnthropicProvider(
                model_name=kwargs.get('anthropic_model', 'claude-3-opus-20240229'),
                api_key=os.getenv("ANTHROPIC_API_KEY")
            ))
            
        # OpenAI
        if os.getenv("OPENAI_API_KEY"):
            self.register_provider('openai', OpenAIProvider(
                model_name=kwargs.get('openai_model', 'gpt-4o'),
                api_key=os.getenv("OPENAI_API_KEY")
            ))
            
        # Gemini
        if os.getenv("GEMINI_API_KEY"):
            self.register_provider('gemini', GeminiProvider(
                model_name=kwargs.get('gemini_model', 'gemini-1.5-flash'),
                api_key=os.getenv("GEMINI_API_KEY")
            ))
            
        # Together
        if os.getenv("TOGETHER_API_KEY"):
            self.register_provider('together', TogetherProvider(
                model_name=kwargs.get('together_model', 'meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8'),
                api_key=os.getenv("TOGETHER_API_KEY")
            ))

        # XAI/Grok
        if os.getenv("XAI_API_KEY"):
            self.register_provider('xai', XAIProvider(
                model_name=kwargs.get('xai_model', 'grok-2-vision'),
                api_key=os.getenv("XAI_API_KEY"),
                api_base=kwargs.get('xai_api_base', 'https://api.x.ai/v1'),
                max_tokens=kwargs.get('xai_max_tokens', 1024),
                temperature=kwargs.get('xai_temperature', 0)
            ))
            
        # Ollama
        ollama_endpoint = kwargs.get('ollama_endpoint', 'http://localhost:11434')
        self.register_provider('ollama', OllamaProvider(
            model_name=kwargs.get('ollama_model', 'llama2'),
            endpoint=ollama_endpoint
        ))