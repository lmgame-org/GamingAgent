# gamingagent/providers/__init__.py

from .base_provider import BaseProvider
from .anthropic import AnthropicProvider
from .openai import OpenAIProvider
from .gemini import GeminiProvider
from .together import TogetherProvider
from .ollama import OllamaProvider
from .api_provider_manager import APIProviderManager

# Create a singleton instance of the API manager
api_manager = APIProviderManager()

__all__ = [
    'BaseProvider',
    'AnthropicProvider',
    'OpenAIProvider',
    'GeminiProvider',
    'TogetherProvider',
    'OllamaProvider',
    'APIProviderManager',
    'api_manager'
]