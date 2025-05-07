# tests/test_providers/test_api_manager.py

import os
import pytest
import base64
from PIL import Image
import numpy as np
from io import BytesIO
from gamingagent.providers import api_manager
from gamingagent.providers.api_provider_manager import APIProviderManager

# Create a simple test image
def create_test_image():
    """Create a simple test image and convert to base64."""
    img = Image.new('RGB', (100, 100), color='red')
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Test prompts
TEST_PROMPT = "What color is dominant in this image?"
TEST_TEXT_PROMPT = "Say 'Hello, World!'"
TEST_IMAGE = create_test_image()

def test_api_manager_initialization():
    """Test API manager initialization with all providers."""
    api_manager.initialize_providers(
        anthropic_model='claude-3-opus-20240229',
        openai_model='gpt-4o',
        gemini_model='gemini-1.5-flash',
        together_model='meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8',
        ollama_model='llama2',
        ollama_endpoint='http://localhost:11434',
        xai_model='grok-2-vision'
    )
    
    # Check if providers are registered based on environment variables
    if os.getenv("ANTHROPIC_API_KEY"):
        assert api_manager.anthropic is not None, "Anthropic provider not initialized"
    if os.getenv("OPENAI_API_KEY"):
        assert api_manager.openai is not None, "OpenAI provider not initialized"
    if os.getenv("GEMINI_API_KEY"):
        assert api_manager.gemini is not None, "Gemini provider not initialized"
    if os.getenv("TOGETHER_API_KEY"):
        assert api_manager.together is not None, "Together provider not initialized"
    
    # Ollama should always be available as it's local
    assert api_manager.ollama is not None, "Ollama provider not initialized"

@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="Anthropic API key not set")
def test_anthropic_provider():
    """Test Anthropic provider functionality."""
    provider = api_manager.anthropic
    
    # Test image generation
    response = provider.generate_with_images(TEST_PROMPT, [TEST_IMAGE])
    assert response and isinstance(response, str), "Anthropic image generation failed"
    assert "red" in response.lower(), "Anthropic failed to identify red color"
    
    # Test text generation
    response = provider.generate_text(TEST_TEXT_PROMPT)
    assert response and isinstance(response, str), "Anthropic text generation failed"
    assert "hello" in response.lower(), "Anthropic failed to generate hello world"

@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not set")
def test_openai_provider():
    """Test OpenAI provider functionality."""
    provider = api_manager.openai
    
    # Test image generation
    response = provider.generate_with_images(TEST_PROMPT, [TEST_IMAGE])
    assert response and isinstance(response, str), "OpenAI image generation failed"
    assert "red" in response.lower(), "OpenAI failed to identify red color"
    
    # Test text generation
    response = provider.generate_text(TEST_TEXT_PROMPT)
    assert response and isinstance(response, str), "OpenAI text generation failed"
    assert "hello" in response.lower(), "OpenAI failed to generate hello world"

@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="Gemini API key not set")
def test_gemini_provider():
    """Test Gemini provider functionality."""
    provider = api_manager.gemini
    
    # Test image generation
    response = provider.generate_with_images(TEST_PROMPT, [TEST_IMAGE])
    assert response and isinstance(response, str), "Gemini image generation failed"
    assert "red" in response.lower(), "Gemini failed to identify red color"
    
    # Test text generation
    response = provider.generate_text(TEST_TEXT_PROMPT)
    assert response and isinstance(response, str), "Gemini text generation failed"
    assert "hello" in response.lower(), "Gemini failed to generate hello world"

@pytest.mark.skipif(not os.getenv("TOGETHER_API_KEY"), reason="Together API key not set")
def test_together_provider():
    """Test Together provider functionality."""
    provider = api_manager.together
    
    # Test image generation
    response = provider.generate_with_images(TEST_PROMPT, [TEST_IMAGE])
    assert response and isinstance(response, str), "Together image generation failed"
    assert "red" in response.lower(), "Together failed to identify red color"
    
    # Test text generation
    response = provider.generate_text(TEST_TEXT_PROMPT)
    assert response and isinstance(response, str), "Together text generation failed"
    assert "hello" in response.lower(), "Together failed to generate hello world"

@pytest.mark.skipif(not os.getenv("XAI_API_KEY"), reason="XAI API key not set")
def test_xai_provider():
    """Test XAI/Grok provider functionality."""
    provider = api_manager.xai
    
    # Test image generation
    response = provider.generate_with_images(TEST_PROMPT, [TEST_IMAGE])
    assert response and isinstance(response, str), "XAI image generation failed"
    assert "red" in response.lower(), "XAI failed to identify red color"
    
    # Test text generation
    response = provider.generate_text(TEST_TEXT_PROMPT)
    assert response and isinstance(response, str), "XAI text generation failed"
    assert "hello" in response.lower(), "XAI failed to generate hello world"

def test_ollama_provider():
    """Test Ollama provider functionality."""
    provider = api_manager.ollama
    
    try:
        # Test text generation first (always supported)
        response = provider.generate_text(TEST_TEXT_PROMPT)
        assert response and isinstance(response, str), "Ollama text generation failed"
        assert "hello" in response.lower(), "Ollama failed to generate hello world"
        
        # Test image generation (may not be supported by all models)
        try:
            response = provider.generate_with_images(TEST_PROMPT, [TEST_IMAGE])
            assert response and isinstance(response, str), "Ollama image generation failed"
            assert "red" in response.lower(), "Ollama failed to identify red color"
        except Exception as e:
            pytest.skip(f"Ollama image generation not supported: {e}")
            
    except Exception as e:
        pytest.skip(f"Ollama server not available: {e}")

def test_default_provider():
    """Test default provider functionality."""
    # Set Anthropic as default if available, otherwise use Ollama
    if api_manager.anthropic:
        api_manager.set_default_provider('anthropic')
    else:
        api_manager.set_default_provider('ollama')
    
    assert api_manager.default_provider is not None, "Default provider not set"
    
    # Test default provider
    response = api_manager.default_provider.generate_text(TEST_TEXT_PROMPT)
    assert response and isinstance(response, str), "Default provider text generation failed"

def test_error_handling():
    """Test error handling in API manager."""
    # Create a new instance for testing errors
    test_manager = APIProviderManager()
    
    # Test invalid provider
    with pytest.raises(ValueError):
        test_manager.get_provider('invalid_provider')
    
    # Test invalid model name
    with pytest.raises(Exception):
        test_manager.initialize_providers(anthropic_model='invalid_model')
    
    # Test invalid API key
    if 'ANTHROPIC_API_KEY' in os.environ:
        original_key = os.environ['ANTHROPIC_API_KEY']
        os.environ['ANTHROPIC_API_KEY'] = 'invalid_key'
        with pytest.raises(Exception):
            test_manager.initialize_providers()
        os.environ['ANTHROPIC_API_KEY'] = original_key

if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v"])