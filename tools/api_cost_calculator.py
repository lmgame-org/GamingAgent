"""
Costs dictionary and utility tool for counting tokens
"""

import os
import tiktoken
import anthropic
from typing import Union, List, Dict
from .constants import TOKEN_COSTS
from decimal import Decimal
import logging
from PIL import Image
import math
import google.generativeai as genai
import os

    

logger = logging.getLogger(__name__)

# Note: cl100k is the openai base tokenizer. Nothing to do with Claude. Tiktoken doesn't have claude yet.
# https://github.com/anthropics/anthropic-tokenizer-typescript/blob/main/index.ts


def get_anthropic_token_count(messages: List[Dict[str, str]], model: str) -> int:
    if not any(
        supported_model in model
        for supported_model in [
            "claude-3-7-sonnet",
            "claude-3-5-sonnet",
            "claude-3-5-haiku",
            "claude-3-haiku",
            "claude-3-opus",
        ]
    ):
        raise ValueError(
            f"{model} is not supported in token counting (beta) API. Use the `usage` property in the response for exact counts."
        )
    try:
        return (
            anthropic.Anthropic()
            .beta.messages.count_tokens(
                model=model,
                messages=messages,
            )
            .input_tokens
        )
    except TypeError as e:
        raise e
    except Exception as e:
        raise e


def strip_ft_model_name(model: str) -> str:
    """
    Finetuned models format: ft:gpt-3.5-turbo:my-org:custom_suffix:id
    We only need the base model name to get cost info.
    """
    if model.startswith("ft:gpt-3.5-turbo"):
        model = "ft:gpt-3.5-turbo"
    return model


def count_message_tokens(messages: List[Dict[str, str]], model: str) -> int:
    """
    Return the total number of tokens in a prompt's messages.
    Args:
        messages (List[Dict[str, str]]): Message format for prompt requests. e.g.:
            [{ "role": "user", "content": "Hello world"},
             { "role": "assistant", "content": "How may I assist you today?"}]
        model (str): Name of LLM to choose encoding for.
    Returns:
        Total number of tokens in message.
    """
    model = model.lower()
    model = strip_ft_model_name(model)

    # Anthropic token counting requires a valid API key
    if "claude-" in model:
        logger.warning(
            "Warning: Anthropic token counting API is currently in beta. Please expect differences in costs!"
        )
        return get_anthropic_token_count(messages, model)

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        logger.warning("Model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        "gpt-4-turbo",
        "gpt-4-turbo-2024-04-09",
        "gpt-4o",
        "gpt-4o-2024-05-13",
    } or model.startswith("o"):
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_message = 4
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        logger.warning(
            "gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613."
        )
        return count_message_tokens(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4o" in model:
        logger.warning(
            "Warning: gpt-4o may update over time. Returning num tokens assuming gpt-4o-2024-05-13."
        )
        return count_message_tokens(messages, model="gpt-4o-2024-05-13")
    elif "gpt-4" in model:
        logger.warning(
            "gpt-4 may update over time. Returning num tokens assuming gpt-4-0613."
        )
        return count_message_tokens(messages, model="gpt-4-0613")
    else:
        raise KeyError(
            f"""num_tokens_from_messages() is not implemented for model {model}.
            See https://github.com/openai/openai-python/blob/main/chatml.md for how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

def convert_string_to_messsage(prompt: str) -> List[Dict[str, str]]:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
            ],
        }
    ]

    return messages


def count_string_tokens(prompt: str, model: str) -> int:
    """
    Returns the number of tokens in a (prompt or completion) text string.

    Args:
        prompt (str): The text string
        model_name (str): The name of the encoding to use. (e.g., "gpt-3.5-turbo")

    Returns:
        int: The number of tokens in the text string.
    """
    model = model.lower()

    if "/" in model:
        model = model.split("/")[-1]

    if "claude-" in model:
        raise ValueError(
            "Warning: Anthropic does not support this method. Please use the `count_message_tokens` function for the exact counts."
        )

    try:
        if "gemini" in model:
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
                gemini_model = genai.GenerativeModel(f"models/{model}")
                token_count = gemini_model.count_tokens([prompt])
                return token_count.total_tokens
        else:
            encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        logger.warning("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(prompt))


def calculate_cost_by_tokens(num_tokens: int, model: str, token_type: str) -> Decimal:
    """
    Calculate the cost based on the number of tokens and the model.

    Args:
        num_tokens (int): The number of tokens.
        model (str): The model name.
        token_type (str): Type of token ('input' or 'output').

    Returns:
        Decimal: The calculated cost in USD.
    """
    model = model.lower()
    if model not in TOKEN_COSTS:
        raise KeyError(
            f"""Model {model} is not implemented.
            Double-check your spelling, or submit an issue/PR"""
        )


    if "gemini-2.5" in model and num_tokens > 200000:
        cost_per_token_key = (
            "input_cost_per_token_above_200k_tokens" if token_type == "input" else "output_cost_per_token_above_200k_tokens"
        )
    else:
        cost_per_token_key = (
            "input_cost_per_token" if token_type == "input" else "output_cost_per_token"
        )
    
    cost_per_token = TOKEN_COSTS[model][cost_per_token_key]


    return Decimal(str(cost_per_token)) * Decimal(num_tokens)


def calculate_prompt_cost(prompt: Union[List[dict], str], model: str) -> Decimal:
    """
    Calculate the prompt's cost in USD.

    Args:
        prompt (Union[List[dict], str]): List of message objects or single string prompt.
        model (str): The model name.

    Returns:
        Decimal: The calculated cost in USD.

    e.g.:
    >>> prompt = [{ "role": "user", "content": "Hello world"},
                  { "role": "assistant", "content": "How may I assist you today?"}]
    >>>calculate_prompt_cost(prompt, "gpt-3.5-turbo")
    Decimal('0.0000300')
    # or
    >>> prompt = "Hello world"
    >>> calculate_prompt_cost(prompt, "gpt-3.5-turbo")
    Decimal('0.0000030')
    """
    model = model.lower()
    model = strip_ft_model_name(model)
    if model not in TOKEN_COSTS:
        raise KeyError(
            f"""Model {model} is not implemented.
            Double-check your spelling, or submit an issue/PR"""
        )
    if not isinstance(prompt, (list, str)):
        raise TypeError(
            f"Prompt must be either a string or list of message objects but found {type(prompt)} instead."
        )
    prompt_tokens = (
        count_string_tokens(prompt, model)
        if isinstance(prompt, str) and "claude-" not in model
        else count_message_tokens(prompt, model)
    )

    return calculate_cost_by_tokens(prompt_tokens, model, "input")


def calculate_completion_cost(completion: str, model: str) -> Decimal:
    """
    Calculate the prompt's cost in USD.

    Args:
        completion (str): Completion string.
        model (str): The model name.

    Returns:
        Decimal: The calculated cost in USD.

    e.g.:
    >>> completion = "How may I assist you today?"
    >>> calculate_completion_cost(completion, "gpt-3.5-turbo")
    Decimal('0.000014')
    """
    model = strip_ft_model_name(model)
    model = model.lower()
    if model not in TOKEN_COSTS:
        raise KeyError(
            f"""Model {model} is not implemented.
            Double-check your spelling, or submit an issue/PR"""
        )

    if not isinstance(completion, str):
        raise TypeError(
            f"Prompt must be a string but found {type(completion)} instead."
        )

    if "claude-" in model:
        completion_list = [{"role": "assistant", "content": completion}]
        # Anthropic appends some 13 additional tokens to the actual completion tokens
        completion_tokens = count_message_tokens(completion_list, model) - 13
    else:
        completion_tokens = count_string_tokens(completion, model)

    return calculate_cost_by_tokens(completion_tokens, model, "output")

def count_image_tokens(image_path: str, model: str):
    """
    Calculate the number of tokens for an image based on OpenAI, Claude, or Gemini token counting rules.
    
    Args:
        image_path (str): Path to the image file
        model (str): The model name
        
    Returns:
        int: Number of tokens for the image
    """
    try:
        # Open the image and get its dimensions
        with Image.open(image_path) as img:
            width, height = img.size
            
            # OpenAI models
            if any(model.startswith(prefix) for prefix in ["gpt-4", "o"]):
                # If image is smaller than or equal to 512x512, use 85 tokens
                if width <= 512 and height <= 512:
                    return 85
                    
                # Scale to fit in 2048x2048 square while maintaining aspect ratio
                if width > 2048 or height > 2048:
                    if width > height:
                        new_width = 2048
                        new_height = int(height * (2048 / width))
                    else:
                        new_height = 2048
                        new_width = int(width * (2048 / height))
                else:
                    new_width, new_height = width, height
                    
                # Scale so shortest side is 768px
                if new_width < new_height:
                    scale = 768 / new_width
                else:
                    scale = 768 / new_height
                    
                new_width = int(new_width * scale)
                new_height = int(new_height * scale)
                
                # Count number of 512px squares needed
                num_squares = math.ceil(new_width / 512) * math.ceil(new_height / 512)
                
                # Calculate total tokens: 170 tokens per square + 85 base tokens
                total_tokens = (num_squares * 170) + 85
                
                return total_tokens
            
            # Claude models
            elif any(model.startswith(prefix) for prefix in ["claude"]):
                # Check if image needs to be resized (Claude limits)
                if width > 8000 or height > 8000:
                    logger.warning(f"Image exceeds Claude's maximum size of 8000x8000. It will be rejected or resized.")
                
                # If dimensions exceed 1568 on either side, it will be resized by Claude
                if width > 1568 or height > 1568:
                    # Resize to keep aspect ratio, with longest side at 1568
                    scale = 1568 / max(width, height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    width, height = new_width, new_height
                
                # Use Claude's token calculation formula: (width * height) / 750
                tokens = math.ceil((width * height) / 750)
                
                # Cap at the maximum for common aspect ratios
                if tokens > 1600:
                    tokens = 1600  # Maximum for images within Claude's optimal size
                
                return tokens
            
            # Gemini models
            elif any(model.startswith(prefix) for prefix in ["gemini", "models/gemini"]):
                try:

                    # Fallback to manual calculation using Gemini 2.0 rules
                    if "2.0" in model or any(version in model for version in ["1.5", "1.0"]):
                        # Gemini 2.0 rules:
                        # - Images ≤384×384px: 258 tokens
                        # - Larger images: Split into 768×768 tiles, 258 tokens per tile
                        if width <= 384 and height <= 384:
                            return 258
                        else:
                            # Calculate number of 768×768 tiles needed
                            num_tiles = math.ceil(width / 768) * math.ceil(height / 768)
                            return num_tiles * 258
                    api_key = os.getenv("GEMINI_API_KEY")
                    if api_key:
                        genai.configure(api_key=api_key)
                        gemini_model = genai.GenerativeModel(f"models/{model}")
                        image_file = genai.upload_file(path=image_path)
                        token_count = gemini_model.count_tokens([image_file])
                        return token_count.total_tokens
                except (ImportError, Exception) as e:
                    logger.warning(f"Couldn't use Google's API for token counting: {e}")
                    return 258
                    
                
            # Default case for other models
            else:
                logger.warning(f"Model {model} is not recognized for image token counting. Using minimal token count.")
                return 85
                
    except Exception as e:
        logger.error(f"Error calculating image tokens: {e}")
        return 0

def calculate_image_cost(image_path: str, model: str):
    """
    Calculate the cost of an image based on the number of tokens and model pricing.
    
    Args:
        image_path (str): Path to the image file
        model (str): The model name
        
    Returns:
        Decimal: The calculated cost in USD
    """
    model = model.lower()
    if model not in TOKEN_COSTS:
        raise KeyError(
            f"""Model {model} is not implemented.
            Double-check your spelling, or submit an issue/PR"""
        )
        
    image_tokens = count_image_tokens(image_path, model)
    cost_per_token = TOKEN_COSTS[model]["input_cost_per_token"]
    
    return Decimal(str(cost_per_token)) * Decimal(image_tokens)

def calculate_all_costs_and_tokens(
    prompt: Union[List[dict], str], completion: str, model: str, image_path: str = None
) -> dict:
    """
    Calculate the prompt and completion costs and tokens in USD.

    Args:
        prompt (Union[List[dict], str]): List of message objects or single string prompt.
        completion (str): Completion string.
        model (str): The model name.

    Returns:
        dict: The calculated cost and tokens in USD.

    e.g.:
    >>> prompt = "Hello world"
    >>> completion = "How may I assist you today?"
    >>> calculate_all_costs_and_tokens(prompt, completion, "gpt-3.5-turbo")
    {'prompt_cost': Decimal('0.0000030'), 'prompt_tokens': 2, 'completion_cost': Decimal('0.000014'), 'completion_tokens': 7}
    """
    if model == "gemini-2.5-pro-preview-03-25":
        model = "gemini-2.5-pro-exp-03-25"
    prompt_cost = calculate_prompt_cost(prompt, model)
    completion_cost = calculate_completion_cost(completion, model)
    prompt_tokens = (
        count_string_tokens(prompt, model)
        if isinstance(prompt, str) and "claude-" not in model
        else count_message_tokens(prompt, model)
    )

    if "claude-" in model:
        logger.warning("Warning: Token counting is estimated for ")
        completion_list = [{"role": "assistant", "content": completion}]
        # Anthropic appends some 13 additional tokens to the actual completion tokens
        completion_tokens = count_message_tokens(completion_list, model) - 13
    else:
        completion_tokens = count_string_tokens(completion, model)
    
    if image_path:
        image_cost = calculate_image_cost(image_path, model)
        image_tokens = count_image_tokens(image_path, model)
        return {
            "prompt_cost": prompt_cost,
            "prompt_tokens": prompt_tokens,
            "completion_cost": completion_cost,
            "completion_tokens": completion_tokens,
            "image_cost": image_cost,
            "image_tokens": image_tokens,
        }
    else:
        return {
            "prompt_cost": prompt_cost,
            "prompt_tokens": prompt_tokens,
            "completion_cost": completion_cost,
            "completion_tokens": completion_tokens
        }