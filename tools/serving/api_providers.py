import os
import random
import time
import functools
import httpx

from openai import OpenAI
from openai import RateLimitError, APITimeoutError, APIConnectionError, APIStatusError, BadRequestError
import anthropic
import google.generativeai as genai
from google.generativeai import types
from together import Together

import requests
import grpc

def estimate_token_count(text: str) -> int:
    """
    Rough estimation of token count for text.
    Uses a simple heuristic of ~4 characters per token.
    """
    if not text:
        return 0
    return len(text) // 4

def _sleep_with_backoff(base_delay: int, attempt: int) -> None:
    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
    print(f"Retrying in {delay:.2f}s … (attempt {attempt + 1})")
    time.sleep(delay)

def retry_on_openai_error(func):
    """
    Retry wrapper for OpenAI SDK calls.
    Retries on: RateLimitError, Timeout, APIConnectionError,
                APIStatusError (5xx), httpx.RemoteProtocolError.
    Immediately raises on: BadRequestError (400).
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        max_retries = kwargs.pop("max_retries", 5)
        base_delay  = kwargs.pop("base_delay", 2)

        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)

            # BadRequestError should NOT be retried - it indicates invalid request
            except BadRequestError as e:
                print(f"OpenAI BadRequestError (not retrying): {e}")
                raise

            # transient issues worth retrying
            except (RateLimitError, APITimeoutError, APIConnectionError,
                    httpx.RemoteProtocolError) as e:
                if attempt < max_retries - 1:
                    print(f"OpenAI transient error: {e}")
                    _sleep_with_backoff(base_delay, attempt)
                    continue
                raise

            # server‑side 5xx response
            except APIStatusError as e:
                if 500 <= e.status_code < 600 and attempt < max_retries - 1:
                    print(f"OpenAI server error {e.status_code}: {e.message}")
                    _sleep_with_backoff(base_delay, attempt)
                    continue
                raise

    return wrapper

def retry_on_overload(func):
    """
    A decorator to retry a function call on anthropic.APIStatusError with 'overloaded_error',
    httpx.RemoteProtocolError, or when the API returns None/empty response.
    It uses exponential backoff with jitter.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        max_retries = 5
        base_delay = 2  # seconds
        for attempt in range(max_retries):
            try:
                result = func(*args, **kwargs)
                
                # Check if result is None or empty string
                if result is None or (isinstance(result, str) and not result.strip()):
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt) + (os.urandom(1)[0] / 255.0)
                        print(f"API returned None/empty response. Retrying in {delay:.2f} seconds... (Attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        continue
                    else:
                        print(f"API still returning None/empty after {max_retries} attempts. Raising an error.")
                        raise RuntimeError("API returned None/empty response after all retry attempts")
                
                # If we got a valid result, return it
                return result
                
            except anthropic.APIStatusError as e:
                if e.body and e.body.get('error', {}).get('type') == 'overloaded_error':
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt) + (os.urandom(1)[0] / 255.0)
                        print(f"Anthropic API overloaded. Retrying in {delay:.2f} seconds... (Attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                    else:
                        print(f"Anthropic API still overloaded after {max_retries} attempts. Raising the error.")
                        raise
                else:
                    # Re-raise if it's not an overload error
                    raise
            except httpx.RemoteProtocolError as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + (os.urandom(1)[0] / 255.0)
                    print(f"Streaming connection closed unexpectedly. Retrying in {delay:.2f} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                else:
                    print(f"Streaming connection failed after {max_retries} attempts. Raising the error.")
                    raise
    return wrapper

@retry_on_overload
def anthropic_completion(system_prompt, model_name, base64_image, prompt, thinking=False, token_limit=30000):
    print(f"anthropic vision-text activated... thinking: {thinking}")
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64_image,
                    },
                },
                {
                    "type": "text",
                    "text": prompt
                },
            ],
        }
    ]

    if "claude-3-5" in model_name:
        print("claude-3-5 only supports 8192 tokens and no thinking")
        thinking = False
        token_limit = 8192
    
    if "claude-3-7" in model_name:
        print("claude-3-7 supports 64000 tokens")
        token_limit = 64000

    if "claude-opus-4" in model_name.lower() and token_limit > 32000:
        print("claude-opus-4 supports 32000 tokens")
        token_limit = 32000

    if "claude-sonnet-4" in model_name.lower() and token_limit > 64000:
        print("claude-sonnet-4 supports 64000 tokens")
        token_limit = 64000

    if thinking:
        with client.messages.stream(
                max_tokens=token_limit,
                thinking={
                    "type": "enabled",
                    "budget_tokens": token_limit - 1
                },
                messages=messages,
                temperature=1,
                system=system_prompt,
                model=model_name, # claude-3-5-sonnet-20241022 # claude-3-7-sonnet-20250219
            ) as stream:
                partial_chunks = []
                try:
                    for chunk in stream.text_stream:
                        partial_chunks.append(chunk)
                except httpx.RemoteProtocolError as e:
                    print(f"Streaming connection closed unexpectedly: {e}")
                    # Return what we have so far
                    return "".join(partial_chunks)
    else:
        with client.messages.stream(
                max_tokens=token_limit,
                messages=messages,
                temperature=0,
                system=system_prompt,
                model=model_name, # claude-3-5-sonnet-20241022 # claude-3-7-sonnet-20250219
            ) as stream:
                partial_chunks = []
                try:
                    for chunk in stream.text_stream:
                        partial_chunks.append(chunk)
                except httpx.RemoteProtocolError as e:
                    print(f"Streaming connection closed unexpectedly: {e}")
                    # Return what we have so far
                    return "".join(partial_chunks)
        
    generated_code_str = "".join(partial_chunks)
    
    return generated_code_str

@retry_on_overload
def anthropic_text_completion(system_prompt, model_name, prompt, thinking=False, token_limit=30000):
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    token_limit =64000 if "claude-3-7" in model_name and token_limit > 64000 else token_limit
    print(f"model_name: {model_name}, token_limit: {token_limit}, thinking: {thinking}")
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
    
    if "claude-3-5" in model_name:
        print("claude-3-5 only supports 8192 tokens and no thinking")
        thinking = False
        token_limit = 8192

    if "claude-opus-4" in model_name.lower() and token_limit > 32000:
        print("claude-opus-4 supports 32000 tokens")
        token_limit = 32000

    if "claude-sonnet-4" in model_name.lower() and token_limit > 64000:
        print("claude-sonnet-4 supports 64000 tokens")
        token_limit = 64000

    if thinking:
        with client.messages.stream(
                max_tokens=token_limit,
                thinking={
                    "type": "enabled",
                    "budget_tokens": token_limit - 1
                },
                messages=messages,
                temperature=1,
                system=system_prompt,
                model=model_name, # claude-3-5-sonnet-20241022 # claude-3-7-sonnet-20250219
            ) as stream:
                partial_chunks = []
                try:
                    for chunk in stream.text_stream:
                        partial_chunks.append(chunk)
                except httpx.RemoteProtocolError as e:
                    print(f"Streaming connection closed unexpectedly: {e}")
                    # Return what we have so far
                    return "".join(partial_chunks)
    else:    
        with client.messages.stream(
                max_tokens=token_limit,
                messages=messages,
                temperature=0,
                system=system_prompt,
                model=model_name, # claude-3-5-sonnet-20241022 # claude-3-7-sonnet-20250219
            ) as stream:
                partial_chunks = []
                try:
                    for chunk in stream.text_stream:
                        partial_chunks.append(chunk)
                except httpx.RemoteProtocolError as e:
                    print(f"Streaming connection closed unexpectedly: {e}")
                    # Return what we have so far
                    return "".join(partial_chunks)
        
    generated_str = "".join(partial_chunks)
    
    return generated_str

@retry_on_overload
def anthropic_multiimage_completion(system_prompt, model_name, prompt, list_content, list_image_base64, token_limit=30000):
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    if "claude-opus-4" in model_name.lower() and token_limit > 32000:
        print("claude-opus-4 supports 32000 tokens")
        token_limit = 32000
    
    if "claude-sonnet-4" in model_name.lower() and token_limit > 64000:
        print("claude-sonnet-4 supports 64000 tokens")
        token_limit = 64000
    
    content_blocks = [] 
    for text_item, base64_image in zip(list_content, list_image_base64):
        content_blocks.append(
            {
                "type": "text",
                "text": text_item,
            }
        )
        content_blocks.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64_image,
                },
            }
        )
    
    content_blocks.append(
        {
            "type": "text",
            "text": prompt
        }
    )

    messages = [
        {
            "role": "user",
            "content": content_blocks,
        }
    ]

    print(f"message size: {len(content_blocks)+1}")

    with client.messages.stream(
            max_tokens=token_limit,
            messages=messages,
            temperature=0,
            system=system_prompt,
            model=model_name, # claude-3-5-sonnet-20241022 # claude-3-7-sonnet-20250219
        ) as stream:
            partial_chunks = []
            try:
                for chunk in stream.text_stream:
                    print(chunk)
                    partial_chunks.append(chunk)
            except httpx.RemoteProtocolError as e:
                print(f"Streaming connection closed unexpectedly: {e}")
                # Return what we have so far
                return "".join(partial_chunks)
        
    generated_str = "".join(partial_chunks)
    
    return generated_str

import httpx

_original_headers_init = httpx.Headers.__init__

def safe_headers_init(self, headers=None, encoding=None):
    # Convert dict values to ASCII
    if isinstance(headers, dict):
        headers = {
            k: (v.encode('ascii', 'ignore').decode() if isinstance(v, str) else v)
            for k, v in headers.items()
        }
    elif isinstance(headers, list):
        # Convert list of tuples: [(k, v), ...]
        headers = [
            (k, v.encode('ascii', 'ignore').decode() if isinstance(v, str) else v)
            for k, v in headers
        ]
    _original_headers_init(self, headers=headers, encoding=encoding)

# Apply the patch
httpx.Headers.__init__ = safe_headers_init

@retry_on_openai_error
def openai_completion(system_prompt, model_name, base64_image, prompt, temperature=1, token_limit=30000, reasoning_effort="medium"):
    print(f"OpenAI vision-text API call: model={model_name}, reasoning_effort={reasoning_effort}")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if "gpt-4o" in model_name:
        print("gpt-4o only supports 16384 tokens")
        token_limit = 16384
    elif "gpt-4.1" in model_name:
        print("gpt-4.1 only supports 32768 tokens")
        token_limit = 32768
    elif "o3" in model_name:
        print("o3 only supports 32768 tokens")
        token_limit = 10000

    # Force-clean headers to prevent UnicodeEncodeError
    client._client._headers.update({
        k: (v.encode('ascii', 'ignore').decode() if isinstance(v, str) else v)
        for k, v in client._client._headers.items()
    })

    base64_image = None if "o3-mini" in model_name else base64_image

    if base64_image is None:
        messages = [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    # Update token parameter logic to include o4 models
    token_param = "max_completion_tokens" if ("o1" in model_name or "o4" in model_name or "o3" in model_name) else "max_tokens"
    request_params = {
        "model": model_name,
        "messages": messages,
        token_param: token_limit,
    }

    # Add reasoning_effort for o1, o3, o4 models, temperature for others
    if "o1" in model_name or "o3" in model_name or "o4" in model_name:
        request_params["reasoning_effort"] = reasoning_effort
    else:
        request_params["temperature"] = temperature

    response = client.chat.completions.create(**request_params)
    return response.choices[0].message.content

@retry_on_openai_error
def openai_text_completion(system_prompt, model_name, prompt, token_limit=30000, reasoning_effort="medium"):
    print(f"OpenAI text-only API call: model={model_name}, reasoning_effort={reasoning_effort}")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if "gpt-4o" in model_name:
        print("gpt-4o only supports 16384 tokens")
        token_limit = 16384
    elif "gpt-4.1" in model_name:
        print("gpt-4.1 only supports 32768 tokens")
        token_limit = 32768
    elif "o3" in model_name:
        print("o3 only supports 32768 tokens")
        token_limit = 10000

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

    # Update token parameter logic to include all o-series models
    token_param = "max_completion_tokens" if ("o1" in model_name or "o4" in model_name or "o3" in model_name) else "max_tokens"
    
    request_params = {
        "model": model_name,
        "messages": messages,
        token_param: token_limit,
    }
    
    # Add reasoning_effort for o1, o3, o4 models, temperature for others
    if "o1" in model_name or "o3" in model_name or "o4" in model_name:
        request_params["reasoning_effort"] = reasoning_effort
    else:
        request_params["temperature"] = 1

    if model_name == "o3-pro":
        messages[0]['content'][0]['type'] = "input_text"
        response = client.responses.create(
            model="o3-pro",
            input=messages,
        )
        generated_str = response.output[1].content[0].text
    else:
        response = client.chat.completions.create(**request_params)
        generated_str = response.choices[0].message.content
    return generated_str

@retry_on_openai_error
def openai_text_reasoning_completion(system_prompt, model_name, prompt, temperature=1, token_limit=30000, reasoning_effort="medium"):
    print(f"OpenAI text-reasoning API call: model={model_name}, reasoning_effort={reasoning_effort}")
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if "gpt-4o" in model_name:
        print("gpt-4o only supports 16384 tokens")
        token_limit = 16384
    elif "gpt-4.1" in model_name:
        print("gpt-4.1 only supports 32768 tokens")
        token_limit = 32768
    elif "o3" in model_name:
        print("o3 only supports 32768 tokens")
        token_limit = 10000
    
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

    # Update token parameter logic to include all o-series models
    token_param = "max_completion_tokens" if ("o1" in model_name or "o4" in model_name or "o3" in model_name) else "max_tokens"
    
    # Prepare request parameters dynamically
    request_params = {
        "model": model_name,
        "messages": messages,
        token_param: token_limit,
    }
    
    # Add reasoning_effort for o1, o3, o4 models, temperature for others
    if "o1" in model_name or "o3" in model_name or "o4" in model_name:
        request_params["reasoning_effort"] = reasoning_effort
    else:
        request_params["temperature"] = temperature

    if model_name == "o3-pro":
        messages[0]['content'][0]['type'] = "input_text"
        response = client.responses.create(
            model="o3-pro",
            input=messages,
        )
        generated_str = response.output[1].content[0].text
    else:
        response = client.chat.completions.create(**request_params)
        generated_str = response.choices[0].message.content
    return generated_str

def deepseek_text_reasoning_completion(system_prompt, model_name, prompt, token_limit=30000):
    print(f"DeepSeek text-reasoning API call: model={model_name}")
    if token_limit > 8192:
        token_limit = 8192
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )


    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    reasoning_content = ""
    content = ""
    response = client.chat.completions.create(
        model= model_name,
        messages = messages,
        stream=True,
        max_tokens=token_limit)
    
    for chunk in response:
        if hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content:
            content += chunk.choices[0].delta.content
    
    # generated_str = response.choices[0].message.content
    
    return content
    

def xai_grok_text_completion(system_prompt, model_name, prompt, reasoning_effort="high", token_limit=30000, temperature=1):
    print(f"XAI Grok text API call: model={model_name}, reasoning_effort={reasoning_effort}")
    from xai_sdk import Client
    from xai_sdk.chat import user, system
    import os

    max_retries = 10
    base_delay = 2
    force_ipv4 = False  # Start with default IP configuration
    
    for attempt in range(max_retries):
        try:
            # Set IPv4 environment variable if needed
            if force_ipv4:
                os.environ["GRPC_DNS_RESOLVER"] = "native"  # Use native DNS resolver
                os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "false"  # Disable fork support which can interfere
                print("Forcing IPv4 connection...")
            
            client = Client(
                api_host="api.x.ai",
                api_key=os.getenv("XAI_API_KEY")
            )

            params = {
                "model": model_name,
                "temperature": temperature,
                "max_tokens": token_limit
            }

            if "grok-3-mini" in model_name:
                params["reasoning_effort"] = reasoning_effort

            chat = client.chat.create(**params)
            chat.append(system(system_prompt))
            chat.append(user(prompt))
            response = chat.sample()
            return response.content
        except Exception as e:
            error_str = str(e)
            error_type = e.__class__.__name__
            error_code = None
            
            # Safely get error code if available
            try:
                if hasattr(e, 'code') and callable(e.code):
                    error_code = e.code()
            except:
                error_code = None

            # Check for IPv6-related issues
            is_ipv6_error = any(x in error_str.lower() for x in ["ipv6", "no route to host"])
            
            # Helper function to determine retry delay
            def get_retry_delay(base_delay, attempt, error_type, is_ipv6=False):
                delay = base_delay * (2 ** attempt)
                if is_ipv6:
                    return delay * 1.2  # 20% more delay for IPv6 issues
                if "Connection reset by peer" in error_str:
                    return delay * 1.5  # 50% more delay for connection resets
                elif error_type == "UNKNOWN":
                    return delay + (attempt * 2)  # Extra delay for server errors
                return delay

            # Check if error is retryable
            is_retryable = False
            error_type = None
            
            if (error_code == grpc.StatusCode.RESOURCE_EXHAUSTED) or ('RESOURCE_EXHAUSTED' in error_str):
                is_retryable = True
                error_type = "RESOURCE_EXHAUSTED"
            elif (error_code == grpc.StatusCode.UNAVAILABLE) or ('UNAVAILABLE' in error_str):
                is_retryable = True
                error_type = "UNAVAILABLE"
                # If this is an IPv6 error and we haven't tried IPv4 yet, force IPv4 for next attempt
                if is_ipv6_error and not force_ipv4:
                    force_ipv4 = True
                    print("Detected IPv6 connectivity issue, will retry with IPv4 only")
            elif (error_code == grpc.StatusCode.DEADLINE_EXCEEDED) or ('DEADLINE_EXCEEDED' in error_str):
                is_retryable = True
                error_type = "DEADLINE_EXCEEDED"
            elif (error_code == grpc.StatusCode.UNKNOWN) or ('UNKNOWN' in error_str):
                is_retryable = True
                error_type = "UNKNOWN"

            if is_retryable and attempt < max_retries - 1:
                delay = get_retry_delay(base_delay, attempt, error_type, is_ipv6_error)
                print(f"Grok API error ({error_type} - {error_str}). Retrying in {delay:.2f} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                continue
            elif is_retryable:
                print(f"Grok API error persists after {max_retries} attempts. Last error ({error_type}): {error_str}")
                raise
            else:
                print(f"Unhandled Grok API error: {error_str}")
                raise

@retry_on_openai_error
def openai_multiimage_completion(system_prompt, model_name, prompt, list_content, list_image_base64, token_limit=30000, reasoning_effort="medium"):
    print(f"OpenAI multi-image API call: model={model_name}, reasoning_effort={reasoning_effort}")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if "gpt-4o" in model_name:
        print("gpt-4o only supports 16384 tokens")
        token_limit = 16384
    elif "gpt-4.1" in model_name:
        print("gpt-4.1 only supports 32768 tokens")
        token_limit = 32768
    elif "o3" in model_name:
        print("o3 only supports 32768 tokens")
        token_limit = 10000

    content_blocks = []
    
    joined_steps = "\n\n".join(list_content)
    content_blocks.append(
        {
            "type": "text",
            "text": joined_steps
        }
    )

    for base64_image in list_image_base64:
        content_blocks.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"
                },
            },
        )

    messages = [
        {
            "role": "user",
            "content": content_blocks,
        }
    ]
    
    # Update token parameter logic to include all o-series models
    token_param = "max_completion_tokens" if ("o1" in model_name or "o4" in model_name or "o3" in model_name) else "max_tokens"
    
    request_params = {
        "model": model_name,
        "messages": messages,
        token_param: token_limit,
    }
    
    # Add reasoning_effort for o1, o3, o4 models, temperature for others
    if "o1" in model_name or "o3" in model_name or "o4" in model_name:
        request_params["reasoning_effort"] = reasoning_effort
    else:
        request_params["temperature"] = 1

    response = client.chat.completions.create(**request_params)
    generated_str = response.choices[0].message.content
    return generated_str


def gemini_text_completion(system_prompt, model_name, prompt, token_limit=30000):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel(model_name=model_name)
    print(f"gemini_text_completion: model_name={model_name}, token_limit={token_limit}")

    messages = [
        prompt,
    ]
            
    try:
        response = model.generate_content(
            messages,
            generation_config=types.GenerationConfig(
                max_output_tokens=token_limit
            )
        )
    except Exception as e:
        print(f"error: {e}")

    try:
        response = model.generate_content(
            messages,
            generation_config=types.GenerationConfig(
                max_output_tokens=token_limit
            )
        )

        # Ensure response is valid and contains candidates
        if not response or not hasattr(response, "candidates") or not response.candidates:
            print("Warning: Empty or invalid response")
            return ""
        
        return response.text  # Access response.text safely

    except Exception as e:
        print(f"Error: {e}")
        return "" 

def gemini_completion(system_prompt, model_name, base64_image, prompt, token_limit=30000):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel(model_name=model_name)
    print(f"gemini_completion: model_name={model_name}, token_limit={token_limit}")
    messages = [
        {
            "mime_type": "image/jpeg",
            "data": base64_image,
        },
        prompt,
    ]
            
    try:
        response = model.generate_content(
            messages,
            generation_config=types.GenerationConfig(
                max_output_tokens=token_limit
            )
        )
    except Exception as e:
        print(f"error: {e}")

    try:
        response = model.generate_content(
            messages,
            generation_config=types.GenerationConfig(
                max_output_tokens=token_limit
            )
        )

        # Ensure response is valid and contains candidates
        if not response or not hasattr(response, "candidates") or not response.candidates:
            print("Warning: Empty or invalid response")
            return ""
        
        return response.text  # Access response.text safely

    except Exception as e:
        print(f"Error: {e}")
        return "" 

def gemini_multiimage_completion(system_prompt, model_name, prompt, list_content, list_image_base64, token_limit=30000):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel(model_name=model_name)

    content_blocks = []
    for base64_image in list_image_base64:
        content_blocks.append(
            {
                "mime_type": "image/jpeg",
                "data": base64_image,
            },
        )
    
    joined_steps = "\n\n".join(list_content)
    content_blocks.append(
        joined_steps
    )

    messages = content_blocks
            
    try:
        response = model.generate_content(
            messages,
            generation_config=types.GenerationConfig(
                max_output_tokens=token_limit
            )
        )
    except Exception as e:
        print(f"error: {e}")

    generated_str = response.text

    return generated_str


def together_ai_completion(system_prompt, model_name, prompt, base64_image=None, temperature=1, token_limit=30000):
    try:
        # Initialize client without explicitly passing API key
        # It will automatically use TOGETHER_API_KEY environment variable
        client = Together()

        if "qwen3" in model_name.lower() and token_limit > 25000:
            token_limit = 25000
            print(f"qwen3 only supports 40960 tokens, setting token_limit={token_limit} safely excluding input tokens")
        
        if base64_image is not None:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                temperature=temperature,
                max_tokens=token_limit
            )
        else:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                temperature=temperature,
                max_tokens=token_limit
            )

        generated_str = response.choices[0].message.content
        return generated_str
    except Exception as e:
        print(f"Error in together_ai_completion: {e}")
        raise

def together_ai_text_completion(system_prompt, model_name, prompt, temperature=1, token_limit=30000):
    print(f"Together AI text-only API call: model={model_name}")
    try:
        # Initialize client without explicitly passing API key
        # It will automatically use TOGETHER_API_KEY environment variable
        client = Together()

        if "qwen3" in model_name.lower() and token_limit > 25000:
            token_limit = 25000
            print(f"qwen3 only supports 40960 tokens, setting token_limit={token_limit} safely excluding input tokens")
        
        # Format messages with system prompt if provided
        messages = []
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=token_limit
        )
        generated_str = response.choices[0].message.content

        # HACK: resolve temporary generation repetition issue for deepseek-ai/DeepSeek-R1-0528
        import re
        def extract_move(text):
            """
            Extracts the content immediately after the first </think> tag,
            then extracts the content after either 'move:' or '### move' up to the next newline.
            Strips whitespace.
            Returns None if not found.
            """
            # Find the first </think>
            think_match = re.search(r"</think>", text)
            if think_match:
                after_think = text[think_match.end():]
            else:
                after_think = text  # If </think> not found, search the whole text
            
            return after_think.strip()
            # Now extract move after 'move:' or '### move'
            #move_match = re.search(r"(?:move:|### move)\s*(.+?)\s*(?:\\n|\n|$)", after_think)
            #if move_match:
            #    return move_match.group(1).strip()
            #return None

        if model_name == "deepseek-ai/DeepSeek-R1" or model_name == "Qwen/Qwen3-235B-A22B-fp8":
            generated_str = extract_move(generated_str)
        
        return generated_str
    except Exception as e:
        print(f"Error in together_ai_text_completion: {e}")
        raise

def together_ai_multiimage_completion(system_prompt, model_name, prompt, list_content, list_image_base64, temperature=1, token_limit=30000):
    print(f"Together AI multi-image API call: model={model_name}")
    try:
        # Initialize client without explicitly passing API key
        # It will automatically use TOGETHER_API_KEY environment variable
        client = Together()
        
        # Prepare message with multiple images and text
        content_blocks = []

        if "qwen3" in model_name.lower() and token_limit > 25000:
            token_limit = 25000
            print(f"qwen3 only supports 40960 tokens, setting token_limit={token_limit} safely excluding input tokens")
        
        # Add text content
        joined_text = "\n\n".join(list_content)
        content_blocks.append({
            "type": "text",
            "text": joined_text
        })
        
        # Add images
        for base64_image in list_image_base64:
            content_blocks.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"}
            })
        
        # Add final prompt text
        content_blocks.append({
            "type": "text",
            "text": prompt
        })
        
        # Format messages with system prompt if provided
        messages = []
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        messages.append({
            "role": "user",
            "content": content_blocks
        })
        
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=token_limit
        )
        generated_str = response.choices[0].message.content

        return generated_str
    except Exception as e:
        print(f"Error in together_ai_multiimage_completion: {e}")
        raise

def parse_vllm_model_name(model_name: str) -> str:
    """
    Extracts the actual model path from a vLLM-prefixed model name.
    For example, 'vllm-mistralai/Mistral-7B-Instruct-v0.2' becomes 'mistralai/Mistral-7B-Instruct-v0.2'.
    """
    if model_name.startswith("vllm-"):
        return model_name[len("vllm-"):]
    return model_name

def vllm_text_completion(
    system_prompt, 
    vllm_model_name, 
    prompt, 
    token_limit=30000, 
    temperature=1, 
    port=8000,
    host="localhost"
):
    url = f"http://{host}:{port}/v1/chat/completions"
    headers = {"Authorization": "Bearer FAKE_TOKEN"}
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    model_name = parse_vllm_model_name(vllm_model_name)
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": token_limit,
        "temperature": temperature,
        "stream": False
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

def vllm_completion(
    system_prompt,
    vllm_model_name,
    prompt,
    base64_image=None,
    token_limit=30000,
    temperature=1.0,
    port=8000,
    host="localhost"
):
    url = f"http://{host}:{port}/v1/chat/completions"
    headers = {"Authorization": "Bearer FAKE_TOKEN"}

    # Construct the user message content
    if base64_image:
        user_content = [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
            {"type": "text", "text": prompt}
        ]
    else:
        user_content = [{"type": "text", "text": prompt}]

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})

    model_name = parse_vllm_model_name(vllm_model_name)
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": token_limit,
        "temperature": temperature,
        "stream": False
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

def vllm_multiimage_completion(
    system_prompt,
    vllm_model_name,
    prompt,
    list_image_base64,
    token_limit=30000,
    temperature=1.0,
    port=8000,
    host="localhost"
):
    url = f"http://{host}:{port}/v1/chat/completions"
    headers = {"Authorization": "Bearer FAKE_TOKEN"}

    # Construct the user message content with multiple images
    user_content = []
    for image_base64 in list_image_base64:
        user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}})
    user_content.append({"type": "text", "text": prompt})

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})

    model_name = parse_vllm_model_name(vllm_model_name)
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": token_limit,
        "temperature": temperature,
        "stream": False
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def parse_modal_model_name(modal_model_name: str) -> str:
    if modal_model_name.startswith("modal-"):
        return modal_model_name[len("modal-"):]
    return modal_model_name

from openai import OpenAI

def modal_vllm_text_completion(
    system_prompt: str,
    model_name: str,
    prompt: str,
    token_limit: int = 30000,
    temperature: float = 1.0,
    api_key: str = "DUMMY_TOKEN",
    port=8000,
    url: str = "https://your-modal-url.modal.run/v1",
):
    model_name = parse_modal_model_name(model_name)

    # Ensure URL ends with /v1
    if not url.endswith('/v1'):
        url = url + '/v1'

    print(f"calling modal_vllm_text_completion...\nmodel_name: {model_name}\nurl: {url}\n")

    if api_key:
        client = OpenAI(api_key=api_key, base_url=url)
    else:
        client = OpenAI(api_key=os.getenv("MODAL_API_KEY"), base_url=url)
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    if "Qwen2.5-7B" in model_name and token_limit > 20000:
        print("Qwen2.5 7B only supports 32768 tokens")
        token_limit = 20000
    
    if "Qwen2.5-14B" in model_name and token_limit > 30000:
        print("Qwen2.5 14B only supports 32768 tokens")
        token_limit = 30000

    if "Qwen2.5-32B" in model_name and token_limit > 10000:
        token_limit = 10000

    if "Qwen2.5-72B" in model_name and token_limit > 8000:
        token_limit = 8000

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=token_limit,
        temperature=temperature,
    )
    return response.choices[0].message.content

def modal_vllm_completion(
    system_prompt: str,
    model_name: str,
    prompt: str,
    base64_image: str = None,
    token_limit: int = 30000,
    temperature: float = 1.0,
    api_key: str = "DUMMY_TOKEN",
    port=8000,
    url: str = "https://your-modal-url.modal.run/v1",
):
    model_name = parse_modal_model_name(model_name)
    
    # Ensure URL ends with /v1
    if not url.endswith('/v1'):
        url = url + '/v1'
    
    print(f"calling modal_vllm_completion...\nmodel_name: {model_name}\nurl: {url}\n")
    
    if api_key:
        client = OpenAI(api_key=api_key, base_url=url)
    else:
        client = OpenAI(api_key=os.getenv("MODAL_API_KEY"), base_url=url)

    user_content = []
    if base64_image:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{base64_image}"},
        })
    user_content.append({"type": "text", "text": prompt})

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    
    if "Qwen-2.5-7B" in model_name and token_limit > 20000:
        print("Qwen-2.5 7B only supports 32768 tokens")
        token_limit = 20000

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=token_limit,
        temperature=temperature,
    )
    return response.choices[0].message.content

def modal_vllm_multiimage_completion(
    system_prompt: str,
    model_name: str,
    prompt: str,
    list_image_base64: list,
    token_limit: int = 30000,
    temperature: float = 1.0,
    api_key: str = "DUMMY_TOKEN",
    port=8000,
    url: str = "https://your-modal-url.modal.run/v1",
):
    model_name = parse_modal_model_name(model_name)
    
    # Ensure URL ends with /v1
    if not url.endswith('/v1'):
        url = url + '/v1'
    
    print(f"calling modal_multiimage_vllm_completion...\nmodel_name: {model_name}\nurl: {url}\n")
    
    if api_key:
        client = OpenAI(api_key=api_key, base_url=url)
    else:
        client = OpenAI(api_key=os.getenv("MODAL_API_KEY"), base_url=url)

    user_content = []
    for base64_image in list_image_base64:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{base64_image}"}
        })
    user_content.append({"type": "text", "text": prompt})

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})

    if "Qwen-2.5-7B" in model_name and token_limit > 20000:
        print("Qwen-2.5 7B only supports 32768 tokens")
        token_limit = 20000

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=token_limit,
        temperature=temperature,
    )
    return response.choices[0].message.content


# ======== MOONSHOT AI KIMI API INTEGRATION ========

def retry_on_moonshot_error(func):
    """
    Retry wrapper for Moonshot AI SDK calls.
    Retries on: RateLimitError, Timeout, APIConnectionError,
                APIStatusError (5xx), httpx.RemoteProtocolError.
    Immediately raises on: BadRequestError (400).
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        max_retries = kwargs.pop("max_retries", 5)
        base_delay  = kwargs.pop("base_delay", 2)

        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)

            # BadRequestError should NOT be retried - it indicates invalid request
            except BadRequestError as e:
                print(f"Moonshot AI BadRequestError (not retrying): {e}")
                raise

            # transient issues worth retrying
            except (RateLimitError, APITimeoutError, APIConnectionError,
                    httpx.RemoteProtocolError) as e:
                if attempt < max_retries - 1:
                    print(f"Moonshot AI transient error: {e}")
                    _sleep_with_backoff(base_delay, attempt)
                    continue
                raise

            # server‑side 5xx response
            except APIStatusError as e:
                if 500 <= e.status_code < 600 and attempt < max_retries - 1:
                    print(f"Moonshot AI server error {e.status_code}: {e.message}")
                    _sleep_with_backoff(base_delay, attempt)
                    continue
                raise

    return wrapper

@retry_on_moonshot_error
def moonshot_text_completion(system_prompt, model_name, prompt, temperature=1, token_limit=30000):
    """
    Moonshot AI Kimi text completion API call.
    
    Args:
        system_prompt (str): System prompt
        model_name (str): Model name (e.g., "moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k")
        prompt (str): User prompt
        temperature (float): Temperature parameter (0-1)
        token_limit (int): Maximum number of tokens for the completion response
        
    Returns:
        str: Generated text
    """
    print(f"Moonshot AI text API call: model={model_name}")
    
    # Use OpenAI client with Moonshot base URL
    client = OpenAI(
        api_key=os.getenv("MOONSHOT_API_KEY"),
        base_url="https://api.moonshot.ai/v1"
    )
    
    # Determine model's total context limit
    total_context_limit = 8000
    if "8k" in model_name:
        total_context_limit = 8000
    elif "32k" in model_name:
        total_context_limit = 32000
    elif "128k" in model_name:
        total_context_limit = 128000
    elif "kimi-k2" in model_name.lower():
        total_context_limit = 128000
    elif "kimi-thinking-preview" in model_name.lower():
        total_context_limit = 128000

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    # Estimate input token count
    total_input_text = (system_prompt or "") + prompt
    estimated_input_tokens = estimate_token_count(total_input_text)
    
    # Calculate available tokens for output (leave some buffer)
    buffer_tokens = 100  # Safety buffer
    available_output_tokens = total_context_limit - estimated_input_tokens - buffer_tokens
    
    # Adjust token_limit to not exceed available space
    if available_output_tokens <= 0:
        # Input is too long, truncate the prompt
        max_input_tokens = total_context_limit - token_limit - buffer_tokens
        if max_input_tokens <= 0:
            max_input_tokens = total_context_limit // 2  # Use half for input, half for output
            token_limit = total_context_limit - max_input_tokens - buffer_tokens
        
        # Truncate the user prompt (keep system prompt intact)
        max_prompt_chars = max_input_tokens * 4 - len(system_prompt or "")
        if max_prompt_chars > 0:
            truncated_prompt = prompt[:max_prompt_chars]
            print(f"Warning: Prompt truncated from {len(prompt)} to {len(truncated_prompt)} characters to fit context limit")
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": truncated_prompt})
        else:
            raise ValueError(f"System prompt too long ({len(system_prompt)} chars) for model {model_name}")
    else:
        # Limit output tokens to available space
        token_limit = min(token_limit, available_output_tokens)
    
    print(f"Model: {model_name}, Context limit: {total_context_limit}, Estimated input tokens: {estimated_input_tokens}, Output token limit: {token_limit}")

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=token_limit,
            temperature=temperature,
        )
        
        # Debug: Print the response structure
        print(f"Moonshot API response: {response}")
        
        # Validate response structure
        if not hasattr(response, 'choices') or not response.choices:
            raise ValueError(f"Invalid response from Moonshot API: no choices found. Response: {response}")
        
        if not hasattr(response.choices[0], 'message') or not hasattr(response.choices[0].message, 'content'):
            raise ValueError(f"Invalid response structure from Moonshot API. Choice: {response.choices[0]}")
        
        return response.choices[0].message.content
        
    except BadRequestError as e:
        if "token limit" in str(e).lower():
            # If we still hit token limit, try with even smaller output limit
            reduced_token_limit = token_limit // 2
            print(f"Hit token limit, retrying with reduced output limit: {reduced_token_limit}")
            
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=reduced_token_limit,
                temperature=temperature,
            )
            return response.choices[0].message.content
        else:
            raise  # Re-raise if it's not a token limit error

@retry_on_moonshot_error  
def moonshot_completion(system_prompt, model_name, base64_image, prompt, temperature=1, token_limit=30000):
    """
    Moonshot AI Kimi vision-text completion API call.
    
    Args:
        system_prompt (str): System prompt
        model_name (str): Model name (e.g., "moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k")
        base64_image (str): Base64-encoded image data
        prompt (str): User prompt
        temperature (float): Temperature parameter (0-1)
        token_limit (int): Maximum number of tokens for the completion response
        
    Returns:
        str: Generated text
    """
    print(f"Moonshot AI vision-text API call: model={model_name}")
    
    # Check if model supports vision
    if "kimi-k2" in model_name.lower():
        raise ValueError(f"Model {model_name} does not support vision functionality. Use text-only completion instead.")
    
    # Use OpenAI client with Moonshot base URL
    client = OpenAI(
        api_key=os.getenv("MOONSHOT_API_KEY"),
        base_url="https://api.moonshot.ai/v1"
    )
    
    # Determine model's total context limit
    total_context_limit = 8000
    if "8k" in model_name:
        total_context_limit = 8000
    elif "32k" in model_name:
        total_context_limit = 32000
    elif "128k" in model_name:
        total_context_limit = 128000
    elif "kimi-k2" in model_name.lower():
        total_context_limit = 128000
    elif "kimi-thinking-preview" in model_name.lower():
        total_context_limit = 128000

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    messages.append({
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
            {"type": "text", "text": prompt},
        ],
    })

    # Estimate input token count (images typically use ~1000-2000 tokens)
    estimated_image_tokens = 1500  # Conservative estimate for image tokens
    total_input_text = (system_prompt or "") + prompt
    estimated_text_tokens = estimate_token_count(total_input_text)
    estimated_input_tokens = estimated_text_tokens + estimated_image_tokens
    
    # Calculate available tokens for output (leave some buffer)
    buffer_tokens = 200  # Larger buffer for vision tasks
    available_output_tokens = total_context_limit - estimated_input_tokens - buffer_tokens
    
    # Adjust token_limit to not exceed available space
    if available_output_tokens <= 0:
        # Input is too long, truncate the prompt
        max_input_tokens = total_context_limit - token_limit - buffer_tokens
        if max_input_tokens <= estimated_image_tokens:
            # Even with truncation, we can't fit the image
            max_input_tokens = total_context_limit // 2  # Use half for input, half for output
            token_limit = total_context_limit - max_input_tokens - buffer_tokens
        
        # Truncate the user prompt (keep system prompt and image intact)
        max_prompt_chars = (max_input_tokens - estimated_image_tokens) * 4 - len(system_prompt or "")
        if max_prompt_chars > 0:
            truncated_prompt = prompt[:max_prompt_chars]
            print(f"Warning: Prompt truncated from {len(prompt)} to {len(truncated_prompt)} characters to fit context limit")
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                    {"type": "text", "text": truncated_prompt},
                ],
            })
        else:
            raise ValueError(f"System prompt and image too large for model {model_name}")
    else:
        # Limit output tokens to available space
        token_limit = min(token_limit, available_output_tokens)
    
    print(f"Model: {model_name}, Context limit: {total_context_limit}, Estimated input tokens: {estimated_input_tokens} (text: {estimated_text_tokens}, image: {estimated_image_tokens}), Output token limit: {token_limit}")

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=token_limit,
            temperature=temperature,
        )
        
        # Debug: Print the response structure
        print(f"Moonshot vision API response: {response}")
        
        # Validate response structure
        if not hasattr(response, 'choices') or not response.choices:
            raise ValueError(f"Invalid response from Moonshot API: no choices found. Response: {response}")
        
        if not hasattr(response.choices[0], 'message') or not hasattr(response.choices[0].message, 'content'):
            raise ValueError(f"Invalid response structure from Moonshot API. Choice: {response.choices[0]}")
        
        return response.choices[0].message.content
        
    except BadRequestError as e:
        if "token limit" in str(e).lower():
            # If we still hit token limit, try with even smaller output limit
            reduced_token_limit = token_limit // 2
            print(f"Hit token limit, retrying with reduced output limit: {reduced_token_limit}")
            
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=reduced_token_limit,
                temperature=temperature,
            )
            return response.choices[0].message.content
        else:
            raise  # Re-raise if it's not a token limit error

@retry_on_moonshot_error
def moonshot_multiimage_completion(system_prompt, model_name, prompt, list_content, list_image_base64, temperature=1, token_limit=30000):
    """
    Moonshot AI Kimi multi-image completion API call.
    
    Args:
        system_prompt (str): System prompt
        model_name (str): Model name (e.g., "moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k")
        prompt (str): User prompt
        list_content (List[str]): List of text content corresponding to each image
        list_image_base64 (List[str]): List of base64-encoded image data
        temperature (float): Temperature parameter (0-1)
        token_limit (int): Maximum number of tokens for the completion response
        
    Returns:
        str: Generated text
    """
    print(f"Moonshot AI multi-image API call: model={model_name}")
    
    # Check if model supports vision
    if "kimi-k2" in model_name.lower():
        raise ValueError(f"Model {model_name} does not support vision functionality. Use text-only completion instead.")
    
    # Use OpenAI client with Moonshot base URL
    client = OpenAI(
        api_key=os.getenv("MOONSHOT_API_KEY"),
        base_url="https://api.moonshot.ai/v1"
    )
    
    # Determine model's total context limit
    total_context_limit = 8000
    if "8k" in model_name:
        total_context_limit = 8000
    elif "32k" in model_name:
        total_context_limit = 32000
    elif "128k" in model_name:
        total_context_limit = 128000
    elif "kimi-k2" in model_name.lower():
        total_context_limit = 128000
    elif "kimi-thinking-preview" in model_name.lower():
        total_context_limit = 128000

    content_blocks = []
    
    # Add text content and corresponding images
    for text_item, base64_image in zip(list_content, list_image_base64):
        content_blocks.append({
            "type": "text",
            "text": text_item,
        })
        content_blocks.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{base64_image}"}
        })
    
    # Add final prompt
    content_blocks.append({
        "type": "text",
        "text": prompt
    })

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    messages.append({
        "role": "user",
        "content": content_blocks,
    })

    # Estimate input token count
    num_images = len(list_image_base64)
    estimated_image_tokens = num_images * 1500  # Conservative estimate per image
    total_input_text = (system_prompt or "") + prompt + " ".join(list_content)
    estimated_text_tokens = estimate_token_count(total_input_text)
    estimated_input_tokens = estimated_text_tokens + estimated_image_tokens
    
    # Calculate available tokens for output (leave some buffer)
    buffer_tokens = 300  # Larger buffer for multi-image tasks
    available_output_tokens = total_context_limit - estimated_input_tokens - buffer_tokens
    
    # Adjust token_limit to not exceed available space
    if available_output_tokens <= 0:
        # Input is too long, try to reduce
        max_input_tokens = total_context_limit - token_limit - buffer_tokens
        if max_input_tokens <= estimated_image_tokens:
            # Even with truncation, we can't fit all images
            max_input_tokens = total_context_limit // 2  # Use half for input, half for output
            token_limit = total_context_limit - max_input_tokens - buffer_tokens
        
        # Try to reduce the number of images or truncate text
        if num_images > 1 and estimated_image_tokens > max_input_tokens // 2:
            # Reduce number of images
            max_images = max(1, (max_input_tokens // 2) // 1500)
            print(f"Warning: Reducing images from {num_images} to {max_images} to fit context limit")
            list_image_base64 = list_image_base64[:max_images]
            list_content = list_content[:max_images]
            
            # Rebuild content blocks with fewer images
            content_blocks = []
            for text_item, base64_image in zip(list_content, list_image_base64):
                content_blocks.append({
                    "type": "text",
                    "text": text_item,
                })
                content_blocks.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                })
            content_blocks.append({
                "type": "text",
                "text": prompt
            })
            
            # Update messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({
                "role": "user",
                "content": content_blocks,
            })
            
            # Recalculate tokens
            estimated_image_tokens = len(list_image_base64) * 1500
            estimated_input_tokens = estimated_text_tokens + estimated_image_tokens
        
        # If still too large, truncate text content
        available_text_tokens = max_input_tokens - estimated_image_tokens
        if estimated_text_tokens > available_text_tokens:
            # Truncate text content proportionally
            reduction_factor = available_text_tokens / estimated_text_tokens
            truncated_content = []
            for text_item in list_content:
                truncated_length = int(len(text_item) * reduction_factor)
                truncated_content.append(text_item[:truncated_length])
            
            truncated_prompt_length = int(len(prompt) * reduction_factor)
            truncated_prompt = prompt[:truncated_prompt_length]
            
            print(f"Warning: Text content truncated by factor {reduction_factor:.2f} to fit context limit")
            
            # Rebuild content blocks with truncated text
            content_blocks = []
            for text_item, base64_image in zip(truncated_content, list_image_base64):
                content_blocks.append({
                    "type": "text",
                    "text": text_item,
                })
                content_blocks.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                })
            content_blocks.append({
                "type": "text",
                "text": truncated_prompt
            })
            
            # Update messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({
                "role": "user",
                "content": content_blocks,
            })
    else:
        # Limit output tokens to available space
        token_limit = min(token_limit, available_output_tokens)
    
    print(f"Model: {model_name}, Context limit: {total_context_limit}, Estimated input tokens: {estimated_input_tokens} (text: {estimated_text_tokens}, images: {estimated_image_tokens}), Output token limit: {token_limit}")

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=token_limit,
            temperature=temperature,
        )
        
        # Debug: Print the response structure
        print(f"Moonshot multiimage API response: {response}")
        
        # Validate response structure
        if not hasattr(response, 'choices') or not response.choices:
            raise ValueError(f"Invalid response from Moonshot API: no choices found. Response: {response}")
        
        if not hasattr(response.choices[0], 'message') or not hasattr(response.choices[0].message, 'content'):
            raise ValueError(f"Invalid response structure from Moonshot API. Choice: {response.choices[0]}")
        
        return response.choices[0].message.content
        
    except BadRequestError as e:
        if "token limit" in str(e).lower():
            # If we still hit token limit, try with even smaller output limit
            reduced_token_limit = max(100, token_limit // 2)  # Ensure minimum viable output
            print(f"Hit token limit, retrying with reduced output limit: {reduced_token_limit}")
            
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=reduced_token_limit,
                temperature=temperature,
            )
            return response.choices[0].message.content
        else:
            raise  # Re-raise if it's not a token limit error

# ======== OPENROUTER KIMI K2 INTEGRATION ========

@retry_on_openai_error
def openrouter_kimi_text_completion(system_prompt, model_name, prompt, temperature=1, token_limit=30000):
    """
    OpenRouter Kimi K2 text completion API call.
    
    Args:
        system_prompt (str): System prompt
        model_name (str): Model name (should be "moonshotai/kimi-k2")
        prompt (str): User prompt
        temperature (float): Temperature parameter (0-1)
        token_limit (int): Maximum number of tokens for the completion response
        
    Returns:
        str: Generated text
    """
    print(f"OpenRouter Kimi K2 text API call: model={model_name}")
    
    # Use OpenAI client with OpenRouter base URL
    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1"
    )
    
    # Determine model's total context limit
    total_context_limit = 128000  # Default for Kimi models
    if "kimi-k2" in model_name.lower():
        total_context_limit = 128000
    elif "kimi-thinking-preview" in model_name.lower():
        total_context_limit = 128000

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    # Estimate input token count
    total_input_text = (system_prompt or "") + prompt
    estimated_input_tokens = estimate_token_count(total_input_text)
    
    # Calculate available tokens for output (leave some buffer)
    buffer_tokens = 100  # Safety buffer
    available_output_tokens = total_context_limit - estimated_input_tokens - buffer_tokens
    
    # Adjust token_limit to not exceed available space
    if available_output_tokens <= 0:
        # Input is too long, truncate the prompt
        max_input_tokens = total_context_limit - token_limit - buffer_tokens
        if max_input_tokens <= 0:
            max_input_tokens = total_context_limit // 2  # Use half for input, half for output
            token_limit = total_context_limit - max_input_tokens - buffer_tokens
        
        # Truncate the user prompt (keep system prompt intact)
        max_prompt_chars = max_input_tokens * 4 - len(system_prompt or "")
        if max_prompt_chars > 0:
            truncated_prompt = prompt[:max_prompt_chars]
            print(f"Warning: Prompt truncated from {len(prompt)} to {len(truncated_prompt)} characters to fit context limit")
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": truncated_prompt})
        else:
            raise ValueError(f"System prompt too long ({len(system_prompt)} chars) for model {model_name}")
    else:
        # Limit output tokens to available space
        token_limit = min(token_limit, available_output_tokens)
    
    print(f"Model: {model_name}, Context limit: {total_context_limit}, Estimated input tokens: {estimated_input_tokens}, Output token limit: {token_limit}")

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=token_limit,
            temperature=temperature,
        )
        
        return response.choices[0].message.content
        
    except BadRequestError as e:
        if "token limit" in str(e).lower():
            # If we still hit token limit, try with even smaller output limit
            reduced_token_limit = token_limit // 2
            print(f"Hit token limit, retrying with reduced output limit: {reduced_token_limit}")
            
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=reduced_token_limit,
                temperature=temperature,
            )
            return response.choices[0].message.content
        else:
            raise  # Re-raise if it's not a token limit error

@retry_on_openai_error
def openrouter_kimi_completion(system_prompt, model_name, base64_image, prompt, temperature=1, token_limit=30000):
    """
    OpenRouter Kimi K2 vision-text completion API call.
    
    Args:
        system_prompt (str): System prompt
        model_name (str): Model name (should be "moonshotai/kimi-k2")
        base64_image (str): Base64-encoded image data
        prompt (str): User prompt
        temperature (float): Temperature parameter (0-1)
        token_limit (int): Maximum number of tokens for the completion response
        
    Returns:
        str: Generated text
    """
    print(f"OpenRouter Kimi K2 vision-text API call: model={model_name}")
    
    # Check if model supports vision
    if "kimi-k2" in model_name.lower():
        raise ValueError(f"Model {model_name} does not support vision functionality. Use text-only completion instead.")
    
    # Use OpenAI client with OpenRouter base URL
    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1"
    )
    
    # Kimi K2 supports up to 63K context, but we'll be conservative with output tokens
    if token_limit > 128000:
        print("Kimi K2 supports up to 63K context, limiting output tokens to 128K for safety")
        token_limit = 128000
    elif "kimi-thinking-preview" in model_name.lower() and token_limit > 128000:
        print("kimi-thinking-preview supports up to 128K context, limiting output tokens to 128K")
        token_limit = 128000

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    messages.append({
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
            {"type": "text", "text": prompt},
        ],
    })

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=token_limit,
        temperature=temperature,
    )
    
    return response.choices[0].message.content

@retry_on_openai_error
def openrouter_kimi_multiimage_completion(system_prompt, model_name, prompt, list_content, list_image_base64, temperature=1, token_limit=30000):
    """
    OpenRouter Kimi K2 multi-image completion API call.
    
    Args:
        system_prompt (str): System prompt
        model_name (str): Model name (should be "moonshotai/kimi-k2")
        prompt (str): User prompt
        list_content (List[str]): List of text content corresponding to each image
        list_image_base64 (List[str]): List of base64-encoded image data
        temperature (float): Temperature parameter (0-1)
        token_limit (int): Maximum number of tokens for the completion response
        
    Returns:
        str: Generated text
    """
    print(f"OpenRouter Kimi K2 multi-image API call: model={model_name}")
    
    # Check if model supports vision
    if "kimi-k2" in model_name.lower():
        raise ValueError(f"Model {model_name} does not support vision functionality. Use text-only completion instead.")
    
    # Use OpenAI client with OpenRouter base URL
    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1"
    )
    
    # Kimi K2 supports up to 63K context, but we'll be conservative with output tokens
    if token_limit > 128000:
        print("Kimi K2 supports up to 128K context, limiting output tokens to 128K for safety")
        token_limit = 128000
    elif "kimi-thinking-preview" in model_name.lower() and token_limit > 128000:
        print("kimi-thinking-preview supports up to 128K context, limiting output tokens to 128K")
        token_limit = 128000

    content_blocks = []
    
    # Add text content and corresponding images
    for text_item, base64_image in zip(list_content, list_image_base64):
        content_blocks.append({
            "type": "text",
            "text": text_item,
        })
        content_blocks.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{base64_image}"}
        })
    
    # Add final prompt
    content_blocks.append({
        "type": "text",
        "text": prompt
    })

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    messages.append({
        "role": "user",
        "content": content_blocks,
    })

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=token_limit,
        temperature=temperature,
    )
    
    return response.choices[0].message.content
