import os

from openai import OpenAI
import anthropic
import google.generativeai as genai
from google.generativeai import types
from together import Together

def anthropic_completion(system_prompt, model_name, base64_image, prompt, thinking=False, token_limit=30000):
    print(f"anthropic vision-text activated... thinking: f{thinking}")
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
                for chunk in stream.text_stream:
                    partial_chunks.append(chunk)
    else:
         
        with client.messages.stream(
                max_tokens=token_limit,
                messages=messages,
                temperature=0,
                system=system_prompt,
                model=model_name, # claude-3-5-sonnet-20241022 # claude-3-7-sonnet-20250219
            ) as stream:
                partial_chunks = []
                for chunk in stream.text_stream:
                    partial_chunks.append(chunk)
        
    generated_code_str = "".join(partial_chunks)
    
    return generated_code_str

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
                for chunk in stream.text_stream:
                    partial_chunks.append(chunk)
    else:    
        with client.messages.stream(
                max_tokens=token_limit,
                messages=messages,
                temperature=0,
                system=system_prompt,
                model=model_name, # claude-3-5-sonnet-20241022 # claude-3-7-sonnet-20250219
            ) as stream:
                partial_chunks = []
                for chunk in stream.text_stream:
                    partial_chunks.append(chunk)
        
    generated_str = "".join(partial_chunks)
    
    return generated_str


def anthropic_multiimage_completion(system_prompt, model_name, prompt, list_content, list_image_base64, token_limit=30000):
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
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
            for chunk in stream.text_stream:
                print(chunk)
                partial_chunks.append(chunk)
        
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


def openai_completion(system_prompt, model_name, base64_image, prompt, temperature=1, token_limit=30000, reasoning_effort="medium"):
    print(f"OpenAI vision-text API call: model={model_name}, reasoning_effort={reasoning_effort}")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if "gpt-4o" in model_name:
        print("gpt-4o only supports 16384 tokens")
        token_limit = 16384
    if "gpt-4.1" in model_name:
        print("gpt-4.1 only supports 32768 tokens")
        token_limit = 32768

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

def openai_text_completion(system_prompt, model_name, prompt, token_limit=30000, reasoning_effort="medium"):
    print(f"OpenAI text-only API call: model={model_name}, reasoning_effort={reasoning_effort}")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if "gpt-4o" in model_name:
        print("gpt-4o only supports 16384 tokens")
        token_limit = 16384
    if "gpt-4.1" in model_name:
        print("gpt-4.1 only supports 32768 tokens")
        token_limit = 32768

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

    response = client.chat.completions.create(**request_params)

    generated_str = response.choices[0].message.content
     
    return generated_str

def openai_text_reasoning_completion(system_prompt, model_name, prompt, temperature=1, token_limit=30000, reasoning_effort="medium"):
    print(f"OpenAI text-reasoning API call: model={model_name}, reasoning_effort={reasoning_effort}")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if "gpt-4o" in model_name:
        print("gpt-4o only supports 16384 tokens")
        token_limit = 16384
    if "gpt-4.1" in model_name:
        print("gpt-4.1 only supports 32768 tokens")
        token_limit = 32768
    
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
        if chunk.choices[0].delta.reasoning_content and chunk.choices[0].delta.reasoning_content:
            reasoning_content += chunk.choices[0].delta.reasoning_content
        elif hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content:
            content += chunk.choices[0].delta.content
    
    # generated_str = response.choices[0].message.content

    print("========== Raw String ==========")
    print(content)
    print("========== Raw String ==========")
    
    return content
    

def xai_grok_completion(system_prompt, model_name, prompt, reasoning_effort="high", token_limit=30000, temperature=1):
    print(f"XAI Grok text API call: model={model_name}, reasoning_effort={reasoning_effort}")
    
    client = OpenAI(
        api_key=os.getenv("XAI_API_KEY"),
        base_url="https://api.x.ai/v1",
    )
    
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]
    
    # Only include reasoning_effort for supported models
    params = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": token_limit
    }
    
    # Add reasoning_effort only for models that support it
    if "grok-3-mini" in model_name:
        params["reasoning_effort"] = reasoning_effort
    
    completion = client.chat.completions.create(**params)
    
    # Return just the content for consistency with other completion functions
    return completion.choices[0].message.content

def openai_multiimage_completion(system_prompt, model_name, prompt, list_content, list_image_base64, token_limit=30000, reasoning_effort="medium"):
    print(f"OpenAI multi-image API call: model={model_name}, reasoning_effort={reasoning_effort}")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if "gpt-4o" in model_name:
        print("gpt-4o only supports 16384 tokens")
        token_limit = 16384
    if "gpt-4.1" in model_name:
        print("gpt-4.1 only supports 32768 tokens")
        token_limit = 32768

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

        print("========== Raw String ==========")
        print(generated_str)
        print("========== Raw String ==========")

        if model_name == "deepseek-ai/DeepSeek-R1":
            generated_str = extract_move(generated_str)

        print("========== Processed String ==========")
        print(generated_str)
        print("========== Processed String ==========")
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