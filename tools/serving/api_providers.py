import os

from openai import OpenAI
import anthropic
import google.generativeai as genai
from google.generativeai import types
from together import Together

def anthropic_completion(system_prompt, model_name, base64_image, prompt, thinking=False):
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
    if thinking:
        with client.messages.stream(
                max_tokens=20000,
                thinking={
                    "type": "enabled",
                    "budget_tokens": 16000
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
                max_tokens=1024,
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

def anthropic_text_completion(system_prompt, model_name, prompt, thinking=False):
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
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
    if thinking:
        with client.messages.stream(
                max_tokens=20000,
                thinking={
                    "type": "enabled",
                    "budget_tokens": 16000
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
                max_tokens=1024,
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


def anthropic_multiimage_completion(system_prompt, model_name, prompt, list_content, list_image_base64):
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
            max_tokens=1024,
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


def openai_completion(system_prompt, model_name, base64_image, prompt, temperature=0):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

    # Set request parameters based on model type
    request_params = {
        "model": model_name,
        "messages": messages,
    }

    # Add the correct token parameter based on model
    if "o3" in model_name or "o4" in model_name or "o1" in model_name:
        request_params["max_completion_tokens"] = 100000
    else:
        request_params["max_tokens"] = 4096

    # Add temperature parameter (only for models that support it)
    if "o3" not in model_name and "o4" not in model_name and "o1" not in model_name:
        request_params["temperature"] = temperature

    response = client.chat.completions.create(**request_params)
    return response.choices[0].message.content

def openai_text_completion(system_prompt, model_name, prompt):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0,
        max_tokens=1024,
    )

    generated_str = response.choices[0].message.content
     
    return generated_str

def openai_text_reasoning_completion(system_prompt, model_name, prompt, temperature=0):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
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

    # Determine correct token parameter
    token_param = "max_completion_tokens" if "o3-mini" in model_name else "max_tokens"
    
    # Prepare request parameters dynamically
    request_params = {
        "model": model_name,
        "messages": messages,
        token_param: 100000,
        "reasoning_effort": "medium"
    }
    
    # Only add 'temperature' if the model supports it
    if "o3-mini" not in model_name:  # Assuming o3-mini doesn't support 'temperature'
        request_params["temperature"] = temperature

    response = client.chat.completions.create(**request_params)

    generated_str = response.choices[0].message.content
     
    return generated_str

def deepseek_text_reasoning_completion(system_prompt, model_name, prompt):
     
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
        max_tokens=8000)
    
    for chunk in response:
        if chunk.choices[0].delta.reasoning_content and chunk.choices[0].delta.reasoning_content:
            reasoning_content += chunk.choices[0].delta.reasoning_content
        elif hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content:
            content += chunk.choices[0].delta.content
    
    # generated_str = response.choices[0].message.content
    print(content)
    return content
    


def openai_multiimage_completion(system_prompt, model_name, prompt, list_content, list_image_base64):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
    
    # Set request parameters based on model type
    request_params = {
        "model": model_name,
        "messages": messages,
        "temperature": 0,
    }
    
    # Add the correct token parameter based on model
    if "o3" in model_name or "o4" in model_name or "o1" in model_name:
        request_params["max_completion_tokens"] = 100000
    else:
        request_params["max_tokens"] = 4096
    
    response = client.chat.completions.create(**request_params)

    generated_str = response.choices[0].message.content
     
    return generated_str


def gemini_text_completion(system_prompt, model_name, prompt):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel(model_name=model_name)

    messages = [
        prompt,
    ]
            
    try:
        response = model.generate_content(
            messages
        )
    except Exception as e:
        print(f"error: {e}")

    try:
        response = model.generate_content(messages)

        # Ensure response is valid and contains candidates
        if not response or not hasattr(response, "candidates") or not response.candidates:
            print("Warning: Empty or invalid response")
            return ""
        
        return response.text  # Access response.text safely

    except Exception as e:
        print(f"Error: {e}")
        return "" 

def anthropic_text_completion(system_prompt, model_name, prompt, thinking=False):
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
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
    if thinking:
        with client.messages.stream(
                max_tokens=20000,
                thinking={
                    "type": "enabled",
                    "budget_tokens": 16000
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
                max_tokens=1024,
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

def gemini_text_completion(system_prompt, model_name, prompt):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel(model_name=model_name)

    messages = [
        prompt,
    ]
            
    try:
        response = model.generate_content(
            messages
        )
    except Exception as e:
        print(f"error: {e}")

    try:
        response = model.generate_content(messages)

        # Ensure response is valid and contains candidates
        if not response or not hasattr(response, "candidates") or not response.candidates:
            print("Warning: Empty or invalid response")
            return ""
        
        return response.text  # Access response.text safely

    except Exception as e:
        print(f"Error: {e}")
        return "" 

def gemini_completion(system_prompt, model_name, base64_image, prompt):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel(model_name=model_name)

    messages = [
        {
            "mime_type": "image/jpeg",
            "data": base64_image,
        },
        prompt,
    ]
            
    try:
        response = model.generate_content(
            messages
        )
    except Exception as e:
        print(f"error: {e}")

    try:
        response = model.generate_content(messages)

        # Ensure response is valid and contains candidates
        if not response or not hasattr(response, "candidates") or not response.candidates:
            print("Warning: Empty or invalid response")
            return ""
        
        return response.text  # Access response.text safely

    except Exception as e:
        print(f"Error: {e}")
        return "" 

def gemini_multiimage_completion(system_prompt, model_name, prompt, list_content, list_image_base64):
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
        )
    except Exception as e:
        print(f"error: {e}")

    generated_str = response.text

    return generated_str


def deepseek_text_reasoning_completion(system_prompt, model_name, prompt):
     
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
        max_tokens=8000)
    
    for chunk in response:
        if chunk.choices[0].delta.reasoning_content and chunk.choices[0].delta.reasoning_content:
            reasoning_content += chunk.choices[0].delta.reasoning_content
        elif hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content:
            content += chunk.choices[0].delta.content
    
    # generated_str = response.choices[0].message.content
    return content


def together_ai_completion(system_prompt, model_name, prompt, base64_image=None, temperature=0):
    client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
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
            temperature=temperature
        )
    else:
        response = client.chat.completions.create(
            model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
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
            temperature=temperature
        )

    generated_str = response.choices[0].message.content
     
    return generated_str


def xai_grok_completion(system_prompt, model_name, prompt, temperature=0, reasoning_effort="medium"):
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
    }
    
    # Add reasoning_effort only for models that support it
    if "grok-3-mini" in model_name:
        params["reasoning_effort"] = reasoning_effort
    
    completion = client.chat.completions.create(**params)
    
    # Return just the content for consistency with other completion functions
    return completion.choices[0].message.content
    