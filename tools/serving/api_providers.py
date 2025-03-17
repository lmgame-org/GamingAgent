import os

from openai import OpenAI
import anthropic
import google.generativeai as genai
from google.genai import types

def openai_completion(system_prompt, model_name, base64_image, prompt, temperature=0):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    base64_image = None if "o3-mini" in model_name else base64_image
    if base64_image is None:
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
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt
                    },
                ],
            }
        ]

    token_param = "max_completion_tokens" if "o3-mini" in model_name else "max_tokens"
    # Prepare request parameters dynamically
    request_params = {
        "model": model_name,
        "messages": messages,
        token_param: 4096
    }
    
    if "o3-mini" not in model_name:  # Assuming o3-mini doesn't support 'temperature'
        request_params["temperature"] = temperature

    response = client.chat.completions.create(**request_params)

    generated_str = response.choices[0].message.content
     
    return generated_str

def openai_vision_reasoning_completion(system_prompt, model_name, base64_image, prompt, temperature=0):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    if base64_image is None:
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
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt
                    },
                ],
            }
        ]

    # Determine correct token parameter
    token_param = "max_completion_tokens"
    
    # Prepare request parameters dynamically
    request_params = {
        "model": model_name,
        "messages": messages,
        token_param: 100000,
        "reasoning_effort": "medium"
    }
    

    response = client.chat.completions.create(**request_params)

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