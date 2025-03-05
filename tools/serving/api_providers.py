import os

from openai import OpenAI
import anthropic
import google.generativeai as genai

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

    # Determine correct token parameter
    token_param = "max_completion_tokens" if "o3-mini" in model_name else "max_tokens"

    # # Correct parameter for different models
    # "reasoning_effort": "low"
    
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
    print(response)

    generated_code_str = response.choices[0].message.content
     
    return generated_code_str

def anthropic_completion(system_prompt, model_name, base64_image, prompt, thinking=False):
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
            messages,
        )
    except Exception as e:
        print(f"error: {e}")

    generated_code_str = response.text

    return generated_code_str