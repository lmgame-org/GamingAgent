"""
API Manager for handling API calls, cost calculation, and logging.

This class provides an object-oriented approach to:
1. Make API calls to various providers (OpenAI, Anthropic, Gemini)
2. Calculate token usage and costs
3. Log API calls, responses, and costs in a structured format
4. Store inputs and outputs in JSON format
"""

import os
import json
import time
import logging
import datetime
import base64
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Union, Any

# Import API providers
from .api_providers import (
    anthropic_completion,
    anthropic_text_completion,
    anthropic_multiimage_completion,
    openai_completion,
    openai_text_completion,
    openai_multiimage_completion,
    gemini_completion,
    gemini_text_completion,
    gemini_multiimage_completion,
    together_ai_completion,
    together_ai_text_completion,
    together_ai_multiimage_completion,
    deepseek_text_reasoning_completion,
    xai_grok_text_completion,
    vllm_text_completion,
    vllm_completion,
    vllm_multiimage_completion,
    modal_vllm_text_completion,
    modal_vllm_completion,
    modal_vllm_multiimage_completion,
)

# Import cost calculator utilities
from .api_cost_calculator import (
    calculate_all_costs_and_tokens,
    count_message_tokens,
    count_string_tokens,
    count_image_tokens,
    calculate_cost_by_tokens,
    calculate_prompt_cost,
    calculate_completion_cost,
    calculate_image_cost,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    # Set handlers to empty list to prevent console output
    handlers=[]
)
logger = logging.getLogger(__name__)


class APIManager:
    """
    Object-oriented manager for API calls with cost tracking and logging.
    
    This class centralizes all API calls, handles token counting, cost calculation,
    and provides structured logging for inputs, outputs, and costs.
    """
    
    def __init__(
        self, 
        game_name: str, 
        base_cache_dir: str = "cache",
        enable_logging: bool = True,
        info: Optional[Dict[str, Any]] = None,
        session_dir: Optional[str] = None,
        vllm_url: Optional[str] = None,
        modal_url: Optional[str] = None
    ):
        """
        Initialize the API Manager.
        
        Args:
            game_name (str): Name of the game/application (e.g., "ace_attorney")
            base_cache_dir (str): Base directory for all cache files
            enable_logging (bool): Whether to enable logging
            info (Dict, optional): Additional information for customizing directory structure
                                   Can include 'model_name', 'modality', 'datetime', etc.
            session_dir (str, optional): Optional path to an existing session directory.
                                         If provided, the directory structure setup is skipped.
        """
        self.game_name = game_name
        self.base_cache_dir = base_cache_dir
        self.enable_logging = enable_logging
        self.info = info or {}

        self.vllm_url = vllm_url
        self.modal_url = modal_url
        
        # Create timestamp for this session (use from info if provided)
        self.timestamp = self.info.get('datetime', datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        
        # Set up session directory
        if session_dir:
            self._set_session_dir(session_dir)
        else:
            # Set up cache directories
            self._setup_directories()
        
        # Configure session logger
        self._setup_logger()
        
        logger.info(f"Initialized API Manager for {game_name}")
    
    def _setup_directories(self):
        """Set up all necessary cache directories."""
        # Base game directory
        self.game_dir = os.path.join(self.base_cache_dir, self.game_name)
        os.makedirs(self.game_dir, exist_ok=True)
        
        # Directory for this session's logs (will be created when needed)
        self.session_dir = None
    
    def _set_session_dir(self, session_dir: str) -> None:
        """
        Set the session directory directly, bypassing the directory creation logic.
        
        Args:
            session_dir (str): Path to the existing session directory
        """
        # Ensure the directory exists
        os.makedirs(session_dir, exist_ok=True)
        
        # Set session directory
        self.session_dir = session_dir
        
        # Extract game directory from session directory path
        # This is a best guess assuming the session_dir follows a similar structure
        self.game_dir = os.path.dirname(os.path.dirname(os.path.dirname(session_dir)))
        
        logger.info(f"Using provided session directory: {session_dir}")
    
    def _setup_logger(self):
        """Set up logger with file handler for this session."""
        if not self.enable_logging:
            return
        
        if self.session_dir:
            # Only create logs in the session directory - no global logs
            log_file = os.path.join(self.session_dir, f"{self.game_name}_api_manager.log")
        else:
            # Create a log file for the manager in the game directory
            log_file = os.path.join(self.game_dir, f"{self.game_name}_api_manager.log")
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        
        # Add file handler to logger
        logger.addHandler(file_handler)
    
    def _get_model_session_dir(self, model_name: str, session_name: Optional[str] = None, modality: str = "default") -> str:
        """
        Get or create the directory for this model session.
        
        Args:
            model_name (str): Name of the model
            session_name (str, optional): Custom session name
            modality (str, optional): Input modality (vision_only, text_only, vision_text)
            
        Returns:
            str: Path to the session directory
        """
        # Use model_name from info if provided, otherwise use the passed parameter
        model_name = self.info.get('model_name', model_name)
        
        # Use modality from info if provided, otherwise use the passed parameter
        modality = self.info.get('modality', modality)
        
        # Clean model name for directory
        clean_model_name = model_name.lower().split('/')[-1] if '/' in model_name else model_name.lower()
        
        # Create model directory directly under the game directory
        model_dir = os.path.join(self.game_dir, clean_model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Create modality directory under model directory
        modality_dir = os.path.join(model_dir, modality)
        os.makedirs(modality_dir, exist_ok=True)
        
        # Create datetime directory under modality directory
        datetime_dir = os.path.join(modality_dir, self.timestamp)
        if session_name:
            datetime_dir = os.path.join(modality_dir, f"{self.timestamp}_{session_name}")
            
        os.makedirs(datetime_dir, exist_ok=True)
        
        # Set the session directory for this call
        self.session_dir = datetime_dir
        
        return datetime_dir
    
    def _log_api_call(
        self, 
        model_name: str,
        input_data: Dict[str, Any],
        output_data: str,
        costs: Dict[str, Any],
        session_name: Optional[str] = None,
        modality: str = "default"
    ) -> Dict[str, str]:
        """
        Log API call details to files.
        
        Args:
            model_name (str): Name of the model
            input_data (Dict): Input data dictionary
            output_data (str): Output response from API
            costs (Dict): Token count and cost information
            session_name (str, optional): Custom session name
            modality (str, optional): Input modality (vision_only, text_only, vision_text)
            
        Returns:
            Dict[str, str]: Paths to created log files
        """
        if not self.enable_logging:
            return {}
        
        # If session_dir is already set, use it directly; otherwise get/create one
        if not self.session_dir:
            session_dir = self._get_model_session_dir(model_name, session_name, modality)
        else:
            session_dir = self.session_dir
        
        # Create file paths
        json_file = os.path.join(session_dir, "api_call.json")
        cost_log_file = os.path.join(session_dir, f"{self.game_name}_api_costs.log")
        
        # Save input and output data as JSON
        with open(json_file, "w", encoding="utf-8") as f:
            # Create a copy of input_data to avoid modifying original
            json_data = {
                "input": input_data,
                "output": output_data,
                "costs": {
                    k: str(v) if isinstance(v, Decimal) else v 
                    for k, v in costs.items()
                },
                "timestamp": time.time(),
                "datetime": datetime.datetime.now().isoformat(),
                "model": model_name,
                "game": self.game_name,
                "modality": modality,
                # Add structured conversation format directly in api_call.json
                "conversation": {
                    "system": input_data.get("system_prompt", ""),
                    "user": input_data.get("prompt", ""),
                    "assistant": output_data
                }
            }
            
            # Handle image base64 data to avoid huge JSON files
            if "base64_image" in input_data:
                json_data["input"]["base64_image"] = "[BASE64_IMAGE_DATA]"
            
            if "list_image_base64" in input_data:
                json_data["input"]["list_image_base64"] = [
                    "[BASE64_IMAGE_DATA]" for _ in input_data["list_image_base64"]
                ]
                
            json.dump(json_data, f, indent=2)
        
        # Format log entry for costs
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cost_log_entry = (
            f"[{timestamp}]\n"
            f"Game: {self.game_name}\n"
            f"Model: {model_name}\n"
            f"Modality: {modality}\n"
            f"Total Input Tokens: {costs.get('prompt_tokens', 0)}\n"
            # Comment out the problematic subtraction that can cause "unsupported operand type(s) for -: 'int' and 'None'"
            # f"Input Text Tokens: {costs.get('prompt_tokens', 0) - costs.get('image_tokens', 0)}\n"
            f"Input Text Tokens: {costs.get('prompt_tokens', 0)}\n"
            f"Input Image Tokens: {costs.get('image_tokens', 0)}\n"
            f"Output Tokens: {costs.get('completion_tokens', 0)}\n"
            f"Total Input Cost: ${costs.get('prompt_cost', 0):.6f}\n"
            f"Total Output Cost: ${costs.get('completion_cost', 0):.6f}\n"
            f"Total Cost: ${Decimal(costs.get('prompt_cost', 0)) + Decimal(costs.get('completion_cost', 0)):.6f}\n"
            f"{'-'*50}\n"
        )
        
        # Write to the session-specific cost log file
        with open(cost_log_file, "a", encoding="utf-8") as f:
            f.write(cost_log_entry)
        
        # No longer write to a main game cost log - keep logs only in session directory
        
        logger.info(f"Logged API call ({modality}) to {json_file} and costs to {cost_log_file}")
        
        return {
            "json_file": json_file,
            "cost_log_file": cost_log_file
        }
    
    def _calculate_costs(
        self, 
        model_name: str, 
        prompt: Union[str, List[Dict]], 
        completion: str,
        image_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate token usage and costs.
        
        Args:
            model_name (str): Name of the model
            prompt: Prompt text or message list
            completion: Completion text
            image_path: Path to image file (if any)
            
        Returns:
            Dict: Token counts and costs
        """
        try:
            costs = calculate_all_costs_and_tokens(
                prompt=prompt,
                completion=completion,
                model=model_name,
                image_path=image_path
            )
            
            logger.info(
                f"Calculated costs for {model_name}: "
                f"input={costs['prompt_tokens']} tokens (${costs['prompt_cost']}), "
                f"output={costs['completion_tokens']} tokens (${costs['completion_cost']})"
            )
            
            return costs
        except Exception as e:
            logger.error(f"Error calculating costs: {e}")
            # Return empty costs as fallback
            return {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "prompt_cost": Decimal("0"),
                "completion_cost": Decimal("0"),
                "image_tokens": 0 if image_path else None,
                "image_cost": Decimal("0") if image_path else None
            }
    
    def _get_base64_from_path(self, image_path: str) -> str:
        """
        Convert image file to base64 string.
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            str: Base64-encoded image data
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            logger.error(f"Error reading image file: {e}")
            raise
    
    def vision_text_completion(
        self, 
        model_name: str, 
        system_prompt: str, 
        prompt: str, 
        image_path: Optional[str] = None,
        base64_image: Optional[str] = None, 
        session_name: Optional[str] = None,
        temperature: float = 1,
        thinking: bool = False,
        reasoning_effort: str = "medium",
        token_limit: int = 30000,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Make a combined vision-text completion API call.
        Both image and text are provided as input to the model.
        
        Args:
            model_name (str): Model name (e.g., "claude-3-opus-20240229")
            system_prompt (str): System prompt
            prompt (str): User prompt text
            image_path (str, optional): Path to image file
            base64_image (str, optional): Base64-encoded image data (alternative to image_path)
            session_name (str, optional): Custom session name
            temperature (float): Temperature parameter (0-1)
            thinking (bool): Whether to enable thinking mode (Anthropic models)
            reasoning_effort (str): Reasoning effort for O-series models ("low"|"medium"|"high")
            token_limit (int): Maximum number of tokens for the completion response
            
        Returns:
            Tuple[str, Dict]: (Generated text, Cost information)
        """
        # Validate inputs
        if not (image_path or base64_image):
            raise ValueError("Either image_path or base64_image must be provided")
        
        # Get base64 image if path is provided
        if image_path and not base64_image:
            base64_image = self._get_base64_from_path(image_path)
        
        # TEMPORARILY SKIPPING COST CALCULATION
        # Original code commented out but preserved
        
        # Prepare input data for logging
        input_data = {
            "system_prompt": system_prompt,
            "prompt": prompt,
            "base64_image": base64_image,
            "model_name": model_name,
            "temperature": temperature,
            "thinking": thinking,
            "reasoning_effort": reasoning_effort,
            "token_limit": token_limit
        }
        
        # Select appropriate API based on model name
        try:
            if "claude" in model_name.lower():
                completion = anthropic_completion(
                    system_prompt=system_prompt,
                    model_name=model_name,
                    base64_image=base64_image,
                    prompt=prompt,
                    thinking=thinking,
                    token_limit=token_limit
                )
            elif "gpt" in model_name.lower() or model_name.startswith("o"):
                completion = openai_completion(
                    system_prompt=system_prompt,
                    model_name=model_name,
                    base64_image=base64_image,
                    prompt=prompt,
                    temperature=temperature,
                    reasoning_effort=reasoning_effort,
                    token_limit=token_limit
                )
            elif "gemini" in model_name.lower():
                completion = gemini_completion(
                    system_prompt=system_prompt,
                    model_name=model_name,
                    base64_image=base64_image,
                    prompt=prompt,
                    token_limit=token_limit
                )
            elif model_name.startswith("vllm-"):
                completion = vllm_completion(
                    system_prompt=system_prompt,
                    model_name=model_name,
                    prompt=prompt,
                    base64_image=base64_image,
                    temperature=temperature,
                    token_limit=token_limit,
                    url=self.vllm_url
                )
            elif model_name.startswith("modal-"):
                # TODO: make different modal backend configurable
                completion = modal_vllm_completion(
                    system_prompt=system_prompt,
                    model_name=model_name,
                    prompt=prompt,
                    base64_image=base64_image,
                    temperature=temperature,
                    token_limit=token_limit,
                    url=self.modal_url
                )
            elif "llama" in model_name.lower() or "meta" in model_name.lower() or (model_name == "deepseek-ai/DeepSeek-R1") or (model_name == "Qwen/Qwen3-235B-A22B-fp8-turbo"):
                completion = together_ai_completion(
                    system_prompt=system_prompt,
                    model_name=model_name,
                    base64_image=base64_image,
                    prompt=prompt,
                    temperature=temperature,
                    token_limit=token_limit
                )
            else:
                raise ValueError(f"Unsupported model: {model_name}")
            
            # TEMPORARILY SKIPPING COST CALCULATION: Create empty costs dict
            empty_costs = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "prompt_cost": "0",
                "completion_cost": "0",
                "image_tokens": 0,
                "image_cost": "0"
            }
            
            # Skip cost calculation and API call logging entirely
            # self._log_api_call(
            #     model_name=model_name,
            #     input_data=input_data,
            #     output_data=completion,
            #     costs=empty_costs,
            #     session_name=session_name,
            #     modality="vision_text"
            # )
            
            # Return completion and empty costs instead of calculated costs
            return completion, empty_costs
            
        except Exception as e:
            logger.error(f"Error in vision-text completion API call: {e}")
            raise
    
    def vision_only_completion(
        self, 
        model_name: str, 
        system_prompt: str,
        image_path: Optional[str] = None,
        base64_image: Optional[str] = None, 
        session_name: Optional[str] = None,
        temperature: float = 0,
        thinking: bool = False,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Make a vision-only completion API call.
        Only image is provided as input (no text prompt).
        
        Args:
            model_name (str): Model name (e.g., "claude-3-opus-20240229")
            system_prompt (str): System prompt
            image_path (str, optional): Path to image file
            base64_image (str, optional): Base64-encoded image data (alternative to image_path)
            session_name (str, optional): Custom session name
            temperature (float): Temperature parameter (0-1)
            thinking (bool): Whether to enable thinking mode (Anthropic models)
            
        Returns:
            Tuple[str, Dict]: (Generated text, Cost information)
        """
        # Validate inputs
        if not (image_path or base64_image):
            raise ValueError("Either image_path or base64_image must be provided")
        
        # Get base64 image if path is provided
        if image_path and not base64_image:
            base64_image = self._get_base64_from_path(image_path)
        
        # Use an empty or default prompt for vision-only analysis
        empty_prompt = "Describe this image in detail."
        
        # Prepare input data for logging
        input_data = {
            "system_prompt": system_prompt,
            "prompt": empty_prompt,  # We're using a default/empty prompt
            "base64_image": base64_image,
            "model_name": model_name,
            "temperature": temperature,
            "thinking": thinking
        }
        
        # Select appropriate API based on model name
        try:
            if "claude" in model_name.lower():
                completion = anthropic_completion(
                    system_prompt=system_prompt,
                    model_name=model_name,
                    base64_image=base64_image,
                    prompt=empty_prompt,
                    thinking=thinking
                )
            elif "gpt" in model_name.lower() or model_name.startswith("o"):
                completion = openai_completion(
                    system_prompt=system_prompt,
                    model_name=model_name,
                    base64_image=base64_image,
                    prompt=empty_prompt,
                    temperature=temperature
                )
            elif "gemini" in model_name.lower():
                completion = gemini_completion(
                    system_prompt=system_prompt,
                    model_name=model_name,
                    base64_image=base64_image,
                    prompt=empty_prompt
                )
            elif model_name.startswith("vllm-"):
                completion = vllm_completion(
                    system_prompt=system_prompt,
                    model_name=model_name,
                    prompt=empty_prompt,
                    base64_image=base64_image,
                    temperature=temperature,
                    url=self.vllm_url
                )
            elif model_name.startswith("modal-"):
                # TODO: make different modal backend configurable
                completion = modal_vllm_completion(
                    system_prompt=system_prompt,
                    model_name=model_name,
                    prompt=empty_prompt,
                    base64_image=base64_image,
                    temperature=temperature,
                    url=self.modal_url
                )
            else:
                raise ValueError(f"Unsupported model: {model_name}")
            
            # Format prompt for cost calculation based on model type
            if "claude" in model_name.lower():
                formatted_prompt = [
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
                                "text": empty_prompt
                            },
                        ],
                    }
                ]
            else:
                formatted_prompt = empty_prompt
            
            costs = self._calculate_costs(
                model_name=model_name,
                prompt=formatted_prompt,
                completion=completion,
                image_path=image_path
            )
            
            # Log API call
            self._log_api_call(
                model_name=model_name,
                input_data=input_data,
                output_data=completion,
                costs=costs,
                session_name=session_name,
                modality="vision_only"
            )
            
            return completion, costs
            
        except Exception as e:
            logger.error(f"Error in vision-only completion API call: {e}")
            raise
    
    def text_only_completion(
        self,
        model_name: str,
        system_prompt: str,
        prompt: str,
        session_name: Optional[str] = None,
        temperature: float = 1,
        thinking: bool = False,
        reasoning_effort: str = "medium",
        token_limit: int = 30000,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Make a text-only completion API call.
        
        Args:
            model_name (str): Model name
            system_prompt (str): System prompt
            prompt (str): User prompt
            session_name (str, optional): Custom session name
            temperature (float): Temperature parameter (0-1)
            thinking (bool): Whether to enable thinking mode (Anthropic models)
            reasoning_effort (str): Reasoning effort for O-series models ("low"|"medium"|"high")
            token_limit (int): Maximum number of tokens for the completion response
            
        Returns:
            Tuple[str, Dict]: (Generated text, Cost information)
        """
        # Prepare input data for logging
        input_data = {
            "system_prompt": system_prompt,
            "prompt": prompt,
            "model_name": model_name,
            "temperature": temperature,
            "thinking": thinking,
            "reasoning_effort": reasoning_effort,
            "token_limit": token_limit
        }
        
        # Select appropriate API based on model name
        try:
            if "claude" in model_name.lower():
                completion = anthropic_text_completion(
                    system_prompt=system_prompt,
                    model_name=model_name,
                    prompt=prompt,
                    thinking=thinking,
                    token_limit=token_limit
                )
            elif "gpt" in model_name.lower() or model_name.startswith("o"):
                completion = openai_text_completion(
                    system_prompt=system_prompt,
                    model_name=model_name,
                    prompt=prompt,
                    reasoning_effort=reasoning_effort,
                    token_limit=token_limit
                )
            elif "gemini" in model_name.lower():
                completion = gemini_text_completion(
                    system_prompt=system_prompt,
                    model_name=model_name,
                    prompt=prompt,
                    token_limit=token_limit
                )
            elif model_name.startswith("vllm-"):
                completion = vllm_text_completion(
                    system_prompt=system_prompt,
                    model_name=model_name,
                    prompt=prompt,
                    temperature=temperature,
                    token_limit=token_limit,
                    url=self.vllm_url
                )
            elif model_name.startswith("modal-"):
                # TODO: make different modal backend configurable
                completion = modal_vllm_text_completion(
                    system_prompt=system_prompt,
                    model_name=model_name,
                    prompt=prompt,
                    temperature=temperature,
                    token_limit=token_limit,
                    url=self.modal_url
                )
            elif "llama" in model_name.lower() or "meta" in model_name.lower() or (model_name == "Qwen/Qwen3-235B-A22B-fp8") or (model_name == "deepseek-ai/DeepSeek-R1"):
                completion = together_ai_text_completion(
                    system_prompt=system_prompt,
                    model_name=model_name,
                    prompt=prompt,
                    temperature=temperature,
                    token_limit=token_limit
                )
            elif "deepseek" in model_name.lower():
                completion = deepseek_text_reasoning_completion(
                    system_prompt=system_prompt,
                    model_name=model_name,
                    prompt=prompt,
                    token_limit=token_limit
                )
            elif "grok" in model_name.lower():
                completion = xai_grok_text_completion(
                    system_prompt=system_prompt,
                    model_name=model_name,
                    prompt=prompt,
                    token_limit=token_limit,
                    temperature=temperature,
                    reasoning_effort=reasoning_effort
                )
            else:
                raise ValueError(f"Unsupported model: {model_name}")
            
            # Format prompt for cost calculation based on model type
            if "claude" in model_name.lower():
                formatted_prompt = [
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
                formatted_prompt = prompt
                
            # Calculate costs
            costs = self._calculate_costs(
                model_name=model_name,
                prompt=formatted_prompt,
                completion=completion
            )
            
            # Log API call
            self._log_api_call(
                model_name=model_name,
                input_data=input_data,
                output_data=completion,
                costs=costs,
                session_name=session_name,
                modality="text_only"
            )
            
            return completion, costs
            
        except Exception as e:
            logger.error(f"Error in text-only completion API call: {e}")
            raise
    
    # Legacy methods for backward compatibility  
    def vision_completion(
        self, 
        model_name: str, 
        system_prompt: str, 
        prompt: str, 
        image_path: Optional[str] = None,
        base64_image: Optional[str] = None, 
        session_name: Optional[str] = None,
        temperature: float = 1,
        thinking: bool = False,
        reasoning_effort: str = "medium",
        token_limit: int = 30000,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Legacy method for vision-based completion API call.
        This is a wrapper around vision_text_completion for backward compatibility.
        
        Args:
            model_name (str): Model name
            system_prompt (str): System prompt
            prompt (str): User prompt
            image_path (str, optional): Path to image file
            base64_image (str, optional): Base64-encoded image data
            session_name (str, optional): Custom session name
            temperature (float): Temperature parameter (0-1)
            thinking (bool): Whether to enable thinking mode
            reasoning_effort (str): Reasoning effort for O-series models ("low"|"medium"|"high")
            token_limit (int): Maximum number of tokens for the completion response
            
        Returns:
            Tuple[str, Dict]: (Generated text, Cost information)
        """
        logger.warning("Using legacy vision_completion - consider using vision_text_completion instead")
        return self.vision_text_completion(
            model_name=model_name,
            system_prompt=system_prompt,
            prompt=prompt,
            image_path=image_path,
            base64_image=base64_image,
            session_name=session_name,
            temperature=temperature,
            thinking=thinking,
            reasoning_effort=reasoning_effort,
            token_limit=token_limit,
            url=self.modal_url
        )
    
    def text_completion(
        self,
        model_name: str,
        system_prompt: str,
        prompt: str,
        session_name: Optional[str] = None,
        temperature: float = 1,
        thinking: bool = False,
        reasoning_effort: str = "medium",
        token_limit: int = 30000,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Legacy method for text-only completion API call.
        This is a wrapper around text_only_completion for backward compatibility.
        
        Args:
            model_name (str): Model name
            system_prompt (str): System prompt
            prompt (str): User prompt
            session_name (str, optional): Custom session name
            temperature (float): Temperature parameter (0-1)
            thinking (bool): Whether to enable thinking mode
            reasoning_effort (str): Reasoning effort for O-series models ("low"|"medium"|"high")
            token_limit (int): Maximum number of tokens for the completion response
            
        Returns:
            Tuple[str, Dict]: (Generated text, Cost information)
        """
        logger.warning("Using legacy text_completion - consider using text_only_completion instead")
        return self.text_only_completion(
            model_name=model_name,
            system_prompt=system_prompt,
            prompt=prompt,
            session_name=session_name,
            temperature=temperature,
            thinking=thinking,
            reasoning_effort=reasoning_effort,
            token_limit=token_limit,
            url=self.modal_url
        )
    
    def multi_image_completion(
        self,
        model_name: str,
        system_prompt: str,
        prompt: str,
        list_content: List[str],
        list_image_paths: Optional[List[str]] = None,
        list_image_base64: Optional[List[str]] = None,
        session_name: Optional[str] = None,
        temperature: float = 1,
        reasoning_effort: str = "medium",
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Make a multi-image completion API call.
        
        Args:
            model_name (str): Model name
            system_prompt (str): System prompt
            prompt (str): User prompt
            list_content (List[str]): List of text content corresponding to each image
            list_image_paths (List[str], optional): List of image file paths
            list_image_base64 (List[str], optional): List of base64-encoded image data
            session_name (str, optional): Custom session name
            temperature (float): Temperature parameter (0-1)
            reasoning_effort (str): Reasoning effort for O-series models ("low"|"medium"|"high")
            
        Returns:
            Tuple[str, Dict]: (Generated text, Cost information)
        """
        # Validate inputs
        if not (list_image_paths or list_image_base64):
            raise ValueError("Either list_image_paths or list_image_base64 must be provided")
            
        if list_image_paths and not list_image_base64:
            # Convert image paths to base64
            list_image_base64 = []
            for image_path in list_image_paths:
                list_image_base64.append(self._get_base64_from_path(image_path))
        
        # Prepare input data for logging
        input_data = {
            "system_prompt": system_prompt,
            "prompt": prompt,
            "list_content": list_content,
            "list_image_base64": list_image_base64,
            "model_name": model_name,
            "temperature": temperature,
            "reasoning_effort": reasoning_effort
        }
        
        # Select appropriate API based on model name
        try:
            if "claude" in model_name.lower():
                completion = anthropic_multiimage_completion(
                    system_prompt=system_prompt,
                    model_name=model_name,
                    prompt=prompt,
                    list_content=list_content,
                    list_image_base64=list_image_base64
                )
            elif "gpt" in model_name.lower() or model_name.startswith("o"):
                completion = openai_multiimage_completion(
                    system_prompt=system_prompt,
                    model_name=model_name,
                    prompt=prompt,
                    list_content=list_content,
                    list_image_base64=list_image_base64,
                    reasoning_effort=reasoning_effort
                )
            elif "gemini" in model_name.lower():
                completion = gemini_multiimage_completion(
                    system_prompt=system_prompt,
                    model_name=model_name,
                    prompt=prompt,
                    list_content=list_content,
                    list_image_base64=list_image_base64
                )
            elif "llama" in model_name.lower() or "meta" in model_name.lower() or (model_name == "Qwen/Qwen3-235B-A22B-fp8") or (model_name == "deepseek-ai/DeepSeek-R1"):
                completion = together_ai_multiimage_completion(
                    system_prompt=system_prompt,
                    model_name=model_name,
                    prompt=prompt,
                    list_content=list_content,
                    list_image_base64=list_image_base64,
                    temperature=temperature
                )
            elif model_name.startswith("vllm-"):
                completion = vllm_multiimage_completion(
                    system_prompt=system_prompt,
                    model_name=model_name,
                    prompt=prompt,
                    base64_image=list_image_base64,
                    temperature=temperature,
                    url=self.vllm_url
                )
            elif model_name.startswith("modal-"):
                # TODO: make different modal backend configurable
                completion = modal_vllm_multiimage_completion(
                    system_prompt=system_prompt,
                    model_name=model_name,
                    prompt=prompt,
                    base64_image=list_image_base64,
                    temperature=temperature,
                    url=self.modal_url,
                )
            else:
                raise ValueError(f"Unsupported model: {model_name}")
                
            # Format message for cost calculation
            # This is simplified - in practice, would need more detailed message formatting
            formatted_prompt = prompt
            
            # Calculate costs (simplified - in practice, would need more detailed cost calculation)
            total_image_tokens = 0
            if list_image_paths:
                for image_path in list_image_paths:
                    total_image_tokens += count_image_tokens(image_path, model_name)
            
            # Calculate basic costs
            costs = self._calculate_costs(
                model_name=model_name,
                prompt=formatted_prompt,
                completion=completion
            )
            
            # Add image tokens manually
            if total_image_tokens > 0:
                costs["image_tokens"] = total_image_tokens
                costs["prompt_tokens"] += total_image_tokens
                image_cost = calculate_cost_by_tokens(total_image_tokens, model_name, "input")
                costs["prompt_cost"] += image_cost
                costs["image_cost"] = image_cost
            
            # Log API call
            self._log_api_call(
                model_name=model_name,
                input_data=input_data,
                output_data=completion,
                costs=costs,
                session_name=session_name,
                modality="multi_image"
            )
            
            return completion, costs
            
        except Exception as e:
            logger.error(f"Error in multi-image completion API call: {e}")
            raise 