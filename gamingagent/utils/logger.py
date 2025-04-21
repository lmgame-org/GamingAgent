import os
import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional, Union
import numpy as np
from pathlib import Path

class Logger:
    """Logger class for handling logging functionality."""
    
    def __init__(
        self,
        name: str = "gamingagent",
        log_dir: str = "logs",
        level: int = logging.INFO,
        console_output: bool = True
    ):
        """
        Initialize logger.
        
        Args:
            name (str): Logger name
            log_dir (str): Directory for log files
            level (int): Logging level
            console_output (bool): Whether to output to console
        """
        self.name = name
        self.log_dir = log_dir
        self.level = level
        self.console_output = console_output
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(levelname)s - %(message)s'
        )
        
        # File handler
        log_file = os.path.join(log_dir, f"{name}_{datetime.now().strftime('%Y%m%d')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler
        if console_output:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
            
    def get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        return datetime.now().isoformat()
        
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self.logger.debug(message, **kwargs)
        
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self.logger.info(message, **kwargs)
        
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self.logger.warning(message, **kwargs)
        
    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self.logger.error(message, **kwargs)
        
    def critical(self, message: str, **kwargs) -> None:
        """Log critical message."""
        self.logger.critical(message, **kwargs)
        
    def log_json(self, data: Dict[str, Any], level: str = "info") -> None:
        """
        Log JSON data.
        
        Args:
            data (Dict[str, Any]): Data to log
            level (str): Logging level
        """
        message = json.dumps(data, indent=2)
        getattr(self, level)(message)
        
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log metrics.
        
        Args:
            metrics (Dict[str, Any]): Metrics to log
            step (Optional[int]): Step number
        """
        if step is not None:
            metrics["step"] = step
        self.log_json(metrics, "info")
        
    def log_exception(self, exception: Exception) -> None:
        """
        Log exception.
        
        Args:
            exception (Exception): Exception to log
        """
        self.error(f"Exception occurred: {str(exception)}", exc_info=True)
        
    # Agent-specific logging methods
    def log_agent_config(self, config: Dict[str, Any]) -> None:
        """
        Log agent configuration.
        
        Args:
            config (Dict[str, Any]): Agent configuration
        """
        self.info("Agent configuration", extra={"config": config})
        
    def log_action(self, action: np.ndarray, step: int, action_dir: str) -> None:
        """
        Log agent action.
        
        Args:
            action (np.ndarray): Action taken
            step (int): Current step
            action_dir (str): Directory to save action file
        """
        action_data = {
            "step": step,
            "action": action.tolist(),
            "timestamp": self.get_timestamp()
        }
        
        # Save to file
        action_path = os.path.join(action_dir, f"action_{step:06d}.json")
        with open(action_path, "w") as f:
            json.dump(action_data, f, indent=4)
            
        # Log action
        self.info(f"Action taken at step {step}", extra={"action": action_data})
        
    def log_state(self, observation: np.ndarray, step: int, state_dir: str) -> None:
        """
        Log agent state/observation.
        
        Args:
            observation (np.ndarray): Current observation
            step (int): Current step
            state_dir (str): Directory to save state file
        """
        state_data = {
            "step": step,
            "observation_shape": observation.shape,
            "timestamp": self.get_timestamp()
        }
        
        # Save to file
        state_path = os.path.join(state_dir, f"state_{step:06d}.json")
        with open(state_path, "w") as f:
            json.dump(state_data, f, indent=4)
            
        # Log state
        self.info(f"State observed at step {step}", extra={"state": state_data})
        
    def log_api_call(self, 
                    model: str, 
                    max_tokens: int, 
                    temperature: float, 
                    prompt_length: int, 
                    image_size: int,
                    response_time: float,
                    response_length: int,
                    response_preview: str) -> None:
        """
        Log API call details.
        
        Args:
            model (str): Model name
            max_tokens (int): Max tokens used
            temperature (float): Temperature setting
            prompt_length (int): Length of prompt
            image_size (int): Size of image
            response_time (float): Time taken for response
            response_length (int): Length of response
            response_preview (str): Preview of response
        """
        api_data = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "prompt_length": prompt_length,
            "image_size": image_size,
            "response_time": response_time,
            "response_length": response_length,
            "response_preview": response_preview
        }
        self.info("API call details", extra={"api_call": api_data})
        
    def log_rate_limit(self, wait_time: float, new_interval: Optional[float] = None) -> None:
        """
        Log rate limiting information.
        
        Args:
            wait_time (float): Time to wait
            new_interval (Optional[float]): New rate limit interval if changed
        """
        rate_data = {"wait_time": wait_time}
        if new_interval is not None:
            rate_data["new_interval"] = new_interval
        self.info("Rate limiting", extra={"rate_limit": rate_data}) 