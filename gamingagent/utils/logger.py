import os
import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional
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