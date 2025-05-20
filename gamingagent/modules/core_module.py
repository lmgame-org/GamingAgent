import json
import os
import datetime
from abc import ABC, abstractmethod
from tools.serving import APIManager

class CoreModule(ABC):
    """
    Core module that serves as the foundation for all other modules.
    Provides common functionality for API calls, logging, and response parsing.
    """
    
    def __init__(self, module_name, model_name="claude-3-7-sonnet-latest", 
                 system_prompt="", prompt="", cache_dir="cache",
                 token_limit=100000, reasoning_effort="medium"):
        """
        Initialize the core module with basic parameters.
        
        Args:
            module_name (str): Name of the module.
            model_name (str): The name of the model to use for inference.
            system_prompt (str): Default system prompt for LLM calls.
            prompt (str): Default user prompt for LLM calls.
            cache_dir (str): Directory for storing logs and cache files.
            token_limit (int): Maximum number of tokens for API calls.
            reasoning_effort (str): Reasoning effort for API calls (low, medium, high).
        """
        self.module_name = module_name
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.prompt = prompt
        self.cache_dir = cache_dir
        self.token_limit = token_limit
        self.reasoning_effort = reasoning_effort
        
        # Initialize API manager
        self.api_manager = APIManager(game_name=module_name.replace("_module", ""))
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize logger file path
        self.module_file = os.path.join(cache_dir, f"{module_name}.json")
        
    def log(self, data):
        """
        Log module data to the module file.
        
        Args:
            data (dict): Data to be logged.
        """
        try:
            # Add timestamp to log entry
            log_entry = {
                "datetime": datetime.datetime.now().isoformat(),
                **data
            }
            
            # Create or append to log file
            existing_logs = []
            if os.path.exists(self.module_file):
                try:
                    with open(self.module_file, 'r') as f:
                        existing_logs = json.load(f)
                except json.JSONDecodeError:
                    existing_logs = []
            
            # Ensure existing_logs is a list
            if not isinstance(existing_logs, list):
                existing_logs = []
            
            existing_logs.append(log_entry)
            
            # Write updated logs back to file
            with open(self.module_file, 'w') as f:
                json.dump(existing_logs, f, indent=2)
                
        except Exception as e:
            print(f"Error logging to {self.module_file}: {e}")
    
    @abstractmethod
    def _parse_response(self, response):
        """
        Parse LLM response to extract structured information.
        
        Args:
            response (str): The raw response from the LLM
            
        Returns:
            dict: Structured information extracted from the response
        """
        pass
