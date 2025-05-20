from abc import abstractmethod
from .core_module import CoreModule

class ReasoningModule(CoreModule):
    """
    Reasoning module that plans actions based on perception and memory.
    
    Game-specific implementations should inherit from this class and implement
    the required abstract methods.
    """
    
    def __init__(self, model_name="claude-3-7-sonnet-latest", cache_dir="cache",
                 system_prompt="", prompt="", token_limit=100000, reasoning_effort="high"):
        """
        Initialize the reasoning module.
        
        Args:
            model_name (str): The name of the model to use for inference.
            cache_dir (str): Directory for storing logs and cache files.
            system_prompt (str): System prompt for LLM calls.
            prompt (str): Default user prompt for LLM calls.
            token_limit (int): Maximum number of tokens for API calls.
            
        Note: 
            Reasoning module always uses "high" reasoning effort regardless of default.
        """
        super().__init__(
            module_name="reasoning_module",
            model_name=model_name,
            system_prompt=system_prompt,
            prompt=prompt,
            cache_dir=cache_dir,
            token_limit=token_limit,
            reasoning_effort=reasoning_effort  # Always use high reasoning effort
        )
    
    @abstractmethod
    def plan_action(self, perception_data, memory_summary, img_path=None):
        """
        Plan the next action sequence based on current perception and memory.
        
        Args:
            perception_data (dict): Output from perception_module.get_perception_summary()
            memory_summary (dict): Output from memory_module.get_memory_summary()
            img_path (str, optional): Path to the current game image (override for perception_data["img_path"])
            
        Returns:
            dict: A dictionary containing action and thought
        """
        # Get the image path (prefer the passed parameter if available)
        image_path = img_path or perception_data.get("img_path")
        
        # Get the symbolic representation from perception data
        symbolic_representation = perception_data.get("symbolic_representation", "")
        
        # Extract memory components
        prev_game_states = memory_summary.get("prev_game_states", "")
        memory_reflection = memory_summary.get("reflection", "")
        
        # Format the memory and perception context for the prompt
        memory_context = f"Memory History:\n{prev_game_states}\n\n"
        perception_context = f"Current Perception:\n{symbolic_representation}\n\n"
        reflection_context = f"Reflection:\n{memory_reflection}" if memory_reflection else ""
        
        # Create the full context
        full_context = memory_context + perception_context + reflection_context
        
        # Choose API call based on whether an image is available
        if image_path:
            response = self._call_vision_api(full_context, image_path)
        else:
            response = self._call_text_api(full_context)
        
        # Parse the response
        parsed_response = self._parse_response(response)
        
        # Log the reasoning process
        self.log({
            "perception_data": perception_data,
            "memory_summary": memory_summary,
            "response": response,
            "thought": parsed_response.get("thought"),
            "action": parsed_response.get("action")
        })
        
        return parsed_response
    
    def _call_vision_api(self, context, img_path):
        """
        Call the vision API with text context and image.
        
        Args:
            context (str): Formatted context with perception and memory
            img_path (str): Path to the current game image
            
        Returns:
            str: Raw response from the API
        """
        # Create user prompt with context
        user_prompt = self.prompt.replace("{context}", context) if "{context}" in self.prompt else context
        
        # Call the vision-text API
        response = self.api_manager.vision_text_completion(
            model_name=self.model_name,
            system_prompt=self.system_prompt,
            prompt=user_prompt,
            image_path=img_path,
            thinking=True,
            reasoning_effort=self.reasoning_effort,
            token_limit=self.token_limit
        )
        
        return response
    
    def _call_text_api(self, context, custom_prompt=None):
        """
        Call the text-only API with context.
        
        Args:
            context (str): Formatted context with perception and memory data
            custom_prompt (str, optional): Custom prompt to use
            
        Returns:
            str: Raw response from the API
        """
        # Create user prompt
        user_prompt = custom_prompt if custom_prompt else self.prompt
        
        # Replace context placeholder if it exists
        if "{context}" in user_prompt:
            user_prompt = user_prompt.replace("{context}", context)
        else:
            # If no placeholder, append the context
            user_prompt = context + "\n\n" + user_prompt
        
        # Call the API
        response = self.api_manager.text_only_completion(
            model_name=self.model_name,
            system_prompt=self.system_prompt,
            prompt=user_prompt,
            thinking=True,
            reasoning_effort=self.reasoning_effort,
            token_limit=self.token_limit
        )
        
        return response
    
    def _prepare_memory_context(self, memory_summary):
        """
        Prepare a concise summary of memory for the prompt.
        
        Args:
            memory_summary (list): List of memory entries
            
        Returns:
            str: A formatted summary of memory for use in prompts
        """
        if not memory_summary:
            return "No memory of past states available."
            
        # Default implementation - can be overridden by game-specific subclasses
        summary_parts = []
        for i, entry in enumerate(memory_summary[-3:]):  # Only include the last 3 entries
            entry_summary = f"State {i+1}:\n"
            
            # Basic memory entry visualization
            if 'last_action' in entry:
                entry_summary += f"- Last action: {entry['last_action']}\n"
            if 'thought' in entry:
                entry_summary += f"- Thought: {entry['thought']}\n"
            if 'reflection' in entry:
                entry_summary += f"- Reflection: {entry['reflection']}\n"
            
            summary_parts.append(entry_summary)
            
        return "\n".join(summary_parts)
    
    def _parse_response(self, response):
        """
        Parse the reasoning response to extract structured action data.
        
        Args:
            response (str): The raw response from the LLM
            
        Returns:
            dict: Structured information extracted from the response
        """
        # Default implementation - should be overridden by game-specific subclasses
        return {
            "action": None,
            "thought": response.strip()
        }
