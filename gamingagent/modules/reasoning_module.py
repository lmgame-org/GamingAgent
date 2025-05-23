from abc import abstractmethod
from .core_module import CoreModule

# TODO: 
# 1.module integration 
# 2.COT thinking mode 

class ReasoningModule(CoreModule):
    """
    Reasoning module that plans actions based on perception and memory.
    
    Game-specific implementations should inherit from this class and implement
    the required abstract methods.
    """
    
    def __init__(self, 
                model_name="claude-3-7-sonnet-latest", 
                observation_mode="vision",
                cache_dir="cache",
                system_prompt="", 
                prompt="", 
                token_limit=100000, 
                reasoning_effort="high"
        ):
        """
        Initialize the reasoning module.
        
        Args:
            model_name (str): The name of the model to use for inference.
            observation_mode (str): Mode for processing observations:
                - "vision": Uses image path as input
                - "text": Uses symbolic representation/textual description as input
                - "both": Uses both image path and text representation as inputs
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

        self.observation_mode = observation_mode

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
        textual_representation = perception_data.get("textual_representation", "")
        
        # Get the description of visual elements from perception module
        processed_visual_description = perception_data.get("processed_visual_description", "")
        
        # Extract game trajectory and reflection memory module
        game_trajectory = memory_summary.get("game_trajectory", "")
        reflection = memory_summary.get("reflection", "")
        
        # Format the memory and perception context, and create full context
        #memory_context = f"Memory History:\n{game_trajectory}\n\n"
        #perception_context = f"Current Perception:\n{textual_representation}\n\n"
        #reflection_context = f"Reflection:\n{memory_reflection}" if memory_reflection else ""
        #full_context = memory_context + perception_context + reflection_context

        use_memory = bool(game_trajectory.strip() and reflection.strip())
        use_perception = bool(processed_visual_description.strip())

        full_context = observation.get_complete_prompt(
            observation_mode=self.observation_mode,
            use_memory_module=use_memory,
            use_perception_module=use_perception,
        )
        
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
            "generation": response.strip()
        }
