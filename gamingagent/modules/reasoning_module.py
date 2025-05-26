from abc import abstractmethod
from .core_module import CoreModule, Observation
from typing import Optional

import re

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
                game_name: Optional[str] = None,
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
            game_name (str, optional): Name of the game, for game-specific logic.
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
        self.game_name = game_name

    def plan_action(self, observation: Observation, perception_data: dict, memory_summary: dict, img_path: Optional[str]=None):
        """
        Plan the next action sequence based on current perception and memory.
        
        Args:
            observation (Observation): An Observation instance, potentially containing all primary data.
            perception_data (dict): Output from perception_module.get_perception_summary()
            memory_summary (dict): Output from memory_module.get_memory_summary()
            img_path (str, optional): Path to the current game image (override for perception_data["img_path"])
            
        Returns:
            dict: A dictionary containing action and thought
        """
        # TODO: CLEAN UP — remove perception_data, memory_summary
        # THEY SHOULD HAVE BEEN PRESENT IN observation

        # Get the image path (prefer the passed parameter if available)
        image_path_to_use = img_path or perception_data.get("img_path")
        
        # Memory data
        game_trajectory = memory_summary.get("game_trajectory", "No game trajectory available.")
        latest_reflection = memory_summary.get("reflection", "No reflection available.")

        full_prompt_str: str

        if self.game_name == "ace_attorney":
            # --- Ace Attorney Specific Prompt Population ---
            # print("[ReasoningModule DEBUG] Using Ace Attorney specific prompt population.")
            
            # Extract detailed perception fields
            p_game_state = perception_data.get("game_state_from_perception", "Unknown")
            p_dialogue = perception_data.get("parsed_dialogue") # This is a dict or None
            p_dialogue_speaker = p_dialogue.get("speaker", "N/A") if p_dialogue else "N/A"
            p_dialogue_text = p_dialogue.get("text", "N/A") if p_dialogue else "N/A"
            p_dialogue_cont = perception_data.get("dialogue_continuation_from_perception", "Unknown")
            p_options = perception_data.get("options_from_perception", "None")
            p_selected_evidence = perception_data.get("selected_evidence_from_perception", "None")
            p_scene_desc = perception_data.get("scene_description_from_perception", "No scene description available.")

            # Use the module's self.prompt (which should be the Ace Attorney reasoning prompt)
            # And replace placeholders.
            prompt_template = self.prompt # This should be loaded from module_prompts.json
            
            full_prompt_str = prompt_template.replace("{perception_game_state}", str(p_game_state)) \
                                        .replace("{perception_dialogue_speaker}", str(p_dialogue_speaker)) \
                                        .replace("{perception_dialogue_text}", str(p_dialogue_text)) \
                                        .replace("{perception_dialogue_continuation}", str(p_dialogue_cont)) \
                                        .replace("{perception_options}", str(p_options)) \
                                        .replace("{perception_selected_evidence}", str(p_selected_evidence)) \
                                        .replace("{perception_scene_description}", str(p_scene_desc)) \
                                        .replace("{memory_game_trajectory}", str(game_trajectory)) \
                                        .replace("{memory_latest_reflection}", str(latest_reflection))
            # print(f"[ReasoningModule DEBUG Ace Attorney] Populated prompt (first 300 chars): {full_prompt_str[:300]}")

        else:
            # --- Generic/Default Prompt Population ---
            # print("[ReasoningModule DEBUG] Using Generic prompt population.")
            # This part reuses the older logic from Observation.get_complete_prompt for non-AA games
            textual_representation = perception_data.get("textual_representation", "")
            processed_visual_description = perception_data.get("processed_visual_description", "")
            
            use_memory = bool(game_trajectory and latest_reflection)
            use_perception_text = bool(processed_visual_description or textual_representation)

            # Constructing context similar to Observation.get_complete_prompt()
            context_parts = []
            if use_memory:
                context_parts.append(f"Memory History:\n{game_trajectory}")
                context_parts.append(f"Reflection:\n{latest_reflection}")
            
            if use_perception_text:
                if textual_representation:
                    context_parts.append(f"Current Symbolic State:\n{textual_representation}")
                if processed_visual_description:
                    context_parts.append(f"Current Visual Description:\n{processed_visual_description}")
            
            combined_context = "\n\n".join(context_parts)
            
            # Replace {context} in the generic prompt, or append if no placeholder
            if "{context}" in self.prompt:
                full_prompt_str = self.prompt.replace("{context}", combined_context)
            else:
                full_prompt_str = f"{combined_context}\n\n{self.prompt}"
            # print(f"[ReasoningModule DEBUG Generic] Populated prompt (first 300 chars): {full_prompt_str[:300]}")

        # Choose API call based on whether an image is available
        response: Optional[str] = None
        if image_path_to_use:
            response = self._call_vision_api(full_prompt_str, image_path_to_use)
        else:
            response = self._call_text_api(full_prompt_str)
        
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
    
    def _call_vision_api(self, user_prompt_str: str, img_path: str):
        """
        Call the vision API with text context and image.
        
        Args:
            user_prompt_str (str): The fully formatted user prompt string.
            img_path (str): Path to the current game image
            
        Returns:
            str: Raw response from the API
        """
        # System prompt is self.system_prompt
        # user_prompt_str is already fully formed here
        # print(f"""
# --- VISION API --- Reasoning Module --- FINAL USER PROMPT ---
# {user_prompt_str}
# --- END FINAL USER PROMPT ---
# """)
        
        response = self.api_manager.vision_text_completion(
            model_name=self.model_name,
            system_prompt=self.system_prompt,
            prompt=user_prompt_str, # Use the fully constructed prompt directly
            image_path=img_path,
            thinking=True,
            reasoning_effort=self.reasoning_effort,
            token_limit=self.token_limit
        )
        
        return response
    
    def _call_text_api(self, user_prompt_str: str, custom_prompt=None):
        """
        Call the text-only API with context.
        
        Args:
            user_prompt_str (str): The fully formatted user prompt string.
            custom_prompt (str, optional): Not typically used if user_prompt_str is already complete.
            
        Returns:
            str: Raw response from the API
        """
        # System prompt is self.system_prompt
        # user_prompt_str is already fully formed here. custom_prompt argument is largely redundant now.
        final_prompt_to_send = user_prompt_str
        if custom_prompt: # Though not expected to be used with the new logic
            # This case would need re-evaluation if custom_prompt is still desired
            # print("[ReasoningModule WARNING] _call_text_api received custom_prompt, but user_prompt_str was already formatted. Prioritizing user_prompt_str.")
            pass 

        # print(f"""
# --- TEXT API --- Reasoning Module --- FINAL USER PROMPT ---
# {final_prompt_to_send}
# --- END FINAL USER PROMPT ---
# """)
        response = self.api_manager.text_only_completion(
            model_name=self.model_name,
            system_prompt=self.system_prompt,
            prompt=final_prompt_to_send, # Use the fully constructed prompt
            thinking=True,
            reasoning_effort=self.reasoning_effort,
            token_limit=self.token_limit
        )
        
        return response
   
    def _parse_response(self, response):
        """
        Parse the response to extract thought and action.
        
        Args:
            response (str): The raw response from the LLM
            
        Returns:
            dict: A dictionary containing action and thought
        """
        if not response:
            return {"action": None, "thought": "No response received"}
        
        # Initialize result with defaults
        result = {
            "action": None,
            "thought": None
        }
        
        # Use regex to find thought and action sections
        # Match patterns like "thought:", "# thought:", "Thought:", etc.
        thought_pattern = r'(?:^|\n)(?:#\s*)?thought:(.+?)(?=(?:\n(?:#\s*)?(?:action|move):)|$)'
        action_pattern = r'(?:^|\n)(?:#\s*)?(?:action|move):(.+?)(?=(?:\n(?:#\s*)?thought:)|$)'
        
        # Find thought section using regex (case insensitive)
        thought_match = re.search(thought_pattern, response, re.DOTALL | re.IGNORECASE)
        if thought_match:
            result["thought"] = thought_match.group(1).strip()
        
        # Find action section using regex (case insensitive)
        action_match = re.search(action_pattern, response, re.DOTALL | re.IGNORECASE)
        if action_match:
            result["action"] = action_match.group(1).strip()
        
        # If no structured format was found, treat the whole response as thought
        if not result["thought"] and not result["action"]:
            result["thought"] = response.strip()
        elif not result["thought"]:  # If only action was found
            # Look for any text before the action as thought
            pre_action = re.split(r'(?:^|\n)(?:#\s*)?(?:action|move):', response, flags=re.IGNORECASE)[0]
            if pre_action and pre_action.strip():
                result["thought"] = pre_action.strip()
            # action is left as none
        
        # If only thought is found, action is left as none
        
        # Normalize action format if needed
        if result["action"]:
            # Process specific action formats if needed
            pass
        
        return result
