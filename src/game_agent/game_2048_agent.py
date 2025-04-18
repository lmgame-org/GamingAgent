"""
Game2048Agent - Specialized agent for playing the 2048 game
"""

import time
import os
import re
import asyncio
import logging
import pyautogui
from typing import Dict, List, Optional, Any, Union

from pathlib import Path
from .game_agent import GameAgent, Message
from tools.utils import encode_image, get_annotate_img

logger = logging.getLogger(__name__)

class Game2048Agent(GameAgent):
    """
    Specialized agent for playing the 2048 game.
    
    This agent extends the base GameAgent class with game-specific functionality
    for playing 2048, including specialized vision processing, action execution,
    and strategy implementation.
    """
    
    def __init__(
        self,
        api_provider: str = "anthropic",
        model_name: str = "claude-3-7-sonnet-20250219",
        modality: str = "vision-text",
        thinking: bool = True,
        cache_dir: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_threads: int = 1,
        max_history_tokens: int = 4000,
        context_window: int = 10,
        log_dir: Optional[Path] = None,
    ):
        """
        Initialize the 2048 game agent.
        
        Args:
            api_provider: LLM API provider to use ("anthropic", "openai", "gemini", etc.)
            model_name: Name of the language model to use
            modality: Modality for input ("vision-text", "text-only")
            thinking: Whether to enable deep thinking in prompts
            cache_dir: Directory to cache game state and screenshots
            system_prompt: System prompt to use for the LLM
            max_threads: Maximum number of threads for parallel reasoning
            max_history_tokens: Maximum number of tokens to keep in history
            context_window: Number of recent interactions to maintain in context
            log_dir: Directory for saving logs and screenshots
        """
        # Use default game-specific system prompt if none provided
        if system_prompt is None:
            system_prompt = (
                "You are an expert AI agent specialized in solving 2048 puzzles optimally. "
                "Your goal is to get the highest score without freezing the game board. The ultimate goal is to reach 2048. "
                "You are particularly skilled at planning ahead and making strategic moves to maximize score."
            )
            
        # Initialize the base GameAgent
        super().__init__(
            game_name="2048",
            api_provider=api_provider,
            model_name=model_name,
            modality=modality,
            thinking=thinking,
            cache_dir=cache_dir,
            system_prompt=system_prompt,
            max_history_tokens=max_history_tokens,
            context_window=context_window,
            log_dir=log_dir,
        )
        
        # 2048-specific attributes
        self.max_threads = max_threads
        self.prev_responses = []  # For tracking previous moves and reasoning
        self.board_state = None
        
    async def read_game_board(self, screenshot_path: str) -> str:
        """
        Extract the 2048 game board from a screenshot using vision.
        
        Args:
            screenshot_path: Path to the screenshot to process
            
        Returns:
            Formatted text representation of the game board
        """
        from tools.serving.api_providers import (
            anthropic_completion,
            anthropic_text_completion,
            openai_completion,
            openai_text_reasoning_completion,
            gemini_completion,
            gemini_text_completion
        )
        
        base64_image = encode_image(screenshot_path)
        
        # Custom prompt for 2048 board extraction
        prompt = (
            "Extract the 2048 puzzel board layout from the provided image. "
            "Use the existing 4 * 4 grid to generate a text table to represent the game board. "
            "For each square block, recognize the value at center of this block. If it is empty just label it as empty "
            "Strictly format the output as: **value (row, column)**. "
            "Each row should reflect the board layout. "
            "Example format: \n2 (0, 0) | 4 (1, 0)| 16 (2, 0) | 8 (3, 0) \nempty (0,1) | 2 (1, 1)| empty (2, 1)... "
        )
        
        # Call the appropriate API based on provider and modality
        self.file_logger.info(f"Using {self.model_name} for text table generation...")
        
        try:
            if self.api_provider == "anthropic" and self.modality == "text-only":
                response = anthropic_text_completion(self.system_prompt, self.model_name, prompt, self.thinking)
            elif self.api_provider == "anthropic":
                response = anthropic_completion(self.system_prompt, self.model_name, base64_image, prompt, self.thinking)
            elif self.api_provider == "openai" and "o3" in self.model_name and self.modality == "text-only":
                response = openai_text_reasoning_completion(self.system_prompt, self.model_name, prompt)
            elif self.api_provider == "openai":
                response = openai_completion(self.system_prompt, self.model_name, base64_image, prompt)
            elif self.api_provider == "gemini" and self.modality == "text-only":
                response = gemini_text_completion(self.system_prompt, self.model_name, prompt)
            elif self.api_provider == "gemini":
                response = gemini_completion(self.system_prompt, self.model_name, base64_image, prompt)
            else:
                raise NotImplementedError(f"API provider: {self.api_provider} is not supported.")
            
            # Process response and format as structured board output
            structured_board = response.strip()
            
            # Generate final text output
            final_output = "\n2048 Puzzle Board Representation:\n" + structured_board
            
            # Save the board state
            self.board_state = structured_board
            
            return final_output
            
        except Exception as e:
            self.file_logger.error(f"Error reading game board: {e}")
            return f"Error reading board: {str(e)}"
            
    def process_move(self, move: str, thought: str) -> None:
        """
        Process a move by logging it and tracking it in memory.
        
        Args:
            move: Direction to move ("up", "down", "left", "right")
            thought: Reasoning behind the move
        """
        # Ensure valid move direction
        if move not in ["up", "down", "left", "right"]:
            self.file_logger.warning(f"Invalid move: {move}")
            return
            
        # Record the move and thought
        self.file_logger.info(f"Move: {move}, Thought: {thought}")
        
        # Save to prev_responses for future context
        latest_response = f"move: {move}, thought: {thought}"
        self.prev_responses.append(latest_response)
        
        # Keep only the most recent responses
        if len(self.prev_responses) > 5:
            self.prev_responses = self.prev_responses[-5:]
            
    def execute_action(self, action: Union[str, Dict[str, Any]]) -> None:
        """
        Execute a game action by simulating keypresses.
        
        Args:
            action: Action to execute, either as a string or a dictionary
        """
        # Extract the move if it's a dictionary
        if isinstance(action, dict):
            move = action.get("action", "")
        else:
            move = action
            
        # Map move to keyboard keys
        key_map = {
            "up": "up",
            "down": "down",
            "left": "left",
            "right": "right",
            "restart": "r"
        }
        
        # Validate and execute the move
        if move.lower() in key_map:
            key = key_map[move.lower()]
            self.file_logger.info(f"Executing action: {move} (key: {key})")
            pyautogui.press(key)
            
            # Allow a moment for animation
            time.sleep(0.2)
        else:
            self.file_logger.warning(f"Invalid action: {move}")
            
    async def get_action(self, observation: Dict[str, Any], prev_action: Optional[str] = None) -> Dict[str, bool]:
        """
        Get the next action based on the current game observation.
        
        Args:
            observation: Dictionary containing 'screen' (PIL Image) and 'buttons' (available buttons)
            prev_action: The previous action that was taken
            
        Returns:
            Dictionary mapping button names to boolean values (pressed/not pressed)
        """
        # Increment step count
        self.step_count += 1
        self.file_logger.info(f"Step {self.step_count}: Getting next action")
        
        # Save and process the screenshot
        image_dir = self.log_dir / "game_screen"
        image_dir.mkdir(exist_ok=True)
        
        # Save the observation image
        screenshot_path = image_dir / f"screenshot_{self.step_count}.png"
        observation['screen'].save(screenshot_path)
        self.file_logger.info(f"Saved observation image to {screenshot_path}")
        
        # Process the image to extract the game board
        annotate_image_path, _, annotate_cropped_image_path = get_annotate_img(
            screenshot_path,
            crop_left=0, crop_right=0, crop_top=0, crop_bottom=0,
            grid_rows=4, grid_cols=4,
            enable_digit_label=False,
            cache_dir=self.cache_dir,
            black=True
        )
        
        # Read the game board from the annotated image
        table = await self.read_game_board(annotate_cropped_image_path)
        self.file_logger.info(f"Game board:\n{table}")
        
        # Construct reasoning prompt with game board and memory context
        prev_response_text = "\n".join(self.prev_responses)
        
        prompt = (
            "## Previous Lessons Learned\n"
            "- The 2048 board is structured as a 4x4 grid where each tile holds a power-of-two number.\n"
            "- You can slide tiles in four directions (up, down, left, right), merging identical numbers when they collide.\n"
            "- Your goal is to maximize the score and reach the highest possible tile, ideally 2048 or beyond.\n"
            "- You are an expert AI agent specialized in solving 2048 optimally, utilizing advanced heuristic strategies.\n"
            "- Before making a move, evaluate all possible board states and consider which action maximizes the likelihood of long-term success.\n"
            "- Prioritize maintaining an ordered grid structure to prevent the board from filling up prematurely.\n"
            "- Always keep the highest-value tile in a stable corner to allow efficient merges and maintain control of the board.\n"
            "- Minimize unnecessary movements that disrupt tile positioning and reduce future merge opportunities.\n"
            
            "**IMPORTANT: You must always try a valid direction that leads to a merge. If there are no available merges in the current direction, moving in that direction is invalid.**\n"

            "## Potential Errors to Avoid:\n"
            "1. Grid Disorder Error: Moving tiles in a way that disrupts the structured arrangement of numbers.\n"
            "2. Edge Lock Error: Moving the highest tile out of a stable corner, reducing long-term strategic control.\n"
            "3. Merge Delay Error: Failing to merge tiles early, causing a filled board with no valid moves.\n"
            "4. Tile Isolation Error: Creating a situation where smaller tiles are blocked from merging.\n"
            "5. Forced Move Error: Reaching a state where only one move is possible, reducing strategic flexibility.\n"

            f"Here is your previous response: {prev_response_text}\n\n"
            "Here is the current state of the 2048 board:\n"
            f"{table}\n\n"

            "### Output Format:\n"
            "move: up/down/left/right, thought: <brief reasoning>\n\n"
            "Example output: move: left, thought: Maintaining the highest tile in the corner while creating merge opportunities."
        )
        
        # Get reasoning from LLM
        from tools.serving.api_providers import (
            anthropic_text_completion,
            anthropic_completion,
            openai_text_reasoning_completion, 
            openai_completion,
            gemini_text_completion,
            gemini_completion
        )
        
        base64_image = encode_image(annotate_cropped_image_path)
        if "o3-mini" in self.model_name:
            base64_image = None
            
        start_time = time.time()
        
        try:
            self.file_logger.info(f"Calling {self.model_name} API...")
            
            if self.api_provider == "anthropic" and self.modality == "text-only":
                response = anthropic_text_completion(self.system_prompt, self.model_name, prompt, self.thinking)
            elif self.api_provider == "anthropic":
                response = anthropic_completion(self.system_prompt, self.model_name, base64_image, prompt, self.thinking)
            elif self.api_provider == "openai" and "o3" in self.model_name and self.modality == "text-only":
                response = openai_text_reasoning_completion(self.system_prompt, self.model_name, prompt)
            elif self.api_provider == "openai":
                response = openai_completion(self.system_prompt, self.model_name, base64_image, prompt)
            elif self.api_provider == "gemini" and self.modality == "text-only":
                response = gemini_text_completion(self.system_prompt, self.model_name, prompt)
            elif self.api_provider == "gemini":
                response = gemini_completion(self.system_prompt, self.model_name, base64_image, prompt)
            else:
                raise NotImplementedError(f"API provider: {self.api_provider} is not supported.")
                
            latency = time.time() - start_time
            self.file_logger.info(f"API response time: {latency:.2f}s")
            
            # Extract moves and thoughts from the response
            pattern = r'move:\s*(\w+),\s*thought:\s*(.*)'
            matches = re.findall(pattern, response, re.IGNORECASE)
            
            move_thought_list = []
            for move, thought in matches:
                move = move.strip().lower()
                thought = thought.strip()
                move_thought_list.append({"move": move, "thought": thought})
                
            # If no valid moves found, use fallback logic
            if not move_thought_list:
                self.file_logger.warning("No valid move found in response. Using fallback logic.")
                
                # Try to extract a valid move from the text
                move_match = re.search(r'(?:should|will|recommend|suggest|best move is|move):\s*([a-z]+)', response, re.IGNORECASE)
                if move_match and move_match.group(1).lower() in ["up", "down", "left", "right"]:
                    move = move_match.group(1).lower()
                    thought = "Extracted move from response using fallback logic."
                    move_thought_list.append({"move": move, "thought": thought})
                else:
                    # Choose a random move if all else fails
                    import random
                    move = random.choice(["up", "down", "left", "right"])
                    thought = "Random move chosen as fallback strategy."
                    move_thought_list.append({"move": move, "thought": thought})
                    
            # Process the first move (or the only move)
            chosen_move = move_thought_list[0]
            
            # Record move and thought in memory
            self.process_move(chosen_move["move"], chosen_move["thought"])
            
            # Execute the move
            self.execute_action(chosen_move["move"])
            
            # Format the response
            button_states = {
                "up": chosen_move["move"] == "up",
                "down": chosen_move["move"] == "down",
                "left": chosen_move["move"] == "left",
                "right": chosen_move["move"] == "right"
            }
            
            return button_states
            
        except Exception as e:
            self.file_logger.error(f"Error in get_action: {e}")
            # Return a default safe move (left) in case of error
            return {"up": False, "down": False, "left": True, "right": False}
            
    async def run_game(self, max_steps: int = 1000) -> None:
        """
        Run the game agent for a specified number of steps.
        
        Args:
            max_steps: Maximum number of steps to take
        """
        self.file_logger.info(f"Starting 2048 game agent with {max_steps} max steps")
        
        prev_action = None
        
        for step in range(max_steps):
            self.file_logger.info(f"Step {step+1}/{max_steps}")
            
            # Take screenshot
            screenshot_path = self.take_screenshot(f"step_{step+1}.png")
            if not screenshot_path:
                self.file_logger.error("Failed to take screenshot. Stopping.")
                break
                
            # Create an observation dict
            from PIL import Image
            observation = {
                "screen": Image.open(screenshot_path),
                "buttons": ["up", "down", "left", "right"]
            }
            
            # Get the next action
            action = await self.get_action(observation, prev_action)
            
            # Determine the chosen direction
            chosen_direction = None
            for direction, pressed in action.items():
                if pressed:
                    chosen_direction = direction
                    break
                    
            prev_action = chosen_direction
            
            # Add a small delay between actions
            await asyncio.sleep(0.5)
            
        self.file_logger.info("Game run completed") 