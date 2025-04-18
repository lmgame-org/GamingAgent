"""
GameAgent - Base class for game agents across different games
"""

from typing import Any, Dict, List, Optional, Union, Callable, Tuple
import time
import os
import re
import numpy as np
import threading
import json
import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Message:
    """Container for a message in the conversation history."""
    
    def __init__(self, role: str, content: Any, has_image: bool = False, tokens: int = 0):
        self.role = role
        self.content = content
        self.has_image = has_image
        self.tokens = tokens  # Approximate token count
        self.timestamp = datetime.now()
        
    def __str__(self):
        if isinstance(self.content, str):
            preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
            return f"{self.role}: {preview}"
        else:
            return f"{self.role}: [Complex content with image={self.has_image}]"


class GameAgent(ABC):
    """
    General GameAgent parent class for all game agents.
    
    This class provides the core functionality needed across different game agents,
    including vision processing, reasoning, memory management, and real-time control.
    """
    
    def __init__(
        self,
        game_name: str,
        api_provider: str = "anthropic",
        model_name: str = "claude-3-7-sonnet-20250219",
        modality: str = "vision-text",
        thinking: bool = True,
        cache_dir: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_history_tokens: int = 4000,
        context_window: int = 10,
        log_dir: Optional[Path] = None,
    ):
        """
        Initialize the game agent.
        
        Args:
            game_name: Name of the game this agent is designed for
            api_provider: LLM API provider to use ("anthropic", "openai", "gemini", etc.)
            model_name: Name of the language model to use
            modality: Modality for input ("vision-text", "text-only")
            thinking: Whether to enable deep thinking in prompts
            cache_dir: Directory to cache game state and screenshots
            system_prompt: System prompt to use for the LLM
            max_history_tokens: Maximum number of tokens to keep in history
            context_window: Number of recent interactions to maintain in context
            log_dir: Directory for saving logs and screenshots
        """
        self.game_name = game_name
        self.api_provider = api_provider
        self.model_name = model_name
        self.modality = modality
        self.thinking = thinking
        self.max_history_tokens = max_history_tokens
        self.context_window = context_window
        
        # Set up cache directory
        if cache_dir is None:
            self.cache_dir = f"cache/{game_name.lower().replace(' ', '_')}"
        else:
            self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Set up log directory
        if log_dir is None:
            model_name_safe = model_name.replace("/", "-").replace(".", "-")
            self.log_dir = Path("logs") / f"{game_name.lower().replace(' ', '_')}" / model_name_safe / datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.log_dir = log_dir
            
        # Set up file logger
        self.file_logger = self._setup_file_logger()
        
        # Set system prompt
        if system_prompt is None:
            self.system_prompt = f"You are an expert AI agent specialized in playing {game_name}. "
        else:
            self.system_prompt = system_prompt
            
        # Memory storage
        self.short_term_memory: List[Message] = []
        self.long_term_memory: List[Dict[str, Any]] = []
        self.reflection_memory = ""
        self.memory_lock = threading.Lock()
        
        # Worker threads
        self.active_workers = {}
        self.worker_lock = threading.Lock()
        self.stop_workers = False
        
        # State tracking
        self.current_state = {}
        self.state_history = []
        self.last_screenshot_path = None
        self.last_action_time = 0
        self.last_vision_time = 0
        self.step_count = 0
        
        # Create consolidated log files
        self.reflection_log_file = self.log_dir / "reflections.txt"
        self.reflection_log_file.touch()
        
        logger.info(f"{self.__class__.__name__} initialized with {model_name} for {game_name}")
        self.file_logger.info(f"Agent initialized. Logging to: {self.log_dir}")
        
    def _setup_file_logger(self) -> logging.Logger:
        """Set up a file logger for this session."""
        file_logger = logging.getLogger(f"{self.__class__.__name__.lower()}_{id(self)}")
        file_logger.setLevel(logging.INFO)
        
        # Create file handler
        log_file = self.log_dir / "agent_session.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        
        # Add handler to logger
        file_logger.addHandler(file_handler)
        
        return file_logger
        
    def vision_worker(
        self, 
        screenshot_path: str, 
        custom_prompt: Optional[str] = None,
        custom_system_prompt: Optional[str] = None,
        callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Process a screenshot to extract game state information.
        
        Args:
            screenshot_path: Path to the screenshot to process
            custom_prompt: Custom prompt to use for the vision model
            custom_system_prompt: Custom system prompt to override the default
            callback: Optional callback function to call with the result
            
        Returns:
            Dictionary containing the extracted game state information
        """
        from tools.utils import encode_image
        from tools.serving.api_providers import (
            anthropic_completion,
            openai_completion,
            gemini_completion,
            anthropic_text_completion,
            openai_text_reasoning_completion,
            gemini_text_completion
        )
        
        # Use the provided system prompt or the default
        system_prompt = custom_system_prompt or self.system_prompt
        
        # Default prompt if none provided
        if custom_prompt is None:
            custom_prompt = (
                f"Analyze this {self.game_name} game screenshot and describe the current state. "
                f"Identify key elements including player position, objects, enemies, and any text visible."
            )
        
        # Encode the image
        base64_image = encode_image(screenshot_path) if self.modality == "vision-text" else None
        
        # Store the screenshot path
        self.last_screenshot_path = screenshot_path
        self.last_vision_time = time.time()
        
        # Call the appropriate API based on provider and modality
        try:
            if self.api_provider == "anthropic" and self.modality == "text-only":
                response = anthropic_text_completion(system_prompt, self.model_name, custom_prompt, self.thinking)
            elif self.api_provider == "anthropic":
                response = anthropic_completion(system_prompt, self.model_name, base64_image, custom_prompt, self.thinking)
            elif self.api_provider == "openai" and "o3" in self.model_name and self.modality == "text-only":
                response = openai_text_reasoning_completion(system_prompt, self.model_name, custom_prompt)
            elif self.api_provider == "openai":
                response = openai_completion(system_prompt, self.model_name, base64_image, custom_prompt)
            elif self.api_provider == "gemini" and self.modality == "text-only":
                response = gemini_text_completion(system_prompt, self.model_name, custom_prompt)
            elif self.api_provider == "gemini":
                response = gemini_completion(system_prompt, self.model_name, base64_image, custom_prompt)
            else:
                raise NotImplementedError(f"API provider: {self.api_provider} is not supported.")
        
            # Process the response
            result = self._process_vision_response(response, screenshot_path)
            
            # Add to memory
            self._add_to_short_term_memory(Message(
                role="vision",
                content=response,
                has_image=True,
                tokens=len(response.split()) * 1.3  # Rough token estimation
            ))
            
            # Call callback if provided
            if callback is not None:
                callback(result)
                
            return result
            
        except Exception as e:
            self.file_logger.error(f"Error in vision worker: {e}")
            return {"error": str(e)}
    
    def reasoning_worker(
        self,
        game_state: Dict[str, Any],
        context: Optional[str] = None,
        custom_prompt: Optional[str] = None,
        custom_system_prompt: Optional[str] = None,
        callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Process game state to decide on the next action.
        
        Args:
            game_state: Current game state information
            context: Additional context information
            custom_prompt: Custom prompt to use for the reasoning model
            custom_system_prompt: Custom system prompt to override the default
            callback: Optional callback function to call with the result
            
        Returns:
            Dictionary containing the next action to take
        """
        from tools.serving.api_providers import (
            anthropic_text_completion,
            openai_text_reasoning_completion,
            gemini_text_completion
        )
        
        # Use the provided system prompt or the default
        system_prompt = custom_system_prompt or self.system_prompt
        
        # Get memory context
        memory_context = self._get_memory_context()
        
        # Update current state
        self.current_state = game_state.copy()
        self.state_history.append(self.current_state)
        
        # Default prompt if none provided
        if custom_prompt is None:
            custom_prompt = (
                f"Based on the current {self.game_name} state, decide on the best move to make. "
                f"Current state: {game_state}\n"
                f"Memory context: {memory_context}\n"
                f"Additional context: {context or 'None'}\n\n"
                f"Respond with your reasoning and the action to take in the format:\n"
                f"move: <action>, thought: <reasoning>"
            )
        
        # Call the appropriate API
        try:
            if self.api_provider == "anthropic":
                response = anthropic_text_completion(system_prompt, self.model_name, custom_prompt, self.thinking)
            elif self.api_provider == "openai" and "o3" in self.model_name:
                response = openai_text_reasoning_completion(system_prompt, self.model_name, custom_prompt)
            elif self.api_provider == "gemini":
                response = gemini_text_completion(system_prompt, self.model_name, custom_prompt)
            else:
                raise NotImplementedError(f"Text reasoning for {self.api_provider} is not supported.")
        
            # Process the response
            result = self._process_reasoning_response(response)
            
            # Track the time
            current_time = time.time()
            
            # Add to memory
            self._add_to_short_term_memory(Message(
                role="reasoning",
                content=response,
                has_image=False,
                tokens=len(response.split()) * 1.3  # Rough token estimation
            ))
            
            # Store the action time
            self.last_action_time = current_time
            
            # Call callback if provided
            if callback is not None:
                callback(result)
                
            return result
            
        except Exception as e:
            self.file_logger.error(f"Error in reasoning worker: {e}")
            return {"error": str(e), "action": "none"}
            
    def memory_worker(
        self,
        query: str,
        memory_type: str = "short",
        limit: int = 5,
        custom_system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a query against the agent's memory.
        
        Args:
            query: Question or query to process against memory
            memory_type: Type of memory to query ("short", "long", "reflection")
            limit: Maximum number of memory items to return
            custom_system_prompt: Custom system prompt to override the default
            
        Returns:
            Dictionary containing the memory query results
        """
        from tools.serving.api_providers import anthropic_text_completion
        
        # Get the appropriate memory content
        if memory_type == "short":
            memory_items = self.short_term_memory[-limit:]
            memory_content = "\n".join([f"{m.role}: {m.content}" for m in memory_items])
        elif memory_type == "long":
            memory_items = self.long_term_memory[-limit:]
            memory_content = json.dumps(memory_items, indent=2)
        elif memory_type == "reflection":
            memory_content = self.reflection_memory
        else:
            raise ValueError(f"Unknown memory type: {memory_type}")
            
        # Use the provided system prompt or the default
        system_prompt = custom_system_prompt or self.system_prompt
        
        # Create the prompt for querying memory
        prompt = (
            f"The following is the agent's {memory_type} memory content:\n\n"
            f"{memory_content}\n\n"
            f"Query: {query}\n\n"
            f"Please provide a comprehensive answer to this query based on the memory content."
        )
        
        try:
            # Call the API
            response = anthropic_text_completion(system_prompt, self.model_name, prompt, False)
            
            # Return the result
            return {
                "query": query,
                "memory_type": memory_type,
                "response": response
            }
            
        except Exception as e:
            self.file_logger.error(f"Error in memory worker: {e}")
            return {"error": str(e)}
            
    def start_real_time_worker(
        self,
        worker_type: str,
        interval: float,
        worker_args: Dict[str, Any] = {},
        worker_id: Optional[str] = None
    ) -> str:
        """
        Start a real-time worker thread that executes periodically.
        
        Args:
            worker_type: Type of worker to start ("vision", "reasoning", "custom")
            interval: Time interval between worker executions in seconds
            worker_args: Arguments to pass to the worker
            worker_id: Optional ID for the worker, generated if not provided
            
        Returns:
            ID of the created worker
        """
        # Generate worker ID if not provided
        if worker_id is None:
            worker_id = f"{worker_type}_{time.time()}"
            
        # Create worker thread
        thread = threading.Thread(
            target=self._run_worker_thread,
            args=(worker_type, interval, worker_args, worker_id),
            daemon=True
        )
        
        # Register worker
        with self.worker_lock:
            self.active_workers[worker_id] = {
                "thread": thread,
                "type": worker_type,
                "interval": interval,
                "args": worker_args,
                "started": time.time()
            }
            
        # Start thread
        thread.start()
        self.file_logger.info(f"Started {worker_type} worker with ID {worker_id} at interval {interval}s")
        
        return worker_id
        
    def _run_worker_thread(
        self,
        worker_type: str,
        interval: float,
        worker_args: Dict[str, Any],
        worker_id: str
    ):
        """
        Worker thread function that runs a specific worker at intervals.
        """
        def worker_thread():
            while not self.stop_workers:
                try:
                    # Execute the appropriate worker based on type
                    if worker_type == "vision":
                        result = self.vision_worker(**worker_args)
                    elif worker_type == "reasoning":
                        result = self.reasoning_worker(**worker_args)
                    elif worker_type == "custom" and "func" in worker_args:
                        custom_func = worker_args.pop("func")
                        result = custom_func(**worker_args)
                    else:
                        self.file_logger.error(f"Unknown worker type: {worker_type}")
                        break
                        
                    # Log the result
                    self.file_logger.info(f"Worker {worker_id} executed with result: {result}")
                    
                except Exception as e:
                    self.file_logger.error(f"Error in worker {worker_id}: {e}")
                    
                # Wait for the next execution
                sleep_time = max(0.1, interval - 0.1)  # Ensure some minimum sleep time
                for _ in range(int(sleep_time * 10)):  # Check stop flag every 0.1 seconds
                    if self.stop_workers:
                        break
                    time.sleep(0.1)
                    
        return worker_thread()
        
    def stop_real_time_worker(self, worker_id: str) -> bool:
        """
        Stop a specific real-time worker thread.
        
        Args:
            worker_id: ID of the worker to stop
            
        Returns:
            True if the worker was stopped, False otherwise
        """
        with self.worker_lock:
            if worker_id in self.active_workers:
                # Get worker thread
                worker = self.active_workers[worker_id]
                
                # Log the stop
                self.file_logger.info(f"Stopping worker {worker_id} of type {worker['type']}")
                
                # Remove worker
                del self.active_workers[worker_id]
                return True
                
        return False
        
    def stop_all_workers(self):
        """Stop all real-time worker threads."""
        self.stop_workers = True
        self.file_logger.info("Stopping all workers")
        time.sleep(0.2)  # Give threads time to notice the stop flag
        
    def custom_worker(
        self,
        custom_func: Callable,
        func_args: Dict[str, Any] = {},
        custom_system_prompt: Optional[str] = None
    ) -> Any:
        """
        Execute a custom worker function.
        
        Args:
            custom_func: Custom function to execute
            func_args: Arguments to pass to the function
            custom_system_prompt: Custom system prompt to override the default
            
        Returns:
            Result of the custom function
        """
        try:
            # Log the execution
            self.file_logger.info(f"Executing custom worker function: {custom_func.__name__}")
            
            # Set system prompt if provided
            if custom_system_prompt is not None:
                old_prompt = self.system_prompt
                self.system_prompt = custom_system_prompt
                
            # Execute function
            start_time = time.time()
            result = custom_func(**func_args)
            execution_time = time.time() - start_time
            
            # Restore system prompt if changed
            if custom_system_prompt is not None:
                self.system_prompt = old_prompt
                
            # Log result
            self.file_logger.info(f"Custom worker executed in {execution_time:.2f}s")
            
            return result
            
        except Exception as e:
            self.file_logger.error(f"Error in custom worker: {e}")
            return {"error": str(e)}
            
    def _process_vision_response(self, response: str, screenshot_path: str) -> Dict[str, Any]:
        """
        Process the vision model's response into a structured format.
        
        Args:
            response: Raw response from the vision model
            screenshot_path: Path to the processed screenshot
            
        Returns:
            Structured dictionary with the extracted information
        """
        # Basic pattern matching to extract structured data
        # In a real implementation, this would be more sophisticated
        elements = {}
        
        # Store full response
        elements["full_text"] = response
        
        # Extract player position if mentioned
        player_match = re.search(r'player .* position:? \(?(\d+),\s*(\d+)\)?', response, re.IGNORECASE)
        if player_match:
            elements["player_position"] = (int(player_match.group(1)), int(player_match.group(2)))
            
        # Extract game objects
        elements["objects"] = []
        object_matches = re.finditer(r'(\w+) (?:at|located at) \(?(\d+),\s*(\d+)\)?', response, re.IGNORECASE)
        for match in object_matches:
            elements["objects"].append({
                "type": match.group(1),
                "position": (int(match.group(2)), int(match.group(3)))
            })
            
        return elements
        
    def _process_reasoning_response(self, response: str) -> Dict[str, Any]:
        """
        Process the reasoning model's response into a structured format.
        
        Args:
            response: Raw response from the reasoning model
            
        Returns:
            Structured dictionary with the extracted information
        """
        # Extract move and thought
        move_match = re.search(r'move:\s*([^,\n]+)[,\n]?\s*thought:\s*(.+)', response, re.IGNORECASE | re.DOTALL)
        
        if move_match:
            action = move_match.group(1).strip()
            thought = move_match.group(2).strip()
            
            # Extract a potential reflection if present
            reflection_match = re.search(r'reflection:\s*(.+)', thought, re.IGNORECASE | re.DOTALL)
            reflection = reflection_match.group(1).strip() if reflection_match else None
            
            return {
                "action": action,
                "thought": thought,
                "reflection": reflection
            }
        else:
            # Fallback to best-effort action extraction
            action_match = re.search(r'(?:should|will|action|move):\s*([^,\n\.]+)', response, re.IGNORECASE)
            action = action_match.group(1).strip() if action_match else "unknown"
            
            return {
                "action": action,
                "thought": response,
                "reflection": None
            }
            
    def _add_to_short_term_memory(self, memory_item: Message):
        """
        Add an item to short-term memory with thread safety.
        
        Args:
            memory_item: Message object to add to memory
        """
        with self.memory_lock:
            self.short_term_memory.append(memory_item)
            
            # Prune memory if too large
            if len(self.short_term_memory) > self.context_window * 2:
                self.short_term_memory = self.short_term_memory[-self.context_window * 2:]
                
    def _add_to_long_term_memory(self, memory_item: Dict[str, Any]):
        """
        Add an item to long-term memory with thread safety.
        
        Args:
            memory_item: Dictionary with memory content
        """
        with self.memory_lock:
            self.long_term_memory.append({
                **memory_item,
                "timestamp": datetime.now().isoformat()
            })
            
    def _get_memory_context(self, limit: int = 5) -> str:
        """
        Get a formatted string of recent memory items.
        
        Args:
            limit: Maximum number of memory items to include
            
        Returns:
            Formatted string with recent memory items
        """
        with self.memory_lock:
            # Get the most recent items
            recent_items = self.short_term_memory[-limit:]
            
            # Format as a string
            memory_lines = []
            for i, item in enumerate(recent_items):
                if item.has_image:
                    memory_lines.append(f"[{i+1}] {item.role}: [Image observation with text: {item.content[:100]}...]")
                else:
                    memory_lines.append(f"[{i+1}] {item.role}: {item.content[:200]}...")
                    
            return "\n".join(memory_lines)
            
    def take_screenshot(self, filename: Optional[str] = None) -> str:
        """
        Take a screenshot of the current game state.
        
        Args:
            filename: Optional custom filename for the screenshot
            
        Returns:
            Path to the saved screenshot
        """
        import pyautogui
        
        # Generate filename if not provided
        if filename is None:
            filename = f"screenshot_{time.time()}.png"
            
        # Ensure the path is within the cache directory
        screenshot_path = os.path.join(self.cache_dir, filename)
        
        # Take the screenshot
        try:
            screenshot = pyautogui.screenshot()
            screenshot.save(screenshot_path)
            self.last_screenshot_path = screenshot_path
            self.file_logger.info(f"Screenshot saved to {screenshot_path}")
            return screenshot_path
        except Exception as e:
            self.file_logger.error(f"Error taking screenshot: {e}")
            return ""
            
    def execute_action(self, action: Union[str, Dict[str, Any]]):
        """
        Execute an action in the game.
        
        Args:
            action: Action to execute, either as a string or a dictionary
            
        This method should be implemented by subclasses to perform
        game-specific actions.
        """
        raise NotImplementedError("Subclasses must implement execute_action")
        
    def reset(self):
        """Reset the agent state."""
        self.file_logger.info("Resetting agent state")
        self.short_term_memory = []
        self.current_state = {}
        self.state_history = []
        self.last_screenshot_path = None
        self.stop_all_workers()
        
    def save_reflection(self) -> None:
        """Save reflection to consolidated file."""
        if self.reflection_memory:
            with open(self.reflection_log_file, "a") as f:
                f.write(f"\n\n-=-=-=-=-=-reflection_{self.step_count}-=-=-=-=-=\n\n")
                f.write(self.reflection_memory)
            self.file_logger.info(f"Updated reflection memory saved to {self.reflection_log_file}")
            
    def update_steps_count(self, count: Optional[int] = None) -> None:
        """Update the steps counter."""
        if count is not None:
            self.step_count = count
        else:
            self.step_count += 1

    @abstractmethod
    async def get_action(self, observation: Dict[str, Any], prev_action: Optional[str] = None) -> Dict[str, bool]:
        """
        Get the next action based on the current game observation.
        
        Args:
            observation: Dictionary containing game state information
            prev_action: The previous action that was taken
            
        Returns:
            Dictionary mapping button names to boolean values (pressed/not pressed)
        """
        pass 