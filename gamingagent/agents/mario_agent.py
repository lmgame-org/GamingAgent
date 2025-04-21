import time
import numpy as np
from typing import Any, Dict, Optional, Tuple, Union, List
from .base_agent import BaseAgent
from gamingagent.providers import APIProviderManager
from gamingagent.utils.utils import encode_image
import threading
import asyncio
import concurrent.futures
from queue import Queue
import json
import os
from datetime import datetime

class MarioAgent(BaseAgent):
    """Mario agent that uses API providers for decision making with async workers."""
    
    def __init__(
        self, 
        env: Any,
        provider_manager: APIProviderManager,
        model_name: str = "claude-3-opus-20240229",
        max_tokens: int = 1000,
        temperature: float = 0.7,
        num_threads: int = 4,
        concurrency_interval: float = 1.0,
        api_response_latency_estimate: float = 7.0,
        record_bk2: bool = False
    ):
        """Initialize Mario agent with API provider."""
        super().__init__(
            env=env,
            game_name="SuperMarioBros-Nes",
            api_provider="anthropic",
            model_name=model_name
        )
        
        self.provider_manager = provider_manager
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.record_bk2 = record_bk2
        
        # Action queues and locks
        self.short_term_queue = Queue(maxsize=5)  # Store up to 5 short-term actions
        self.long_term_queue = Queue(maxsize=5)   # Store up to 5 long-term actions
        self.action_lock = threading.Lock()
        
        # Store current observation
        self.current_observation = None
        self.observation_lock = threading.Lock()
        
        # Game state control
        self.is_running = True
        self.game_done = False
        self.state_lock = threading.Lock()
        
        # Rate limiting
        self.last_api_call_time = 0
        self.min_api_call_interval = 1.0
        
        # Set number of threads and calculate offsets
        self.num_threads = num_threads
        self.offsets = [i * concurrency_interval for i in range(self.num_threads)]
        
        # Event loop for async operations
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        # Create API logs directory
        self.api_logs_dir = os.path.join(self.cache_dir, "api_logs")
        os.makedirs(self.api_logs_dir, exist_ok=True)
        
        # Create BK2 recording directory if needed
        if self.record_bk2:
            self.bk2_dir = os.path.join(self.cache_dir, "recordings")
            os.makedirs(self.bk2_dir, exist_ok=True)
            self.current_recording_path = None
        
        # Start worker threads
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads)
        self.start_workers()
        
    def _log_api_interaction(self, prompt: str, response: str, thread_type: str, step: int) -> None:
        """Log API interaction details to file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_data = {
                "timestamp": timestamp,
                "thread_type": thread_type,
                "step": step,
                "prompt": prompt,
                "response": response,
                "model": self.model_name,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
            
            # Save to JSON file
            log_file = os.path.join(self.api_logs_dir, f"api_call_{timestamp}_{thread_type}_{step}.json")
            with open(log_file, "w") as f:
                json.dump(log_data, f, indent=4)
                f.flush()  # Ensure data is written to disk
                os.fsync(f.fileno())  # Force sync to disk
                
        except Exception as e:
            print(f"Error logging API interaction: {str(e)}")
            
    async def _make_api_call(self, prompt: str, image: str, thread_type: str, step: int) -> Optional[str]:
        """Make an async API call with rate limiting and logging."""
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call_time
        
        if time_since_last_call < self.min_api_call_interval:
            wait_time = self.min_api_call_interval - time_since_last_call
            print(f"Rate limiting: waiting {wait_time:.2f}s")
            await asyncio.sleep(wait_time)
            
        try:
            print(f"Making API call to {self.model_name} (thread: {thread_type}, step: {step})")
            start_time = time.time()
            response = await self.loop.run_in_executor(
                None,
                lambda: self.provider_manager.anthropic.generate_with_images(
                    prompt=prompt,
                    images=[image],
                    model=self.model_name,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
            )
            end_time = time.time()
            duration = end_time - start_time
            
            self.last_api_call_time = time.time()
            
            print(f"API call completed in {duration:.2f}s")
            print(f"Response length: {len(response)}")
            print(f"Response preview: {response[:200]}...")
            
            # Log full interaction
            self._log_api_interaction(prompt, response, thread_type, step)
            
            return response
        except Exception as e:
            print(f"API call error: {str(e)}")
            if "rate_limit" in str(e).lower():
                self.min_api_call_interval *= 1.5
                print(f"Rate limit hit, increasing interval to {self.min_api_call_interval}s")
            return None
            
    async def _short_term_worker(self, thread_id: int, offset: float):
        """Async worker for short-term decisions."""
        await asyncio.sleep(offset)
        print(f"[Thread {thread_id} - SHORT] Starting after {offset}s delay...")
        
        step = 0
        while True:
            try:
                with self.observation_lock:
                    observation = self.current_observation
                    
                if observation is None:
                    await asyncio.sleep(0.1)
                    continue
                    
                print(f"[Thread {thread_id} - SHORT] Making API call at step {step}")
                prompt = self._format_short_term_prompt()
                encoded_image = encode_image(observation)
                
                response = await self._make_api_call(prompt, encoded_image, "short_term", step)
                if response:
                    print(f"[Thread {thread_id} - SHORT] Received response at step {step}")
                    action = self.parse_response(response)
                    if not self.short_term_queue.full():
                        self.short_term_queue.put(action)
                        print(f"[Thread {thread_id} - SHORT] Action queued at step {step}")
                    else:
                        print(f"[Thread {thread_id} - SHORT] Queue full at step {step}")
                else:
                    print(f"[Thread {thread_id} - SHORT] No response received at step {step}")
                        
                step += 1
                await asyncio.sleep(0.1)
            except Exception as e:
                print(f"[Thread {thread_id} - SHORT] Error: {str(e)}")
                await asyncio.sleep(0.1)
                
    async def _long_term_worker(self, thread_id: int, offset: float):
        """Async worker for long-term decisions."""
        await asyncio.sleep(offset)
        print(f"[Thread {thread_id} - LONG] Starting after {offset}s delay...")
        
        step = 0
        while True:
            try:
                with self.observation_lock:
                    observation = self.current_observation
                    
                if observation is None:
                    await asyncio.sleep(0.1)
                    continue
                    
                print(f"[Thread {thread_id} - LONG] Making API call at step {step}")
                prompt = self._format_long_term_prompt()
                encoded_image = encode_image(observation)
                
                response = await self._make_api_call(prompt, encoded_image, "long_term", step)
                if response:
                    print(f"[Thread {thread_id} - LONG] Received response at step {step}")
                    action = self.parse_response(response)
                    if not self.long_term_queue.full():
                        self.long_term_queue.put(action)
                        print(f"[Thread {thread_id} - LONG] Action queued at step {step}")
                    else:
                        print(f"[Thread {thread_id} - LONG] Queue full at step {step}")
                else:
                    print(f"[Thread {thread_id} - LONG] No response received at step {step}")
                        
                step += 1
                await asyncio.sleep(0.1)
            except Exception as e:
                print(f"[Thread {thread_id} - LONG] Error: {str(e)}")
                await asyncio.sleep(0.1)
                
    def start_workers(self):
        """Start async worker tasks."""
        print("Starting worker threads...")
        for i in range(self.num_threads):
            if i % 2 == 0:
                print(f"Starting long-term worker {i}")
                self.loop.create_task(self._long_term_worker(i, self.offsets[i]))
            else:
                print(f"Starting short-term worker {i}")
                self.loop.create_task(self._short_term_worker(i, self.offsets[i]))
                
        # Start the event loop in a separate thread
        def run_loop():
            print("Starting event loop...")
            self.loop.run_forever()
            
        self.worker_thread = threading.Thread(target=run_loop, daemon=True)
        self.worker_thread.start()
        print("Worker threads started successfully")
        
    async def select_action_async(self, observation: Union[np.ndarray, Tuple[np.ndarray, Dict]]) -> np.ndarray:
        """Asynchronously select an action based on the current observation."""
        if isinstance(observation, tuple):
            obs = observation[0]
        else:
            obs = observation
            
        # Update current observation
        with self.observation_lock:
            self.current_observation = obs
            print(f"Updated observation")
            
        # Get action from appropriate queue
        try:
            # Try to get short-term action first
            if not self.short_term_queue.empty():
                action = self.short_term_queue.get_nowait()
                print(f"Got short-term action from queue")
            # Fall back to long-term action
            elif not self.long_term_queue.empty():
                action = self.long_term_queue.get_nowait()
                print(f"Got long-term action from queue")
            else:
                # Default action if no actions in queues
                print("No actions in queues, using default action")
                action = np.array([False, False, False, False, False, False, False, True, False], dtype=np.uint8)
                
            # Log the action
            self.log_action(action, 0)
            self.log_state(obs, 0)
            
            return action
        except Exception as e:
            print(f"Error selecting action: {str(e)}")
            return np.array([False, False, False, False, False, False, False, True, False], dtype=np.uint8)
        
    async def reset_async(self) -> None:
        """Asynchronously reset the agent's state."""
        # Clear action queues
        while not self.short_term_queue.empty():
            self.short_term_queue.get_nowait()
        while not self.long_term_queue.empty():
            self.long_term_queue.get_nowait()
            
        # Reset observation
        with self.observation_lock:
            self.current_observation = None
            
        # Reset rate limiting
        self.last_api_call_time = 0
        self.min_api_call_interval = 1.0
        
        # Stop any ongoing recording
        self.stop_recording()
        
    async def close_async(self) -> None:
        """Asynchronously clean up resources."""
        # Stop any ongoing recording
        self.stop_recording()
        
        # Stop the event loop
        print("Stopping event loop...")
        self.loop.stop()
        
        # Cancel all running tasks
        for task in asyncio.all_tasks(self.loop):
            task.cancel()
            
        # Wait for tasks to complete
        await asyncio.gather(*asyncio.all_tasks(self.loop), return_exceptions=True)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
    def select_action(self, observation: Union[np.ndarray, Tuple[np.ndarray, Dict]]) -> np.ndarray:
        """Synchronously select an action based on the current observation."""
        if isinstance(observation, tuple):
            obs = observation[0]
        else:
            obs = observation
            
        # Update current observation
        with self.observation_lock:
            self.current_observation = obs
            
        # Get action from appropriate queue
        try:
            # Try to get short-term action first
            if not self.short_term_queue.empty():
                action = self.short_term_queue.get_nowait()
            # Fall back to long-term action
            elif not self.long_term_queue.empty():
                action = self.long_term_queue.get_nowait()
            else:
                # Default action if no actions in queues
                action = np.array([False, False, False, False, False, False, False, True, False], dtype=np.uint8)
                
            # Log the action
            self.log_action(action, 0)
            self.log_state(obs, 0)
            
            return action
        except Exception as e:
            print(f"Error selecting action: {str(e)}")
            return np.array([False, False, False, False, False, False, False, True, False], dtype=np.uint8)
        
    def reset(self) -> None:
        """Synchronously reset the agent's state."""
        super().reset()
        
    def close(self) -> None:
        """Synchronously clean up resources."""
        super().close()

    def _format_short_term_prompt(self) -> str:
        """Format prompt for short-term decisions."""
        return """Analyze the current game state and generate the next action for Mario for the next 1 second.
Focus on immediate obstacles, enemies, and hazards.

Game State Analysis:
1. Look for:
   - Immediate obstacles or enemies
   - Gaps or pits that need immediate jumping
   - Power-ups that can be quickly collected
   - Enemies that are about to hit Mario

2. Short-term Strategy:
   - React quickly to immediate threats
   - Make small adjustments to avoid obstacles
   - Use quick jumps to avoid enemies
   - Collect nearby power-ups
   - If in immediate danger, move left to avoid

Controls:
- A (index 8): Jump - Use for quick jumps over enemies or gaps
- B (index 0): Run - Use for quick acceleration
- RIGHT (index 7): Move right - Use for forward movement
- LEFT (index 6): Move left - Use to avoid immediate threats
- UP (index 4): Look up - Use to check for overhead threats
- DOWN (index 5): Crouch - Use to avoid projectiles

Output Format:
Return ONLY a Python list of 9 boolean values: [B, null, SELECT, START, UP, DOWN, LEFT, RIGHT, A]"""
        
    def _format_long_term_prompt(self) -> str:
        """Format prompt for long-term decisions."""
        return """Analyze the current game state and generate the next action for Mario for the next 2 seconds.
Focus on strategic planning and progress.

Game State Analysis:
1. Look for:
   - Upcoming obstacles or enemies
   - Strategic positions for jumps
   - Power-up locations
   - Safe paths forward
   - Areas that need preparation

2. Long-term Strategy:
   - Plan ahead for upcoming obstacles
   - Position Mario for optimal jumps
   - Look for opportunities to collect power-ups
   - Identify safe paths forward
   - Prepare for upcoming challenges
   - If unsure, take a defensive position

Controls:
- A (index 8): Jump - Use for planned jumps over gaps
- B (index 0): Run - Use for building momentum
- RIGHT (index 7): Move right - Use for steady progress
- LEFT (index 6): Move left - Use for strategic positioning
- UP (index 4): Look up - Use to plan ahead
- DOWN (index 5): Crouch - Use for strategic positioning

Output Format:
Return ONLY a Python list of 9 boolean values: [B, null, SELECT, START, UP, DOWN, LEFT, RIGHT, A]"""
                
    def parse_response(self, response: str) -> np.ndarray:
        """Parse the API response into a valid action."""
        try:
            print(f"Received response: {response}")
            
            # Find the last occurrence of '[' in the response
            start = response.rfind('[')
            end = response.rfind(']') + 1
            
            if start >= 0 and end > start:
                action_str = response[start:end]
                print(f"Extracted action string: {action_str}")
                
                # Clean up the action string by removing any non-bracket characters
                action_str = ''.join(c for c in action_str if c in '[],TrueFalse')
                
                try:
                    action = eval(action_str)
                    if isinstance(action, list) and len(action) == 9:
                        print(f"Parsed action: {action}")
                        # Print the model's analysis
                        if start > 0:
                            print("Model's analysis:")
                            print(response[:start].strip())
                        return np.array(action, dtype=np.uint8)
                except Exception as e:
                    print(f"Error evaluating action string: {e}")
                    print(f"Action string was: {action_str}")
        except Exception as e:
            print(f"Error parsing response: {e}")
            
        # Default to moving right if parsing fails
        print("Using default action: moving right")
        return np.array([False, False, False, False, False, False, False, True, False], dtype=np.uint8)

    def start_recording(self, episode: int) -> None:
        """Start recording a new BK2 file."""
        if self.record_bk2:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.current_recording_path = os.path.join(
                self.bk2_dir, 
                f"episode_{episode}_{timestamp}.bk2"
            )
            self.env.unwrapped.record_movie(self.current_recording_path)
            print(f"Started recording episode {episode} to {self.current_recording_path}")
            
    def stop_recording(self) -> None:
        """Stop recording the current BK2 file."""
        if self.record_bk2 and self.current_recording_path:
            self.env.unwrapped.stop_record()
            print(f"Stopped recording to {self.current_recording_path}")
            self.current_recording_path = None
