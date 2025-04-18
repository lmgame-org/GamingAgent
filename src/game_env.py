from typing import Optional, Type, Dict, Any, Tuple
from game_agents.base_agent import BaseGameAgent
import importlib
import os
import datetime
import shutil
from PIL import ImageGrab  # For screenshot capability

class GameEnv:
    """
    Game environment class that manages game setup and execution.
    Similar to OpenAI Gym interface for reinforcement learning environments.
    """
    
    # Dictionary mapping game names to their corresponding agent classes
    GAME_AGENT_MAP = {
        "super_mario_bros": "SuperMarioAgent",
        "2048": "Tile2048Agent",
        "ace_attorney": "AceAttorneyAgent",
        "candy_crush": "CandyCrushAgent",
        "sokoban": "SokobanAgent",
        "tetris": "TetrisAgent"
    }
    
    def __init__(
        self,
        game_name: str,
        model_name: str,
        modality: str,
        api_provider: str,
        thinking: bool = False,
        custom_agent: Optional[Type[BaseGameAgent]] = None,
        base_cache_dir: str = "cache",
    ):
        """
        Initialize the game environment.
        
        Args:
            game_name: Name of the game to play
            model_name: Name of the model to use
            modality: Type of input/output (e.g., "text", "image")
            api_provider: Provider of the API (e.g., "openai", "anthropic")
            thinking: Whether to use thinking mode (special for Claude)
            custom_agent: Optional custom agent class to use instead of default
            base_cache_dir: Base directory for caching game data and API responses
            capture_frequency: How often to capture observations (every N steps)
        """
        self.game_name = game_name
        self.model_name = model_name
        self.modality = modality
        self.api_provider = api_provider
        self.thinking = thinking
        self.custom_agent = custom_agent
        self.base_cache_dir = base_cache_dir
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Tracking variables
        self.is_open = True
        self.step_count = 0
        self.screenshots_dir = None

        # Set up cache directories
        self._setup_cache_directories()
        
        # Initialize the appropriate agent
        self.agent = self._initialize_agent()
        
        print(f"Game environment initialized for {game_name}. Cache directory: {self.session_dir}")
    
    def _setup_cache_directories(self):
        """Set up the cache directory structure consistent with APIManager."""
        # Create base cache directory if it doesn't exist
        os.makedirs(self.base_cache_dir, exist_ok=True)
        
        # Create game-specific directory
        self.game_dir = os.path.join(self.base_cache_dir, self.game_name)
        os.makedirs(self.game_dir, exist_ok=True)
        
        # Create model-specific directory
        clean_model_name = self.model_name.lower().split('/')[-1] if '/' in self.model_name else self.model_name.lower()
        self.model_dir = os.path.join(self.game_dir, clean_model_name)
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Create modality-specific directory
        self.modality_dir = os.path.join(self.model_dir, self.modality)
        os.makedirs(self.modality_dir, exist_ok=True)
        
        # Create timestamp-based session directory
        self.session_dir = os.path.join(self.modality_dir, self.timestamp)
        os.makedirs(self.session_dir, exist_ok=True)
        
        # Create screenshots directory inside session directory
        self.screenshots_dir = os.path.join(self.session_dir, "screenshots")
        os.makedirs(self.screenshots_dir, exist_ok=True)
    
    def _initialize_agent(self) -> BaseGameAgent:
        """
        Initialize the appropriate agent based on game name or custom agent.
        
        Returns:
            BaseGameAgent: The initialized agent instance
        """
        if self.custom_agent:
            # Use the provided custom agent
            return self.custom_agent()
        
        # Get the agent class name from the mapping
        agent_class_name = self.GAME_AGENT_MAP.get(self.game_name)
        if not agent_class_name:
            raise ValueError(f"No agent found for game: {self.game_name}")
        
        # Dynamically import and instantiate the agent
        try:
            # Assuming agents are in the game_agents package
            module = importlib.import_module(f"game_agents.{self.game_name}_agent")
            agent_class = getattr(module, agent_class_name)
            return agent_class()
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Failed to initialize agent for {self.game_name}: {str(e)}")
    
    def _get_step_params(self, img_path: str) -> Dict[str, Any]:
        """
        Create a dictionary of parameters for the step method.
        
        Args:
            img_path: Path to the current screenshot
            
        Returns:
            Dict[str, Any]: Dictionary containing all step parameters
        """
        return {
            "img_path": img_path,
            "session_dir": self.session_dir,
            "model_name": self.model_name,
            "api_provider": self.api_provider,
            "modality": self.modality,
            "game_name": self.game_name,
            "thinking": self.thinking,
            "timestamp": self.timestamp
        }
    
    def step(self, img_path: str) -> Dict[str, Any]:
        """
        Execute one step in the game using the agent's step method.
        Similar to the OpenAI Gym interface.
        
        Args:
            img_path: Path to the current screenshot
            
        Returns:
            Tuple containing:
                - observation (Dict): Current state observation
                - reward (float): Reward from the action
                - done (bool): Whether the episode is finished
                - info (Dict): Additional information
        """
        if not self.is_open:
            raise RuntimeError("Environment is closed. Please create a new environment instance.")
        
        # Increment the step counter
        self.step_count += 1
        
        # Get step parameters
        step_params = self._get_step_params(img_path)
        
        # Execute the agent's step method with unpacked parameters
        step_result = self.agent.step(**step_params)

        return step_result
    
    def observation(self) -> Dict[str, Any]:
        """
        Capture the current state of the environment as an observation.
        Takes a screenshot and saves it to the screenshots directory.
        
        Returns:
            Dict[str, Any]: Dictionary containing observation data
        """
        if not self.is_open:
            raise RuntimeError("Environment is closed. Please create a new environment instance.")
        
        try:
            # Take a screenshot
            screenshot = ImageGrab.grab()
            
            # Save screenshot to the screenshots directory
            screenshot_path = os.path.join(
                self.screenshots_dir, 
                f"screenshot_{self.step_count:06d}.png"
            )
            screenshot.save(screenshot_path)
            
            return screenshot_path
        except Exception as e:
            print(f"Error capturing observation: {e}")
            return None
    
    def reset(self) -> str:
        """
        Reset the environment to the initial state.
        Creates a new session directory with a new timestamp.
        
        Returns:
            str: Path to the initial screenshot
        """
        if not self.is_open:
            raise RuntimeError("Environment is closed. Please create a new environment instance.")
        
        # Reset step counter
        self.step_count = 0
        
        # Create new timestamp and session directory
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(self.modality_dir, self.timestamp)
        os.makedirs(self.session_dir, exist_ok=True)
        
        # Recreate screenshots directory
        self.screenshots_dir = os.path.join(self.session_dir, "screenshots")
        os.makedirs(self.screenshots_dir, exist_ok=True)
        
        # Take initial observation
        initial_screenshot = self.observation()
        
        print(f"Environment reset. New session directory: {self.session_dir}")
        
        return initial_screenshot
    
    def close(self):
        """
        Close the environment and clean up resources.
        This method should be called when done using the environment.
        """
        if not self.is_open:
            print("Environment already closed.")
            return
        
        try:
            # Mark as closed
            self.is_open = False
            
            # Perform any necessary cleanup
            print(f"Closing environment. Total steps: {self.step_count}")
            print(f"Session data stored in: {self.session_dir}")
            
            # Additional cleanup can be added here if needed
            # For example:
            # - Closing any open files
            # - Terminating subprocesses
            # - Finalizing logs
            
        except Exception as e:
            print(f"Error during environment cleanup: {e}")
    
    def __enter__(self):
        """Support for 'with' statement."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Automatically close the environment when exiting a 'with' block."""
        self.close()
    
    def get_agent_info(self) -> Dict[str, Any]:
        """
        Get information about the current agent and environment setup.
        
        Returns:
            Dict[str, Any]: Dictionary containing environment and agent information
        """
        return {
            "game_name": self.game_name,
            "model_name": self.model_name,
            "modality": self.modality,
            "api_provider": self.api_provider,
            "thinking": self.thinking,
            "agent_type": type(self.agent).__name__,
            "is_custom_agent": self.custom_agent is not None,
            "cache_dir": self.session_dir,
            "screenshots_dir": self.screenshots_dir,
            "step_count": self.step_count,
            "is_open": self.is_open
        }
