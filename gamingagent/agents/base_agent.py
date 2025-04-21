from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
import numpy as np

class BaseAgent(ABC):
    """Base class for all game-playing agents."""
    
    def __init__(self, env: Any):
        """
        Initialize agent.
        
        Args:
            env: Game environment the agent will interact with
        """
        self.env = env
        
    @abstractmethod
    def reset(self) -> None:
        """Reset agent's internal state."""
        pass
        
    @abstractmethod
    def select_action(self, observation: Any) -> Any:
        """
        Select action based on current observation.
        
        Args:
            observation: Current environment observation
            
        Returns:
            Action to take in the environment
        """
        pass
        
    def update(self, observation: Any, reward: float, done: bool, info: Dict) -> None:
        """
        Update agent's internal state after taking an action.
        
        Args:
            observation: Current observation
            reward: Reward received
            done: Whether episode is done
            info: Additional information
        """
        pass

class LLMAgent(BaseAgent):
    """Base class for LLM-based agents."""
    
    def __init__(self, env: Any, provider: Any, **kwargs):
        """
        Initialize LLM agent.
        
        Args:
            env: Game environment
            provider: LLM provider for decision making
            **kwargs: Additional configuration
        """
        super().__init__(env)
        self.provider = provider
        self.conversation_history = []
        
    def reset(self) -> None:
        """Reset conversation history."""
        self.conversation_history = []
        
    @abstractmethod
    def format_prompt(self, observation: Any) -> str:
        """
        Format observation into LLM prompt.
        
        Args:
            observation: Environment observation
            
        Returns:
            str: Formatted prompt for LLM
        """
        pass
        
    @abstractmethod
    def parse_response(self, response: str) -> Any:
        """
        Parse LLM response into action.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Valid action for the environment
        """
        pass
        
    def select_action(self, observation: Any) -> Any:
        """Get action from LLM based on observation."""
        prompt = self.format_prompt(observation)
        response = self.provider.generate(prompt)
        return self.parse_response(response)