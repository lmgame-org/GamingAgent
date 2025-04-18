from abc import ABC, abstractmethod
from typing import Any, List, Optional

class BaseGameAgent(ABC):
    """
    Abstract base class for game agents.
    Defines the core interface that all game agents must implement.
    """
    
    def __init__(self):
        self.workers: List[Any] = []
    
    @abstractmethod
    def worker(self, *args, **kwargs) -> Any:
        """
        Abstract method for defining a worker function.
        Each agent can implement multiple workers for different tasks.
        
        Returns:
            Any: The result of the worker's computation
        """
        pass
    
    @abstractmethod
    def step(self, *args, **kwargs) -> Any:
        """
        Abstract method that defines how to run workers together
        to play one step in the game.
        
        Returns:
            Any: The result of the game step
        """
        pass
    
    def add_worker(self, worker: Any) -> None:
        """
        Add a worker to the agent's worker list.
        
        Args:
            worker: The worker to add
        """
        self.workers.append(worker)
    
    def get_workers(self) -> List[Any]:
        """
        Get all workers associated with this agent.
        
        Returns:
            List[Any]: List of workers
        """
        return self.workers 