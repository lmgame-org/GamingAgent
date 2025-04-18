from abc import ABC, abstractmethod
from typing import Any, List, Optional, Dict, Tuple
from dataclasses import dataclass

@dataclass
class Worker:
    name: str
    function: Any

class BaseGameAgent(ABC):
    """
    Abstract base class for game agents.
    Defines the core interface that all game agents must implement.
    """
    
    def __init__(
        self,
        game_name: str,
        model_name: str,
        modality: str,
        api_provider: str,
        thinking: bool = False,
        session_dir: Optional[str] = None,
        game_config: Optional[Dict[str, Any]] = None
    ):
        # Environment and model context
        self.game_name = game_name
        self.model_name = model_name
        self.modality = modality
        self.api_provider = api_provider
        self.thinking = thinking
        self.session_dir = session_dir
        self.game_config = game_config or {}

        # Runtime state
        self.step_count: int = 0
        self.done: bool = False
        self.last_observation: Any = None
        self.last_action: Any = None
        self.last_reward: float = 0.0
    
    @abstractmethod
    def worker(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Abstract method for defining a worker function.
        Each agent can implement multiple workers for different tasks.
        
        Returns:
            api_response: str - The API response from the worker
        """
        pass
    
    def step(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Execute all workers in sequence and combine their results.
        
        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
            
        Returns:
            Dict[str, Any]: Combined results from all workers with keys:
                - thought: str - Combined thoughts from all workers
                - action: str - Final action to take
                - reward: str - Combined reward (empty string by default)
                - done: bool - Whether the game is complete
        """
        # Reset state for new step
        self.thought = ""
        self.action = ""
        self.reward = ""
        self.done = False
        
        # Execute all workers in sequence
        for worker in self.workers:
            result = worker.function(*args, **kwargs)
            
            # Combine thoughts
            if result.get("thought"):
                self.thought += f"{worker.name}: {result['thought']}\n"
            
            # Update action (last worker's action takes precedence)
            if result.get("action"):
                self.action = result["action"]
            
            # Combine rewards
            if result.get("reward"):
                self.reward += f"{worker.name}: {result['reward']}\n"
            
            # If any worker says we're done, we're done
            if result.get("done", False):
                self.done = True
        
        return {
            "thought": self.thought.strip(),
            "action": self.action,
            "reward": self.reward.strip(),
            "done": self.done
        }
    
    def add_worker(self, name: str, worker_function: Any) -> None:
        """
        Add a worker to the agent's worker list.
        
        Args:
            name: The name of the worker (e.g., "vision_worker", "reasoning_worker")
            worker_function: The worker function to add
        """
        self.workers.append(Worker(name=name, function=worker_function))
    
    def get_workers(self) -> List[Worker]:
        """
        Get all workers associated with this agent.
        
        Returns:
            List[Worker]: List of workers with their names
        """
        return self.workers
    
    def get_worker_graph(self) -> List[str]:
        """
        Get the computational graph of workers as a list of worker names.
        
        Returns:
            List[str]: List of worker names in execution order
        """
        return [worker.name for worker in self.workers] 