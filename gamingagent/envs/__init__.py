# Main adapter for integrating gym environments with the agent's expected interaction loop
from .gym_env_adapter import GymEnvAdapter

# Utility functions for environments, e.g., image generation
from .env_utils import create_board_image_2048


__all__ = [
    "GymEnvAdapter",
    "create_board_image_2048",
]
