from .base_env import BaseGameEnv
from .custom_01_2048.twentyFortyEight_env import TwentyFortyEightEnvWrapper

GAME_ENV_WRAPPERS = {
    "twenty_forty_eight": TwentyFortyEightEnvWrapper,
    # Add other game wrappers here as they are created
    # "sokoban": SokobanEnvWrapper,
}

def get_game_env_wrapper(game_name: str, *args, **kwargs) -> BaseGameEnv:
    """Factory function to get a game environment wrapper."""
    # Ensure game_name key lookup is consistent (e.g. twenty_forty_eight)
    lookup_key = game_name.lower().replace(" ", "_").replace("-", "_")
    wrapper_class = GAME_ENV_WRAPPERS.get(lookup_key)
    if wrapper_class:
        # Pass the original game_name (which might have specific casing/hyphens needed for config loading)
        return wrapper_class(game_name=game_name, *args, **kwargs)
    else:
        raise ValueError(f"No environment wrapper found for game: {game_name} (looked for key: {lookup_key})")
