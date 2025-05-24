from gymnasium.envs.registration import register

register(id="TileMatch-v0", entry_point="tile_match_gym.tile_match_env:TileMatchEnv")
