from src.executor import Executor

SUPPORTED_GAMES = [
    "sokoban",
    "mario",
    "tetris",
    "2048",
    "candy",
    "ace"
]

def run_game_env(game, model_name, provider, max_steps=100, headless=False):
    games_to_run = SUPPORTED_GAMES if game == "all" else [game.lower()]

    for game_name in games_to_run:
        print(f"\n[ENV] Starting: {game_name}")
        env = GameEnv(
            game_name=game_name,
            model_name=model_name,
            provider=provider,
            max_steps=max_steps,
            headless=headless,
        )
        env.run()

class GameEnv:
    def __init__(self, game_name, model_name, provider, max_steps=100, headless=False):
        self.game_name = game_name.lower()
        self.model_name = model_name
        self.provider = provider
        self.max_steps = max_steps
        self.headless = headless
        self.step_count = 0

        self.agent = self._load_agent()
        self.executor = Executor(mode="pygame", game=self.game_name)

    def _load_agent(self):
        if self.game_name == "sokoban":
            from src.game_agents.sokoban_agent import SokobanAgent
            return SokobanAgent(self.model_name, self.provider)
        elif self.game_name == "mario":
            from src.game_agents.super_mario_agent import SuperMarioAgent
            return SuperMarioAgent(self.model_name, self.provider)
        elif self.game_name == "tetris":
            from src.game_agents.tetris_agent import TetrisAgent
            return TetrisAgent(self.model_name, self.provider)
        elif self.game_name == "2048":
            from src.game_agents.tile_2048_agent import Tile2048Agent
            return Tile2048Agent(self.model_name, self.provider)
        elif self.game_name == "candy":
            from src.game_agents.candy_crush_agent import CandyCrushAgent
            return CandyCrushAgent(self.model_name, self.provider)
        elif self.game_name == "ace":
            from src.game_agents.ace_attorney_agent import AceAttorneyAgent
            return AceAttorneyAgent(self.model_name, self.provider)
        else:
            raise ValueError(f"[GameEnv] Unsupported game: {self.game_name}")

    def run(self):
        while self.step_count < self.max_steps:
            move = self.agent.step()
            if move is None:
                print(f"[{self.game_name.upper()}] Step {self.step_count}: No move")
                self.step_count += 1
                continue

            self.executor.execute(move)

            # Custom game termination check (can improve this)
            if move == "solve" or move == "win":
                print(f"[{self.game_name.upper()}] Game completed.")
                break

            self.step_count += 1
            print(f"[{self.game_name.upper()}] Step {self.step_count}: move = {move}")
