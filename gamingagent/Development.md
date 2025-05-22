# GamingAgent Development Map

## 1. Project File Layout

```
GamingAgent/
├── configs/
│   ├── <game_name>/                 # Example: twenty_forty_eight
│   │   ├── config.yaml              # Main configuration for the game and agent defaults
│   │   └── module_prompts.json      # Prompts for different agent modules
│   └── ...                          # Other game configurations
├── gamingagent/
│   ├── agents/
│   │   ├── __init__.py
│   │   └── base_agent.py            # Abstract Base Class for all agents
│   ├── envs/
│   │   ├── __init__.py              # Factory function get_game_env_wrapper()
│   │   ├── base_env.py              # Abstract Base Class for game environment wrappers
│   │   └── custom_01_2048/          # Example custom game environment
│   │       ├── __init__.py
│   │       ├── twentyFortyEight_env.py # Wrapper for the 2048 game
│   │       └── game_env_config.json    # Specific overrides for 2048 env
│   │   └── <other_game_envs>/       # Wrappers for other games (custom or retro)
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── core_module.py           # Base class for all modules, handles API and logging
│   │   ├── base_module.py           # Simple module for direct action (no harness)
│   │   ├── perception_module.py     # Handles game state perception
│   │   ├── memory_module.py         # Handles agent's memory
│   │   └── reasoning_module.py      # Handles planning and decision making
│   ├── __init__.py
│   └── runner.py                    # Main script to run game episodes with an agent
├── tools/
│   ├── serving/                     # LLM API management
│   │   ├── __init__.py
│   │   ├── api_manager.py           # Manages calls to different LLM providers
│   │   ├── api_providers.py         # Implementations for specific LLM APIs (Anthropic, OpenAI, etc.)
│   ├── __init__.py
│   └── utils.py                     # Utility functions (e.g., convert_numpy_to_python)
├── tests/                           # Unit and integration tests
│   └── ...
├── runs_output/                     # Default directory for logs and agent cache
│   └── <agent_cache_subfolder>/     # Created by BaseAgent (e.g., <timestamp>_<game>...)
│       ├── agent_config.json
│       ├── observations/            # Saved game state images for vision-based agents
│       │   └── ep<episode_id>_st<step_num>_<game_name>_board.png
│       ├── <module_name>.json       # Logs for each module (e.g., base_module.json)
│       ├── episode_XXX_log.jsonl    # Detailed log for each step in an episode
│       └── run_summary.json         # Summary of the entire run (multiple episodes)
├── README.md
├── requirements.txt
├── setup_env.sh                     #stores api keys; in .gitignore
├── pyproject.toml
└── .gitignore
```

## 2. Config File Templates

### `configs/<game_name>/config.yaml`

```yaml
game_env:
  name: "twenty_forty_eight"
  description: "2048 game environment"
  env_type: "custom"
  render_mode: "human"
  max_steps: 1000
  seed: 42
  num_runs: 3

agent:
  # Global agent settings
  name: "2048_agent"
  model_name: "claude-3-5-sonnet-latest"
  cache_dir: "cache/twenty_forty_eight"
  reasoning_effort: "high"
  token_limit: 100000
  harness: false  # Whether to use the perception-memory-reasoning pipeline
  
  # Module-specific settings
  modules:
    base_module:
      observation_mode: "vision"  # Options: "vision", "text", "both"
    
    perception_module:

    memory_module:
      max_memory: 10  # Maximum number of memory entries to store
    
    reasoning_module:

```

### `configs/<game_name>/module_prompts.json`

```json
{
  "base_module": {
    "system_prompt": "You are an AI agent playing a game. Analyze the game state and decide the best move. Your response MUST be in the format:\nthought: [your detailed thought process]\nmove: [your chosen move string]",
    "prompt": "Current game state:\n{context}\n\nPrevious action: {last_action}\nPrevious thought: {last_thought}\n\nConsider the board and your previous attempt. What is your next move and reasoning?"
  },
  "perception_module": {
    "system_prompt": "You are an AI module designed to interpret game states. Extract key information from the provided game observation.",
    "prompt": "Analyze the following game observation (text or image) and provide a symbolic representation and detailed game state:\n{context}\n\nFormat:\nsymbolic_representation: [Structured data or concise text summary]\ngame_state_details: [Detailed observations or analysis]"
  },
  "memory_module": {
    "system_prompt": "You are an AI module responsible for managing game history and generating reflections to improve strategy.",
    "prompt_reflection": "Reflect on the past sequence of game states, actions, and outcomes:\n{history}\nWhat insights can be gained? What strategies were effective or ineffective? Provide a concise reflection."
  },
  "reasoning_module": {
    "system_prompt": "You are an AI agent playing a game. Based on your perception and memory, plan your next action. Your response MUST be in the format:\nthought: [your detailed thought process]\nmove: [your chosen move string]",
    "prompt": "Current Perception:\n{perception_summary}\n\nMemory Summary:\n{memory_summary}\n\nBased on the current situation and past experiences, decide the optimal next move."
  }
}
```

### `gamingagent/envs/<custom_game_folder>/game_env_config.json` (Example for 2048)

```json
{
    "env_id": "gymnasium_2048:gymnasium_2048/TwentyFortyEight-v0",
    "env_init_kwargs": {
        "size": 4,
        "max_pow": 16
    },
    "action_mapping": {
        "up": 0,
        "right": 1,
        "down": 2,
        "left": 3
    },
    "render_mode_gym_make": "human",
    "max_unchanged_steps_for_termination": 10
}
```

## 3. Defined Functions and Methods

### Core Components:

**`gamingagent.modules.core_module.CoreModule(ABC)`**
  *   `__init__(self, module_name, model_name, system_prompt, prompt, cache_dir, token_limit, reasoning_effort)`
  *   `log(self, data)`: Logs data to the module's JSON file.
  *   `_parse_response(self, response)`: (Abstract) Parses LLM response.
  *   `api_manager`: Instance of `APIManager` for LLM calls.

**`gamingagent.modules.base_module.BaseModule(CoreModule)`**
  *   `__init__(self, ..., observation_mode, ...)`
  *   `execute(self, observation: Observation, last_action: Optional[str] = None, last_thought: Optional[str] = None) -> dict`: Executes the module logic (calls LLM).
  *   `_call_llm(self, context: str, img_path: Optional[str] = None) -> str`: Makes the actual LLM call.
  *   `_parse_response(self, response: str) -> dict`: Parses LLM response for `thought` and `action`.

**`gamingagent.modules.perception_module.PerceptionModule(CoreModule)`**
  *   `__init__(self, ..., observation: Observation, ...)`
  *   `update_observation(self, new_observation: Observation)`
  *   `get_perception_summary(self) -> dict`: (Abstract) Analyzes current observation.
  *   `_parse_response(self, response: str) -> dict`: Parses LLM response for perception data.

**`gamingagent.modules.memory_module.MemoryModule(CoreModule)`**
  *   `__init__(self, ..., max_memory: int, ...)`
  *   `add_memory(self, game_state: Any, action: str, reward: float, thought: str, reflection: Optional[str] = None)`
  *   `generate_reflection(self, prev_perception: str, current_perception: str, last_action: str, last_thought: str) -> str`: (Abstract) Generates reflection on past experiences.
  *   `get_memory_summary(self) -> dict`: (Abstract) Provides a summary of relevant memories.
  *   `_parse_response(self, response: str) -> dict`: Parses LLM response for reflection.

**`gamingagent.modules.reasoning_module.ReasoningModule(CoreModule)`**
  *   `__init__(self, ...)`
  *   `plan_action(self, perception_data: dict, memory_summary: dict, img_path: Optional[str] = None) -> dict`: (Abstract) Plans the next action.
  *   `_call_vision_api(self, context: str, img_path: str) -> str`
  *   `_call_text_api(self, context: str, custom_prompt: Optional[str] = None) -> str`
  *   `_parse_response(self, response: str) -> dict`: Parses LLM response for `thought` and `action`.

**`gamingagent.agents.base_agent.BaseAgent(ABC)`**
  *   `__init__(self, game_name, model_name, observation_mode, config_root_dir, cache_dir_prefix, harness, max_memory, custom_modules=None)`
  *   `get_action(self, observation: Observation) -> dict`: Main method to get agent's action.
  *   `_get_action_harness(self, observation: Observation) -> dict`: Implements Perception -> Memory -> Reasoning pipeline.
  *   `_get_action_direct(self, observation: Observation) -> dict`: Uses `BaseModule` for direct action.
  *   `_initialize_modules(self, custom_modules=None) -> dict`: Sets up agent modules.
  *   `_save_agent_config(self)`: Saves agent's configuration.
  *   `_load_prompts(self, config: dict) -> dict`: Loads prompts using `AgentConfigLoader`.
  *   `observations_dir`: Property for path to save observation images.
  *   `cache_dir`: Property for agent's cache directory.

**`gamingagent.envs.base_env.BaseGameEnv(ABC)`**
  *   `__init__(self, game_name, observation_mode, agent_observations_base_dir, env_type, config_root_dir, log_root_dir)`
  *   `reset(self, seed: Optional[int] = None, episode_id: int = 1) -> Tuple[Observation, Dict[str, Any]]`: Resets the environment.
  *   `step(self, agent_action_str: Optional[str], thought_process: str, time_taken_s: float) -> Tuple[Observation, float, bool, bool, Dict[str, Any]]`: Takes a step in the environment.
  *   `close(self)`: Closes the environment and log files.
  *   `render_human(self)`: Renders the environment for human viewing if applicable.
  *   `extract_observation(self, raw_observation: Any, info: Dict[str, Any]) -> Observation`: (Abstract) Converts raw env observation to agent `Observation`.
  *   `verify_termination(self, observation: Observation, current_terminated: bool, current_truncated: bool) -> Tuple[bool, bool]`: Verifies if episode should terminate (e.g., due to being stuck). Can be overridden.
  *   `_initialize_env(self) -> None`: Concrete method to initialize the gym/retro environment using `self.env_id`, `self.env_type`, `self.env_init_kwargs`, `self.render_mode_for_make`.
  *   `_load_config(self)`: Loads `config.yaml`.
  *   `_log_and_print_step_data(...)`: Logs step data to file and console.
  *   `_create_agent_observation_path(self, episode_id: int, step_num: int) -> str`: Generates path for saving observation images.
  *   `map_agent_action_to_env_action(self, agent_action_str: Optional[str]) -> Optional[int]`: Maps string action to environment action ID.
  *   `map_env_action_to_agent_action(self, env_action_idx: int) -> Optional[str]`: Maps environment action ID to string.
  *   `get_current_episode_step_num(self) -> Tuple[int, int]`.

**`gamingagent.envs.custom_01_2048.twentyFortyEight_env.TwentyFortyEightEnvWrapper(BaseGameEnv)`** (Example)
  *   `__init__(self, ...)`: Loads 2048-specific `game_env_config.json`, sets up `_max_unchanged_steps`.
  *   `extract_observation(self, raw_observation: Any, info: Dict[str, Any]) -> Observation`: Implements 2048-specific observation extraction (text, image using `create_board_image_2048`).
  *   `get_board_state(self, raw_observation: Any, info: Dict[str, Any]) -> np.ndarray`: Extracts 2048 board.
  *   `verify_termination(self, observation: Observation, current_terminated: bool, current_truncated: bool) -> Tuple[bool, bool]`: Implements "stuck" detection for 2048.
  *   `create_board_image_2048(board_powers: np.ndarray, save_path: str, size: int = 400)`: (Helper function, could be static or outside class)

## 4. Runner Execution Command

To run the agent, use the `runner.py` script. For the default 2048 game:

```bash
python ./gamingagent/runner.py --game_name twenty_forty_eight
```

**Common optional arguments for `runner.py`:**
*   `--model_name <model_id>`: Specify the LLM model (e.g., "claude-3-opus-20240229", "gpt-4").
*   `--observation_mode <mode>`: "text", "vision", or "both".
*   `--harness <true_false>`: `true` for full P-M-R pipeline, `false` for BaseModule.
*   `--num_episodes <number>`: Number of episodes to run.
*   `--max_steps_per_episode <number>`: Max steps per episode.
*   `--cache_dir_prefix <prefix>`: Prefix for the agent's output/cache directory.
*   `--seed <number>`: Seed for environment reproducibility.
*   `--env_type <type>`: "custom" or "retro". Default is "custom".

Example with more options:
```bash
python ./gamingagent/runner.py --game_name twenty_forty_eight --model_name claude-3-haiku-20240307 --observation_mode vision --harness false --num_episodes 5
```

## 5. Notes

*   **Agent & Modules Standardization**: The `BaseAgent` and the `CoreModule`-derived modules (`BaseModule`, `PerceptionModule`, `MemoryModule`, `ReasoningModule`) are designed to be highly standardized. Game-specific logic within these components is primarily handled through prompts loaded from configuration files (`module_prompts.json`). The core Python code of the agent and modules aims to be game-agnostic.
*   **Environment Game-Specificity**: The `BaseGameEnv` provides a standard interface for the agent to interact with any game. However, concrete implementations (like `TwentyFortyEightEnvWrapper`) are highly game-specific. They are responsible for:
    *   Initializing the actual game environment (e.g., a specific Gymnasium or Retro environment).
    *   Mapping abstract agent actions to game-specific action IDs.
    *   Extracting observations from the raw game state into a standardized `Observation` object (text, image path, symbolic data). This often involves game-specific parsing or rendering (e.g., creating a board image for 2048).
    *   Optionally, implementing custom logic for `verify_termination` if the game has specific conditions for being "stuck" or unrecoverable.
