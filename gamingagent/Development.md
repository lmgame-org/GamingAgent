# GamingAgent Development Map

## 1. Project File Layout

```
GamingAgent/
├── gamingagent/                     # Main package
│   ├── agents/
│   │   ├── __init__.py
│   │   └── base_agent.py            # BaseAgent class - main agent implementation
│   ├── envs/                        # Game environment adapters
│   │   ├── __init__.py              # Imports GymEnvAdapter and utilities
│   │   ├── gym_env_adapter.py       # Universal adapter for Gymnasium environments
│   │   ├── env_utils.py             # Utility functions (e.g., image generation)
│   │   ├── README.md                # Environment documentation
│   │   ├── custom_01_2048/          # 2048 game implementation
│   │   │   ├── __init__.py
│   │   │   ├── twentyFortyEightEnv.py
│   │   │   └── game_env_config.json # Action mappings, termination settings
│   │   ├── custom_02_sokoban/       # Sokoban game implementation
│   │   ├── custom_03_candy_crush/   # Candy Crush game implementation
│   │   ├── custom_04_tetris/        # Tetris game implementation
│   │   ├── retro_01_super_mario_bros/ # Super Mario Bros (RetroGym)
│   │   ├── retro_02_ace_attorney/   # Ace Attorney (RetroGym)
│   │   └── retro_03_1942/           # 1942 game (RetroGym)
│   ├── modules/                     # Agent reasoning modules
│   │   ├── __init__.py              # Module imports and Observation class
│   │   ├── core_module.py           # CoreModule base class and Observation dataclass
│   │   ├── base_module.py           # BaseModule for direct action (non-harness mode)
│   │   ├── perception_module.py     # PerceptionModule for visual/text interpretation
│   │   ├── memory_module.py         # MemoryModule for game history and reflection
│   │   ├── reasoning_module.py      # ReasoningModule for action planning
│   │   └── prompt_graph.py          # Prompt graph utilities
│   ├── configs/                     # Game-specific configurations
│   │   ├── custom_01_2048/
│   │   │   ├── config.yaml          # Main game and agent configuration
│   │   │   └── module_prompts.json  # Prompts for each module
│   │   ├── custom_02_sokoban/
│   │   ├── custom_03_candy_crush/
│   │   ├── custom_04_tetris/
│   │   ├── retro_01_super_mario_bros/
│   │   ├── retro_02_ace_attorney/
│   │   └── retro_03_1942/
│   ├── Development.md               # This file
│   └── __init__.py
├── lmgame-bench/                    # Main execution engine
│   ├── custom_runner.py             # Primary runner script for all games
│   ├── run.py                       # Alternative runner
│   ├── evaluate_all.sh              # Batch evaluation script
│   ├── README.md                    # Runner documentation
│   └── __pycache__/
├── tools/                           # Utilities and API management
│   ├── serving/                     # LLM API management
│   │   ├── __init__.py
│   │   ├── api_manager.py           # APIManager class for LLM calls
│   │   ├── api_providers.py         # Provider implementations (Anthropic, OpenAI, etc.)
│   │   ├── api_cost_calculator.py   # Cost tracking and calculation
│   │   ├── constants.py             # API constants and configuration
│   │   └── model_prices.json        # Model pricing data
│   ├── modal/                       # Modal deployment utilities
│   ├── utils.py                     # General utilities (image scaling, type conversion)
│   └── __init__.py
├── eval/                            # Evaluation and analysis tools
│   ├── perf/                        # Performance analysis
│   ├── configs/                     # Evaluation configurations
│   ├── assets/                      # Evaluation assets
│   ├── video_samples/               # Sample videos
│   ├── lmgame_Bench_Evaluation_Pipeline.ipynb # Main evaluation notebook
│   ├── notebook_utils.py            # Jupyter notebook utilities
│   ├── replay_utils.py              # Game replay functionality
│   ├── video_generation_script.py   # Video generation from logs
│   └── __init__.py
├── computer_use/                    # Computer use integration
│   ├── games/                       # Computer use games
│   ├── README.md                    # Computer use documentation
│   └── __init__.py
├── tests/                           # Unit and integration tests
├── cache/                           # Default directory for agent logs and cache
├── logs/                            # System logs
├── assets/                          # Project assets
├── docs/                            # Documentation (currently empty)
├── pyproject.toml                   # Project configuration and dependencies
├── requirements.txt                 # Python dependencies
├── setup_env.sh                     # Environment setup script
├── credentials.sh                   # API credentials setup
├── LICENSE                          # MIT License
├── README.md                        # Main project README
└── .gitignore                       # Git ignore rules
```

## 2. Core Architecture

### 2.1 Agent System (`gamingagent/agents/`)

**`BaseAgent` (base_agent.py)**
- Main agent class that coordinates all modules
- Supports two modes:
  - **Harness mode** (`harness=True`): Uses Perception → Memory → Reasoning pipeline
  - **Non-harness mode** (`harness=False`): Uses BaseModule for direct action
- Manages configuration loading, module initialization, and caching
- Key methods:
  - `get_action(observation)`: Main method to get agent's next action
  - `_initialize_modules()`: Sets up required modules based on configuration
  - `_load_config()`: Loads prompts and configuration from JSON files

### 2.2 Environment System (`gamingagent/envs/`)

**`GymEnvAdapter` (gym_env_adapter.py)**
- Universal adapter that bridges Gymnasium environments with the agent framework
- Handles observation creation, action mapping, logging, and termination detection
- Key features:
  - Observation mode support ("vision", "text", "both")
  - Stuck detection via observation hashing
  - Episode logging to JSONL files
  - Performance scoring and run summarization
  - Action mapping from string commands to environment actions

**Game-Specific Implementations:**
- **Custom Games**: 2048, Sokoban, Candy Crush, Tetris
- **Retro Games**: Super Mario Bros, Ace Attorney, 1942
- Each game has its own wrapper class and `game_env_config.json`

### 2.3 Module System (`gamingagent/modules/`)

**`CoreModule` (core_module.py)**
- Abstract base class for all modules
- Provides API management via `APIManager`
- Handles logging to module-specific JSON files
- Contains the `Observation` dataclass with comprehensive state representation

**`Observation` dataclass:**
- `img_path`: Path to visual observation image
- `textual_representation`: Text-based game state
- `processed_visual_description`: AI-generated description of visual elements
- `game_trajectory`: Historical game states (via `GameTrajectory` class)
- `reflection`: Memory module's reflection on past actions
- `background`: Static episode information

**Module Types:**
1. **`BaseModule`**: Direct action planning (non-harness mode)
2. **`PerceptionModule`**: Visual and textual state interpretation
3. **`MemoryModule`**: Game history management and reflection generation
4. **`ReasoningModule`**: Strategic action planning using perception and memory

### 2.4 Configuration System (`gamingagent/configs/`)

**`config.yaml` structure:**
```yaml
game_env:
  name: "game_name"
  description: "Game description"
  env_type: "custom" | "retro"
  render_mode: "human" | "rgb_array"
  max_steps: 1000
  seed: 42
  num_runs: 3

agent:
  name: "agent_name"
  model_name: "claude-3-5-sonnet-latest"
  cache_dir: "cache/game_name"
  reasoning_effort: "high" | "medium" | "low"
  token_limit: 100000
  harness: true | false
  observation_mode: "vision" | "text" | "both"
  
  modules:
    base_module:
    perception_module:
    memory_module:
      max_memory: 10
    reasoning_module:
```

**`module_prompts.json`**: Contains system prompts and user prompts for each module type

**`game_env_config.json`**: Game-specific environment configuration
```json
{
    "env_init_kwargs": { "size": 4, "max_pow": 16 },
    "action_mapping": { "up": 0, "right": 1, "down": 2, "left": 3 },
    "render_mode_gym_make": "human",
    "max_unchanged_steps_for_termination": 10
}
```

## 3. Execution System

### 3.1 Main Runner (`lmgame-bench/custom_runner.py`)

Primary execution script that:
- Parses command-line arguments and YAML configurations
- Creates game environments and agents
- Runs multiple episodes with comprehensive logging
- Supports all implemented games (custom and retro)
- Generates run summaries and performance metrics

**Usage:**
```bash
python lmgame-bench/custom_runner.py --game_name tetris --model_name claude-3-haiku-20240307 --observation_mode vision --harness true --num_runs 5
```

**Key Arguments:**
- `--game_name`: Game to play (twenty_forty_eight, sokoban, tetris, etc.)
- `--model_name`: LLM model to use
- `--harness`: Enable/disable perception-memory-reasoning pipeline
- `--observation_mode`: "vision", "text", or "both"
- `--num_runs`: Number of episodes to run
- `--vllm_url`, `--modal_url`: Custom inference endpoints

### 3.2 Game Support

**Implemented Games:**
- **custom_01_2048**: 2048 sliding tile puzzle
- **custom_02_sokoban**: Sokoban box-pushing puzzle
- **custom_03_candy_crush**: Tile-matching game
- **custom_04_tetris**: Classic Tetris via tetris-gymnasium
- **retro_01_super_mario_bros**: Super Mario Bros via stable-retro
- **retro_02_ace_attorney**: Ace Attorney visual novel
- **retro_03_1942**: 1942 shoot-em-up arcade game

## 4. API and Model Support

### 4.1 Supported Providers (`tools/serving/`)

**`APIManager` supports:**
- **Anthropic**: Claude models with vision support
- **OpenAI**: GPT models with vision support
- **Google**: Gemini models
- **Together AI**: Open source models
- **vLLM**: Self-hosted inference
- **Modal**: Cloud deployment

**Features:**
- Automatic cost tracking and calculation
- Vision + text multimodal support
- Token limit management
- Reasoning effort control
- Retry logic and error handling

## 5. Data Flow and Logging

### 5.1 Episode Logging
Each episode generates:
- `episode_XXX_log.jsonl`: Step-by-step action log
- `observations/`: Saved game state images
- `<module_name>.json`: Module-specific logs
- `gym_run_summary.json`: Final run statistics

### 5.2 Observation Flow
1. **Game Environment** generates raw observation
2. **GymEnvAdapter** creates standardized `Observation` object
3. **Agent** processes observation through modules:
   - **Harness mode**: Perception → Memory → Reasoning → Action
   - **Non-harness mode**: BaseModule → Action
4. **Action** is mapped back to environment action space

## 6. Evaluation and Analysis (`eval/`)

### 6.1 Analysis Tools
- **Jupyter notebook pipeline**: Comprehensive analysis of agent performance
- **Video generation**: Create videos from episode logs
- **Replay utilities**: Replay games from logs
- **Performance metrics**: Statistical analysis across runs

### 6.2 Benchmarking
- Batch evaluation scripts for systematic testing
- Cross-model performance comparison
- Game-specific performance metrics

## 7. Development Guidelines

### 7.1 Adding New Games
1. Create game-specific environment wrapper in `gamingagent/envs/custom_XX_gamename/`
2. Implement environment class inheriting from base patterns
3. Add `game_env_config.json` with action mappings
4. Create configuration directory in `gamingagent/configs/custom_XX_gamename/`
5. Add game support to `lmgame-bench/custom_runner.py`

### 7.2 Adding New Modules
1. Inherit from `CoreModule` in `gamingagent/modules/`
2. Implement required abstract methods (`_parse_response`)
3. Add module initialization logic to `BaseAgent`
4. Create module-specific prompts in configuration files

### 7.3 Model Integration
1. Add provider support in `tools/serving/api_providers.py`
2. Update `APIManager` to handle new provider
3. Add model pricing information to `model_prices.json`
4. Test with existing games to ensure compatibility

## 8. Key Dependencies

**Core Requirements:**
- `anthropic>=0.49.0`: Claude API access
- `openai>=1.65.4`: GPT API access
- `google-generativeai>=0.8.4`: Gemini API access
- `gymnasium`: Modern RL environment interface
- `stable-retro`: Classic game emulation
- `pygame>=2.6.1`: Graphics and game rendering
- `pyyaml>=6.0.2`: Configuration file parsing

**Game-Specific:**
- `gymnasium_2048`: 2048 game environment
- `tile_match_gym`: Tile matching games (Candy Crush)
- Additional retro game ROMs (user-provided)

## 9. Performance and Scaling

### 9.1 Caching Strategy
- Agent cache directories organized by model and timestamp
- Observation images saved for vision-based replay
- Module logs enable detailed debugging and analysis

### 9.2 Resource Management
- Token limit enforcement per API call
- Cost tracking across runs
- Background/foreground mode support for long runs

This development map reflects the current state of the GamingAgent repository as of the latest codebase analysis.
