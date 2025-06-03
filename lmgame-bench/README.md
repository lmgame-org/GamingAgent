<p align="center">
<img src="../assets/img/logo.png" alt="lmgame-bench" width="100%" align="center">
</p>

<div align="center"> <h1>LMGame Bench</h1> </div> 

## Setup

### Gym and Retro Interface

##### Gymnasium Envrionments

We standardize our gaming envrionment interfaces following [Gymnasium](https://github.com/Farama-Foundation/Gymnasium).

Currently our evaluation suite composes of the following games using gym envrionments:

- Sokoban
- Tetris
- 2048
- Candy Crush

all runnable out-of-the-box with no additional setup.

##### Retro Envrionments

Gym [Retro](https://github.com/Farama-Foundation/stable-retro) is a library that enables classic video game emulation through a wide range of supported systems, providing a standardized interface via Gymnasium.

To run classical games implemented on Retro, you need to legally obtain the games files and import them with [this instruction](https://retro.readthedocs.io/en/latest/getting_started.html#importing-roms):

```
python3 -m retro.import /path/to/your/ROMs/directory/
```

Currently, our evaluation suite includes the following games from Retro environments:
- Super Mario Bros 1985


We have also integrated additional Retro environments that are not included in stable-retro.
For these games, no `retro.import` is required. To enable the envrionments, simply place the ROM file into the designated directory.

For example, for Ace Attorney: Phoenix Wright, place the ROM file into:
```
gamingagent/envs/retro_02_ace_attorney/AceAttorney-GbAdvance
```

Additional games we integrated:
- Ace Attorney: Phoenix Wright

### UI-only Interface

Coming Soon!


## Single-Model Performance

Launch multiple evaluation instances (in parallel) for a model on different games with the following commands:

```
python3 lmgame-bench/run.py --model_name {model_name} --game_names {list_of_games} --harness_mode false
```

To multiple models in parallel, run the following script:

```
bash lmgame-bench/evaluate_all.sh
```

## Agentic Performance

Evaluate a model's performance in gaming agent (with gaming harness support), run the following command:

```
python3 lmgame-bench/run.py --model_name {model_name} --game_names {list_of_games} --harness_mode true
```

##### Command options

```
--harness_mode: if to evaluate the model using agentic workflow, choice of ["true", "false", "both"].
--max_parallel_procs: max parallel instances to run.
--game_names: list of games to evaluated on, e.g. "sokoban,tetris,candy_crush,twenty_forty_eight".

Currently supported games:
- sokoban
- tetris
- candy_crush
- twenty_forty_eight
- super_mario_bros
- ace_attorney
```

## Customize Your Settings

`run.py` launches multiple instances of `custom_runner.py`. To run single model in a single game setting, run `python3 lmgame-bench/custom_runner.py --game_name {game_name} --model_name {model_name} --config_root_dir {path_to_gaming_agent_config} (--harness)`. 

Adjust gaming-agent related configurations in `gamingagent/configs/{game_env_dir}/config.yaml`. 

Propmts can be found in `gamingagent/configs/{game_env_dir}/module_prompts.json`.
