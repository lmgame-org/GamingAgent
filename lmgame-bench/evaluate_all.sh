#!/bin/bash

GAME_NAMES="sokoban,tetris,candy_crush,twenty_forty_eight,super_mario_bros,ace_attorney"
HARNESS_MODE="both"
MAX_PROCS=12

MODELS=(
  # OpenAI
  "o4-mini"
  "o3-mini"
  "o3"
  "o1"
  "gpt-4o"
  "gpt-4o-mini"

  # Anthropic
  "claude-4-opus"
  "claude-4-sonnet--thinking"
  "claude-3-7-sonnet--thinking"
  "claude-3-5-haiku"
  "claude-3-5-sonnet"

  # Gemini
  "gemini-2.5-pro"
  "gemini-2.5-flash"
  "gemini-2.0-flash-thinking-exp"
  "gemini-2.0-pro"
  "gemini-2.0-flash"
  "gemini-1.5-pro"

  # xAI
  "grok-3-mini"

  # Deepseek
  "deepseek-reasoner"
  "deepseek-chat"
)

mkdir -p logs

for MODEL in "${MODELS[@]}"; do
  BASE_MODEL=$(echo $MODEL | cut -d'--' -f1)
  EXTRA_ARGS=""

  echo "Launching: $BASE_MODEL. outputs will be logged to logs/{game_name}_${BASE_MODEL}_{timestamp}.log upon finish."
  python3 run.py \
    --model_name "$BASE_MODEL" \
    --game_names "$GAME_NAMES" \
    --harness_mode "$HARNESS_MODE" \
    --max_parallel_procs "$MAX_PROCS" \
    $EXTRA_ARGS &
done

wait
