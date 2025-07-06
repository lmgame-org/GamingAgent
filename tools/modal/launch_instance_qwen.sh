#!/bin/bash

# === CONFIGURATION ===
export N_GPU=2
export GPU_TYPE="H100"
export MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
export MODEL_REVISION="a09a354"
export API_KEY="DUMMY_TOKEN"
export VLLM_PORT=8000
export HF_CACHE_VOL="huggingface-cache"
export VLLM_CACHE_VOL="vllm-cache"
export MINUTES=60
export HF_TOKEN=""

# === DEPLOY MODAL INSTANCE ===
modal deploy serve_instance_qwen.py