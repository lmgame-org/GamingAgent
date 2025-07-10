#!/bin/bash

# === CONFIGURATION ===
export N_GPU=1
export GPU_TYPE="H100"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export MODEL_REVISION="main"
export API_KEY="DUMMY_TOKEN"
export VLLM_PORT=8000
export HF_CACHE_VOL="huggingface-cache"
export VLLM_CACHE_VOL="vllm-cache"
export MINUTES=60
export HF_TOKEN=""  # Replace with your actual Hugging Face token

# === DEPLOY MODAL INSTANCE ===
modal deploy serve_instance_qwen.py