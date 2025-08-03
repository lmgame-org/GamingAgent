#!/bin/bash

# === CONFIGURATION ===
export N_GPU=8
export GPU_TYPE="H100"
export MODEL_NAME="Qwen/Qwen3-235B-A22B-Instruct-2507"
export API_KEY="DUMMY_TOKEN"
export VLLM_PORT=8000
export HF_CACHE_VOL="huggingface-cache"
export VLLM_CACHE_VOL="vllm-cache"
export MINUTES=60
export HF_TOKEN="your_huggingface_token"  # Replace with your actual Hugging Face token

# === DEPLOY MODAL INSTANCE ===
modal deploy serve_instance_qwen.py