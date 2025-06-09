#!/bin/bash

# === CONFIGURATION ===
export N_GPU=8
export GPU_TYPE="H100"
export MODEL_NAME="google/gemma-3-27b-it"
export MODEL_REVISION="main"
export API_KEY="DUMMY_TOKEN"
export VLLM_PORT=8000
export HF_CACHE_VOL="huggingface-cache"
export VLLM_CACHE_VOL="vllm-cache"
export MINUTES=60
export HF_TOKEN="your_huggingface_token"  # Replace with your actual Hugging Face token

# === DEPLOY MODAL INSTANCE ===
modal deploy serve_instance.py