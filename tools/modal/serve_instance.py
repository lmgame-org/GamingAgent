import modal
import os

# --------
# Get config from environment variables
# --------
def get_env_args():
    return {
        "gpus": int(os.environ.get("N_GPU", 8)),
        "gpu_type": os.environ.get("GPU_TYPE", "H100"),
        "model": os.environ.get("MODEL_NAME", "google/gemma-3-27b-it"),
        "revision": os.environ.get("MODEL_REVISION", "005ad3404e59d6023443cb575daa05336842228a"),
        "api_key": os.environ.get("API_KEY", "DUMMY_TOKEN"),
        "port": int(os.environ.get("VLLM_PORT", 8000)),
        "hf_cache_vol": os.environ.get("HF_CACHE_VOL", "huggingface-cache"),
        "vllm_cache_vol": os.environ.get("VLLM_CACHE_VOL", "vllm-cache"),
        "minutes": int(os.environ.get("MINUTES", 60)),
        "hf_token": str(os.environ.get("HF_TOKEN", "your_huggingface_token")),
    }

args = get_env_args()

# --------
# Modal image definition
# --------
vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm==0.8.2",
        "huggingface_hub[hf_transfer]==0.32.0",
        extra_index_url="https://flashinfer.ai/whl/cu124/torch2.5",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_USE_V1": "1"})
)

hf_cache_vol = modal.Volume.from_name(args["hf_cache_vol"], create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name(args["vllm_cache_vol"], create_if_missing=True)

app = modal.App("vllm-serving-engine-gemma3-27b-it-8h100-test")
@app.function(
    image=vllm_image,
    gpu=f"{args['gpu_type']}:{args['gpus']}",
    scaledown_window=60 * args['minutes'],
    timeout=120 * args['minutes'],
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    secrets=[modal.Secret.from_name("lmgame-secret")]
)
@modal.concurrent(max_inputs=100)
@modal.web_server(port=args['port'], startup_timeout=120 * args['minutes'])
def serve():
    import subprocess

    hf_cmd = f"huggingface-cli login --token {args['hf_token']}"
    vllm_cmd = (
        f"vllm serve "
        f"--uvicorn-log-level=info "
        f"{args['model']} "
        f"--revision {args['revision']} "
        f"--enable-chunked-prefill "
        f"--tensor_parallel_size {args['gpus']} "
        f"--host 0.0.0.0 "
        f"--port {args['port']} "
        f"--api-key {os.environ['LMGAME_SECRET']}"
    )

    full_cmd = f"{hf_cmd} && {vllm_cmd}"

    print("Running merged command:", full_cmd)
    proc = subprocess.Popen(full_cmd, shell=True)