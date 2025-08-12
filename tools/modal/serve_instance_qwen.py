import modal
import os

# --------
# Get config from environment variables
# --------
def get_env_args():
    return {
        "gpus": int(os.environ.get("N_GPU", 8)),
        "gpu_type": os.environ.get("GPU_TYPE", "H100"),
        "model": os.environ.get("MODEL_NAME", "Qwen/Qwen3-235B-A22B-Instruct-2507"),
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
        "vllm==0.9.2",
        "huggingface_hub[hf_transfer]",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "VLLM_USE_V1": "1",
        "HF_TOKEN": args["hf_token"],
        "HUGGINGFACE_HUB_TOKEN": args["hf_token"],
    })
)

hf_cache_vol = modal.Volume.from_name(args["hf_cache_vol"], create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name(args["vllm_cache_vol"], create_if_missing=True)

app = modal.App("vllm-serving-qwen3-2507-it-8h100")
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
    vllm_cmd = (
        f"vllm serve "
        f"--uvicorn-log-level=info "
        f"{args['model']} "
        f"--enable-chunked-prefill "
        f"--tensor_parallel_size {args['gpus']} "
        f"--host 0.0.0.0 "
        f"--port {args['port']} "
        f"--api-key {os.environ['LMGAME_SECRET']} "
        f"--max-model-len 262144 "
        f"--gpu-memory-utilization 0.95"
    )

    full_cmd = f"{vllm_cmd}"

    print("Running merged command:", full_cmd)
    proc = subprocess.Popen(full_cmd, shell=True)