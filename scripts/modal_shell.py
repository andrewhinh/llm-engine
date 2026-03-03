import os
import subprocess
from pathlib import Path

import modal

app = modal.App("llm-engine-dev-shell")

model_volume = modal.Volume.from_name("llm-engine-model", create_if_missing=True)
cargo_cache_volume = modal.Volume.from_name(
    "llm-engine-cargo-cache", create_if_missing=True
)
volumes = {
    "/huggingface": model_volume,
    "/root/.cargo/registry": cargo_cache_volume,
}


def download_model(repo_id, local_dir):
    import os

    from huggingface_hub import snapshot_download

    if os.path.exists(local_dir) and os.listdir(local_dir):
        print(f"Model already exists at {local_dir}, skipping download")
        return
    snapshot_download(repo_id=repo_id, local_dir=local_dir)


image = (
    modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install(
        "git",
        "build-essential",
        "pkg-config",
        "wget",
        "openssl",
        "libssl-dev",
        "ca-certificates",
        "curl",
    )
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable"
    )
    .env(
        {
            "PATH": "/root/.cargo/bin:/usr/local/cuda/bin:${PATH}",
            "CUDA_ROOT": "/usr/local/cuda",
            "CUDA_PATH": "/usr/local/cuda",
            "CUDA_TOOLKIT_ROOT_DIR": "/usr/local/cuda",
            "CARGO_HOME": "/root/.cargo",
        }
    )
    .uv_pip_install("huggingface-hub==1.5.0")
    .run_function(
        download_model,
        volumes=volumes,
        args=("Qwen/Qwen3-0.6B", "/huggingface/Qwen3-0.6B"),
    )
    .run_commands("ln -sf /huggingface ~/huggingface")
    .add_local_dir(Path(__file__).parent.parent, "/", ignore=["target", ".git"])
)


def _gpu_from_env() -> str | None:
    gpu_type = os.environ.get("GPU_TYPE", "H100").strip()
    gpu_count = os.environ.get("GPU_COUNT", "1").strip()
    return f"{gpu_type}:{gpu_count}"


gpu_spec = _gpu_from_env()
function_kwargs = {
    "image": image,
    "volumes": volumes,
    "timeout": int(os.environ.get("MODAL_TIMEOUT_SECONDS", "3600")),
}
if gpu_spec is not None:
    function_kwargs["gpu"] = gpu_spec


@app.function(**function_kwargs)
def dev_shell() -> None:
    pass


@app.function(**function_kwargs)
def run_cmd(command: str) -> None:
    subprocess.run(["bash", "-lc", command], check=True, cwd="/")


@app.function(**function_kwargs)
def run_verify() -> None:
    subprocess.run(["bash", "/scripts/modal_verify.sh"], check=True, cwd="/")
