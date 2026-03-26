import os

import modal

SECONDS = 1
MINUTES = 60 * SECONDS
HOURS = 60 * MINUTES

# Resources
NNODES = int(os.environ.get("NNODES", 1))
if not (1 <= NNODES <= 4):
    raise ValueError(f"nnodes must be >= 1 and <= 4, got {NNODES}.")
N_GPU = int(os.environ.get("N_GPU", "1"))
if not (1 <= N_GPU <= 8):
    raise ValueError(f"n_gpu must be >= 1 and <= 8, got {N_GPU}.")
GPU_TYPE = os.environ.get("GPU_TYPE", "a100").strip().lower()
RDMA = os.environ.get("RDMA", "0").lower() == "1"


def get_startup_metrics_dict() -> modal.Dict:
    return modal.Dict.from_name(
        "llm-engine-startup-metrics",
        create_if_missing=True,
    )


# Secrets
hf_secret = modal.Secret.from_name("huggingface-secret")

# Volumes
HF_CACHE_PATH = "/root/.cache/huggingface"

hf_cache_volume = modal.Volume.from_name("llm-engine-hf-cache", create_if_missing=True)


app = modal.App("llm-engine", volumes={HF_CACHE_PATH: hf_cache_volume})
NVIDIA_NCCL_LIB = "/usr/local/lib/python3.12/site-packages/nvidia/nccl/lib"


def _image_factory(
    cuda_base: str,
    engine: str = "llmeng",
    extra_packages: tuple[str, ...] = (),
) -> modal.Image:
    return (
        modal.Image.from_registry(
            cuda_base,
            add_python="3.12",
        )
        .apt_install(
            "git",
            "g++",
            "ibverbs-providers",
            "libnuma1",
            "libibverbs1",
            "rdma-core",
        )
        # install PyTorch first so other dependencies are forced to use it
        .uv_pip_install(
            "torch==2.9.1+cu128",
            index_url="https://download.pytorch.org/whl/cu128",
        )
        .uv_pip_install(
            "accelerate",
            "cuda-python",
            "fastapi",
            "flashinfer-python>=0.5.3",
            "huggingface_hub[hf_transfer]",
            "ml-dtypes",
            "modelscope",
            "msgpack",
            "nvidia-cutlass-dsl",
            "openai",
            "prompt_toolkit",
            "pyarrow",
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "pyzmq",
            "quack-kernels",
            "setuptools",
            "sgl_kernel>=0.3.17.post1",
            "transformers>=4.56.0,<=4.57.3",
            "uvicorn",
            "wheel",
            *extra_packages,
        )
        .run_commands(
            f"ln -sf {NVIDIA_NCCL_LIB}/libnccl.so.2 /usr/local/lib/libnccl.so || true"
        )
        .env(
            {
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
                "LD_LIBRARY_PATH": f"{NVIDIA_NCCL_LIB}:/usr/local/cuda/lib64",
                "LIBRARY_PATH": f"{NVIDIA_NCCL_LIB}:/usr/local/cuda/lib64",
            }
        )
        .add_local_python_source("llmeng", ignore=["kernel*"])
        .add_local_dir(
            "llmeng/kernel_new" if engine == "llmeng" else "llmeng/kernel_old",
            "/root/llmeng/kernel",
        )
    )


def get_runtime_image(
    engine: str = "llmeng",
    gpu_type: str = GPU_TYPE,
    extra_packages: tuple[str, ...] = (),
) -> modal.Image:
    normalized_gpu_type = gpu_type.strip().lower()
    if normalized_gpu_type in {"b200", "b200+", "rtx-pro-6000"}:
        return _image_factory(
            "nvidia/cuda:13.1.0-devel-ubuntu24.04",
            engine,
            (*extra_packages, "numba-cuda==0.15.1", "cuda-tile", "nccl4py[cu13]"),
        )
    return _image_factory(
        "nvidia/cuda:12.8.1-devel-ubuntu24.04",
        engine,
        (*extra_packages, "numba-cuda", "nccl4py[cu12]"),
    )
