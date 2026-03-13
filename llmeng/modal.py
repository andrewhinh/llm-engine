import os

import modal

MINUTES = 60  # seconds

NNODES = int(os.environ.get("NNODES", 1))
if not (1 <= NNODES <= 4):
    raise ValueError(f"nnodes must be >= 1 and <= 4, got {NNODES}.")
N_GPU = int(os.environ.get("N_GPU", "1"))
if not (1 <= N_GPU <= 8):
    raise ValueError(f"n_gpu must be >= 1 and <= 8, got {N_GPU}.")
GPU_TYPE = os.environ.get("GPU_TYPE", "a100").strip().lower()
RDMA = os.environ.get("RDMA", "0").lower() == "1"

MODEL_DIR = "/models"
MODEL_VOLUME_NAME = "llm-engine-models"
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
app = modal.App("llm-engine", volumes={MODEL_DIR: model_volume})

DEFAULT_NVSHMEM_INC = "/usr/include/nvshmem_12"
DEFAULT_NVSHMEM_LIB = "/usr/lib/x86_64-linux-gnu/nvshmem/12"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu24.04",
        add_python="3.12",
    )
    .apt_install(
        "git",
        "g++",
        "ibverbs-providers",
        "libnuma1",
        "libibverbs1",
        "libnvshmem3-cuda-12",
        "libnvshmem3-dev-cuda-12",
        "libnvshmem3-static-cuda-12",
        "rdma-core",
    )
    .uv_pip_install(
        "torch==2.9.1+cu128",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .uv_pip_install(
        "accelerate",
        "apache-tvm-ffi>=0.1.4",
        "fastapi",
        "flashinfer-python>=0.5.3",
        "huggingface_hub[hf_transfer]",
        "modelscope",
        "msgpack",
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
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "NVSHMEM_INC": DEFAULT_NVSHMEM_INC,
            "NVSHMEM_LIB": DEFAULT_NVSHMEM_LIB,
            "LD_LIBRARY_PATH": f"{DEFAULT_NVSHMEM_LIB}:/usr/local/cuda/lib64",
        }
    )
    .add_local_python_source("llmeng")
    .add_local_dir("llmeng/kernel/csrc", remote_path="/root/llmeng/kernel/csrc")
)
